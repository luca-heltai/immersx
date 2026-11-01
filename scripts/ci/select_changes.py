#!/usr/bin/env python3
"""Select expensive PR jobs; component suggestions are observation-only."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


DOC_FILES = {"README.md", "CHANGELOG.md", "doc/conf.py", "doc/Doxyfile",
             "doc/requirements.txt"}
DOC_SUFFIXES = {".md", ".rst", ".bib", ".css", ".js", ".png", ".jpg", ".jpeg",
                ".svg", ".gif", ".webp", ".ico", ".woff", ".woff2"}


def documentation_path(name):
    """Only known documentation inputs can exclude numerical jobs."""
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts and
            (name in DOC_FILES or
             (name.startswith("doc/") and path.suffix in DOC_SUFFIXES)))


def parse_changes(data):
    """Read git --name-status -z, retaining both sides of copies and renames."""
    if not data:
        return []
    fields = data.split(b"\0")
    if fields.pop() != b"":
        raise ValueError("unterminated git diff")
    changes = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii")
        index += 1
        if not re.fullmatch(r"(?:[ADMT]|[RC](?:100|[0-9]{1,2}))", status):
            raise ValueError("unknown git status")
        count = 2 if status[0] in "RC" else 1
        if index + count > len(fields):
            raise ValueError("missing git path")
        paths = [os.fsdecode(name) for name in fields[index:index + count]]
        index += count
        if any(not name or PurePosixPath(name).is_absolute() or
               ".." in PurePosixPath(name).parts for name in paths):
            raise ValueError("invalid repository path")
        changes.append({"status": status, "paths": paths})
    return changes


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def changed_files(repo, base, head):
    """Compare the PR head with its actual merge base, including stacked PRs."""
    if not base or not head:
        raise ValueError("base and head are required")
    base_sha = git(repo, "rev-parse", "--verify", "--end-of-options",
                   base + "^{commit}").decode("ascii").strip()
    head_sha = git(repo, "rev-parse", "--verify", "--end-of-options",
                   head + "^{commit}").decode("ascii").strip()
    merge_bases = git(repo, "merge-base", "--all", base_sha, head_sha).splitlines()
    if len(merge_bases) != 1:
        raise ValueError("a unique merge base is required")
    merge_base = merge_bases[0].decode("ascii")
    changes = parse_changes(git(repo, "diff", "--name-status", "--find-renames",
                                "-z", merge_base, head_sha, "--"))
    return merge_base, head_sha, changes


def observed_components(paths):
    """Record ownership hints, never an assertion that other tests are unaffected.

    Shared infrastructure and runtime inputs deliberately request all components.
    Actual exclusion needs compiler/link dependencies and fixture ownership first.
    """
    components = set()
    for name in paths:
        if documentation_path(name):
            continue
        path = PurePosixPath(name)
        if path.suffix not in {".h", ".cc", ".cpp"}:
            return ["all"]
        if name.startswith(("include/immersx/physics/", "source/", "apps/", "gtests/")):
            stem = path.stem.removeprefix("app_")
            if stem.startswith("fiber_reinforced_"):
                components.update(("fiber-reinforcement", "elasticity"))
            elif stem.startswith(("elastic", "lifted_elastic", "coupled_elastic")):
                components.add("elasticity")
            elif stem.startswith(("poisson", "reduced_poisson", "lifted_poisson", "laplacian")):
                components.add("poisson")
            elif stem.startswith("coupled_poisson_elasticity"):
                components.update(("poisson", "elasticity"))
            elif stem.startswith("coupled_poisson"):
                components.add("poisson")
            elif stem.startswith("navier_stokes"):
                components.add("navier-stokes")
            elif stem.startswith("metric_flow_x"):
                components.add("metric-flow-x")
            else:
                return ["all"]
        elif name.startswith("coral/"):
            components.add("coral")
        else:
            return ["all"]
    return sorted(components)


def select(repo, event, base=None, head=None, force_full=False):
    plan = {"schema_version": 1, "mode": "full-pr", "docs_only": False,
            "run_numeric": True, "run_coral": True, "run_benchmarks": True,
            "full_suite": bool(force_full or event != "pull_request"),
            "component_selection": "observation-only", "observed_components": ["all"],
            "changes": [], "reasons": []}
    if plan["full_suite"]:
        plan["mode"] = "full-suite"
        plan["reasons"] = ["forced-full" if force_full else "non-pr-event"]
        return plan
    try:
        merge_base, head_sha, changes = changed_files(repo, base, head)
    except (OSError, subprocess.CalledProcessError, ValueError, UnicodeError):
        plan["reasons"] = ["diff-unavailable"]
        return plan
    plan.update(merge_base=merge_base, head=head_sha, changes=changes)
    paths = sorted({name for change in changes for name in change["paths"]})
    if not paths:
        plan["reasons"] = ["empty-diff"]
    elif all(documentation_path(name) for name in paths):
        plan.update(mode="docs-only", docs_only=True, run_numeric=False,
                    run_coral=False, run_benchmarks=False, observed_components=[])
        plan["reasons"] = ["known-documentation-inputs-only"]
    else:
        plan["observed_components"] = observed_components(paths)
        plan["reasons"] = ["non-documentation-changes",
                           "component-exclusions-not-enabled"]
    return plan


def write_github_outputs(plan, output=None, summary=None):
    """Only fixed keys and controlled values enter the Actions output protocol."""
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            for key in ("run_numeric", "run_coral", "run_benchmarks", "full_suite", "docs_only"):
                stream.write(f"{key}={str(plan[key]).lower()}\n")
            stream.write(f"mode={plan['mode']}\n")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("## CI selection\n\n")
            stream.write("Component suggestions are recorded only; C++ test exclusions are disabled.\n\n")
            stream.write("```json\n" + json.dumps(plan, indent=2, ensure_ascii=True) + "\n```\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = select(args.repo, args.event, args.base, args.head, args.force_full)
    serialized = json.dumps(plan, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    write_github_outputs(plan, os.environ.get("GITHUB_OUTPUT"),
                         os.environ.get("GITHUB_STEP_SUMMARY"))
    print(serialized, end="")


if __name__ == "__main__":
    main()
