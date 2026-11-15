#!/usr/bin/env python3
"""Validate per-PR fragments, render the changelog, or consolidate a release."""

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import subprocess

DIRECTORIES = {
    "added": "Added",
    "changed": "Changed",
    "fix": "Fixed",
    "removed": "Removed",
    "docs": "Documentation",
    "tutorials": "Tutorial",
    "build": "Build and packaging",
    "testing": "Testing and CI",
}
CATEGORIES = tuple(DIRECTORIES.values())
PR_URL = "https://github.com/luca-heltai/immersx/pull/"
FRAGMENT_NAME = re.compile(r"([0-9]{4})_([0-9]{2})_([0-9]{2})_[a-z0-9]+(?:_[a-z0-9]+)*\.md\Z")
PR_LINE = re.compile(r"PR: #([1-9][0-9]*)\Z")
VERSION = re.compile(r"v?[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?\Z")


@dataclass(frozen=True)
class Fragment:
    path: Path
    pr: int
    started: date
    category: str
    entries: list
    original: str


def parse_fragment(path, category):
    name = FRAGMENT_NAME.fullmatch(path.name)
    if path.is_symlink() or not path.is_file() or not name:
        raise ValueError(f"{path}: use YYYY_MM_DD_short_description.md")
    try:
        started = date(*(int(value) for value in name.groups()))
    except ValueError as error:
        raise ValueError(f"{path}: invalid activity start date") from error
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    match = PR_LINE.fullmatch(lines[0]) if lines else None
    if not match:
        raise ValueError(f"{path}: first line must be PR: #<positive pull request number>")
    entries = []
    current = []

    def finish():
        if current:
            entry = "\n".join(current).rstrip()
            if not entry[2:].strip() or entry[2:].startswith("[PR #"):
                raise ValueError(f"{path}: write a description without the generated PR link")
            if entry in entries:
                raise ValueError(f"{path}: duplicate entry in {category}")
            entries.append(entry)
            current.clear()

    for line in lines[1:]:
        if line.startswith("- "):
            finish()
            current.append(line)
        elif not line.strip():
            if current:
                current.append(line)
        elif line.startswith("  ") and current:
            current.append(line)
        else:
            raise ValueError(f"{path}: expected a bullet or indented continuation: {line!r}")
    finish()
    if not entries:
        raise ValueError(f"{path}: include at least one nonempty bullet")
    return Fragment(path, int(match.group(1)), started, category, entries, text)


def fragments(repo):
    directory = repo / "changelog"
    result = []
    if directory.is_symlink():
        raise ValueError(f"{directory}: the fragment directory must not be a symlink")
    if directory.exists():
        for category_directory in sorted(directory.iterdir()):
            if (category_directory.is_symlink() or not category_directory.is_dir() or
                    category_directory.name not in DIRECTORIES):
                raise ValueError(f"{category_directory}: use a category directory from "
                                 + ", ".join(DIRECTORIES))
            category = DIRECTORIES[category_directory.name]
            for path in sorted(category_directory.iterdir()):
                result.append(parse_fragment(path, category))
    seen = set()
    for fragment in result:
        key = fragment.pr, fragment.category
        if key in seen:
            raise ValueError(f"More than one fragment for PR #{fragment.pr} in {fragment.category}")
        seen.add(key)

    return sorted(result, key=lambda fragment: (-fragment.started.toordinal(), fragment.path.name))


def read_history(repo):
    path = repo / "CHANGELOG.md"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: history must be a regular file")
    return path.read_text(encoding="utf-8")


def split_history(text):
    markers = list(re.finditer(r"(?m)^## \[Unreleased\]\n", text))
    if len(markers) != 1:
        raise ValueError("CHANGELOG.md must contain exactly one ## [Unreleased] heading")
    marker = markers[0]
    prefix, remaining = text[:marker.start()], text[marker.end():]
    match = re.search(r"(?m)^## ", remaining)
    if match:
        return prefix, remaining[:match.start()], remaining[match.start():]
    return prefix, remaining, ""


def render(repo):
    """Add fragments to Unreleased while retaining the historical text."""
    repo = Path(repo)
    history = read_history(repo)
    prefix, pending, released = split_history(history)
    additions = {category: [] for category in CATEGORIES}
    for fragment in fragments(repo):
        if re.search(r"\[PR #" + str(fragment.pr) + r"\]\(", history):
            raise ValueError(f"PR #{fragment.pr} already has an entry in CHANGELOG.md")
        additions[fragment.category].extend(
            f"- [PR #{fragment.pr}]({PR_URL}{fragment.pr}) {entry[2:]}"
            for entry in fragment.entries)
    for category in CATEGORIES:
        if not additions[category]:
            continue
        block = "\n\n".join(additions[category]) + "\n\n"
        heading = f"### {category}\n"
        if pending.count(heading) > 1:
            raise ValueError(f"Repeated {category} category in CHANGELOG.md")
        if heading in pending:
            before, after = pending.split(heading, 1)
            pending = before + heading + "\n" + block + after.lstrip("\n")
        else:
            # Older histories can omit empty categories. Insert at their
            # canonical position without changing existing category contents.
            later = [f"### {name}\n" for name in CATEGORIES[CATEGORIES.index(category) + 1:]]
            positions = [pending.index(name) for name in later if name in pending]
            at = min(positions) if positions else len(pending)
            pending = pending[:at].rstrip("\n") + "\n\n" + heading + "\n" + block + pending[at:]
    return prefix + "## [Unreleased]\n" + pending + released


def check(repo, pr=None, added=None):
    items = fragments(repo)
    render(repo)
    if pr is not None:
        if added is None and not any(fragment.pr == pr for fragment in items):
            raise ValueError(f"No changelog fragment with PR: #{pr}")
        for fragment in items:
            if (added is not None and fragment.path.relative_to(repo).as_posix() in added
                    and fragment.pr != pr):
                raise ValueError(f"{fragment.path}: new fragments must reference PR: #{pr}")
    return items


def release(repo, version, release_date, write=False):
    """Return a release preview; mutate files only with explicit write=True."""
    repo = Path(repo)
    if not VERSION.fullmatch(version):
        raise ValueError("Version must be a semantic version, for example 0.5.0")
    if date.fromisoformat(release_date).isoformat() != release_date:
        raise ValueError("Release date must use YYYY-MM-DD")
    original = read_history(repo)
    normalized = version.removeprefix("v")
    versions = re.findall(r"(?m)^## \[([^]]+)\]", original)
    if normalized in [value.removeprefix("v") for value in versions]:
        raise ValueError(f"Release {version} already exists")
    items = check(repo)
    prefix, pending, released = split_history(render(repo))
    if not re.search(r"(?m)^- ", pending):
        raise ValueError("No unreleased entries to consolidate")
    empty = "\n" + "\n".join(f"### {category}\n" for category in CATEGORIES) + "\n"
    result = (prefix + "## [Unreleased]\n" + empty +
              f"## [{version}] - {release_date}\n" + pending + released)
    if write:
        if read_history(repo) != original:
            raise ValueError("CHANGELOG.md changed during release preparation")
        for fragment in items:
            if fragment.path.is_symlink() or fragment.path.read_text(encoding="utf-8") != fragment.original:
                raise ValueError(f"{fragment.path} changed during release preparation")
        target = repo / "CHANGELOG.md"
        temporary = target.with_suffix(".md.tmp")
        # Exclusive creation leaves an existing temporary file untouched.
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(result)
        temporary.replace(target)
        for fragment in items:
            fragment.path.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("check", help="validate fragments and historical ownership")
    validate.add_argument("--pr", type=int, help="check metadata for this PR")
    validate.add_argument("--base", help="base revision for newly added fragment checks")
    validate.add_argument("--head", help="head revision for newly added fragment checks")
    commands.add_parser("render", help="print combined Markdown without changing files")
    consolidate = commands.add_parser("release", help="preview a release; --write also consolidates fragments")
    consolidate.add_argument("--version", required=True)
    consolidate.add_argument("--date", required=True)
    consolidate.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "check":
            added = None
            if args.base or args.head:
                if not (args.base and args.head and args.pr):
                    raise ValueError("Use --base and --head together with --pr")
                names = subprocess.check_output(
                    ["git", "diff", "--no-ext-diff", "--name-only", "--diff-filter=A",
                     "-z", f"{args.base}...{args.head}", "--", "changelog/"], cwd=args.repo)
                added = set(names.decode("utf-8").rstrip("\0").split("\0"))
            items = check(args.repo, args.pr, added)
            print(f"Validated {len(items)} changelog fragment(s)")
        elif args.command == "render":
            print(render(args.repo), end="")
        else:
            print(release(args.repo, args.version, args.date, args.write), end="")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"changelog: {error}\n")


if __name__ == "__main__":
    main()
