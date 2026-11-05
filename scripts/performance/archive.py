#!/usr/bin/env python3
"""Read or fast-forward the separate performance-data branch; never force push."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from history import attempts, load
from pipeline import git, merge

BRANCH = "performance-data"


def checkout(repo, destination):
    url = git(repo, "remote", "get-url", "origin")
    git(destination, "init")
    git(destination, "remote", "add", "origin", url)
    # In Actions the write job's checkout supplies an HTTP extraheader. Do not
    # copy it into artifacts or the archive; use it only in this temporary repo.
    headers = subprocess.run(["git", "-C", str(repo), "config", "--get-regexp",
                              r"http\..*\.extraheader"], text=True, capture_output=True)
    for line in headers.stdout.splitlines():
        key, value = line.split(" ", 1)
        git(destination, "config", key, value)
    refs = git(destination, "ls-remote", "--heads", "origin", "refs/heads/" + BRANCH)
    if refs:
        git(destination, "fetch", "--depth=1", "origin", "refs/heads/" + BRANCH)
        git(destination, "checkout", "-B", BRANCH, "FETCH_HEAD")
    else:
        git(destination, "checkout", "--orphan", BRANCH)
    return bool(refs)


def validate_publication(args):
    """Actions writes require a batch from this trusted master run."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return  # Local recovery requires an operator to verify the original batch.
    if (os.environ.get("GITHUB_REF") != "refs/heads/master" or
            os.environ.get("GITHUB_EVENT_NAME") not in {"push", "schedule", "workflow_dispatch"}):
        raise ValueError("official publication requires a trusted master event")
    plan = load(args.plan)
    sha = os.environ.get("GITHUB_SHA")
    if not sha or plan["source_sha"] != sha or plan["harness_sha"] != sha:
        raise ValueError("batch source or harness does not match the publishing run")
    run = os.environ.get("GITHUB_RUN_ID")
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    for item in attempts(args.incoming):
        if (item["sha"] not in plan["selected"] or item["harness_sha"] != sha or
                item["run"] != run or item["attempt"] != attempt or
                item["id"] != f"{item['sha']}-{run}-{attempt}"):
            raise ValueError("attempt provenance does not match the publishing run")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["read", "publish"])
    p.add_argument("--repo", default=".")
    p.add_argument("--output")
    p.add_argument("--incoming")
    p.add_argument("--plan")
    args = p.parse_args()
    if args.action == "publish":
        validate_publication(args)
        # The archive checkout contains only performance data. Ancestry must
        # be proved against the caller's complete source repository instead.
        if git(args.repo, "rev-parse", "--is-shallow-repository") != "false":
            raise ValueError("publication requires the full source Git history")
    for retry in range(5):
        with tempfile.TemporaryDirectory() as tmp:
            checkout(args.repo, tmp)
            if args.action == "read":
                dest = Path(args.output)
                dest.mkdir(parents=True, exist_ok=True)
                for name in ("baseline.json", "plan.json", "attempts", "raw"):
                    source = Path(tmp) / name
                    if source.is_dir():
                        shutil.copytree(source, dest / name, dirs_exist_ok=True)
                    elif source.is_file():
                        shutil.copyfile(source, dest / name)
                return
            # A competing publisher may have measured a source commit created
            # after this job's checkout. Fetch its ancestry without moving HEAD.
            git(args.repo, "fetch", "--no-tags", "origin", "refs/heads/master")
            args.archive = tmp
            merge(args)
            git(tmp, "config", "user.name", "github-actions[bot]")
            git(tmp, "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
            git(tmp, "add", ".")
            if not git(tmp, "status", "--porcelain"):
                return
            git(tmp, "commit", "-m", "Archive performance attempts")
            # A competing publisher can win. Re-read and merge immutable data,
            # rather than force updating or losing its measurements.
            if subprocess.run(["git", "-C", tmp, "push", "origin", "HEAD:refs/heads/" + BRANCH]).returncode == 0:
                return
    raise RuntimeError("archive publication lost five races; recover from the batch artifact")


if __name__ == "__main__":
    main()
