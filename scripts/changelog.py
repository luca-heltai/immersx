#!/usr/bin/env python3
"""Validate development fragments, render history, or archive an explicit release."""

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
ARCHIVE_NAME = re.compile(r"version_([0-9]{2})_([0-9]{2})_v(.+)\.md\Z")
ARCHIVE_HEADING = re.compile(r"## \[v([^]]+)\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})\Z")
PR_LINE = re.compile(r"PR: #([1-9][0-9]*)\Z")
VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]*\Z")
LINKED_ENTRY = re.compile(r"- \[PR #([1-9][0-9]*)\]\(https://github\.com/luca-heltai/immersx/pull/\1\)\s+\S")


@dataclass(frozen=True)
class Fragment:
    path: Path
    pr: int
    started: date
    category: str
    entries: list
    original: str


@dataclass(frozen=True)
class Archive:
    path: Path
    version: str
    released: date
    prs: set
    original: str


def version_tag(value):
    if not VERSION.fullmatch(value):
        raise ValueError("Use a version tag such as 0.5.0 or v0.5.0, without path separators")
    value = value.removeprefix("v")
    if not value or value.startswith("v") or ".." in value or value.endswith((".", ".lock")):
        raise ValueError("Invalid version tag")
    return value


def iso_date(value):
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


def regular_text(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: expected a regular file")
    return path.read_text(encoding="utf-8")


def bullets(lines, path, linked=False):
    result, current = [], []

    def finish():
        if current:
            entry = "\n".join(current).rstrip()
            if not entry[2:].strip():
                raise ValueError(f"{path}: empty bullet")
            if linked:
                if not LINKED_ENTRY.match(entry):
                    raise ValueError(f"{path}: archived entries need matching numeric PR links")
            elif entry[2:].lstrip().startswith("[PR #"):
                raise ValueError(f"{path}: omit the generated PR link")
            if entry in result:
                raise ValueError(f"{path}: duplicate entry")
            result.append(entry)
            current.clear()

    for line in lines:
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
    if not result:
        raise ValueError(f"{path}: include at least one nonempty bullet")
    return result


def layout(repo):
    directory = repo / "changelog"
    if directory.is_symlink():
        raise ValueError(f"{directory}: the changelog directory must not be a symlink")
    if not directory.exists():
        return [], []
    fragment_paths, archive_paths = [], []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise ValueError(f"{path}: symlinks are not changelog data")
        if path.is_dir() and path.name in DIRECTORIES:
            fragment_paths.extend((child, DIRECTORIES[path.name]) for child in sorted(path.iterdir()))
        elif path.is_file() and ARCHIVE_NAME.fullmatch(path.name):
            archive_paths.append(path)
        else:
            raise ValueError(f"{path}: use a known category directory or version_YY_MM_vTAG.md archive")
    return fragment_paths, archive_paths


def parse_fragment(path, category):
    name = FRAGMENT_NAME.fullmatch(path.name)
    if not name:
        raise ValueError(f"{path}: use YYYY_MM_DD_short_description.md")
    try:
        started = date(*(int(value) for value in name.groups()))
    except ValueError as error:
        raise ValueError(f"{path}: invalid activity start date") from error
    text = regular_text(path)
    lines = text.splitlines()
    match = PR_LINE.fullmatch(lines[0]) if lines else None
    if not match:
        raise ValueError(f"{path}: first line must be PR: #<positive pull request number>")
    return Fragment(path, int(match.group(1)), started, category, bullets(lines[1:], path), text)


def fragments(repo):
    result = [parse_fragment(path, category) for path, category in layout(Path(repo))[0]]
    seen = set()
    for fragment in result:
        key = fragment.pr, fragment.category
        if key in seen:
            raise ValueError(f"More than one fragment for PR #{fragment.pr} in {fragment.category}")
        seen.add(key)
    return sorted(result, key=lambda fragment: (-fragment.started.toordinal(), fragment.path.name))


def parse_archive(path):
    return archive_text(path, regular_text(path))


def archive_text(path, text):
    """Validate both existing archives and a prospective release before writing."""
    lines = text.splitlines()
    heading = ARCHIVE_HEADING.fullmatch(lines[0]) if lines else None
    if not heading:
        raise ValueError(f"{path}: first line must be ## [vTAG] - YYYY-MM-DD")
    version, released = version_tag("v" + heading.group(1)), iso_date(heading.group(2))
    expected = f"version_{released:%y_%m}_v{version}.md"
    if path.name != expected:
        raise ValueError(f"{path}: release heading requires filename {expected}")
    sections, current, category = {}, [], None

    def finish():
        if category is not None:
            sections[category] = bullets(current, path, linked=True)
            current.clear()

    for line in lines[1:]:
        if line.startswith("### "):
            finish()
            category = line[4:]
            if category not in CATEGORIES or category in sections:
                raise ValueError(f"{path}: unknown or repeated archived category {category!r}")
        elif category is None and not line.strip():
            continue
        elif category is None:
            raise ValueError(f"{path}: entries must follow a category heading")
        else:
            current.append(line)
    finish()
    if not sections:
        raise ValueError(f"{path}: release archive contains no categories")
    prs = {int(LINKED_ENTRY.match(entry).group(1)) for entries in sections.values() for entry in entries}
    return Archive(path, version, released, prs, text)


def archives(repo):
    result = [parse_archive(path) for path in layout(Path(repo))[1]]
    versions, prs = set(), set()
    for archive in result:
        if archive.version in versions:
            raise ValueError(f"Release v{archive.version} already exists in another archive")
        if archive.prs & prs:
            raise ValueError(f"{archive.path}: PR entries are already recorded in another archive")
        versions.add(archive.version)
        prs.update(archive.prs)
    return sorted(result, key=lambda archive: (-archive.released.toordinal(), archive.path.name))


def read_history(repo):
    return regular_text(repo / "CHANGELOG.md")


def historical_versions(history):
    return {value.removeprefix("v") for value in re.findall(r"(?m)^## \[([^]]+)\]", history)
            if value != "Unreleased"}


def check(repo, pr=None, added=None):
    repo = Path(repo)
    items, recorded = fragments(repo), archives(repo)
    history = read_history(repo)
    for archive in recorded:
        if archive.version in historical_versions(history):
            raise ValueError(f"Release v{archive.version} already exists in legacy history")
    historical_prs = {int(value) for value in re.findall(r"\[PR #([1-9][0-9]*)\]\(", history)}
    archived_prs = set().union(*(archive.prs for archive in recorded))
    if historical_prs & archived_prs:
        raise ValueError("A release archive repeats PR entries from the legacy history")
    for fragment in items:
        if fragment.pr in historical_prs | archived_prs:
            raise ValueError(f"PR #{fragment.pr} already has an entry in released or historical changes")
    if pr is not None:
        if added is None and not any(fragment.pr == pr for fragment in items):
            raise ValueError(f"No changelog fragment with PR: #{pr}")
        for fragment in items:
            if (added is not None and fragment.path.relative_to(repo).as_posix() in added and fragment.pr != pr):
                raise ValueError(f"{fragment.path}: new fragments must reference PR: #{pr}")
    return items


def development_body(items):
    sections = []
    for category in CATEGORIES:
        entries = [f"- [PR #{item.pr}]({PR_URL}{item.pr}) {entry[2:]}"
                   for item in items if item.category == category for entry in item.entries]
        if entries:
            sections.append(f"### {category}\n\n" + "\n\n".join(entries) + "\n")
    return "\n".join(sections)


def legacy_body(history):
    """Keep old entries visible without treating legacy Unreleased as development."""
    lines = history.splitlines(keepends=True)
    if lines and lines[0].rstrip() == "# Changelog":
        lines = lines[1:]
    result = []
    for line in lines:
        if line.rstrip() == "## [Unreleased]":
            result.append("### Legacy changes recorded before fragment adoption\n")
        elif re.match(r"^#{2,5} ", line):
            result.append("#" + line)
        else:
            result.append(line)
    return "".join(result).lstrip("\n")


def render(repo):
    repo = Path(repo)
    items = check(repo)
    development = development_body(items) or "No development entries.\n"
    released = "\n".join(archive.original.rstrip() + "\n" for archive in archives(repo))
    return ("# Changelog\n\n## Development version\n\n" + development + "\n" + released +
            "\n## Historical changes\n\n" + legacy_body(read_history(repo)))


def release(repo, version, release_date, write=False):
    """Preview or create one release archive, consuming only development fragments."""
    repo = Path(repo)
    version, released = version_tag(version), iso_date(release_date)
    items = check(repo)
    if (version in historical_versions(read_history(repo)) or
            version in [archive.version for archive in archives(repo)]):
        raise ValueError(f"Release v{version} already exists")
    if not items:
        raise ValueError("No development fragments to archive")
    result = f"## [v{version}] - {released.isoformat()}\n\n" + development_body(items)
    target = repo / "changelog" / f"version_{released:%y_%m}_v{version}.md"
    archive_text(target, result)
    if write:
        for item in items:
            if regular_text(item.path) != item.original:
                raise ValueError(f"{item.path} changed during release preparation")
        with target.open("x", encoding="utf-8") as stream:
            stream.write(result)
        for item in items:
            item.path.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("check", help="validate development fragments and release archives")
    validate.add_argument("--pr", type=int, help="check metadata for this PR")
    validate.add_argument("--base", help="base revision for newly added fragment checks")
    validate.add_argument("--head", help="head revision for newly added fragment checks")
    commands.add_parser("render", help="print combined Markdown without changing files")
    consolidate = commands.add_parser("release", help="preview a release; --write also archives development files")
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
                revisions = [subprocess.check_output(
                    ["git", "rev-parse", "--verify", "--end-of-options", value + "^{commit}"],
                    cwd=args.repo, text=True).strip() for value in (args.base, args.head)]
                names = subprocess.check_output(
                    ["git", "diff", "--no-ext-diff", "--name-only", "--diff-filter=A", "-z",
                     "...".join(revisions), "--", "changelog/"], cwd=args.repo)
                added = set(names.decode("utf-8").rstrip("\0").split("\0"))
            print(f"Validated {len(check(args.repo, args.pr, added))} development fragment(s)")
        elif args.command == "render":
            print(render(args.repo), end="")
        else:
            print(release(args.repo, args.version, args.date, args.write), end="")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"changelog: {error}\n")


if __name__ == "__main__":
    main()
