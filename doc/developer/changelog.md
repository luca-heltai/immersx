# Changelog and releases

Development notes live in dated files under `changelog/<category>/`. When
preparing a release, an explicit command collects them into one versioned
archive and removes those individual files. The documentation publishes current
development notes, release archives, and older legacy history together.

## Add development fragments

The directory selects the category:

| Directory | Rendered section |
| --- | --- |
| `added` | Added |
| `changed` | Changed |
| `fix` | Fixed |
| `removed` | Removed |
| `docs` | Documentation |
| `tutorials` | Tutorial |
| `build` | Build and packaging |
| `testing` | Testing and CI |

Name each file `YYYY_MM_DD_short_description.md`, using the date work began.
For example, `changelog/fix/2026_11_15_cache_assembly.md` can contain:

```markdown
PR: #427

- Rebuild the cached matrix when coefficients change.
```

The date is the start of the activity, not the merge date. Keep the filename
stable when opening or rebasing the PR. If another PR has the same date and
description in that category, choose a distinct meaningful description. No
commit hash, UUID, or PR number is needed in the filename.

Use the actual PR number in the metadata. Before opening the PR, a local draft
can use `PR: pending`; replace it immediately once the number is known. The
validator and documentation build require the real number. One PR can add a
fragment in several categories; within a category, use one file with multiple
bullets when necessary.

Copy `.github/CHANGELOG_TEMPLATE.md`. Write descriptions without category
headings or a leading PR link: the containing directory supplies the category,
and the renderer supplies the link from `PR: #N`. Indent continuation lines by
two spaces.

Individual category files belong only to the current development version.
There is no extra bot commit that incorporates or deletes them after every PR
merge; they remain separate until explicit release preparation.

## Validate and preview

Run these commands from the repository root:

```bash
python3 scripts/changelog.py check
python3 scripts/changelog.py render
```

`check` validates all development fragments and release archives. CI additionally
checks that newly added development fragments refer to the current PR; editing
an existing entry retains its original PR reference. Release preparation needs
no extra fragment solely to archive the existing ones. `render` prints combined
Markdown and changes no file.

Within each development category, fragments appear newest activity date first,
then alphabetically by filename for equal dates. PR numbers do not affect
ordering. Release archives follow, newest full release date first, then filename
for equal dates. The old `CHANGELOG.md` appears last under Historical changes.
Its old `[Unreleased]` section is presented as legacy notes recorded before
fragment adoption, without inventing a version or release date for those notes.
Existing entries and their historical ordering remain unchanged.

`./scripts/build_doc.sh` builds this combined page using the same renderer.
The Sphinx source-read hook renders it in memory, including on incremental
builds. Documentation generation neither rewrites history or archives nor
modifies or deletes development fragments.

## Archive an explicit release

Preview a release without changing the checkout:

```bash
python3 scripts/changelog.py release --version 0.5.0 --date 2026-11-15
```

Review the Markdown, then explicitly write the archive:

```bash
python3 scripts/changelog.py release --version 0.5.0 --date 2026-11-15 --write
```

This creates `changelog/version_26_11_v0.5.0.md` with a heading containing the
full date and tag:

```markdown
## [v0.5.0] - 2026-11-15

### Fixed

- [PR #427](https://github.com/luca-heltai/immersx/pull/427) Rebuild the cached matrix when coefficients change.
```

`0.5.0` and `v0.5.0` mean the same version; two-component tags such as `v0.5`
are also accepted. Tag names must be portable filename components, without path
separators or `..`. The `YY_MM` filename prefix comes from the explicit release
date. The full ISO date in the header controls ordering across years.

Only after writing the archive does the command remove all validated current
fragments from the category directories. Empty directories may remain locally.
Older archives and the global `CHANGELOG.md` remain untouched. A later PR adds
new category files for the next development version.

The command refuses malformed data, existing archive paths, and versions already
recorded in another archive or the legacy history. With `--write`, review and
commit the new archive and removed fragments yourself. Preview and `--write`
work offline and do not invoke Git or GitHub.

To prepare the commit and pull request automatically instead:

```bash
python3 scripts/changelog.py release --version 0.5.0 --date 2026-11-15 --commit
```

`--commit` implies `--write`. It requires GitHub CLI (`gh`) installed and
authenticated, a clean checkout including staged and untracked files, and the
repository's default branch checked out at the current `origin` commit. It
checks these prerequisites before modifying the changelog. Existing release
branches or PRs are rejected rather than overwritten.

The command creates `codex/release-v0.5.0`, archives the notes, commits only the
new archive and removed fragments with message `Prepare changelog for v0.5.0`,
pushes normally, and opens a PR against the default branch. It preserves user
Git hooks and signing settings and checks the resulting commit before pushing.
This changelog-only commit changes no C++ source and does not run a formatter
on unrelated files. No mode creates a Git tag or a GitHub release.

If pushing or opening the PR fails, the command preserves the release branch
and commit and reports their identity with recovery commands. Inspect that
branch and retry the push or PR creation; do not rerun release consolidation.
Check for an existing PR before retrying `gh pr create` to avoid duplicates.
