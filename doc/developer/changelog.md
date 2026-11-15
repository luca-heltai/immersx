# Changelog and releases

Each pull request records its changes in dated files under `changelog/`.
Concurrent PRs therefore add separate files rather than editing the same
`[Unreleased]` section. The root `CHANGELOG.md` retains existing entries and
historical releases.

## Add fragments

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

## Validate and preview

Run these commands from the repository root:

```bash
python3 scripts/changelog.py check
python3 scripts/changelog.py render
```

`check` validates the whole collection. CI additionally checks that newly
added fragments refer to the current PR; editing an existing entry retains
its original PR reference. A release PR does not need an extra fragment solely
to consolidate the existing ones. `render` prints combined Markdown and changes
no file.

Within each category, fragments appear newest activity date first, then
alphabetically by filename for equal dates. PR numbers do not affect ordering.
Existing historical text and its ordering remain unchanged.

`./scripts/build_doc.sh` builds the combined changelog page using the same
renderer. The Sphinx source-read hook renders the page in memory, including
on incremental builds. Documentation generation neither rewrites the global
changelog nor modifies or deletes fragments.

## Prepare a release

Fragments remain separate after their PRs merge. There is no extra bot commit
that incorporates and deletes them after every merge.

Preview a release without changing the checkout:

```bash
python3 scripts/changelog.py release --version 0.5.0 --date 2026-11-15
```

Review the resulting Markdown, then explicitly apply it:

```bash
python3 scripts/changelog.py release --version 0.5.0 --date 2026-11-15 --write
```

The write moves the combined pending entries into the dated release, leaves
an empty `[Unreleased]` category skeleton, and removes the consolidated
fragment files. Older release sections are retained. The command refuses an
already recorded version, malformed fragments, or PR metadata already present
in the historical changelog. Review and commit the result as part of the normal
release preparation; the command does not create a commit or tag.
