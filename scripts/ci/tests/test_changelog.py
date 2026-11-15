"""Exercise fragments, documentation rendering, and release writes in isolation."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/changelog.py"
spec = importlib.util.spec_from_file_location("fragment_changelog", SCRIPT)
changelog = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = changelog
spec.loader.exec_module(changelog)

HISTORY = """# Changelog

Historical introduction stays unchanged.

## [Unreleased]

### Added

- [PR #11](https://github.com/luca-heltai/immersx/pull/11) Existing unreleased entry.

### Fixed

- [PR #10](https://github.com/luca-heltai/immersx/pull/10) Existing fix.

## [0.1.0] - 2025-01-02

### Added

- Historical release text stays unchanged.
"""


class Fragments(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        (self.repo / "CHANGELOG.md").write_text(HISTORY)

    def fragment(self, name="2026_11_15_change.md", category="fix", pr=42, body="- Correct the result.\n"):
        path = self.repo / "changelog" / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"PR: #{pr}\n\n{body}")
        return path

    def snapshot(self):
        return {path.relative_to(self.repo).as_posix(): path.read_bytes()
                for path in self.repo.rglob("*") if path.is_file()}

    def test_no_fragments_leave_history_identical(self):
        self.assertEqual(changelog.render(self.repo), HISTORY)

    def test_render_links_categories_and_preserves_history_without_writes(self):
        self.fragment(body="- Correct the result.\n  Preserve the boundary case.\n")
        self.fragment(category="tutorials", body="- Explain the workflow.\n")
        before = self.snapshot()
        text = changelog.render(self.repo)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(text.count("### Fixed\n"), 1)
        self.assertEqual(text.count("### Tutorial\n"), 1)
        self.assertIn("- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Correct the result.\n  Preserve", text)
        self.assertLess(text.index("Correct the result"), text.index("Existing fix"))
        self.assertTrue(text.endswith(HISTORY.split("## [0.1.0]", 1)[1]))
        self.assertEqual(text.count("# Changelog\n"), 1)

    def test_date_then_filename_orders_entries_independently_of_pr_numbers(self):
        self.fragment("2026_10_30_old.md", pr=999, body="- Month-old change.\n")
        self.fragment("2026_11_15_beta.md", pr=1, body="- Same-day beta.\n")
        self.fragment("2026_11_15_alpha.md", pr=2, body="- Same-day alpha.\n")
        text = changelog.render(self.repo)
        self.assertLess(text.index("Same-day alpha"), text.index("Same-day beta"))
        self.assertLess(text.index("Same-day beta"), text.index("Month-old change"))

    def test_metadata_is_independent_of_filename(self):
        path = self.fragment("2024_02_29_cache_assembly.md", pr=701)
        item = changelog.parse_fragment(path, "Fixed")
        self.assertEqual(item.pr, 701)
        self.assertEqual(item.started.isoformat(), "2024-02-29")

    def test_invalid_dates_and_names_are_rejected(self):
        for name in ("2026_02_29_change.md", "2026_13_15_change.md", "0000_01_01_change.md",
                     "2026_11_15.md", "427.md", "2026_11_15_Upper.md", "2026_11_15_change.txt"):
            with self.subTest(name=name):
                path = self.fragment(name)
                with self.assertRaises(ValueError):
                    changelog.fragments(self.repo)
                path.unlink()

    def test_unknown_category_or_nested_directory_is_rejected(self):
        for category in ("typo", "fix/nested"):
            with self.subTest(category=category):
                directory = self.repo / "changelog"
                path = self.fragment(category=category)
                with self.assertRaises(ValueError):
                    changelog.fragments(self.repo)
                path.unlink()
                for child in sorted(directory.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()

    def test_malformed_metadata_and_entries_are_rejected(self):
        path = self.fragment()
        for text in ("PR: pending\n\n- Change.\n", "PR: #0\n\n- Change.\n",
                     "PR: #42\n", "PR: #42\n\n## Fixed\n\n- Change.\n",
                     "PR: #42\n\n- [PR #99](url) Wrong prefix.\n",
                     "PR: #42\n\n- Repeated.\n\n- Repeated.\n",
                     "PR: #42\n\nUnstructured prose.\n", "PR: #42\n\n- \n"):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    changelog.fragments(self.repo)

    def test_one_fragment_per_pr_per_category(self):
        self.fragment()
        self.fragment("2026_11_15_other.md")
        with self.assertRaisesRegex(ValueError, "More than one fragment"):
            changelog.check(self.repo)

    def test_same_pr_can_have_fragments_in_multiple_categories(self):
        self.fragment()
        self.fragment(category="docs")
        self.assertEqual(len(changelog.check(self.repo)), 2)

    def test_pr_already_recorded_in_history_is_rejected(self):
        self.fragment(pr=11)
        with self.assertRaisesRegex(ValueError, "already has an entry"):
            changelog.render(self.repo)

    def test_symlink_is_not_read_or_removed(self):
        external = self.repo / "external.md"
        external.write_text("PR: #42\n\n- External.\n")
        target = self.repo / "changelog/fix/2026_11_15_link.md"
        target.parent.mkdir(parents=True)
        target.symlink_to(external)
        with self.assertRaises(ValueError):
            changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        self.assertTrue(target.is_symlink())
        self.assertEqual((self.repo / "CHANGELOG.md").read_text(), HISTORY)

    def test_fragment_root_and_history_symlinks_are_rejected(self):
        external = self.repo / "external"
        external.mkdir()
        (self.repo / "changelog").symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            changelog.check(self.repo)
        (self.repo / "changelog").unlink()
        history = self.repo / "CHANGELOG.md"
        history.unlink()
        target = external / "history.md"
        target.write_text(HISTORY)
        history.symlink_to(target)
        with self.assertRaises(ValueError):
            changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        self.assertEqual(target.read_text(), HISTORY)

    def test_empty_release_is_not_created(self):
        (self.repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n### Added\n")
        with self.assertRaisesRegex(ValueError, "No unreleased entries"):
            changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)

    def test_current_pr_applies_only_to_newly_added_fragments_in_ci(self):
        old = self.fragment(pr=12)
        new = self.fragment(category="docs", pr=42)
        added = {new.relative_to(self.repo).as_posix()}
        self.assertEqual(len(changelog.check(self.repo, 42, added)), 2)
        with self.assertRaisesRegex(ValueError, "new fragments must reference PR: #43"):
            changelog.check(self.repo, 43, added)
        self.assertEqual(len(changelog.check(self.repo, 43, set())), 2)
        self.assertTrue(old.exists())

    def test_release_preview_is_read_only_and_write_preserves_old_releases(self):
        path = self.fragment()
        before = self.snapshot()
        preview = changelog.release(self.repo, "0.2.0", "2026-11-15")
        self.assertEqual(self.snapshot(), before)
        self.assertIn("## [0.2.0] - 2026-11-15\n", preview)
        self.assertIn("Existing unreleased entry.", preview)
        self.assertTrue(preview.endswith(HISTORY.split("## [0.1.0]", 1)[1]))
        written = changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        self.assertEqual(written, preview)
        self.assertEqual((self.repo / "CHANGELOG.md").read_text(), preview)
        self.assertFalse(path.exists())
        self.assertEqual(changelog.render(self.repo), preview)
        self.assertFalse(list(self.repo.rglob("*.tmp")))
        # A release PR need not add another fragment for its own consolidation.
        self.assertEqual(changelog.check(self.repo, 500, set()), [])

    def test_indented_heading_example_is_not_a_release_section(self):
        self.fragment(body="- Explain the heading:\n  ## [Unreleased]\n")
        result = changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        self.assertEqual(changelog.render(self.repo), result)

    def test_release_refuses_existing_versions_without_changing_files(self):
        self.fragment()
        before = self.snapshot()
        for version in ("0.1.0", "v0.1.0"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "already exists"):
                changelog.release(self.repo, version, "2026-11-15", write=True)
        self.assertEqual(self.snapshot(), before)

    def test_release_refuses_bad_version_date_and_existing_temporary_file(self):
        self.fragment()
        for version, day in (("0.2.0]", "2026-11-15"), ("0.2.0", "2026-02-30")):
            before = self.snapshot()
            with self.assertRaises(ValueError):
                changelog.release(self.repo, version, day, write=True)
            self.assertEqual(self.snapshot(), before)
        temporary = self.repo / "CHANGELOG.md.tmp"
        temporary.write_text("Preserve user work")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        self.assertEqual(self.snapshot(), before)

    def test_sphinx_hook_uses_repo_path_and_refreshes_incremental_changes(self):
        self.fragment()
        spec = importlib.util.spec_from_file_location("fragment_documentation", ROOT / "doc/conf.py")
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)
        with mock.patch.object(conf, "_REPO_ROOT", self.repo):
            source = ["Original placeholder"]
            conf.render_changelog(None, "changes", source)
            self.assertIn("Correct the result.", source[0])
            unchanged = ["Other page"]
            conf.render_changelog(None, "other", unchanged)
            self.assertEqual(unchanged, ["Other page"])
            self.assertEqual(conf.refresh_changelog(None, None, set(), set(), set()), ["changes"])
        app = mock.Mock()
        conf.setup(app)
        self.assertEqual(app.connect.call_count, 2)

    def test_cli_compares_new_fragments_but_not_existing_edits(self):
        self.fragment(pr=12)
        subprocess.run(["git", "init", "-q", "-b", "master"], cwd=self.repo, check=True)
        def commit():
            subprocess.run(["git", "add", "--all"], cwd=self.repo, check=True)
            subprocess.run(["git", "-c", "user.name=Fragment test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "fixture"], cwd=self.repo, check=True)
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        base = commit()
        self.fragment(category="docs", pr=42)
        self.fragment(pr=12, body="- Correct an old description.\n")
        commit()
        command = [sys.executable, str(SCRIPT), "--repo", str(self.repo), "check", "--pr", "42",
                   "--base", base, "--head", "HEAD"]
        subprocess.run(command, cwd="/tmp", check=True, capture_output=True)
        wrong = subprocess.run(command[:command.index("42")] + ["43"] + command[command.index("42") + 1:],
                               cwd="/tmp", capture_output=True, text=True)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("new fragments must reference PR: #43", wrong.stderr)


if __name__ == "__main__":
    unittest.main()
