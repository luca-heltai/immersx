"""Exercise development fragments, release archives, and documentation in isolation."""

import importlib.util
import json
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

- [PR #11](https://github.com/luca-heltai/immersx/pull/11) Existing legacy entry.
  ### Fixed

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

    def archive(self, name="version_26_11_v0.2.0.md", version="0.2.0", day="2026-11-15", pr=42, body=None):
        path = self.repo / "changelog" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body = body or f"### Fixed\n\n- [PR #{pr}]({changelog.PR_URL}{pr}) Archived fix.\n"
        path.write_text(f"## [v{version}] - {day}\n\n{body}")
        return path

    def snapshot(self):
        return {path.relative_to(self.repo).as_posix(): path.read_bytes()
                for path in self.repo.rglob("*") if path.is_file()}

    def test_no_fragments_leave_legacy_visible_once_as_historical(self):
        text = changelog.render(self.repo)
        self.assertIn("## Development version\n\nNo development entries.", text)
        self.assertIn("## Historical changes", text)
        self.assertIn("### Legacy changes recorded before fragment adoption", text)
        self.assertNotIn("\n## [Unreleased]", text)
        self.assertEqual(text.count("Existing legacy entry."), 1)
        self.assertEqual(text.count("Historical release text stays unchanged."), 1)
        self.assertEqual(text.count("# Changelog\n"), 1)
        self.assertEqual((self.repo / "CHANGELOG.md").read_text(), HISTORY)

    def test_render_links_categories_and_preserves_inputs_without_writes(self):
        self.fragment(body="- Correct the result.\n  Preserve the boundary case.\n")
        self.fragment(category="tutorials", body="- Explain the workflow.\n")
        before = self.snapshot()
        text = changelog.render(self.repo)
        self.assertEqual(self.snapshot(), before)
        self.assertIn("- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Correct the result.\n  Preserve", text)
        self.assertIn("### Tutorial", text)
        self.assertLess(text.index("Correct the result"), text.index("## Historical changes"))
        self.assertIn("  ### Fixed\n", text)

    def test_date_then_filename_orders_development_independently_of_pr_number(self):
        self.fragment("2026_10_30_old.md", pr=999, body="- Month-old change.\n")
        self.fragment("2026_11_15_beta.md", pr=1, body="- Same-day beta.\n")
        self.fragment("2026_11_15_alpha.md", pr=2, body="- Same-day alpha.\n")
        text = changelog.render(self.repo)
        self.assertLess(text.index("Same-day alpha"), text.index("Same-day beta"))
        self.assertLess(text.index("Same-day beta"), text.index("Month-old change"))

    def test_metadata_is_independent_of_filename(self):
        item = changelog.parse_fragment(self.fragment("2024_02_29_cache_assembly.md", pr=701), "Fixed")
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

    def test_unknown_category_nested_directory_and_root_file_are_rejected(self):
        for category in ("typo", "fix/nested"):
            with self.subTest(category=category):
                path = self.fragment(category=category)
                with self.assertRaises(ValueError):
                    changelog.fragments(self.repo)
                path.unlink()
                for child in sorted((self.repo / "changelog").rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
        (self.repo / "changelog/notes.md").write_text("Do not delete user work")
        with self.assertRaises(ValueError):
            changelog.check(self.repo)

    def test_malformed_metadata_and_entries_are_rejected(self):
        path = self.fragment()
        for text in ("PR: pending\n\n- Change.\n", "PR: #0\n\n- Change.\n", "PR: #42\n",
                     "PR: #42\n\n## Fixed\n\n- Change.\n", "PR: #42\n\n- [PR #99](url) Wrong prefix.\n",
                     "PR: #42\n\n- Repeated.\n\n- Repeated.\n", "PR: #42\n\nUnstructured prose.\n", "PR: #42\n\n- \n"):
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

    def test_pr_already_historical_or_released_cannot_be_development_again(self):
        path = self.fragment(pr=11)
        with self.assertRaisesRegex(ValueError, "already has an entry"):
            changelog.check(self.repo)
        path.unlink()
        self.archive(pr=42)
        self.fragment(pr=42)
        with self.assertRaisesRegex(ValueError, "already has an entry"):
            changelog.check(self.repo)

    def test_symlink_paths_are_not_read_or_removed(self):
        external = self.repo / "external.md"
        external.write_text("PR: #42\n\n- External.\n")
        target = self.repo / "changelog/fix/2026_11_15_link.md"
        target.parent.mkdir(parents=True)
        target.symlink_to(external)
        with self.assertRaises(ValueError):
            changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        self.assertTrue(target.is_symlink())
        self.assertEqual(external.read_text(), "PR: #42\n\n- External.\n")
        self.assertEqual((self.repo / "CHANGELOG.md").read_text(), HISTORY)

    def test_fragment_root_and_legacy_symlinks_are_rejected(self):
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
            changelog.render(self.repo)
        self.assertEqual(target.read_text(), HISTORY)

    def test_gitignore_keeps_all_changelog_markdown_as_source(self):
        (self.repo / ".gitignore").write_text((ROOT / ".gitignore").read_text())
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        paths = [self.fragment(category=category, body="- Preserve checkpoint restart.\n")
                 for category in changelog.DIRECTORIES]
        paths += [self.fragment("2026_11_15_checkpoint_restart.md", category=category)
                  for category in changelog.DIRECTORIES]
        paths.append(self.archive(name="version_26_11_v0.5-checkpoint.md", version="0.5-checkpoint"))
        for path in paths:
            with self.subTest(path=path):
                result = subprocess.run(["git", "check-ignore", "-q", "--", str(path)], cwd=self.repo)
                self.assertEqual(result.returncode, 1)
        unrelated = self.repo / "checkpoint-output.md"
        unrelated.write_text("Generated output stays ignored.\n")
        result = subprocess.run(["git", "check-ignore", "-q", "--", str(unrelated)], cwd=self.repo)
        self.assertEqual(result.returncode, 0)

    def test_current_pr_applies_only_to_new_fragments_in_ci(self):
        old = self.fragment(pr=12)
        new = self.fragment(category="docs", pr=42)
        added = {new.relative_to(self.repo).as_posix()}
        self.assertEqual(len(changelog.check(self.repo, 42, added)), 2)
        with self.assertRaisesRegex(ValueError, "new fragments must reference PR: #43"):
            changelog.check(self.repo, 43, added)
        self.assertEqual(len(changelog.check(self.repo, 43, set())), 2)
        self.assertTrue(old.exists())

    def test_release_preview_is_read_only_write_creates_archive_and_consumes_all_categories(self):
        fix = self.fragment()
        docs = self.fragment(category="docs", body="- Explain the correction.\n")
        before = self.snapshot()
        preview = changelog.release(self.repo, "v0.2.0", "2026-11-15")
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(preview.startswith("## [v0.2.0] - 2026-11-15\n"))
        self.assertNotIn("Existing legacy entry.", preview)
        self.assertIn("### Documentation", preview)
        changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        archive = self.repo / "changelog/version_26_11_v0.2.0.md"
        self.assertEqual(archive.read_text(), preview)
        self.assertFalse(fix.exists())
        self.assertFalse(docs.exists())
        self.assertEqual((self.repo / "CHANGELOG.md").read_text(), HISTORY)
        self.assertEqual(changelog.fragments(self.repo), [])
        self.assertEqual(changelog.check(self.repo, 500, set()), [])
        rendered = changelog.render(self.repo)
        self.assertIn("No development entries.", rendered)
        self.assertEqual(rendered.count("Existing legacy entry."), 1)

    def test_accepted_whitespace_and_continuations_round_trip_through_release(self):
        for body in ("-  Correct the result.\n", "- \n  Correct the result.\n"):
            with self.subTest(body=body):
                path = self.fragment(body=body)
                changelog.check(self.repo)
                changelog.release(self.repo, "0.5", "2026-11-15", write=True)
                self.assertFalse(path.exists())
                self.assertEqual(changelog.check(self.repo), [])
                self.assertIn("Correct the result.", changelog.render(self.repo))
                (self.repo / "changelog/version_26_11_v0.5.md").unlink()

    def test_two_same_month_archives_and_new_development_are_all_visible_in_order(self):
        self.fragment(pr=42, body="- First release correction.\n")
        changelog.release(self.repo, "0.2", "2026-11-15", write=True)
        self.fragment(pr=43, body="- Second release correction.\n")
        changelog.release(self.repo, "0.3", "2026-11-20", write=True)
        self.fragment(pr=44, body="- Current development correction.\n")
        text = changelog.render(self.repo)
        self.assertLess(text.index("Current development correction"), text.index("## [v0.3]"))
        self.assertLess(text.index("## [v0.3]"), text.index("## [v0.2]"))
        self.assertLess(text.index("## [v0.2]"), text.index("## Historical changes"))
        for entry in ("First release correction.", "Second release correction.", "Existing legacy entry."):
            self.assertEqual(text.count(entry), 1)
        self.assertTrue((self.repo / "changelog/version_26_11_v0.2.md").exists())
        self.assertTrue((self.repo / "changelog/version_26_11_v0.3.md").exists())

    def test_version_alias_and_duplicate_archive_version_never_overwrite(self):
        self.fragment()
        changelog.release(self.repo, "v0.5", "2026-11-15", write=True)
        self.fragment(pr=43)
        before = self.snapshot()
        for version in ("0.5", "v0.5"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "already exists"):
                changelog.release(self.repo, version, "2027-01-02", write=True)
        self.assertEqual(self.snapshot(), before)

    def test_archive_duplicate_version_in_another_month_is_rejected(self):
        self.archive(version="0.5", name="version_26_11_v0.5.md")
        self.archive(version="0.5", name="version_27_01_v0.5.md", day="2027-01-02", pr=43)
        with self.assertRaisesRegex(ValueError, "already exists"):
            changelog.check(self.repo)

    def test_archive_header_date_filename_categories_and_links_are_validated(self):
        path = self.archive()
        valid = path.read_text()
        malformed = (valid.replace("2026-11-15", "2026-12-15"), valid.replace("2026-11-15", "2026-02-30"),
                     valid.replace("[v0.2.0]", "[vv0.2.0]"), valid.replace("### Fixed", "### Unknown"),
                     valid + "\n### Fixed\n\n- [PR #43](https://github.com/luca-heltai/immersx/pull/43) Repeat.\n",
                     valid.replace("### Fixed\n\n- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Archived fix.", "### Fixed"),
                     valid.replace("pull/42", "pull/43"), valid.replace("- [PR #42](https://github.com/luca-heltai/immersx/pull/42) ", "- "))
        for text in malformed:
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    changelog.check(self.repo)

    def test_archived_pr_and_legacy_version_cannot_be_repeated(self):
        first = self.archive()
        second = self.archive(name="version_26_12_v0.3.md", version="0.3", day="2026-12-01")
        with self.assertRaisesRegex(ValueError, "already recorded"):
            changelog.check(self.repo)
        second.unlink()
        first.unlink()
        self.archive(name="version_26_11_v0.1.0.md", version="0.1.0")
        with self.assertRaisesRegex(ValueError, "already exists in legacy"):
            changelog.check(self.repo)

    def test_release_refuses_legacy_versions_empty_development_and_bad_tags(self):
        self.fragment()
        before = self.snapshot()
        for version in ("0.1.0", "v0.1.0"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "already exists"):
                changelog.release(self.repo, version, "2026-11-15", write=True)
        for version in ("../../other", "0..5", "0.5/other", "0.5\\other", "vv0.5", "0.5\n"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                changelog.release(self.repo, version, "2026-11-15", write=True)
        self.assertEqual(self.snapshot(), before)
        self.fragment().unlink()
        with self.assertRaisesRegex(ValueError, "No development fragments"):
            changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)

    def test_indented_heading_examples_stay_inside_their_entries(self):
        self.fragment(body="- Explain headings:\n  ## [Unreleased]\n  ### Fixed\n")
        changelog.release(self.repo, "0.2.0", "2026-11-15", write=True)
        text = changelog.render(self.repo)
        self.assertIn("  ## [Unreleased]\n  ### Fixed\n", text)
        self.assertIn("Existing legacy entry.\n  ### Fixed", text)

    def test_sphinx_hook_refreshes_incremental_changes_and_uses_explicit_repo(self):
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

    def publication_fixture(self):
        self.fragment()
        self.remote_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.remote_temp.cleanup)
        remote = Path(self.remote_temp.name)
        self.origin = remote / "origin.git"
        self.gh_log = remote / "gh.jsonl"
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()
        self.git = git
        git("init", "-q", "-b", "master")
        git("config", "user.name", "Release test")
        git("config", "user.email", "test@example.invalid")
        git("config", "commit.gpgsign", "false")
        git("add", "--all")
        git("commit", "-qm", "initial")
        subprocess.run(["git", "init", "--bare", "-q", "-b", "master", str(self.origin)], check=True)
        git("remote", "add", "origin", str(self.origin))
        git("push", "-q", "origin", "master")
        self.initial_head = git("rev-parse", "HEAD")
        fake = remote / "gh"
        fake.write_text("#!" + sys.executable + "\n" + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['GH_TEST_LOG'], 'a') as log:
    log.write(json.dumps(args)+'\\n')
mode = os.environ.get('GH_TEST_MODE', '')
if args[:2] == ['auth', 'status']:
    sys.exit(1 if mode == 'authfail' else 0)
if args[:2] == ['repo', 'view']:
    print(json.dumps({'nameWithOwner':'example/project', 'defaultBranchRef':{'name':'master'}}))
elif args[:2] == ['pr', 'list']:
    print('[{"number":99}]' if mode == 'existingpr' else '[]')
elif args[:2] == ['pr', 'create']:
    assert '## Overview' in Path(args[-1]).read_text()
    sys.exit(1) if mode == 'prfail' else print('https://github.com/example/project/pull/7')
""")
        fake.chmod(0o755)
        self.fake_gh = fake
        environment = {"PATH": str(remote) + os.pathsep + os.environ["PATH"], "GH_TEST_LOG": str(self.gh_log)}
        self.environment = mock.patch.dict(os.environ, environment)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_commit_release_creates_only_archive_commit_push_and_pr_without_tag(self):
        self.publication_fixture()
        result = changelog.commit_release(self.repo, "v0.5", "2026-11-15")
        self.assertIn("https://github.com/example/project/pull/7", result)
        self.assertEqual(self.git("branch", "--show-current"), "codex/release-v0.5")
        self.assertEqual(self.git("tag", "--list"), "")
        self.assertEqual(self.git("status", "--porcelain"), "")
        changed = self.git("diff", "--name-only", self.initial_head, "HEAD").splitlines()
        self.assertEqual(set(changed), {"changelog/fix/2026_11_15_change.md", "changelog/version_26_11_v0.5.md"})
        self.assertEqual(self.git("rev-parse", "origin/master"), self.initial_head)
        self.assertEqual(self.git("rev-parse", "origin/codex/release-v0.5"), self.git("rev-parse", "HEAD"))
        calls = [json.loads(line) for line in self.gh_log.read_text().splitlines()]
        create = next(call for call in calls if call[:2] == ["pr", "create"])
        self.assertEqual(create[create.index("--base") + 1], "master")
        self.assertIn("--body-file", create)

    def test_commit_cli_implies_write(self):
        self.publication_fixture()
        subprocess.run([sys.executable, "-B", str(SCRIPT), "--repo", str(self.repo), "release",
                        "--version", "0.5", "--date", "2026-11-15", "--commit"], check=True, capture_output=True)
        self.assertTrue((self.repo / "changelog/version_26_11_v0.5.md").exists())
        self.assertFalse((self.repo / "changelog/fix/2026_11_15_change.md").exists())

    def test_commit_rejects_dirty_untracked_and_staged_work_without_mutation(self):
        self.publication_fixture()
        unrelated = self.repo / "unrelated.txt"
        for staged in (False, True):
            unrelated.write_text("Preserve user work.")
            if staged:
                self.git("add", "unrelated.txt")
            before = self.snapshot()
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                changelog.commit_release(self.repo, "0.5", "2026-11-15")
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.git("rev-parse", "HEAD"), self.initial_head)
            if not staged:
                unrelated.unlink()

    def test_commit_missing_gh_and_failed_auth_do_not_change_checkout(self):
        self.publication_fixture()
        before = self.snapshot()
        with mock.patch.object(changelog.shutil, "which", return_value=None), self.assertRaisesRegex(ValueError, "requires GitHub CLI"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.assertEqual(self.snapshot(), before)
        with mock.patch.dict(os.environ, {"GH_TEST_MODE": "authfail"}), self.assertRaises(subprocess.CalledProcessError):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.assertEqual(self.snapshot(), before)

    def test_commit_rejects_wrong_branch_existing_branch_and_existing_pr(self):
        self.publication_fixture()
        self.git("switch", "-qc", "feature")
        with self.assertRaisesRegex(ValueError, "default branch"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.git("switch", "-q", "master")
        self.git("branch", "codex/release-v0.5")
        with self.assertRaisesRegex(ValueError, "already exists locally"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.git("branch", "-D", "codex/release-v0.5")
        self.git("push", "-q", "origin", "master:refs/heads/codex/release-v0.5")
        with self.assertRaisesRegex(ValueError, "exists on origin"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.git("push", "-q", "origin", ":refs/heads/codex/release-v0.5")
        with mock.patch.dict(os.environ, {"GH_TEST_MODE": "existingpr"}), self.assertRaisesRegex(ValueError, "release PR"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.assertFalse((self.repo / "changelog/version_26_11_v0.5.md").exists())

    def test_commit_rejects_head_not_equal_to_remote_default(self):
        self.publication_fixture()
        self.git("commit", "--allow-empty", "-qm", "ahead")
        with self.assertRaisesRegex(ValueError, "HEAD to equal"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.assertFalse((self.repo / "changelog/version_26_11_v0.5.md").exists())
        self.git("push", "-q", "origin", "master")
        self.git("update-ref", "refs/heads/master", self.initial_head)
        with self.assertRaisesRegex(ValueError, "HEAD to equal"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")

    def test_commit_hook_cannot_publish_unrelated_changes(self):
        self.publication_fixture()
        hook = self.repo / ".git/hooks/pre-commit"
        hook.write_text("#!/bin/sh\nprintf 'unexpected' > unrelated.txt\ngit add unrelated.txt\n")
        hook.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "unexpected paths"):
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.assertTrue((self.repo / "unrelated.txt").exists())
        self.assertEqual(self.git("ls-remote", "--heads", "origin", "refs/heads/codex/release-v0.5"), "")
        self.assertEqual(self.git("tag", "--list"), "")

    def test_failed_pr_creation_keeps_published_commit_and_reports_recovery(self):
        self.publication_fixture()
        with mock.patch.dict(os.environ, {"GH_TEST_MODE": "prfail"}), self.assertRaisesRegex(ValueError, "gh pr create") as error:
            changelog.commit_release(self.repo, "0.5", "2026-11-15")
        self.assertEqual(self.git("branch", "--show-current"), "codex/release-v0.5")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.git("rev-parse", "origin/codex/release-v0.5"))
        self.assertIn(self.git("rev-parse", "HEAD"), str(error.exception))
        self.assertEqual(self.git("tag", "--list"), "")

    def test_cli_git_diff_is_read_only_and_only_checks_new_fragment_metadata(self):
        self.fragment(pr=12)
        subprocess.run(["git", "init", "-q", "-b", "master"], cwd=self.repo, check=True)
        def commit():
            subprocess.run(["git", "add", "--all"], cwd=self.repo, check=True)
            subprocess.run(["git", "-c", "user.name=Fragment test", "-c", "user.email=test@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        base = commit()
        self.fragment(category="docs", pr=42)
        self.fragment(pr=12, body="- Correct an old description.\n")
        commit()
        command = [sys.executable, "-B", str(SCRIPT), "--repo", str(self.repo), "check", "--pr", "42",
                   "--base", base, "--head", "HEAD"]
        subprocess.run(command, cwd="/tmp", check=True, capture_output=True)
        before = self.snapshot()
        command[command.index(base)] = "--output=" + str(self.repo / "unexpected_output")
        bad = subprocess.run(command, cwd="/tmp", capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
