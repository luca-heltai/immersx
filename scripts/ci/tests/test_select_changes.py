#!/usr/bin/env python3
"""Exercise selection against real Git history and untrusted path names."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "select_changes.py"
spec = importlib.util.spec_from_file_location("select_changes", SCRIPT)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class GitSelection(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.git("init", "-q", "-b", "master")
        self.write("doc/guide.md", "initial guide\n" * 20)
        self.write("source/solver.cc", "initial solver\n" * 20)
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo).decode().strip()

    def write(self, path, content="changed\n"):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(self):
        self.git("add", "--all")
        self.git("-c", "user.name=CI test", "-c", "user.email=ci@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def plan(self, **kwargs):
        return selector.select(self.repo, "pull_request", self.base, "HEAD", **kwargs)

    def assert_all_pr_jobs(self, plan):
        for key in ("run_numeric", "run_coral", "run_benchmarks"):
            self.assertTrue(plan[key], key)
        self.assertFalse(plan["docs_only"])
        self.assertFalse(plan["full_suite"])

    def test_documentation_allowlist_excludes_expensive_jobs(self):
        for name in ("README.md", "CHANGELOG.md", "doc/conf.py", "doc/Doxyfile",
                     "doc/requirements.txt", "doc/nested/guide.rst", "doc/assets/logo.svg"):
            with self.subTest(name=name):
                self.assertTrue(selector.documentation_path(name))
        self.write("doc/guide.md")
        self.write("doc/assets/figura-é.png")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["mode"], "docs-only")
        for key in ("run_numeric", "run_coral", "run_benchmarks", "full_suite"):
            self.assertFalse(plan[key], key)

    def test_unknown_executable_or_runtime_inputs_cannot_exclude_tests(self):
        for name in ("unknown/file", "doc/snippet.cc", "doc/example.prm.in",
                     "scripts/ci/select_changes.py", "data/mesh.msh",
                     "gtests/parameters/elasticity.prm.in", "CMakeLists.txt"):
            with self.subTest(name=name):
                self.assertFalse(selector.documentation_path(name))
                self.assertEqual(selector.observed_components([name]), ["all"])
        self.write("unknown/file")
        self.commit()
        self.assert_all_pr_jobs(self.plan())

    def test_viewer_browser_checks_only_exclude_jobs_without_solver_changes(self):
        self.write("scripts/performance/tests/viewer.py")
        self.commit()
        plan = self.plan()
        self.assertEqual(plan["mode"], "docs-only")
        for key in ("run_numeric", "run_coral", "run_benchmarks"):
            self.assertFalse(plan[key], key)

        self.write("source/solver.cc")
        self.commit()
        self.assert_all_pr_jobs(self.plan())

        for path in ("scripts/performance/collect.py",
                     "scripts/performance/tests/test_report.py"):
            with self.subTest(path=path):
                self.assertFalse(selector.documentation_path(path))

    def test_cpp_selection_is_observation_only(self):
        self.write("include/immersx/physics/elasticity.h")
        self.commit()
        plan = self.plan()
        self.assert_all_pr_jobs(plan)
        self.assertEqual(plan["observed_components"], ["elasticity"])
        self.assertEqual(plan["component_selection"], "observation-only")

    def test_shared_header_expands_observation(self):
        self.write("include/immersx/core/domain.h")
        self.commit()
        plan = self.plan()
        self.assert_all_pr_jobs(plan)
        self.assertEqual(plan["observed_components"], ["all"])

    def test_rename_from_source_into_documentation_keeps_old_path(self):
        self.git("mv", "source/solver.cc", "doc/solver.md")
        self.commit()
        plan = self.plan()
        self.assert_all_pr_jobs(plan)
        rename = next(c for c in plan["changes"] if c["status"].startswith("R"))
        self.assertEqual(rename["paths"], ["source/solver.cc", "doc/solver.md"])

    def test_documentation_rename_and_deletion_are_safe(self):
        self.git("mv", "doc/guide.md", "doc/renamed.md")
        self.commit()
        self.assertTrue(self.plan()["docs_only"])
        (self.repo / "doc/renamed.md").unlink()
        self.commit()
        plan = self.plan()
        self.assertTrue(plan["docs_only"])
        self.assertEqual(plan["changes"], [{"status": "D", "paths": ["doc/guide.md"]}])

    def test_source_deletion_requires_numerical_jobs(self):
        (self.repo / "source/solver.cc").unlink()
        self.commit()
        self.assert_all_pr_jobs(self.plan())

    def test_stacked_pr_uses_its_base_and_not_master(self):
        self.git("checkout", "-q", "-b", "parent-pr")
        self.write("source/parent.cc")
        stacked_base = self.commit()
        self.git("checkout", "-q", "-b", "child-pr")
        self.write("doc/child.md")
        child = self.commit()
        self.git("checkout", "-q", "-b", "updated-base", stacked_base)
        self.write("source/unrelated-base-update.cc")
        self.commit()
        plan = selector.select(self.repo, "pull_request", "updated-base", child)
        self.assertEqual(plan["merge_base"], stacked_base)
        self.assertEqual(plan["head"], child)
        self.assertTrue(plan["docs_only"])
        self.assertEqual(plan["changes"], [{"status": "A", "paths": ["doc/child.md"]}])

    def test_missing_history_and_empty_diff_fail_open_to_all_pr_jobs(self):
        for base, head in ((None, "HEAD"), (self.base, None),
                           ("missing", "HEAD"), (self.base, "missing")):
            with self.subTest(base=base, head=head):
                plan = selector.select(self.repo, "pull_request", base, head)
                self.assert_all_pr_jobs(plan)
                self.assertEqual(plan["reasons"], ["diff-unavailable"])
        self.assert_all_pr_jobs(self.plan())
        self.assertEqual(self.plan()["reasons"], ["empty-diff"])

    def test_malformed_diff_enables_all_pr_jobs(self):
        sha = b"a" * 40 + b"\n"
        with mock.patch.object(selector, "git", side_effect=[sha, sha, sha, b"M\0truncated"]):
            plan = self.plan()
        self.assert_all_pr_jobs(plan)
        self.assertEqual(plan["reasons"], ["diff-unavailable"])

    def test_full_override_and_non_pr_events_always_run_full_suite(self):
        self.write("doc/guide.md")
        self.commit()
        for event, force in (("pull_request", True), ("push", False),
                             ("schedule", False), ("workflow_dispatch", False), ("", False)):
            with self.subTest(event=event, force=force):
                plan = selector.select(self.repo, event, self.base, "HEAD", force)
                self.assertTrue(plan["full_suite"])
                self.assertFalse(plan["docs_only"])
                self.assertTrue(all(plan[key] for key in
                                    ("run_numeric", "run_coral", "run_benchmarks")))

    def test_unusual_names_cannot_inject_actions_outputs(self):
        name = "doc/é\nrun_numeric=true.md"
        self.write(name)
        self.commit()
        plan = self.plan()
        self.assertTrue(plan["docs_only"])
        output = self.repo / "outputs"
        summary = self.repo / "summary"
        selector.write_github_outputs(plan, output, summary)
        self.assertEqual(output.read_text(),
                         "run_numeric=false\nrun_coral=false\nrun_benchmarks=false\n"
                         "full_suite=false\ndocs_only=true\nmode=docs-only\n")
        self.assertIn("\\nrun_numeric=true.md", summary.read_text())
        self.assertNotIn("\nrun_numeric=true.md", summary.read_text())
        self.assertEqual(json.loads(json.dumps(plan))["changes"][0]["paths"], [name])

    def test_cli_writes_json_and_github_outputs(self):
        self.write("doc/guide.md")
        self.commit()
        output = self.repo / "github-output"
        summary = self.repo / "github-summary"
        plan_file = self.repo / "artifacts/plan.json"
        env = dict(os.environ, GITHUB_OUTPUT=str(output), GITHUB_STEP_SUMMARY=str(summary))
        result = subprocess.check_output(
            ["python3", str(SCRIPT), "--repo", str(self.repo), "--event", "pull_request",
             "--base", self.base, "--head", "HEAD", "--output", str(plan_file)], env=env)
        self.assertEqual(json.loads(result), json.loads(plan_file.read_text()))
        self.assertIn("docs_only=true\n", output.read_text())


class DiffParser(unittest.TestCase):
    def test_copy_and_rename_keep_both_paths(self):
        changes = selector.parse_changes(b"R100\0source/old.cc\0doc/new.md\0C80\0a\0b\0")
        self.assertEqual(changes[0]["paths"], ["source/old.cc", "doc/new.md"])
        self.assertEqual(changes[1]["paths"], ["a", "b"])

    def test_malformed_status_or_paths_raise(self):
        for data in (b"M\0doc/a.md", b"R100\0doc/a.md\0", b"M\0\0",
                     b"X\0doc/a.md\0", b"R101\0a\0b\0", b"R\0a\0b\0",
                     b"M\0../doc/a.md\0", b"M\0/doc/a.md\0", b"\xff\0a\0"):
            with self.subTest(data=data):
                with self.assertRaises((ValueError, UnicodeError)):
                    selector.parse_changes(data)


if __name__ == "__main__":
    unittest.main()
