#!/usr/bin/env python3
"""Exercise workflow gates with representative GitHub job results."""

import json
import os
from pathlib import Path
import re
import subprocess
import textwrap
from types import SimpleNamespace
import unittest


ROOT = Path(os.environ.get("IMMERSX_CI_WORKFLOW_ROOT", Path(__file__).resolve().parents[3]))


def job(workflow, name):
    content = (ROOT / ".github/workflows" / workflow).read_text()
    match = re.search(r"^  " + re.escape(name) + r":\n(.*?)(?=^  [\w-]+:\n|\Z)",
                      content, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"Missing workflow job {workflow}:{name}")
    return match.group(1)


def run_block(workflow, job_name, step_name):
    content = job(workflow, job_name)
    lines = content.splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.strip() == "- name: " + step_name)
    run = next(index for index in range(start + 1, len(lines))
               if lines[index].strip() == "run: |")
    indentation = len(lines[run]) - len(lines[run].lstrip())
    end = run + 1
    while end < len(lines) and (not lines[end].strip() or
                               len(lines[end]) - len(lines[end].lstrip()) > indentation):
        end += 1
    return textwrap.dedent("\n".join(lines[run + 1:end]))


def execute_gate(workflow, step_name, needs, event="pull_request", ref="refs/pull/1/merge"):
    env = dict(os.environ, NEEDS=json.dumps(needs), EVENT=event, REF=ref)
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c",
                             run_block(workflow, "ci", step_name)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stderr


def condition(workflow, job_name, event, ref, archive="skipped"):
    """Evaluate the small Actions expression subset used for job selection."""
    expression = next(line.strip()[4:] for line in job(workflow, job_name).splitlines()
                      if line.strip().startswith("if: "))
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    context = {"github": SimpleNamespace(event_name=event, ref=ref),
               "needs": SimpleNamespace(archive=SimpleNamespace(result=archive)),
               "cancelled": lambda: False, "always": lambda: True, "success": lambda: False,
               "contains": lambda sequence, value: value in sequence,
               "fromJSON": json.loads}
    return eval(expression, {"__builtins__": {}}, context)


class NumericGate(unittest.TestCase):
    step = "Require all selected checks"

    def needs(self, selected=True):
        return {"plan": {"result": "success", "outputs": {"run_numeric": str(selected).lower()}},
                **{name: {"result": "success" if selected else "skipped"}
                   for name in ("debug", "release", "package")}}

    def test_selected_success_and_intentional_skips_pass(self):
        for selected in (True, False):
            code, error = execute_gate("tests.yml", self.step, self.needs(selected))
            self.assertEqual(code, 0, error)

    def test_selected_failure_cancel_or_skip_is_rejected(self):
        for name in ("debug", "release", "package"):
            for result in ("failure", "cancelled", "skipped"):
                with self.subTest(name=name, result=result):
                    needs = self.needs()
                    needs[name]["result"] = result
                    code, _ = execute_gate("tests.yml", self.step, needs)
                    self.assertNotEqual(code, 0)

    def test_selection_failure_or_malformed_output_is_rejected(self):
        for result in ("failure", "cancelled", "skipped"):
            needs = self.needs(False)
            needs["plan"]["result"] = result
            self.assertNotEqual(execute_gate("tests.yml", self.step, needs)[0], 0)
        for output in ("", "tru", "null"):
            needs = self.needs(False)
            needs["plan"]["outputs"]["run_numeric"] = output
            self.assertNotEqual(execute_gate("tests.yml", self.step, needs)[0], 0)
        needs = self.needs(False)
        del needs["plan"]["outputs"]["run_numeric"]
        self.assertNotEqual(execute_gate("tests.yml", self.step, needs)[0], 0)

    def test_unselected_job_must_be_intentionally_skipped(self):
        for name in ("debug", "release", "package"):
            for result in ("success", "failure", "cancelled"):
                needs = self.needs(False)
                needs[name]["result"] = result
                self.assertNotEqual(execute_gate("tests.yml", self.step, needs)[0], 0)

    def test_final_gate_runs_even_when_an_upstream_job_fails(self):
        self.assertTrue(condition("tests.yml", "ci", "pull_request", "refs/pull/1/merge"))


class CoralGate(unittest.TestCase):
    step = "Require all selected Coral checks"

    def needs(self, selected=True):
        return {"plan": {"result": "success", "outputs": {"run_coral": str(selected).lower()}},
                **{name: {"result": "success" if selected else "skipped"}
                   for name in ("debug", "release")}}

    def test_selected_success_and_intentional_skips_pass(self):
        for selected in (True, False):
            code, error = execute_gate("coral.yml", self.step, self.needs(selected))
            self.assertEqual(code, 0, error)

    def test_each_configuration_must_match_the_selection(self):
        for selected in (True, False):
            expected = "success" if selected else "skipped"
            for name in ("debug", "release"):
                for result in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(selected=selected, name=name, result=result):
                        needs = self.needs(selected)
                        needs[name]["result"] = result
                        code, error = execute_gate("coral.yml", self.step, needs)
                        self.assertEqual(code == 0, result == expected, error)

    def test_failed_plan_or_invalid_output_is_rejected(self):
        for result in ("failure", "cancelled", "skipped"):
            needs = self.needs(False)
            needs["plan"]["result"] = result
            self.assertNotEqual(execute_gate("coral.yml", self.step, needs)[0], 0)
        for output in ("", "tru", "null", None):
            needs = self.needs(False)
            needs["plan"]["outputs"]["run_coral"] = output
            self.assertNotEqual(execute_gate("coral.yml", self.step, needs)[0], 0)
        needs = self.needs(False)
        del needs["plan"]["outputs"]["run_coral"]
        self.assertNotEqual(execute_gate("coral.yml", self.step, needs)[0], 0)

    def test_final_gate_runs_even_when_an_upstream_job_fails(self):
        self.assertTrue(condition("coral.yml", "ci", "pull_request", "refs/pull/1/merge"))


class DocumentationGate(unittest.TestCase):
    step = "Require documentation and the applicable performance checks"

    def needs(self, official=False):
        return {**{name: {"result": "success"} for name in ("harness", "build")},
                **{name: {"result": "success" if official else "skipped"}
                   for name in ("performance", "publish-history", "archive")}}

    def test_pr_preview_and_official_history_success_pass(self):
        cases = (("pull_request", "refs/pull/1/merge", False),
                 ("push", "refs/heads/master", True),
                 ("schedule", "refs/heads/master", True),
                 ("workflow_dispatch", "refs/heads/master", True),
                 ("workflow_dispatch", "refs/heads/feature", False))
        for event, ref, official in cases:
            with self.subTest(event=event, ref=ref):
                code, error = execute_gate("doxygen.yml", self.step, self.needs(official), event, ref)
                self.assertEqual(code, 0, error)

    def test_documentation_or_harness_cannot_be_silently_skipped(self):
        for name in ("harness", "build"):
            for result in ("failure", "cancelled", "skipped"):
                needs = self.needs()
                needs[name]["result"] = result
                self.assertNotEqual(execute_gate("doxygen.yml", self.step, needs)[0], 0)

    def test_each_official_history_stage_is_required(self):
        for name in ("performance", "publish-history", "archive"):
            for result in ("failure", "cancelled", "skipped"):
                needs = self.needs(True)
                needs[name]["result"] = result
                code, _ = execute_gate("doxygen.yml", self.step, needs,
                                       "push", "refs/heads/master")
                self.assertNotEqual(code, 0)

    def test_pr_cannot_publish_official_results(self):
        for name in ("performance", "publish-history", "archive"):
            needs = self.needs()
            needs[name]["result"] = "success"
            self.assertNotEqual(execute_gate("doxygen.yml", self.step, needs)[0], 0)

    def test_build_runs_without_measurements_for_pr_and_manual_preview(self):
        for event, ref in (("pull_request", "refs/pull/1/merge"),
                           ("workflow_dispatch", "refs/heads/feature")):
            with self.subTest(event=event, ref=ref):
                self.assertTrue(condition("doxygen.yml", "build", event, ref))

    def test_final_gate_runs_even_when_an_upstream_job_fails(self):
        self.assertTrue(condition("doxygen.yml", "ci", "pull_request", "refs/pull/1/merge"))

    def test_official_measurements_only_run_on_master(self):
        for event, ref, expected in (("pull_request", "refs/pull/1/merge", False),
                                     ("push", "refs/heads/master", True),
                                     ("schedule", "refs/heads/master", True),
                                     ("workflow_dispatch", "refs/heads/master", True),
                                     ("workflow_dispatch", "refs/heads/feature", False)):
            with self.subTest(event=event, ref=ref):
                self.assertEqual(condition("doxygen.yml", "performance", event, ref), expected)
        self.assertFalse(condition("doxygen.yml", "build", "push", "refs/heads/master", "failure"))
        self.assertTrue(condition("doxygen.yml", "build", "push", "refs/heads/master", "success"))


if __name__ == "__main__":
    unittest.main()
