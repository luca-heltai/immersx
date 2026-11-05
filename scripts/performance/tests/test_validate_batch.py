"""CI validates the triggering source; old failures remain immutable history."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_batch import validate_batch
from history import expected_cases, normalize
from test_particle_penalty import penalty_raw

HERE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())


def complete_records():
    # Schema fixtures from committed reports, never published measurements.
    reports = HERE.parents[1] / "benchmarks/reports"
    references = ("weakterm-review-release-dealii-9.8.0-rc1-arm64.jsonl",
                  "lifted-review-release-np1-dealii-9.8.0-rc1-arm64.jsonl")
    raw_records = [json.loads(line) for name in references
                   for line in (reports / name).read_text().splitlines() if line.strip()]
    raw_records.extend({**penalty_raw(case), "ranks": rank,
                        "active_particle_ranks": rank}
                       for case, _, rank in expected_cases(CONFIG)
                       if case.startswith("particle_penalty_"))
    return [normalize({**raw, "repetitions": CONFIG["repetitions"]}, CONFIG)
            for raw in raw_records]


def item(sha, status="passed", run="1"):
    return {"id": f"{sha}-{run}-1", "sha": sha, "run": run, "attempt": 1,
            "started_at": f"2026-11-05T00:00:0{run}+00:00",
            "status": status, "errors": [], "missing_cases": [],
            "configuration": copy.deepcopy(CONFIG), "records": complete_records()}


class BatchValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name) / "batch"
        self.archive = Path(self.tmp.name) / "history"
        self.batch.mkdir()
        self.archive.mkdir()

    def plan(self, selected):
        (self.batch / "plan.json").write_text(json.dumps(
            {"source_sha": "current", "selected": selected, "configuration": CONFIG}))

    def save(self, directory, value):
        dest = directory / "attempts" / (value["id"] + ".json")
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(value))
        return dest

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            validate_batch(self.batch, self.archive)

    def test_valid_current_does_not_hide_or_delete_historical_failure(self):
        self.plan(["old", "current"])
        failed = self.save(self.batch, item("old", "failed"))
        original = failed.read_bytes()
        self.save(self.batch, item("current"))
        self.check()
        self.assertEqual(failed.read_bytes(), original)

    def test_current_failure_is_fatal(self):
        self.plan(["old", "current"])
        self.save(self.batch, item("old"))
        self.save(self.batch, item("current", "failed"))
        with self.assertRaisesRegex(ValueError, "Current source benchmark"):
            self.check()

    def test_incomplete_or_inconsistent_current_result_is_fatal(self):
        for update in ({"errors": [{"reason": "Malformed JSON"}]},
                       {"missing_cases": ["mass_q1"]},
                       {"records": []}, {"records": [{"status": "failed"}]}):
            with self.subTest(update=update):
                self.plan(["current"])
                self.save(self.batch, {**item("current"), **update})
                with self.assertRaisesRegex(ValueError, "Current source benchmark"):
                    self.check()

    def test_fixture_covers_every_current_serial_and_mpi_case(self):
        records = complete_records()
        self.assertEqual(len(records), 39)
        self.assertEqual({(r["case"], r["dimension"], r["ranks"]) for r in records},
                         set(expected_cases(CONFIG)))

    def test_declared_success_with_one_or_38_cases_is_rejected(self):
        for count in (1, 38):
            for directory in (self.batch, self.archive):
                with self.subTest(count=count, directory=directory):
                    self.plan(["current"] if directory == self.batch else [])
                    for root in (self.batch, self.archive):
                        for path in (root / "attempts").glob("*.json"):
                            path.unlink()
                    result = item("current")
                    result["records"] = result["records"][:count]
                    self.save(directory, result)
                    with self.assertRaisesRegex(ValueError, "Current source benchmark"):
                        self.check()

    def test_duplicate_case_cannot_replace_a_missing_case(self):
        self.plan(["current"])
        result = item("current")
        result["records"][-1] = copy.deepcopy(result["records"][0])
        self.save(self.batch, result)
        with self.assertRaisesRegex(ValueError, "Current source benchmark"):
            self.check()

    def test_current_configuration_must_match_the_plan(self):
        self.plan(["current"])
        result = item("current")
        result["configuration"]["repetitions"] += 1
        self.save(self.batch, result)
        with self.assertRaisesRegex(ValueError, "Current source benchmark configuration"):
            self.check()

    def test_raw_numerical_failure_cannot_be_hidden_by_passed_summary(self):
        self.plan(["current"])
        result = item("current")
        target = next(r for r in result["records"] if r["case"] == "mass_q1")
        target["raw"]["relative_matrix_error"] = 1.e-3
        self.save(self.batch, result)
        with self.assertRaisesRegex(ValueError, "Current source benchmark"):
            self.check()

    def test_raw_case_identity_must_match_the_summary(self):
        self.plan(["current"])
        result = item("current")
        target = next(r for r in result["records"] if r["case"] == "mass_q1")
        target["raw"]["dimension"] = 2 if target["dimension"] == 1 else 1
        self.save(self.batch, result)
        with self.assertRaisesRegex(ValueError, "Current source benchmark"):
            self.check()

    def test_every_selected_commit_must_have_exactly_one_attempt(self):
        for values in ([item("old")], [item("old"), item("extra")],
                       [item("old"), item("old", run="2")]):
            with self.subTest(values=values):
                for path in (self.batch / "attempts").glob("*.json"):
                    path.unlink()
                self.plan(["old", "current"])
                for value in values:
                    self.save(self.batch, value)
                with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
                    self.check()

    def test_duplicate_plan_and_invalid_status_are_fatal(self):
        self.plan(["current", "current"])
        with self.assertRaisesRegex(ValueError, "Duplicate commits"):
            self.check()
        self.plan(["current"])
        self.save(self.batch, item("current", "skipped"))
        with self.assertRaisesRegex(ValueError, "Invalid measurement status"):
            self.check()

    def test_scheduled_backfill_requires_a_current_archived_measurement(self):
        self.plan(["old"])
        self.save(self.batch, item("old", "failed"))
        with self.assertRaisesRegex(ValueError, "Current source has no measurement"):
            self.check()
        self.save(self.archive, item("current"))
        self.check()

    def test_empty_batch_can_reuse_current_success(self):
        self.plan([])
        self.save(self.archive, item("current"))
        self.check()

    def test_late_success_cannot_hide_a_newer_current_failure(self):
        self.plan(["current"])
        self.save(self.batch, item("current"))
        self.save(self.archive, item("current", "failed", run="2"))
        with self.assertRaisesRegex(ValueError, "Current source benchmark"):
            self.check()


if __name__ == "__main__":
    unittest.main()
