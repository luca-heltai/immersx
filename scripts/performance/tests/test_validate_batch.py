"""CI validates the triggering source; old failures remain immutable history."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_batch import validate_batch


def item(sha, status="passed", run="1"):
    return {"id": f"{sha}-{run}-1", "sha": sha, "run": run, "attempt": 1,
            "started_at": f"2026-11-05T00:00:0{run}+00:00",
            "status": status, "errors": [], "missing_cases": [],
            "records": [{"status": status}]}


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
            {"source_sha": "current", "selected": selected}))

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
