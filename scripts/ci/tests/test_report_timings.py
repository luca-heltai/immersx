import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


spec = importlib.util.spec_from_file_location(
    "report_timings", Path(__file__).resolve().parents[1] / "report_timings.py"
)
report_timings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_timings)


class Timings(unittest.TestCase):
    def test_incomplete_final_ninja_record_preserves_case_and_build_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            results = build / "gtests" / "test-results"
            results.mkdir(parents=True)
            (results / "group.xml").write_text(
                '<testsuites><testcase classname="Suite" name="Case" '
                'time="0.5"/></testsuites>'
            )
            (build / ".ninja_log").write_text(
                "# ninja log v5\n0\t1500\t1\tcomplete.o\tcompile\n1500\t"
            )
            report = report_timings.collect(build)
            self.assertEqual(report["cases"][0]["test"], "Suite.Case")
            self.assertEqual(report["build_commands"],
                             [{"seconds": 1.5, "outputs": ["complete.o"]}])
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("incomplete final Ninja record", report["warnings"][0])
            self.assertIn(report["warnings"][0], report_timings.summary(report))

    def test_malformed_complete_ninja_record_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / ".ninja_log").write_text("# ninja log v5\n0\t1000\n")
            with self.assertRaisesRegex(ValueError, "malformed Ninja record"):
                report_timings.collect(build)

    def test_build_timings_keep_latest_outputs_and_count_multioutput_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / ".ninja_log").write_text(
                "# ninja log v5\n"
                "0\t9000\t1\told.o\tcompile\n"
                "0\t1000\t2\told.o\tcompile\n"
                "1000\t3000\t3\tprogram\tlink\n"
                "1000\t3000\t3\ttest-registry.cmake\tlink\n"
            )
            commands = report_timings.collect(build)["build_commands"]
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0]["seconds"], 2)
            self.assertEqual(set(commands[0]["outputs"]),
                             {"program", "test-registry.cmake"})
            self.assertEqual(commands[1], {"seconds": 1, "outputs": ["old.o"]})

    def test_case_budget_fails_after_preserving_the_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            results = build / "gtests" / "test-results"
            results.mkdir(parents=True)
            source = results / "group.xml"
            output = build / "timings.json"
            for elapsed, expected in ((120, 0), (120.001, 1)):
                with self.subTest(elapsed=elapsed):
                    source.write_text(
                        '<testsuites><testsuite><testcase classname="Suite" '
                        f'name="Case" time="{elapsed}"/></testsuite></testsuites>'
                    )
                    result = subprocess.run(
                        [sys.executable, str(Path(report_timings.__file__)),
                         "--build-dir", str(build), "--output", str(output)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    report = json.loads(output.read_text())
                    self.assertEqual(len(report["over_budget"]), expected)
                    self.assertEqual(report["cases"][0]["seconds"], elapsed)

    def test_case_budget_rejects_nonfinite_and_nonpositive_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            output = build / "timings.json"
            for limit in ("nan", "inf", "-inf", "0", "-1"):
                with self.subTest(limit=limit):
                    result = subprocess.run(
                        [sys.executable, str(Path(report_timings.__file__)),
                         "--build-dir", str(build), "--output", str(output),
                         "--max-case-seconds=" + limit],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("must be finite and positive", result.stderr)
                    self.assertFalse(output.exists())

    def test_nonroot_watchdog_timeout_preserves_passing_root_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            results = build / "gtests" / "test-results"
            results.mkdir(parents=True)
            (results / "ImmersX_integration_MPI_Suite-abcd1234.xml").write_text(
                '<testsuites><testcase classname="Suite" name="BOTH_Case" '
                'time="0.25"/></testsuites>'
            )
            logs = build / "Testing" / "Temporary"
            logs.mkdir(parents=True)
            text = (
                "2/10 Testing: ImmersX.integration.MPI.Suite\n"
                "[       OK ] Suite.BOTH_Case (250 ms)\n"
                "[ RUN      ] Suite.LaterCase\n"
                "GoogleTest Suite.BOTH_Case exceeded the 120 second "
                "correctness-test budget.\n"
            )
            for name in ("LastTest.log", "LastTest.log.tmp"):
                (logs / name).write_text(text)
            (build / "ctest-results.xml").write_text(
                '<testsuite><testcase name="ImmersX.integration.MPI.Suite" '
                'time="120.1"><failure/></testcase></testsuite>'
            )
            for limit in (120, 240):
                with self.subTest(limit=limit):
                    output = build / "timings.json"
                    result = subprocess.run(
                        [sys.executable, str(Path(report_timings.__file__)),
                         "--build-dir", str(build), "--output", str(output),
                         "--max-case-seconds", str(limit)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    report = json.loads(output.read_text())
                    self.assertEqual(len(report["cases"]), 2)
                    self.assertEqual(report["cases"][0]["status"], "timed_out")
                    self.assertEqual(report["cases"][0]["seconds"], 120)
                    self.assertEqual(report["cases"][0]["test"],
                                     "Suite.BOTH_Case")
                    self.assertIsNone(report["cases"][0]["rank"])
                    self.assertEqual(report["cases"][1]["status"], "passed")
                    self.assertEqual(report["cases"][1]["rank"], 0)
                    self.assertEqual(len(report["over_budget"]), 1)
                    self.assertIn("Suite.BOTH_Case", result.stdout)
                    self.assertIn("timed_out", result.stdout)

    def test_xml_ranks_status_and_partial_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            results = build / "gtests" / "test-results"
            results.mkdir(parents=True)
            (results / "group.rank-1.xml").write_text(
                '<testsuites><testsuite><testcase classname="Suite" name="BOTH_Case" '
                'time="1.25"><failure/></testcase><testcase classname="Suite" '
                'name="Skipped" time="0"><skipped/></testcase></testsuite></testsuites>'
            )
            logs = build / "Testing" / "Temporary"
            logs.mkdir(parents=True)
            (logs / "LastTest.log").write_text(
                "[       OK ] Suite.BOTH_Case (1250 ms)\n"
                "[  FAILED  ] Suite.Partial (3400 ms)\n"
                "[  FAILED  ] 1 test, listed below:\n"
            )
            cases = report_timings.collect(build)["cases"]
            self.assertEqual([c["test"] for c in cases],
                             ["Suite.Partial", "Suite.BOTH_Case", "Suite.Skipped"])
            self.assertEqual(cases[0]["seconds"], 3.4)
            self.assertEqual(cases[0]["status"], "failed")
            self.assertEqual(cases[1]["rank"], 1)
            self.assertEqual(cases[1]["status"], "failed")
            self.assertEqual(cases[2]["status"], "skipped")

    def test_no_completed_tests_after_build_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = report_timings.collect(Path(temporary))
            self.assertEqual(report["cases"], [])
            self.assertIn("No completed case timings", report_timings.summary(report))

    def test_partial_mpi_timing_is_retained_when_serial_xml_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            results = build / "gtests" / "test-results"
            results.mkdir(parents=True)
            (results / "ImmersX_integration_Serial_Suite-abcd1234.xml").write_text(
                '<testsuites><testsuite><testcase classname="Suite" name="BOTH_Case" '
                'time="1"/></testsuite></testsuites>'
            )
            logs = build / "Testing" / "Temporary"
            logs.mkdir(parents=True)
            (logs / "LastTest.log.tmp").write_text(
                "2/10 Testing: ImmersX.integration.MPI.Suite\n"
                "[       OK ] Suite.BOTH_Case (2000 ms)\n"
            )
            cases = report_timings.collect(build)["cases"]
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]["seconds"], 2)
            self.assertEqual(cases[0]["group"], "ImmersX_integration_MPI_Suite")

    def test_corrupt_xml_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "gtests" / "test-results"
            results.mkdir(parents=True)
            (results / "broken.xml").write_text("<testsuites>")
            with self.assertRaises(ET.ParseError):
                report_timings.collect(Path(temporary))

    def test_nonroot_failure_is_not_hidden_by_passing_rank_zero_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            results = build / "gtests" / "test-results"
            results.mkdir(parents=True)
            (results / "group.xml").write_text(
                '<testsuites><testsuite><testcase classname="Suite" name="MPI_Case" '
                'time="1"/></testsuite></testsuites>'
            )
            (build / "ctest-results.xml").write_text(
                '<testsuite><testcase name="ImmersX.integration.MPI.Suite" time="2">'
                '<failure message="MPI rank 1 failed"/></testcase></testsuite>'
            )
            report = report_timings.collect(build)
            self.assertEqual(report["cases"][0]["status"], "passed")
            self.assertEqual(report["groups"][0]["status"], "failed")
            self.assertIn("CTest group failures", report_timings.summary(report))


if __name__ == "__main__":
    unittest.main()
