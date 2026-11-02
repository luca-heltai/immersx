import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


spec = importlib.util.spec_from_file_location(
    "report_timings", Path(__file__).resolve().parents[1] / "report_timings.py"
)
report_timings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_timings)


class Timings(unittest.TestCase):
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
