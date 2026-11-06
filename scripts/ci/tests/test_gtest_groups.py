from collections import defaultdict
import fnmatch
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = """Typed/0.  # TypeParam = int
  Unit
BoundaryConditions.
  Serial
  MPI_Distributed
  BOTH_Shared
KnownTerm.
  Serial
Validation.
  BOTH_Oracle
ValidationTyped/0.  # TypeParam = double
  BOTH_Oracle
ValidationMany.
""" + "".join(f"  BOTH_Oracle{i}\n" for i in range(18)) + """  MPI_Distributed
  DISABLED_BOTH_Held
Values/ParameterizedValidation.
  BOTH_Oracle/Value0
Application.
  Run
OtherApplication.
  BOTH_Run
"""
CASES = {
    "Typed/0.Unit", "BoundaryConditions.Serial",
    "BoundaryConditions.MPI_Distributed", "BoundaryConditions.BOTH_Shared",
    "KnownTerm.Serial", "Validation.BOTH_Oracle", "ValidationTyped/0.BOTH_Oracle",
    "Application.Run", "OtherApplication.BOTH_Run",
    "ValidationMany.MPI_Distributed", "ValidationMany.DISABLED_BOTH_Held",
    "Values/ParameterizedValidation.BOTH_Oracle/Value0",
} | {f"ValidationMany.BOTH_Oracle{i}" for i in range(18)}
ORDERED_CASES = [
    "Typed/0.Unit", "BoundaryConditions.Serial",
    "BoundaryConditions.MPI_Distributed", "BoundaryConditions.BOTH_Shared",
    "KnownTerm.Serial", "Validation.BOTH_Oracle", "ValidationTyped/0.BOTH_Oracle",
] + [f"ValidationMany.BOTH_Oracle{i}" for i in range(18)] + [
    "ValidationMany.MPI_Distributed", "ValidationMany.DISABLED_BOTH_Held",
    "Values/ParameterizedValidation.BOTH_Oracle/Value0",
    "Application.Run", "OtherApplication.BOTH_Run",
]


def matches(name, expression):
    positive, _, negative = expression.partition("-")
    return any(fnmatch.fnmatchcase(name, p) for p in positive.split(":")) and not any(
        fnmatch.fnmatchcase(name, p) for p in negative.split(":") if p
    )


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ctest"),
                     "CMake and CTest are required")
class Discovery(unittest.TestCase):
    def test_compiled_inventory_is_partitioned_without_empty_mpi_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "registry.py"
            executable.write_text("#!/usr/bin/env python3\nprint(" + repr(REGISTRY) + ")\n")
            executable.chmod(0o755)
            categories = directory / "categories.cmake"
            categories.write_text(
                "set(IMMERSX_TEST_PRIMARY_CATEGORIES unit integration validation application)\n"
                "set(IMMERSX_UNIT_TEST_SUITES Typed/*)\n"
                "set(IMMERSX_INTEGRATION_TEST_SUITES BoundaryConditions KnownTerm)\n"
                "set(IMMERSX_VALIDATION_TEST_SUITES Validation ValidationTyped/* "
                "ValidationMany Values/ParameterizedValidation)\n"
                "set(IMMERSX_APPLICATION_TEST_SUITES Application OtherApplication)\n"
            )
            groups = directory / "groups.cmake"
            command = ["cmake", f"-DTEST_EXECUTABLE={executable}",
                       f"-DCATEGORY_FILE={categories}", "-DTEST_NAME_PREFIX=ImmersX.Debug",
                       f"-DTEST_RESULTS_DIR={directory / 'results'}", "-DMPIEXEC=/bin/true",
                       f"-DOUTPUT_FILE={groups}", "-P",
                       str(ROOT / "gtests" / "discover_test_groups.cmake")]
            subprocess.run(command, check=True, capture_output=True, text=True)
            (directory / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\nproject(Groups NONE)\n"
                "enable_testing()\ninclude(groups.cmake)\n"
            )
            subprocess.run(["cmake", "-S", str(directory), "-B", str(directory / "build")],
                           check=True, capture_output=True, text=True)
            manifest = json.loads(subprocess.check_output(
                ["ctest", "--test-dir", str(directory / "build"), "--show-only=json-v1"],
                text=True))
            serial, mpi, outputs = [], [], []
            batches = defaultdict(list)
            previous = {}
            for group in manifest["tests"]:
                expression = next(a.split("=", 1)[1] for a in group["command"]
                                  if a.startswith("GTEST_FILTER="))
                selected = [name for name in ORDERED_CASES if matches(name, expression)]
                self.assertTrue(selected, group["name"])
                labels = next(p["value"] for p in group["properties"] if p["name"] == "LABELS")
                self.assertIn("ImmersX.TestOutput", next(p["value"] for p in group["properties"]
                                                       if p["name"] == "RESOURCE_LOCK"))
                (mpi if "mpi" in labels else serial).extend(selected)
                outputs.extend(a for a in group["command"] if a.startswith("GTEST_OUTPUT="))
                if "integration" in labels:
                    self.assertEqual(len({name.split(".", 1)[0] for name in selected}), 1)
                    self.assertTrue(any(label.startswith("suite:") for label in labels))
                if "validation" in labels or "application" in labels:
                    self.assertLessEqual(len(selected), 15)
                    self.assertEqual(next(p["value"] for p in group["properties"]
                                          if p["name"] == "TIMEOUT"), 115)
                    category = "validation" if "validation" in labels else "application"
                    mode = "MPI" if "mpi" in labels else "Serial"
                    key = category, mode
                    batches[key].extend(selected)
                    dependencies = next((p["value"] for p in group["properties"]
                                         if p["name"] == "DEPENDS"), [])
                    self.assertEqual(dependencies, [previous[key]] if key in previous else [])
                    previous[key] = group["name"]
                if ".integration." in group["name"]:
                    self.assertIn("quick", labels)
                    self.assertTrue(any(label.startswith("suite:") for label in labels))
            expected_serial = {name for name in CASES if ".MPI_" not in name}
            expected_mpi = {name for name in CASES if ".MPI_" in name or ".BOTH_" in name}
            self.assertEqual(set(serial), expected_serial)
            self.assertEqual(len(serial), len(expected_serial))
            self.assertEqual(set(mpi), expected_mpi)
            self.assertEqual(len(mpi), len(expected_mpi))
            self.assertEqual(len(outputs), len(manifest["tests"]))
            self.assertEqual(len(set(outputs)), len(outputs))
            self.assertEqual(set(ORDERED_CASES), CASES)
            category_prefixes = {"validation": ("Validation", "Values/"),
                                 "application": ("Application.", "OtherApplication.")}
            for (category, mode), selected in batches.items():
                expected = [name for name in ORDERED_CASES
                            if name.startswith(category_prefixes[category])
                            and name in (expected_mpi if mode == "MPI" else expected_serial)]
                self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
