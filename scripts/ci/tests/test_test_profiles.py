"""Keep every quick registration in the quick build, including future suites."""

import fnmatch
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[3]


def cmake_list(path, name):
    text = path.read_text()
    match = re.search(r"set\(" + name + r"\s+([^)]*)\)", text)
    if not match:
        raise AssertionError(f"Missing CMake list {name}")
    return re.sub(r"#[^\n]*", "", match.group(1)).split()


def registered_suites(text):
    result = set()
    for macro, suite in re.findall(
            r"\b(GTEST_TEST|TEST(?:_[A-Z]+)*|TYPED_TEST(?:_[A-Z]+)*)\s*\(\s*(\w+)", text):
        if macro in ("TYPED_TEST_SUITE", "TYPED_TEST_SUITE_P"):
            continue
        if macro not in ("TEST", "GTEST_TEST", "TEST_F", "TEST_P"):
            raise AssertionError(f"Unsupported registration macro {macro}")
        if macro == "TEST_P":
            prefixes = re.findall(
                r"INSTANTIATE_TEST_SUITE_P\s*\(\s*(\w+)\s*,\s*" + suite + r"\s*,", text)
            if not prefixes:
                raise AssertionError(f"Unresolved parameterized suite {suite}")
            result.update(prefix + "/" + suite for prefix in prefixes)
        else:
            result.add(suite)
    return result


def check_nightly_source(text, patterns):
    suites = registered_suites(text)
    if not suites:
        raise AssertionError("An empty/unknown source must remain in quick")
    for suite in suites:
        matches = [category for category, entries in patterns.items()
                   if any(fnmatch.fnmatchcase(suite, p) for p in entries)]
        if len(matches) != 1 or matches[0] not in ("validation", "application"):
            raise AssertionError(f"Nightly source has quick/unclassified suite {suite}: {matches}")
    return suites


class Profiles(unittest.TestCase):
    def test_nightly_translation_units_contain_no_quick_or_unknown_suites(self):
        categories = ROOT / "gtests/test_categories.cmake"
        patterns = {category: cmake_list(categories, f"IMMERSX_{category.upper()}_TEST_SUITES")
                    for category in ("unit", "integration", "validation", "application")}
        sources = cmake_list(ROOT / "gtests/nightly_sources.cmake", "IMMERSX_NIGHTLY_TEST_SOURCES")
        self.assertEqual(len(sources), len(set(sources)))
        for source in sources:
            with self.subTest(source=source):
                check_nightly_source((ROOT / "gtests" / source).read_text(), patterns)


    def test_new_quick_or_unknown_registration_in_nightly_file_is_rejected(self):
        patterns = {"unit": ["Unit"], "integration": ["Integration"],
                    "validation": ["Validation"], "application": ["Application"]}
        original = "TEST(Validation, Oracle) {}"
        self.assertEqual(check_nightly_source(original, patterns), {"Validation"})
        for registration in ("TEST(Unit, Added) {}", "TEST_F(Integration, Added) {}",
                             "GTEST_TEST(Integration, Added) {}", "TEST(Unknown, Added) {}",
                             "TYPED_TEST_P(Unknown, Added) {}"):
            with self.subTest(registration=registration), self.assertRaises(AssertionError):
                check_nightly_source(original + "\n" + registration, patterns)

    def test_typed_instantiation_cannot_hide_a_quick_specialization(self):
        patterns = {"validation": ["Typed/0"], "integration": ["Typed/1"]}
        with self.assertRaises(AssertionError):
            check_nightly_source("TYPED_TEST(Typed, Oracle) {}", patterns)

    def test_registration_parser_covers_multiline_and_parameterized_suites(self):
        self.assertEqual(registered_suites("TEST(Quick, A) {}\nTEST_F(Fixture, B) {}"),
                         {"Quick", "Fixture"})
        self.assertEqual(registered_suites("TEST_P(Param, A) {}\n"
            "INSTANTIATE_TEST_SUITE_P(Validation,\n Param, Values(1));"), {"Validation/Param"})
        with self.assertRaises(AssertionError):
            registered_suites("TYPED_TEST(Typed, A) {}")
        with self.assertRaises(AssertionError):
            registered_suites("TEST_P(Unresolved, A) {}")
        with self.assertRaises(AssertionError):
            registered_suites("TYPED_TEST_P(Unresolved, A) {}")



if __name__ == "__main__":
    unittest.main()
