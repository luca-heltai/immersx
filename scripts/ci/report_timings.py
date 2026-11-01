#!/usr/bin/env python3
"""Collect GoogleTest case timings, including partial CTest failure logs."""

import argparse
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def xml_cases(path):
    root = ET.parse(path).getroot()
    rank_match = re.search(r"\.rank-(\d+)\.xml$", path.name)
    rank = int(rank_match.group(1)) if rank_match else None
    group = re.sub(r"(?:\.rank-\d+)?\.xml$", "", path.name)
    group = re.sub(r"-[a-f0-9]{8}$", "", group)
    if rank is None and re.search(r"_(unit|integration|validation|application)_MPI(?:_|$)", group):
        rank = 0
    for case in root.iter("testcase"):
        status = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.find("skipped") is not None or case.get("status") == "notrun":
            status = "skipped"
        yield {
            "test": case.get("classname", "") + "." + case.attrib["name"],
            "seconds": float(case.attrib.get("time", "0")),
            "status": status,
            "rank": rank,
            "group": group,
            "source": str(path),
        }


def log_cases(path):
    # CTest retains successful-group output here too. Interrupted groups may
    # have completed cases even when GoogleTest never wrote its final XML.
    expression = re.compile(
        r"^\[\s*(OK|FAILED)\s*\]\s+(\S+)\s+\((\d+) ms\)\s*$"
    )
    group = None
    for line in path.read_text(errors="replace").splitlines():
        header = re.match(r"^\d+/\d+ Testing: (.+)$", line)
        if header:
            group = re.sub(r"[^a-zA-Z0-9_]", "_", header.group(1))
        match = expression.match(line)
        if match:
            yield {
                "test": match.group(2),
                "seconds": int(match.group(3)) / 1000,
                "status": "passed" if match.group(1) == "OK" else "failed",
                "rank": None,
                "group": group,
                "source": str(path),
            }


def collect(build_dir):
    cases = []
    for path in sorted((build_dir / "gtests" / "test-results").glob("*.xml")):
        cases.extend(xml_cases(path))
    # Keep XML as the authoritative result where available. Serial and MPI
    # remain distinct observations; root-only log output cannot recover rank
    # identities and is used only for cases with no XML result at all.
    reported = {(case["group"], case["test"]) for case in cases}
    reported_tests = {case["test"] for case in cases}
    logs = build_dir / "Testing" / "Temporary"
    for name in ("LastTest.log", "LastTest.log.tmp"):
        path = logs / name
        if path.exists():
            for case in log_cases(path):
                key = (case["group"], case["test"])
                if key not in reported and (case["group"] is not None or
                                           case["test"] not in reported_tests):
                    cases.append(case)
                    reported.add(key)
    cases.sort(key=lambda case: (-case["seconds"], case["test"], case["source"]))
    groups = []
    junit = build_dir / "ctest-results.xml"
    if junit.exists():
        for case in ET.parse(junit).getroot().iter("testcase"):
            status = "passed"
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
            elif case.find("skipped") is not None:
                status = "skipped"
            groups.append({"test": case.attrib["name"], "status": status,
                           "seconds": float(case.attrib.get("time", "0"))})
    return {"build_dir": str(build_dir), "cases": cases, "groups": groups}


def summary(report):
    lines = ["### GoogleTest case timings", ""]
    failures = [group["test"] for group in report["groups"] if group["status"] == "failed"]
    if failures:
        lines += ["CTest group failures: " + ", ".join(f"`{name}`" for name in failures), ""]
    lines += ["MPI case XML describes rank zero; CTest group status covers every rank.", ""]
    if not report["cases"]:
        return "\n".join(lines + ["No completed case timings are available.", ""])
    lines += ["| Case | Seconds | Rank | Result |", "| --- | ---: | ---: | --- |"]
    for case in report["cases"][:20]:
        name = case["test"].replace("|", "\\|").replace("\n", " ")
        rank = case["rank"] if case["rank"] is not None else "serial/log"
        lines.append(
            f"| `{name}` | {case['seconds']:.3f} | {rank} | {case['status']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.build_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    rendered = summary(report)
    print(rendered)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(rendered)


if __name__ == "__main__":
    main()
