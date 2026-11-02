#!/usr/bin/env python3
"""Run benchmark correctness checks without recording performance history."""

import argparse
import subprocess

from history import expected_cases, parse_jsonl


def validate(text, family):
    configuration = {"repetitions": 1, "refinements": 1,
                     "quadrature_recipe": "CI correctness smoke"}
    records, errors = parse_jsonl(text, configuration)
    required = {case for case in expected_cases(configuration)
                if (case[0] == "lifted_geometry_rebuild" if family == "lifted"
                    else case[0] != "lifted_geometry_rebuild" and
                    not case[0].startswith("particle_penalty_"))}
    present = {(r["case"], r["dimension"], r["ranks"]) for r in records}
    if errors or present != required or any(r["status"] != "passed" for r in records):
        raise ValueError(f"Invalid {family} benchmark smoke: errors={errors}, "
                         f"missing={sorted(required - present)}, extra={sorted(present - required)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("weak", "lifted"), required=True)
    parser.add_argument("executable")
    args = parser.parse_args()
    run = subprocess.run([args.executable, "--refinements=1", "--repetitions=1"],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(run.stdout, end="", flush=True)
    print(run.stderr, end="", flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)
    validate(run.stdout, args.family)


if __name__ == "__main__":
    main()
