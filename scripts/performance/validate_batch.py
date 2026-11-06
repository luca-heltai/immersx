#!/usr/bin/env python3
"""Require a complete batch and a valid current measurement, retaining old failures."""
import argparse
from pathlib import Path

from history import attempts, expected_cases, load, normalize, selected_attempt


def validate_batch(batch, archive):
    batch = Path(batch)
    plan = load(batch / "plan.json")
    selected = plan["selected"]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate commits in the measurement plan")
    measured = attempts(batch)
    if (len(measured) != len(selected) or
            {item["sha"] for item in measured} != set(selected)):
        raise ValueError("Missing or unexpected measurement attempts")
    for item in measured:
        if item["status"] not in {"passed", "failed"}:
            raise ValueError("Invalid measurement status")
        if item["status"] == "failed":
            print(f"Recorded failed measurement: {item['sha']} ({item['id']})")

    current = selected_attempt([item for item in [*attempts(archive), *measured]
                                if item["sha"] == plan["source_sha"]])
    if current is None:
        raise ValueError("Current source has no measurement")
    if (current["status"] != "passed" or current.get("errors") or
            current.get("missing_cases") or not current.get("records") or
            any(record["status"] != "passed" for record in current["records"])):
        raise ValueError("Current source benchmark validation failed; see archived attempts")
    configuration = plan["configuration"]
    if current.get("configuration") != configuration:
        raise ValueError("Current source benchmark configuration differs from the plan")
    try:
        records = [normalize(record["raw"], configuration)
                   for record in current["records"]]
        keys = [(record["case"], record["dimension"], record["ranks"])
                for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Current source benchmark records are invalid") from exc
    if (records != current["records"] or
            any(record["status"] != "passed" for record in records) or
            len(keys) != len(set(keys)) or
            set(keys) != set(expected_cases(configuration))):
        raise ValueError("Current source benchmark cases are incomplete or inconsistent")
    print(f"Current source benchmark validation passed: {plan['source_sha']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    validate_batch(args.batch, args.archive)


if __name__ == "__main__":
    main()
