#!/usr/bin/env python3
"""Run serial spherical CCO cases from a JSON plan, preserving each run."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name("cco.prm.in").read_text()
    plan = json.loads(args.plan.read_text())
    environment = os.environ.copy()
    for name in ("DEAL_II_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    metadata = {
        "host": platform.node(),
        "executable": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "plan": plan,
        "thread_environment": {name: environment[name] for name in ("DEAL_II_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    if hasattr(os, "sched_getaffinity"):
        metadata["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        metadata["cpu_models"] = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")})
    cache = executable.parent / "CMakeCache.txt"
    if cache.exists():
        metadata["build_configuration"] = [line for line in cache.read_text().splitlines() if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS_RELEASE:", "DEAL_II_DIR:", "ENABLE_COUPLED_PROBLEMS:"))]
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    for case in plan:
        directory = args.output / case["name"]
        directory.mkdir(exist_ok=False)
        prm = directory / "generate.prm"
        mode = case.get("mode", "random")
        overrides = {
            "Input and ports": {"Mode": "generate", "Evaluate reference": "true", "Seed radius": case.get("radius", .1)},
            "Sampling": {"Terminal sampling": mode, "Terminal count": case.get("k", 1), "Random seed": case.get("seed", 23), "Density mesh exponent": case.get("beta", 3), "Sample budget per insertion": case.get("budget", 4000)},
            "Growth": {"Candidate vessels": case.get("candidates", 12)},
            "Objectives": {"Maximum service distance": case.get("service_distance", 1), "Coverage quadrature order": case.get("coverage_order", 2)},
            "Domain": {"Shape": "sphere", "Sphere center": "0,0,0", "Sphere radius": 1, "Uniform refinements": case.get("refinement", 3), "Refinement focus": "none"},
            "Output": {"Directory": directory.resolve() / "generation", "Save physical archive": "true", "Write tables": "true", "Write meshes": case.get("meshes", "false")},
        }
        text = template + "\nsubsection Vascularization\n"
        for section, values in overrides.items():
            text += f"  subsection {section}\n"
            text += "".join(f"    set {key} = {value}\n" for key, value in values.items())
            text += "  end\n"
        prm.write_text(text + "end\n")
        start = time.monotonic()
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with (directory / "stdout.log").open("w") as log:
            result = subprocess.run([str(executable), str(prm.resolve())], stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record = dict(case, exit_code=result.returncode, process_seconds=time.monotonic() - start,
                      user_seconds=usage_after.ru_utime - usage_before.ru_utime,
                      system_seconds=usage_after.ru_stime - usage_before.ru_stime)
        summary = directory / "generation" / "summary.csv"
        if summary.exists():
            with summary.open() as stream:
                record.update(next(csv.DictReader(stream)))
        (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        fields = list(dict.fromkeys(key for item in records for key in item))
        with (args.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({key: record.get(key) for key in ("name", "exit_code", "arterial_terminals", "arterial_coverage", "generation_seconds", "process_seconds")}), flush=True)


if __name__ == "__main__":
    main()
