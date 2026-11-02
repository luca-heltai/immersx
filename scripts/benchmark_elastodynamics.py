#!/usr/bin/env python3
"""Record native and adapter elastodynamics timings from existing executables."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_text(source, output, refinement, variant):
    native = variant != "elasticity"
    name = "elastodynamics" if native else "elasticity"
    text = (source / "gtests/parameters" / f"issue_209_{name}.prm.in").read_text()
    text = text.replace("@TEST_OUTPUT_DIR@", str(output))
    text = re.sub(r"(set Initial refinement\s*=)\s*3", rf"\g<1> {refinement}", text)
    text = re.sub(r"(set Dirichlet boundary ids\s*=)\s*0\s*\n",
                  r"\1 0,1,2,3\n", text)
    text = text.replace("sin(2*pi*50*t); 0", "0; 0")
    text = text.replace("set Enable computation of the errors = true",
                        "set Enable computation of the errors = false")
    prefix = "subsection Elastodynamics\n" if native else ""
    suffix = "end\n" if native else ""
    text += (f"\n{prefix}subsection Functions\n"
             "  subsection Initial displacement\n"
             "    set Function expression = sin(10*pi*x)*sin(10*pi*y); 0\n"
             f"  end\nend\n{suffix}")
    if variant.startswith("adapter"):
        text += ("\nsubsection Elastodynamics\n  subsection IDA\n"
                 "    set Print current statistics = true\n"
                 "    subsection Running parameters\n"
                 f"      set Maximum order of BDF = {variant[-1]}\n"
                 "    end\n  end\nend\n")
    return text


def measurements(text):
    timers = {}
    for line in text.splitlines():
        match = re.match(r"\| ([^|]+?)\s*\|\s*(\d+)\s*\|\s*([\d.e+-]+)s\s*\|", line)
        if match:
            timers[match[1].strip()] = {"calls": int(match[2]),
                                         "seconds": float(match[3])}
    steps = re.search(r"finished after (\d+) internal step", text)
    return {
        "timers": timers,
        "application_seconds": [float(x) for x in re.findall(
            r"Total wallclock time elapsed since start\s*\|\s*([\d.e+-]+)s", text)],
        "steps": int(steps[1]) if steps else None,
        "linear_iterations": [int(x) for x in re.findall(
            r"(?:GMRES iterations: |FGMRES iterations: |FGMRES converged in )(\d+)", text)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-revision", required=True,
                        help="Revision actually compiled; not inferred from the current checkout")
    parser.add_argument("--refinements", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=["elasticity", "elastodynamics",
                                                         "adapter1", "adapter5"],
                        choices=["elasticity", "elastodynamics", "adapter1", "adapter5"])
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.repetitions < 1 or any(r < 0 for r in args.refinements):
        parser.error("Repetitions must be positive and refinements nonnegative")
    args.build = args.build.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    report = {"source_revision": args.source_revision, "host": os.uname().nodename,
              "ranks": 1, "thread_limit": 1, "build": str(args.build),
              "executables": {}, "records": []}
    for variant in args.variants:
        app = "elastodynamics_adapter" if variant.startswith("adapter") else variant
        executable = args.build / app
        report["executables"][app] = digest(executable)
    library = args.build / "libimmersx_lib.so"
    if library.exists():
        report["library_sha256"] = digest(library)
    failures = 0
    for refinement in args.refinements:
        for repetition in range(args.repetitions):
            for variant in args.variants:
                case = args.output / f"r{refinement}-{variant}-{repetition + 1}"
                case.mkdir()
                parameter = case / "input.prm"
                parameter.write_text(parameter_text(source, case / "results", refinement, variant))
                app = "elastodynamics_adapter" if variant.startswith("adapter") else variant
                command = [str(args.build / app), str(parameter)]
                with (case / "run.log").open("w") as stream:
                    try:
                        result = subprocess.run(command, cwd=case, env=environment,
                                                stdout=stream, stderr=subprocess.STDOUT,
                                                timeout=args.timeout, check=False)
                        status = result.returncode
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                record = {"variant": variant, "refinement": refinement,
                          "repetition": repetition + 1, "exit_status": status,
                          "input_sha256": digest(parameter),
                          "log": str((case / "run.log").relative_to(args.output)),
                          "command": command,
                          **measurements((case / "run.log").read_text())}
                if not variant.startswith("adapter") and status == 0:
                    record["steps"] = 8
                report["records"].append(record)
                (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
                failures += status != 0
                print(f"{case.name}: {status}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
