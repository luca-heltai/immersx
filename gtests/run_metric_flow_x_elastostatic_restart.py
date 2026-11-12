#!/usr/bin/env python3
"""Compare a coupled run with a restart after removing its disposable inputs."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def set_entry(text, name, value):
    pattern = re.compile(r"^(\s*set\s+" + re.escape(name) + r"\s*=).*$", re.MULTILINE)
    result, count = pattern.subn(lambda match: match.group(1) + " " + str(value), text)
    if not count:
        raise ValueError(f"Missing parameter entry: {name}")
    return result


def read_state(filename):
    blocks = []
    for line in filename.read_text().splitlines():
        if line.strip():
            block = [float(value) for value in line.split()]
            if not all(math.isfinite(value) for value in block):
                raise AssertionError(f"{filename}: nonfinite state coefficient")
            blocks.append(block)
    if len(blocks) != 3 or any(not block for block in blocks):
        raise AssertionError(f"{filename}: expected three nonempty state blocks")
    return blocks


def compare_states(expected_file, actual_file, mode, absolute=1e-8, relative=1e-5):
    expected = read_state(expected_file)
    actual = read_state(actual_file)
    for name, first, second in zip(("displacement", "flow", "multiplier"), expected, actual):
        if len(first) != len(second):
            raise AssertionError(f"{mode} {name}: block size changed")
        error = max(abs(a - b) for a, b in zip(first, second))
        scale = max(max(abs(a), abs(b)) for a, b in zip(first, second))
        tolerance = absolute + relative * scale
        print(f"{mode} {name}: max error {error:.6g}, tolerance {tolerance:.6g}")
        if error > tolerance:
            raise AssertionError(f"{mode} {name}: restart coefficient error exceeds tolerance")


def run(command, state_file, log_file, timeout):
    env = os.environ.copy()
    env["IMMERSX_ELASTOSTATIC_STATE_OUTPUT"] = str(state_file)
    with log_file.open("w") as output:
        try:
            result = subprocess.run(command, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Application timed out; inspect {log_file}") from error
    if result.returncode:
        unsupported = (
            "Restart requires a MetricFlowX package with native checkpoint hooks.",
            "Checkpoint writing requires a MetricFlowX package with native checkpoint hooks.",
        )
        if any(message in log_file.read_text() for message in unsupported):
            print(f"SKIP: executable lacks native MetricFlowX checkpoint hooks; inspect {log_file}")
            raise SystemExit(77)
        raise RuntimeError(f"Application exited with {result.returncode}; inspect {log_file}")
    steps = re.search(r"accepted IDA steps:\s*([0-9]+)", log_file.read_text())
    if steps is None or int(steps.group(1)) == 0:
        raise AssertionError(f"Application did not advance; inspect {log_file}")
    if not state_file.is_file():
        raise AssertionError(f"Missing final state diagnostic: {state_file}")
    read_state(state_file)


def check_mode(args, base_text, original_vtk, feedback):
    mode = "feedback" if feedback else "one-way"
    # Retain earlier logs and checkpoints when CTest reruns the test.
    directory = Path(tempfile.mkdtemp(prefix=mode + "-", dir=args.output_directory))
    input_directory = directory / "disposable-inputs"
    input_directory.mkdir()
    source_parameter = input_directory / "source.prm"
    source_vtk = input_directory / "source.vtk"
    shutil.copyfile(original_vtk, source_vtk)
    text = set_entry(base_text, "Vtk file path for mesh input", source_vtk)
    # A restart starts a new IDA history. Keep integration error below the
    # coefficient-comparison threshold without relaxing that threshold.
    text = set_entry(text, "Absolute error tolerance", "1e-12")
    text = set_entry(text, "Relative error tolerance", "1e-10")
    text += ("\nsubsection MetricFlowX coupling\n"
             f"  set Pressure feedback = {'true' if feedback else 'false'}\n"
             "end\n")
    command = [str(args.executable)]
    if args.mpiexec:
        command = [str(args.mpiexec), "-np", str(args.ranks)] + command
    full_directory = directory / "uninterrupted"
    full_directory.mkdir()
    source_parameter.write_text(set_entry(text, "Output directory", full_directory))
    full_state = full_directory / "state.txt"
    run(command + [str(source_parameter), "--final-time", "0.002"],
        full_state, full_directory / "run.log", args.timeout)
    split_directory = directory / "split"
    split_directory.mkdir()
    source_parameter.write_text(set_entry(text, "Output directory", split_directory))
    checkpoint = directory / "checkpoint"
    run(command + [str(source_parameter), "--final-time", "0.001",
                   "--checkpoint", str(checkpoint)],
        split_directory / "state.txt", split_directory / "first.log", args.timeout)
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise AssertionError(f"No checkpoint files were created: {checkpoint}")
    full_blocks = read_state(full_state)
    split_blocks = read_state(split_directory / "state.txt")
    if any(len(a) != len(b) for a, b in zip(full_blocks, split_blocks)):
        raise AssertionError("Checkpoint and final state block sizes differ")
    elapsed_change = max(abs(a - b)
                         for first, second in zip(full_blocks, split_blocks)
                         for a, b in zip(first, second))
    if elapsed_change <= 1e-12:
        raise AssertionError("The split-time state did not differ from the final state")
    # The second process must use only saved metadata, mesh, and coefficients.
    source_parameter.unlink()
    source_vtk.unlink()
    input_directory.rmdir()
    restarted_state = split_directory / "restarted-state.txt"
    run(command + ["--restart", str(checkpoint), "--final-time", "0.002"],
        restarted_state, split_directory / "restart.log", args.timeout)
    if source_parameter.exists() or source_vtk.exists():
        raise AssertionError("Restart unexpectedly recreated the removed input files")
    compare_states(split_directory / "state.txt",
                   Path(str(restarted_state) + ".initial"), mode + " restored",
                   absolute=1e-12, relative=1e-12)
    compare_states(full_state, restarted_state, mode + " continued")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--parameter", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--mpiexec", type=Path)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--feedback", choices=("one-way", "two-way", "both"), default="both")
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    args.parameter = args.parameter.resolve()
    args.output_directory = args.output_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.ranks < 1 or args.timeout <= 0:
        parser.error("Ranks and timeout must be positive")
    base_text = args.parameter.read_text()
    match = re.search(r"^\s*set\s+Vtk file path for mesh input\s*=\s*(.*?)\s*$",
                      base_text, re.MULTILINE)
    if not match:
        parser.error("The parameter file must contain a VTK mesh input")
    original_vtk = Path(match.group(1))
    if not original_vtk.is_absolute():
        original_vtk = args.parameter.parent / original_vtk
    original_vtk = original_vtk.resolve(strict=True)
    modes = {"one-way": (False,), "two-way": (True,), "both": (False, True)}
    for feedback in modes[args.feedback]:
        check_mode(args, base_text, original_vtk, feedback)
    print("Restarted and uninterrupted final states agree")


if __name__ == "__main__":
    main()
