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


STATE_NAMES = ("displacement", "flow", "multiplier")
RATE_NAMES = ("displacement rate", "flow rate", "multiplier rate")
REFERENCE_NAMES = ("reference area",)


def read_state(filename, names=STATE_NAMES):
    blocks = []
    for line in filename.read_text().splitlines():
        if line.strip():
            block = [float(value) for value in line.split()]
            if not all(math.isfinite(value) for value in block):
                raise AssertionError(f"{filename}: nonfinite state coefficient")
            blocks.append(block)
    if len(blocks) != len(names) or any(not block for block in blocks):
        raise AssertionError(f"{filename}: expected {len(names)} nonempty blocks: {names}")
    return blocks


def compare_states(expected_file, actual_file, mode, absolute=1e-8, relative=1e-5,
                   names=STATE_NAMES, skip=()):
    expected = read_state(expected_file, names)
    actual = read_state(actual_file, names)
    for name, first, second in zip(names, expected, actual):
        if name in skip:
            continue
        if len(first) != len(second):
            raise AssertionError(f"{mode} {name}: block size changed")
        error = max(abs(a - b) for a, b in zip(first, second))
        scale = max(max(abs(a), abs(b)) for a, b in zip(first, second))
        tolerance = absolute + relative * scale
        print(f"{mode} {name}: max error {error:.6g}, tolerance {tolerance:.6g}")
        if error > tolerance:
            raise AssertionError(f"{mode} {name}: restart coefficient error exceeds tolerance")


def run(command, state_file, log_file, timeout, cwd):
    env = os.environ.copy()
    env["IMMERSX_ELASTOSTATIC_STATE_OUTPUT"] = str(state_file)
    with log_file.open("w") as output:
        try:
            result = subprocess.run(command, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False, cwd=cwd)
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
    for snapshot in (state_file, Path(str(state_file) + ".initial")):
        read_state(snapshot)
        read_state(Path(str(snapshot) + ".rate"), RATE_NAMES)
        read_state(Path(str(snapshot) + ".reference"), REFERENCE_NAMES)
    for label in ("Static stiffness assemblies", "Static AMG setups"):
        values = re.findall(re.escape(label) + r":\s*([0-9]+)", log_file.read_text())
        if not values or any(int(value) != 1 for value in values):
            raise AssertionError(f"{label} must remain one; inspect {log_file}")


def reject_checkpoint(command, checkpoint, directory, message, timeout):
    directory.mkdir()
    log_file = directory / "rejection.log"
    env = os.environ.copy()
    env.pop("IMMERSX_ELASTOSTATIC_STATE_OUTPUT", None)
    with log_file.open("w") as output:
        try:
            result = subprocess.run(command + ["--restart", str(checkpoint),
                                               "--final-time", "0.002"],
                                    env=env, cwd=directory, stdout=output,
                                    stderr=subprocess.STDOUT,
                                    timeout=min(timeout, 30), check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Checkpoint rejection timed out; inspect {log_file}") from error
    if result.returncode == 0 or message not in log_file.read_text():
        raise AssertionError(
            f"Expected checkpoint rejection '{message}'; inspect {log_file}")
    print(f"Checkpoint rejected as expected: {message}; log {log_file}")


def check_invalid_checkpoints(args, command, checkpoint, directory):
    failed = directory / "invalid-checkpoints"
    failed.mkdir()
    missing = failed / "missing-complete"
    shutil.copytree(checkpoint, missing)
    (missing / "COMPLETE").unlink()
    reject_checkpoint(command, missing, failed / "missing-cwd",
                      "Missing complete coupled checkpoint.", args.timeout)
    truncated = failed / "truncated-solid"
    shutil.copytree(checkpoint, truncated)
    solid_files = sorted(path for path in truncated.iterdir()
                         if path.is_file() and path.name.startswith("solid")
                         and path.stat().st_size > 0)
    if not solid_files:
        raise AssertionError("Checkpoint has no nonempty solid file to truncate")
    payload = solid_files[0]
    with payload.open("r+b") as output:
        output.truncate(payload.stat().st_size - 1)
    reject_checkpoint(command, truncated, failed / "truncated-cwd",
                      "Missing or truncated checkpoint file", args.timeout)
    if args.mpiexec and args.ranks > 1:
        wrong_ranks = failed / "wrong-mpi-size"
        shutil.copytree(checkpoint, wrong_ranks)
        reject_checkpoint([str(args.executable)], wrong_ranks,
                          failed / "wrong-size-cwd",
                          "Checkpoint requires the original MPI size.",
                          args.timeout)
    else:
        print("Wrong-MPI-size rejection is checked by the multi-rank driver")


def check_mode(args, base_text, original_vtk, feedback):
    mode = "feedback" if feedback else "one-way"
    # Retain earlier logs and checkpoints when CTest reruns the test.
    directory = Path(tempfile.mkdtemp(prefix=mode + "-", dir=args.output_directory))
    input_directory = directory / "disposable-inputs"
    input_directory.mkdir()
    source_parameter = input_directory / "source.prm"
    source_vtk = input_directory / "source.vtk"
    source_csv = input_directory / "inflow.csv"
    shutil.copyfile(original_vtk, source_vtk)
    text = set_entry(base_text, "Vtk file path for mesh input", source_vtk)
    if args.csv_inflow:
        samples = [(0.0, 0.0), (0.0001, 0.0)]
        samples += [(time, 5e-8 * (1 - math.cos(314.1592653589793 * time)))
                    for time in (0.0005, 0.001, 0.0015, 0.002, 0.02)]
        source_csv.write_text("time_s,flow_m3_s\n" + "".join(
            f"{time:.17g},{flow:.17g}\n" for time, flow in samples))
        text += ("\nsubsection MetricFlowSystem<1, 3>\n"
                 f"  set Inflow CSV file = {source_csv}\n"
                 "  set Inflow CSV period = 0\n"
                 "end\n")
    # A restart starts a new IDA history. Keep integration error below the
    # coefficient-comparison threshold without relaxing that threshold.
    text = set_entry(text, "Absolute error tolerance",
                     "1e-13" if args.co_refinement else "1e-12")
    text = set_entry(text, "Relative error tolerance", "1e-10")
    text += ("\nsubsection MetricFlowX coupling\n"
             f"  set Pressure feedback = {'true' if feedback else 'false'}\n"
             "  set Wall pressure absolute tolerance = 1e-7\n"
             "end\n")
    final_time = "0.002"
    split_time = "0.001"
    if args.co_refinement:
        text += ("\nsubsection MetricFlowX coupling\n"
                 "  subsection Co-refinement\n"
                 "    set Enabled = true\n"
                 "    set Refinement strategy = both\n"
                 "    set Lower factor = 0.000001\n"
                 "    set Upper factor = 0.001\n"
                 "    set Max refinement level = 1\n"
                 "    set Space post-refinement cycles = 1\n"
                 "    set Max section refinement level = 2\n"
                 "  end\nend\n")
    command = [str(args.executable)]
    if args.mpiexec:
        command = [str(args.mpiexec), "-np", str(args.ranks)] + command
    full_directory = directory / "uninterrupted"
    full_directory.mkdir()
    source_parameter.write_text(set_entry(text, "Output directory", full_directory))
    full_state = full_directory / "state.txt"
    full_cwd = full_directory / "unrelated-cwd"
    full_cwd.mkdir()
    run(command + [str(source_parameter), "--final-time", final_time],
        full_state, full_directory / "run.log", args.timeout, full_cwd)
    split_directory = directory / "split"
    split_directory.mkdir()
    source_parameter.write_text(set_entry(text, "Output directory", split_directory))
    checkpoint = directory / "checkpoint"
    split_cwd = split_directory / "unrelated-cwd"
    split_cwd.mkdir()
    run(command + [str(source_parameter), "--final-time", split_time,
                   "--checkpoint", str(checkpoint)],
        split_directory / "state.txt", split_directory / "first.log",
        args.timeout, split_cwd)
    if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise AssertionError(f"No checkpoint files were created: {checkpoint}")
    if args.co_refinement:
        first_log = (split_directory / "first.log").read_text()
        report = re.search(r"Co-refinement solid cells: (\d+) -> (\d+); fluid cells: (\d+) -> (\d+); section level: (\d+)", first_log)
        if report is None:
            raise AssertionError("Missing co-refinement report")
        solid_before, solid_after, flow_before, flow_after, section = map(int, report.groups())
        if solid_after <= solid_before or flow_after <= flow_before or section != 2:
            raise AssertionError("Regression did not refine both meshes and the section")
        metadata = (checkpoint / "coupled.bin").read_bytes()
        override = f"set Refinement level = {section}\n".encode()
        if override not in metadata:
            raise AssertionError("Checkpoint lacks the resolved section-level override")
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
    if args.csv_inflow:
        source_csv.unlink()
    input_directory.rmdir()
    relocated = directory / "relocated"
    relocated.mkdir()
    original_checkpoint = checkpoint
    checkpoint = relocated / "checkpoint"
    original_checkpoint.rename(checkpoint)
    if original_checkpoint.exists():
        raise AssertionError("Checkpoint was not relocated")
    if not feedback:
        check_invalid_checkpoints(args, command, checkpoint, directory)
    restart_cwd = relocated / "unrelated-cwd"
    restart_cwd.mkdir()
    restarted_state = split_directory / "restarted-state.txt"
    run(command + ["--restart", str(checkpoint), "--final-time", final_time],
        restarted_state, split_directory / "restart.log", args.timeout, restart_cwd)
    if (source_parameter.exists() or source_vtk.exists() or source_csv.exists()
            or original_checkpoint.exists()):
        raise AssertionError("Restart unexpectedly recreated the removed input files")
    if args.co_refinement and "Co-refinement solid cells:" in (split_directory / "restart.log").read_text():
        raise AssertionError("Restart unexpectedly repeated co-refinement")
    compare_states(split_directory / "state.txt",
                   Path(str(restarted_state) + ".initial"), mode + " restored",
                   absolute=0, relative=0)
    compare_states(Path(str(split_directory / "state.txt") + ".rate"),
                   Path(str(restarted_state) + ".initial.rate"),
                   mode + " restored rates", absolute=0, relative=0,
                   names=RATE_NAMES, skip=("displacement rate",))
    reference = Path(str(full_state) + ".reference")
    for snapshot in (full_state, split_directory / "state.txt", restarted_state):
        for suffix in (".reference", ".initial.reference"):
            candidate = Path(str(snapshot) + suffix)
            if candidate == reference:
                continue
            compare_states(reference, candidate,
                           mode + " frozen reference area", absolute=0,
                           relative=0, names=REFERENCE_NAMES)
    compare_states(full_state, restarted_state, mode + " continued")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--parameter", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--mpiexec", type=Path)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--co-refinement", action="store_true", help="Exercise refined native meshes and resolved section checkpoint data")
    parser.add_argument("--csv-inflow", action="store_true",
                        help="Checkpoint CSV inflow and delete the original table before restart")
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
