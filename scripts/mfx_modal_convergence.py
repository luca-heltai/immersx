#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the modal MMS and report reproducible exported-field errors as JSON."""
import argparse
import csv
import itertools
import json
import math
import shutil
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

# Keep this source directory unchanged when the shared decoder is imported.
sys.dont_write_bytecode = True
from test_metric_flow_x_robin import data_array

ELL, A0 = 0.120685, 3.0605e-4
R2 = A0 / math.pi
GAUSS = ((0.5 * (1 - math.sqrt(3 / 5)), 5 / 18),
         (0.5, 4 / 9),
         (0.5 * (1 + math.sqrt(3 / 5)), 5 / 18))
CASES = {
    "local": "mms_cube_local", "local-refined": "mms_cube_local_refined",
    "local-finer": "mms_cube_local_finer", "local-tight": "mms_cube_local_tight",
    "base": "mms_cube", "refined": "mms_cube_refined",
    "finer": "mms_cube_finer", "finer-original": "mms_cube_finer_original_pairing",
    "tight": "mms_cube_tight",
    "radial": "mms_cube_radial", "neumann": "mms_cube_neumann",
    "iterative": "mms_cube_iterative", "mpi": "mms_cube_mpi",
    "neumann-refined": "mms_cube_neumann_refined",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def decode(array, tree):
    # The shared decoder handles compressed production output.
    if array.attrib.get("format") == "ascii":
        values = [float(value) for value in (array.text or "").split()]
        require(all(math.isfinite(value) for value in values), "Nonfinite VTK value")
        return values
    return data_array(array, tree)


def pieces(path):
    tree = ET.parse(path).getroot()
    if tree.attrib["type"] == "PUnstructuredGrid":
        for item in tree.findall(".//Piece"):
            yield from pieces(path.parent / item.attrib["Source"])
        return
    require(tree.attrib["type"] == "UnstructuredGrid", f"Unsupported VTK: {path}")
    for piece in tree.findall("./UnstructuredGrid/Piece"):
        if int(piece.attrib["NumberOfPoints"]) == 0:
            continue
        coords = decode(piece.find("./Points/DataArray"), tree)
        points = [coords[i:i + 3] for i in range(0, len(coords), 3)]
        fields = {}
        for array in piece.findall("./PointData/DataArray"):
            count = int(array.attrib.get("NumberOfComponents", "1"))
            values = decode(array, tree)
            fields[array.attrib["Name"]] = [
                values[i:i + count] for i in range(0, len(values), count)]
        arrays = {a.attrib["Name"]: a for a in piece.findall("./Cells/DataArray")}
        indices = decode(arrays["connectivity"], tree)
        offsets = decode(arrays["offsets"], tree)
        start = 0
        cells = []
        for end in offsets:
            cells.append([int(i) for i in indices[start:int(end)]])
            start = int(end)
        yield points, fields, cells


def collection(path):
    datasets = ET.parse(path).findall(".//DataSet")
    require(bool(datasets), f"No states in {path}")
    return [(float(item.attrib["timestep"]), path.parent / item.attrib["file"])
            for item in datasets]


def solid_target(point, time, radial=False, velocity=False):
    x, y, z = point
    g = 3 * time**2 if velocity else time**3
    h = 0 if radial else (8 * time**3 if velocity else 2 * time**4)
    factor = R2 / max(y*y + z*z, R2)
    return (h*x, g*y*factor, g*z*factor)


def integrate(path, dimension, targets):
    error = {name: 0. for name in targets}
    norm = dict(error)
    integrals = dict(error)
    measure = 0.
    axial_error, radial_error = dict(error), dict(error)
    axial_norm, radial_norm = dict(error), dict(error)
    for points, fields, cells in pieces(path):
        for cell in cells:
            require(len(cell) == 2**dimension, f"Only Cartesian Q1 cells supported: {path}")
            vertices = [points[i] for i in cell]
            low = [min(v[d] for v in vertices) for d in range(3)]
            high = [max(v[d] for v in vertices) for d in range(3)]
            active = [d for d in range(3) if high[d] - low[d] > 1.e-12]
            require(len(active) == dimension, f"Non-Cartesian cell: {path}")
            volume = math.prod(high[d] - low[d] for d in active)
            corners = []
            for vertex in vertices:
                bits = tuple(int(abs(vertex[d] - high[d]) < 1.e-10) for d in active)
                require(all(abs(vertex[d] - (high[d] if bit else low[d])) < 1.e-10
                            for d, bit in zip(active, bits)), "Non-Cartesian vertex")
                corners.append(bits)
            require(len(set(corners)) == len(cell), "Duplicate cell corner")
            for quadrature in itertools.product(GAUSS, repeat=dimension):
                unit = [item[0] for item in quadrature]
                weight = volume * math.prod(item[1] for item in quadrature)
                point = list(low)
                for d, coordinate in zip(active, unit):
                    point[d] += coordinate * (high[d] - low[d])
                shape = [math.prod(coordinate if bit else 1 - coordinate
                                   for coordinate, bit in zip(unit, corner))
                         for corner in corners]
                measure += weight
                for name, target in targets.items():
                    require(name in fields, f"Missing field {name}: {path}")
                    values = fields[name]
                    expected = target(point)
                    actual = [sum(shape[j] * values[index][component]
                                  for j, index in enumerate(cell))
                              for component in range(len(expected))]
                    error[name] += weight * sum((a-b)**2 for a, b in zip(actual, expected))
                    norm[name] += weight * sum(value**2 for value in expected)
                    integrals[name] += weight * actual[0]
                    if dimension == 3 and len(expected) == 3:
                        axial_error[name] += weight * (actual[0]-expected[0])**2
                        axial_norm[name] += weight * expected[0]**2
                        radial_error[name] += weight * sum(
                            (actual[d]-expected[d])**2 for d in (1, 2))
                        radial_norm[name] += weight * sum(
                            expected[d]**2 for d in (1, 2))
    require(measure > 0, f"Empty integration mesh: {path}")
    result = {name: {"l2_error": math.sqrt(error[name]),
                   "relative_l2_error": math.sqrt(error[name]/norm[name])
                   if norm[name] else None,
                   "integral": integrals[name], "measure": measure} for name in targets}
    if dimension == 3:
        for name in targets:
            for label, errors, norms in (
                    ("axial", axial_error, axial_norm),
                    ("radial", radial_error, radial_norm)):
                result[name][label+"_l2_error"] = math.sqrt(errors[name])
                result[name][label+"_relative_l2_error"] = (
                    math.sqrt(errors[name]/norms[name]) if norms[name] else None)
    return result


def oracle(time, radial=False):
    g, gd = time**3, 3*time**2
    h, hd = (0., 0.) if radial else (2*time**4, 8*time**3)
    exterior = 4*ELL**2 - A0
    integral = A0 - R2**2*(math.pi+2)/(2*ELL**2)
    length = 2*ELL
    # Integral of |W_perp|^2 on the square, including the full interior disk.
    catalan = .915965594177219015
    radial_mass = R2**2*(math.pi/2 + 2*math.pi*math.log(2*ELL/math.sqrt(R2))-4*catalan)
    return {
        "multiplier_Pa": h + .2*hd - 2*g - .5*gd,
        "retained_kinetic_energy_J": .5*(hd*hd*8*ELL**5/3 + gd*gd*length*radial_mass),
        "effective_elastic_energy_J": length*(2*g*g*integral + 1.5*h*h*exterior),
        "removed_elastic_energy_J": A0*length*(2*g*g+h*h+.5*(2*g+h)**2),
        "effective_viscous_dissipation_W":
            length*(gd*gd*integral + .7*hd*hd*exterior),
        "added_volume_m3": length*A0*((1+g)**2-1),
    }


def assess(directory, radial=False, local_refinement=False, expected_time=.1):
    solid_states = collection(directory/"solid"/"solid.pvd")
    flow_states = collection(directory/"flow"/"flow.pvd")
    multiplier_states = collection(directory/"flow"/"pressure_jump.pvd")
    time = multiplier_states[-1][0]
    require(math.isclose(time, expected_time, abs_tol=1.e-12),
            f"Did not reach T={expected_time}: {time}")
    require(all(math.isclose(states[-1][0], time, abs_tol=1.e-12)
                for states in (solid_states, flow_states)), "Inconsistent output times")
    exact = oracle(time, radial)
    result = {"time_s": time, "continuum_oracle": exact}
    result["solid"] = integrate(solid_states[-1][1], 3, {
        "displacement": lambda p: solid_target(p, time, radial),
        "velocity": lambda p: solid_target(p, time, radial, True)})
    result["flow"] = integrate(flow_states[-1][1], 1, {
        "area": lambda p: (A0*(1+time**3)**2,),
        "velocity": lambda p: ((ELL-p[0])*6*time**2/(1+time**3),)})
    result["multiplier"] = integrate(multiplier_states[-1][1], 1, {
        "pressure_jump": lambda p: (exact["multiplier_Pa"],)})["pressure_jump"]
    require(math.isclose(result["solid"]["displacement"]["measure"], (2*ELL)**3,
                         rel_tol=1.e-6), "Incomplete solid output mesh")
    require(all(math.isclose(value["measure"], 2*ELL, rel_tol=1.e-6)
                for value in (result["flow"]["area"], result["multiplier"])),
            "Incomplete centerline output mesh")
    result["multiplier"]["mean_Pa"] = (
        result["multiplier"]["integral"]/result["multiplier"]["measure"])
    result["additional_pressure_mean_Pa"] = -result["multiplier"]["mean_Pa"]
    initial = integrate(flow_states[0][1], 1, {"area": lambda p: (A0,)})["area"]["integral"]
    result["added_volume_m3"] = result["flow"]["area"]["integral"] - initial
    result["volume_defect_m3"] = result["added_volume_m3"] - exact["added_volume_m3"]
    result["relative_volume_defect"] = result["volume_defect_m3"]/exact["added_volume_m3"]
    energy = directory/"solid"/"energies.csv"
    if energy.exists():
        with energy.open() as stream:
            rows = list(csv.DictReader(stream))
        require(bool(rows), "Empty energy diagnostics")
        result["energy_diagnostics"] = {k: float(v) for k, v in rows[-1].items()}
        diagnostics = result["energy_diagnostics"]
        if "sampled_lumen_volume_m3" in diagnostics:
            result["sampled_volume_relative_error"] = (
                diagnostics["sampled_lumen_volume_m3"]/(2*ELL*A0)-1)
        expected_modes = {
            "modal_g": time**3, "modal_g_dot": 3*time**2,
            "modal_h": 0. if radial else 2*time**4,
            "modal_h_dot": 0. if radial else 8*time**3}
        result["projected_mode_errors"] = {}
        for name, expected in expected_modes.items():
            if name in diagnostics:
                actual = diagnostics[name]
                result["projected_mode_errors"][name] = {
                    "actual": actual, "expected": expected,
                    "absolute_error": abs(actual-expected),
                    "relative_error": abs((actual-expected)/expected) if expected else None}
        result["energy_errors"] = {}
        for column, target in (
                ("effective_elastic_and_robin_energy_J", "effective_elastic_energy_J"),
                ("removed_modal_elastic_energy_J", "removed_elastic_energy_J"),
                ("kinetic_energy_J", "retained_kinetic_energy_J"),
                ("effective_dissipation_W", "effective_viscous_dissipation_W")):
            if column in diagnostics:
                result["energy_errors"][column] = {
                    "absolute_error": abs(diagnostics[column]-exact[target]),
                    "relative_error": abs((diagnostics[column]-exact[target])/exact[target])}

    mesh_report = directory/"flow"/"co_refinement.csv"
    require(not local_refinement or mesh_report.exists(),
            "Missing local co-refinement diagnostics")
    if mesh_report.exists():
        with mesh_report.open() as stream:
            rows = list(csv.DictReader(stream))
        result["co_refinement"] = {
            "active_cells_by_level": {
                row["level"]: int(row["active_cells"]) for row in rows},
            "solid_dofs": int(rows[0]["solid_dofs"]),
            "hanging_dofs": int(rows[0]["hanging_dofs"]),
        }
        if local_refinement:
            mesh = result["co_refinement"]
            integers = (
                "initial_solid_cells", "initial_fluid_cells", "fluid_cells",
                "solid_passes", "fluid_passes", "section_passes", "section_level",
                "point_cell_pairs", "outside_points",
                "axial_lower_violations", "section_lower_violations",
                "axial_upper_violations", "section_upper_violations")
            for key in integers:
                mesh[key] = int(rows[0][key])
            for key in ("axial_min_ratio", "axial_max_ratio",
                        "section_min_ratio", "section_max_ratio",
                        "lower_factor", "upper_factor"):
                mesh[key] = float(rows[0][key])
            require(mesh["initial_solid_cells"] == 1 and
                    mesh["initial_fluid_cells"] == 1,
                    "Local MMS must start from one solid and one fluid cell")
            require(mesh["fluid_cells"] > 1 and mesh["solid_passes"] > 0 and
                    mesh["fluid_passes"] > 0,
                    "The local MMS did not refine both physical meshes")
            require(mesh["point_cell_pairs"] > 0 and mesh["outside_points"] == 0,
                    "The lift is not covered by the solid")
            for direction in ("axial", "section"):
                require(mesh[direction+"_lower_violations"] == 0 and
                        mesh[direction+"_upper_violations"] == 0,
                        "Unresolved "+direction+" co-refinement violations")
                require(mesh["lower_factor"] < mesh[direction+"_min_ratio"] <=
                        mesh[direction+"_max_ratio"] < mesh["upper_factor"],
                        "Reported "+direction+" ratios lie outside the interval")
            counts = result["co_refinement"]["active_cells_by_level"]
            require(sum(count > 0 for count in counts.values()) > 1,
                    "The local case did not produce different active cell levels")
            require(result["co_refinement"]["hanging_dofs"] > 0,
                    "The local case did not exercise hanging-node constraints")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, default=1,
                        help="MPI ranks for iterative cases (default: serial)")
    parser.add_argument("--cases", choices=CASES, nargs="+", default=["base", "refined", "finer"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=7200.)
    parser.add_argument("--final-time", type=float,
                        help="Final time for a run/check (default: 0.1, or saved metadata when reprocessing)")
    args = parser.parse_args()
    if args.final_time is not None and (
            not math.isfinite(args.final_time) or args.final_time <= 0):
        parser.error("--final-time must be positive and finite")
    if args.mpi_ranks < 1:
        parser.error("--mpi-ranks must be positive")
    launcher = None
    if args.run and args.mpi_ranks > 1:
        if any(case not in {"finer", "finer-original", "iterative", "mpi", "local-finer"} for case in args.cases):
            parser.error("MPI requires iterative inputs; select finer, finer-original, iterative, mpi or local-finer cases")
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        if launcher is None:
            parser.error("MPI execution requires mpirun or mpiexec from the matching MPI installation")
    build = args.build.resolve()
    output = (args.output or build/"test_output"/"mfx-modal-convergence").resolve()
    executable = (args.executable or build/"metric_flow_x_elastodynamics").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"quadrature": "three-point Gauss on exported Cartesian Q1 cells",
              "cases": {}, "failures": {}}
    for case in args.cases:
        directory = output/case
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if args.run:
                template = build/"tutorials"/"metric_flow_x_elastodynamics"/(CASES[case]+".prm")
                require(template.is_file(), f"Missing generated input: {template}")
                parameters = directory/"run.prm"
                final_time = .1 if args.final_time is None else args.final_time
                interval = min(.05, final_time/2)
                parameters.write_text(
                    f"include {template}\n"
                    f"subsection Elastodynamics\n  set Output directory = {directory/'solid'}\nend\n"
                    f"subsection MetricFlowSystem<1, 3>\n"
                    f"  set Output directory = {directory/'flow'}\n"
                    f"  subsection IDA parameters\n"
                    f"    set Final time = {final_time}\n"
                    f"    set Time interval between each output = {interval}\n"
                    f"  end\nend\n")
                command = [str(executable), str(parameters)]
                if launcher:
                    command = [launcher, "-np", str(args.mpi_ranks), *command]
                metadata = {"mpi_ranks": args.mpi_ranks, "command": command,
                            "parameters": str(parameters), "final_time_s": final_time}
                (directory/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
                with (directory/"application.log").open("w") as log:
                    with subprocess.Popen(command, cwd=directory, stdout=log,
                                          stderr=subprocess.STDOUT) as process:
                        try:
                            process.wait(timeout=args.timeout)
                        except subprocess.TimeoutExpired:
                            # Let the MPI launcher terminate its ranks before forcing exit.
                            process.terminate()
                            try:
                                process.wait(timeout=5.)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise
                require(process.returncode == 0, f"Application failed; see {directory/'application.log'}")
            metadata_path = directory/"run_metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            expected_time = (args.final_time if args.final_time is not None
                             else metadata.get("final_time_s", .1))
            report["cases"][case] = assess(directory, case == "radial",
                                          case.startswith("local"), expected_time)
            if metadata:
                report["cases"][case]["execution"] = metadata
        except (RuntimeError, OSError, ValueError, AssertionError, KeyError, IndexError, ET.ParseError, subprocess.TimeoutExpired) as error:
            report["failures"][case] = str(error)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    report["spatial_log2_error_reductions"] = {}
    for coarse, fine in (("base", "refined"), ("refined", "finer"),
                         ("local", "local-refined"), ("local-refined", "local-finer")):
        if coarse in report["cases"] and fine in report["cases"]:
            if not math.isclose(report["cases"][coarse]["time_s"],
                                report["cases"][fine]["time_s"], abs_tol=1.e-12):
                report["failures"][coarse+"_to_"+fine] = "Cannot compare different final times"
                continue
            values = {}
            for label, section, field in (
                    ("displacement", "solid", "displacement"),
                    ("solid_velocity", "solid", "velocity"),
                    ("area", "flow", "area"),
                    ("flow_velocity", "flow", "velocity"),
                    ("multiplier", "multiplier", None)):
                first = report["cases"][coarse][section]
                last = report["cases"][fine][section]
                if field:
                    first, last = first[field], last[field]
                first, last = first["l2_error"], last["l2_error"]
                values[label] = math.log2(first/last) if first > 0 and last > 0 else None
            report["spatial_log2_error_reductions"][coarse+"_to_"+fine] = values
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
