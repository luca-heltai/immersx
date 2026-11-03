#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Add explicit wall and outlet data to a serial CCO sphere result."""

import argparse
import csv
import json
import math
from pathlib import Path


def read_table(directory, name):
    with (directory / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Physical data must be finite")
    return result


def positive(value):
    result = finite(value)
    if result <= 0:
        raise ValueError("Radii, lengths, wall material and resistance must be positive")
    return result


def write_scalars(stream, name, values):
    stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
    for value in values:
        stream.write(f"{value:.17g}\n")


def export(args):
    if args.output.suffix.lower() != ".vtk":
        raise ValueError("Native network output must use the .vtk extension")
    (summary,) = read_table(args.generation, "summary.csv")
    if summary["valid"] != "1" or summary["processes"] != "1":
        raise ValueError("This tutorial exporter requires a valid single-process result")
    nodes = sorted(
        read_table(args.generation, "nodes-0.csv"), key=lambda row: int(row["id"])
    )
    vessels = sorted(
        read_table(args.generation, "vessels-0.csv"), key=lambda row: int(row["id"])
    )
    terminals = sorted(
        read_table(args.generation, "terminals-0.csv"), key=lambda row: int(row["id"])
    )
    if any(row["system"] != "arterial" for row in vessels + terminals):
        raise ValueError("This tutorial exporter supports a single arterial tree")
    for name, rows in (
        ("nodes", nodes), ("vessels", vessels), ("arterial_terminals", terminals)
    ):
        if len(rows) != int(summary[name]) or len(
            {int(row["id"]) for row in rows}
        ) != len(rows):
            raise ValueError(f"Incomplete or duplicate {name} records")
    node_indices = {int(row["id"]): i for i, row in enumerate(nodes)}
    inlets = [int(row["id"]) for row in nodes if int(row["port"]) == 1]
    if len(inlets) != 1:
        raise ValueError("The sphere example requires exactly one arterial inlet")
    coordinates = [[finite(row[axis]) for axis in ("x", "y", "z")] for row in nodes]
    cells = [
        [node_indices[int(row[end])] for end in ("node0", "node1")]
        for row in vessels
    ]
    radii = [positive(row["radius0"]) for row in vessels]
    if any(radius != positive(row["radius1"]) for radius, row in zip(radii, vessels)):
        raise ValueError(
            "This coupled tutorial requires constant radius along each physical branch"
        )
    viscosity = [finite(row["viscosity"]) for row in vessels]
    if any(value < 0 for value in viscosity):
        raise ValueError("Dynamic viscosity must be nonnegative")
    lengths = [positive(row["length"]) for row in vessels]

    boundary_ids = [255] * len(nodes)
    boundary_ids[node_indices[inlets[0]]] = 0
    terminal_boundaries = {}
    for number, row in enumerate(terminals, 1):
        boundary = number if number < 255 else number + 1
        node = node_indices[int(row["node"])]
        if boundary_ids[node] != 255:
            raise ValueError("Invalid or duplicate terminal boundary identifier")
        boundary_ids[node] = boundary
        terminal_boundaries[int(row["id"])] = boundary

    # Junctions retain shared point indices. Only physical leaves receive RCR data.
    degree = [0] * len(nodes)
    for cell in cells:
        for vertex in cell:
            degree[vertex] += 1
    if any(
        (count == 1) != (boundary != 255)
        for count, boundary in zip(degree, boundary_ids)
    ):
        raise ValueError("Inlet and terminal records do not match the physical leaves")
    areas = [math.pi * radius * radius for radius in radii]
    cell_data = {
        "vessel_id": list(range(len(vessels))),
        "a0": areas,
        "a_d": areas,
        "r_d": radii,
        "r_in": radii,
        "r_out": radii,
        "L": lengths,
        "viscosity": viscosity,
        "E": [args.young_modulus] * len(vessels),
        "h_wall": [args.wall_thickness] * len(vessels),
        "p_d": [args.reference_pressure] * len(vessels),
        "p0": [0.] * len(vessels),
    }
    point_data = {
        "boundary_id": boundary_ids,
        "R1": [0.] * len(nodes),
        "R2": [
            args.terminal_resistance if value not in (0, 255) else 0.
            for value in boundary_ids
        ],
        "C": [0.] * len(nodes),
        "P_out": [
            args.outlet_pressure if value not in (0, 255) else 0.
            for value in boundary_ids
        ],
    }
    metadata = {
        "wall_model": "Square root",
        "source_generation": str(args.generation.resolve()),
        "physical_vessel_ids": [int(row["id"]) for row in vessels],
        "physical_node_ids": [int(row["id"]) for row in nodes],
        "terminal_boundary_ids": terminal_boundaries,
        "inlet_boundary_id": 0,
        "junction_boundary_id": 255,
        "wall_and_outlet_parameters": {
            name: value for name, value in vars(args).items()
            if not isinstance(value, Path)
        },
    }
    sidecar = args.output.with_suffix(".json")
    if args.output.exists() or sidecar.exists():
        raise ValueError("VTK output and its identifier sidecar must be new files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(
            "# vtk DataFile Version 3.0\n"
            "CCO sphere with explicit MetricFlowX physics\n"
            "ASCII\nDATASET UNSTRUCTURED_GRID\n"
        )
        stream.write(f"POINTS {len(nodes)} double\n")
        for point in coordinates:
            stream.write(" ".join(f"{value:.17g}" for value in point) + "\n")
        stream.write(f"CELLS {len(cells)} {3 * len(cells)}\n")
        for start, end in cells:
            stream.write(f"2 {start} {end}\n")
        stream.write(f"CELL_TYPES {len(cells)}\n" + "3\n" * len(cells))
        stream.write(f"CELL_DATA {len(cells)}\n")
        for name, values in cell_data.items():
            write_scalars(stream, name, values)
        stream.write(f"POINT_DATA {len(nodes)}\n")
        for name, values in point_data.items():
            write_scalars(stream, name, values)
    with sidecar.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(
        f"Wrote {args.output}: {len(vessels)} physical vessels, "
        f"{len(terminals)} RCR outlets"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "generation", type=Path,
        help="Directory containing summary.csv and the owned physical CSV tables"
    )
    parser.add_argument("output", type=Path, help="New native MetricFlowX legacy .vtk file")
    parser.add_argument("--young-modulus", type=positive, required=True, help="Pa")
    parser.add_argument("--wall-thickness", type=positive, required=True, help="m")
    parser.add_argument(
        "--reference-pressure", type=finite, required=True,
        help="p_d in Pa; p0 is zero"
    )
    parser.add_argument(
        "--terminal-resistance", type=positive, required=True,
        help="R2 in Pa s/m^3; R1 and C are zero"
    )
    parser.add_argument("--outlet-pressure", type=finite, required=True, help="P_out in Pa")
    args = parser.parse_args()
    try:
        export(args)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
