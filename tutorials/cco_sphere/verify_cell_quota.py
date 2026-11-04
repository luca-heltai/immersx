#!/usr/bin/env python3
"""Check arterial terminal quotas in the single-seed sphere tutorial using VTK."""

import argparse
import csv
import json
from pathlib import Path

import vtk


def evaluate(cell, point):
    closest, reference = [0.] * 3, [0.] * 3
    subcell, distance = vtk.reference(0), vtk.reference(0.)
    weights = [0.] * cell.GetNumberOfPoints()
    inside = cell.EvaluatePosition(point, closest, subcell, reference, distance, weights)
    if cell.GetCellType() in (vtk.VTK_HEXAHEDRON, vtk.VTK_VOXEL):
        margin = min(min(x, 1. - x) for x in reference)
    elif cell.GetCellType() == vtk.VTK_TETRA:
        margin = min(*reference, 1. - sum(reference))
    else:
        raise ValueError("The verifier supports linear hexahedra and tetrahedra.")
    return inside, float(distance), margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k must be positive")
    with (args.generation / "summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    if summary["valid"] != "1":
        parser.error("A complete, valid generation is required.")
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(args.generation / "background.pvtu"))
    reader.Update()
    mesh = reader.GetOutput()
    if not mesh.GetNumberOfCells():
        parser.error("The background mesh is empty; enable Write meshes.")
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    counts = [0] * mesh.GetNumberOfCells()
    boundary, outside, unassigned = [], [], []
    terminals = 0
    for filename in sorted(args.generation.glob("terminals-*.csv")):
        with filename.open() as stream:
            for row in csv.DictReader(stream):
                if row["system"] != "arterial":
                    continue
                terminals += 1
                point = [float(row[key]) for key in ("x", "y", "z")]
                cell = locator.FindCell(point)
                if cell < 0:
                    outside.append(row["id"])
                elif evaluate(mesh.GetCell(cell), point)[2] < 1.e-10:
                    boundary.append((row["id"], point))
                else:
                    counts[cell] += 1
    # The axial seed tip lies on mesh interfaces. Its slot is the unique
    # containing cell still missing a terminal after all interior cloud points.
    tolerance = 1.e-10 * max(1., mesh.GetLength())
    for identifier, point in boundary:
        candidates = vtk.vtkIdList()
        bounds = [v for x in point for v in (x - tolerance, x + tolerance)]
        locator.FindCellsWithinBounds(bounds, candidates)
        choices = []
        for i in range(candidates.GetNumberOfIds()):
            cell = candidates.GetId(i)
            inside, distance, _ = evaluate(mesh.GetCell(cell), point)
            if inside >= 0 and (inside == 1 or distance <= tolerance * tolerance):
                if counts[cell] < args.k:
                    choices.append(cell)
        if len(choices) == 1:
            counts[choices[0]] += 1
        else:
            unassigned.append({"terminal": identifier, "candidate_cells": choices})
    report = {
        "valid_quota": not outside and not unassigned and all(n == args.k for n in counts),
        "cells": len(counts), "terminals": terminals, "k": args.k,
        "occupied_cells": sum(n > 0 for n in counts),
        "minimum_per_cell": min(counts), "maximum_per_cell": max(counts),
        "boundary_terminals": boundary, "unassigned": unassigned,
        "outside": outside, "vtk_version": vtk.vtkVersion.GetVTKVersion(),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["valid_quota"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
