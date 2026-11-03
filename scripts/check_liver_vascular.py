#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check a liver tutorial result against its supplied rooted 1D meshes."""
import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def records(directory, name, key="id"):
    result = {}
    paths = sorted(directory.glob(f"{name}-*.csv"))
    require(paths, f"Missing {name} tables")
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                identity = int(row[key])
                require(identity not in result, f"Duplicate owned {name} record {identity}")
                result[identity] = row
    return result


def check(args):
    import vtk

    nodes = records(args.result, "nodes")
    vessels = records(args.result, "vessels")
    terminals = records(args.result, "terminals")
    pairs = records(args.result, "pairs", "territory")
    position = {i: tuple(float(row[k]) for k in ("x", "y", "z")) for i, row in nodes.items()}
    imported_edges = {}
    generated = [0, 0, 0] if args.triplets else [0, 0]
    for identity, row in vessels.items():
        a, b = int(row["node0"]), int(row["node1"])
        require(a in nodes and b in nodes, f"Missing endpoints for vessel {identity}")
        system = int(row["system"])
        require(system in range(len(generated)), "Invalid system")
        if int(row["provenance"]) == 1:
            generated[system] += 1
        else:
            key = (system, min(a, b), max(a, b))
            require(key not in imported_edges, "Repeated imported edge")
            imported_edges[key] = row
    offset = 0
    expected_edges = 0
    filenames = ["arterial.vtk", "venous.vtk"] + (["portal.vtk"] if args.triplets else [])
    for system, filename in enumerate(filenames):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(str(args.input / filename))
        reader.ReadAllScalarsOn()
        reader.Update()
        grid = reader.GetOutput()
        radius = grid.GetPointData().GetArray("Radius")
        root = grid.GetPointData().GetArray("Root")
        require(radius is not None and root is not None, "Missing input arrays")
        for i in range(grid.GetNumberOfPoints()):
            identity = offset + i
            require(identity in nodes, f"Missing imported node {identity}")
            require(math.dist(position[identity], grid.GetPoint(i)) < 1.e-12,
                    f"Moved imported node {identity}")
            require(int(nodes[identity]["provenance"]) == 0, "Changed input node provenance")
            expected_port = (1, 2, 4)[system] if root.GetTuple1(i) else 0
            require(int(nodes[identity]["port"]) == expected_port, "Changed input port")
        for c in range(grid.GetNumberOfCells()):
            cell = grid.GetCell(c)
            require(cell.GetNumberOfPoints() == 2, "Expected individual input line cells")
            a, b = (offset + cell.GetPointId(j) for j in (0, 1))
            key = (system, min(a, b), max(a, b))
            require(key in imported_edges, f"Missing imported edge {key}")
            row = imported_edges[key]
            for endpoint in (0, 1):
                identity = int(row[f"node{endpoint}"])
                expected = radius.GetTuple1(identity - offset)
                require(math.isclose(float(row[f"radius{endpoint}"]), expected,
                                     rel_tol=1.e-12, abs_tol=1.e-14), "Changed imported radius")
            require(math.isclose(float(row["viscosity"]), args.viscosity, rel_tol=1.e-12),
                    "Changed imported viscosity")
            expected_edges += 1
        offset += grid.GetNumberOfPoints()
    require(len(imported_edges) == expected_edges, "Unexpected imported edges")
    require(sum(int(n["provenance"]) == 0 for n in nodes.values()) == offset,
            "Unexpected imported nodes")
    require(all(count > 0 for count in generated), "Every supplied system must grow")
    require(len(pairs) == (0 if args.triplets else args.pairs), "Wrong number of tissue pairs")
    used = set()
    for territory, pair in pairs.items():
        a, v = int(pair["arterial"]), int(pair["venous"])
        require(a in terminals and v in terminals and a != v, "Missing/distinct paired terminals")
        require(a not in used and v not in used, "Terminal paired more than once")
        used.update((a, v))
        require(int(terminals[a]["system"]) == 0 and int(terminals[v]["system"]) == 1,
                "Pair must join an artery and vein")
        require(all(int(terminals[t]["territory"]) == territory for t in (a, v)),
                "Inconsistent common territory")
        require(float(pair["resistance"]) > 0., "Nonpositive bed resistance")
        require(math.isclose(float(terminals[a]["demand"]), float(terminals[v]["demand"]),
                             rel_tol=1.e-12), "Unequal paired demand")
        points = [position[int(terminals[t]["node"])] for t in (a, v)]
        require(math.isclose(math.dist(*points), args.separation, rel_tol=1.e-10, abs_tol=1.e-12),
                "Wrong arterial-venous separation")

    if args.triplets:
        triplets = records(args.result, "triplets", "territory")
        require(len(triplets) == args.triplets, "Wrong number of common tissue compartments")
        for territory, t in triplets.items():
            ids = [int(t[k]) for k in ("arterial", "portal", "venous")]
            require(len(set(ids)) == 3 and all(i in terminals and i not in used for i in ids),
                    "Missing or repeated compartment terminal")
            used.update(ids)
            require([int(terminals[i]["system"]) for i in ids] == [0, 2, 1], "Wrong compartment roles")
            require(all(int(terminals[i]["territory"]) == territory for i in ids), "Inconsistent common territory")
            require(all(float(terminals[i]["demand"]) == float(terminals[ids[0]]["demand"]) for i in ids),
                    "Unequal compartment demand")
            points = [position[int(terminals[i]["node"])] for i in ids]
            require(math.isclose(math.dist(points[0], points[1]), args.portal_separation, rel_tol=1.e-10, abs_tol=1.e-12),
                    "Wrong arterial-portal separation")
            require(math.isclose(math.dist(points[0], points[2]), args.separation, rel_tol=1.e-10, abs_tol=1.e-12),
                    "Wrong arterial-venous separation")
            qa, qp, qv = [float(t[k]) for k in ("arterial_flow", "portal_flow", "venous_flow")]
            require(min(qa, qp, qv) > 0. and math.isclose(qa + qp, qv, rel_tol=1.e-9), "Invalid compartment balance")
            require(all(float(t[k]) > 0. for k in ("arterial_resistance", "portal_resistance", "venous_resistance")),
                    "Invalid compartment resistance")
            require(math.isfinite(float(t["compartment_pressure"])), "Invalid compartment pressure")
            if args.portal_flow_fraction is not None:
                require(math.isclose(qp / qv, args.portal_flow_fraction, rel_tol=1.e-8), "Wrong supply split")

    # Independent sample check against the exported volume, in addition to the
    # continuous geometry certification required by GenerationResult.
    pvtu = args.result / "liver_0.pvtu"
    pieces = []
    for piece in ET.parse(pvtu).findall(".//Piece"):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(pvtu.parent / piece.attrib["Source"]))
        reader.Update()
        if reader.GetOutput().GetNumberOfCells():
            pieces.append(reader.GetOutput())
    require(pieces, "No nonempty tissue mesh pieces")
    background = vtk.vtkUnstructuredGrid()
    merge = vtk.vtkMergeCells()
    merge.SetUnstructuredGrid(background)
    merge.SetTotalNumberOfDataSets(len(pieces))
    merge.SetTotalNumberOfPoints(sum(g.GetNumberOfPoints() for g in pieces))
    merge.SetTotalNumberOfCells(sum(g.GetNumberOfCells() for g in pieces))
    # DataOut duplicates patch vertices even in a serial file. Merge exact
    # duplicates before extracting the physical exterior, including one piece.
    merge.SetPointMergeTolerance(0.)
    for grid in pieces:
        require(merge.MergeDataSet(grid) == 0, "Cannot merge output mesh pieces")
    merge.Finish()
    require(background.GetNumberOfCells() > 0, "Empty background")
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(background)
    locator.BuildLocator()
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(background)
    surface.Update()
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(surface.GetOutput())
    minimum_margin = math.inf
    for row in vessels.values():
        if int(row["provenance"]) != 1:
            continue
        a, b = position[int(row["node0"])], position[int(row["node1"])]
        for t in (0., .25, .5, .75, 1.):
            point = tuple((1. - t) * x + t * y for x, y in zip(a, b))
            require(locator.FindCell(point) >= 0, "Generated centreline sample outside tissue")
            radius = (1. - t) * float(row["radius0"]) + t * float(row["radius1"])
            margin = abs(distance.EvaluateFunction(point)) - radius
            minimum_margin = min(minimum_margin, margin)
            require(margin >= -1.e-9, "Generated wall sample outside tissue")
    summary = dict(imported_nodes=offset, imported_vessels=expected_edges,
                   generated_arteries=generated[0], generated_veins=generated[1],
                   tissue_pairs=len(pairs), background_cells=background.GetNumberOfCells(),
                   minimum_sampled_wall_margin_m=minimum_margin)
    if args.triplets:
        summary.update(generated_portal=generated[2], tissue_triplets=args.triplets)
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--triplets", type=int, default=0, help="Check this number of three-terminal compartments instead of pairs")
    parser.add_argument("--portal-separation", type=float, default=.003)
    parser.add_argument("--portal-flow-fraction", type=float)
    parser.add_argument("--separation", type=float, default=.003)
    parser.add_argument("--viscosity", type=float, default=.004)
    parser.add_argument("--report", type=Path)
    check(parser.parse_args())


if __name__ == "__main__":
    main()
