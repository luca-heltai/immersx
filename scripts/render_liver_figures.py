#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Render the liver tutorial figures from checked native application outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import vtk

from check_liver_vascular import check, records


def volume_mesh(path):
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    raw = reader.GetOutput()
    # DataOut repeats patch vertices, even in a single piece. Merge exact
    # duplicates before extracting the exterior; otherwise interior faces show.
    grid = vtk.vtkUnstructuredGrid()
    merge = vtk.vtkMergeCells()
    merge.SetUnstructuredGrid(grid)
    merge.SetTotalNumberOfDataSets(1)
    merge.SetTotalNumberOfCells(raw.GetNumberOfCells())
    merge.SetTotalNumberOfPoints(raw.GetNumberOfPoints())
    merge.SetPointMergeTolerance(0.)
    merge.MergeDataSet(raw)
    merge.Finish()
    return grid


def overview(args, background):
    nodes = records(args.result, "nodes")
    vessels = list(records(args.result, "vessels").values())
    terminals = records(args.result, "terminals")
    pairs = records(args.result, "triplets" if args.triplets else "pairs", "territory")
    positions = {i: np.array([float(row[k]) for k in ("x", "y", "z")]) * 1000
                 for i, row in nodes.items()}
    exterior = vtk.vtkDataSetSurfaceFilter()
    exterior.SetInputData(background)
    exterior.Update()
    surface = exterior.GetOutput()
    vertices = np.array([surface.GetPoint(i) for i in range(surface.GetNumberOfPoints())]) * 1000
    faces = [vertices[[surface.GetCell(i).GetPointId(j)
                       for j in range(surface.GetCell(i).GetNumberOfPoints())]]
             for i in range(surface.GetNumberOfCells())]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    middle, extent = (low + high) / 2, (high - low).max() / 2
    imported = sum(int(row["provenance"]) == 0 for row in vessels)
    generated = [sum(int(row["system"]) == system and int(row["provenance"]) == 1
                     for row in vessels) for system in ((0, 1, 2) if args.triplets else (0, 1))]
    fig = plt.figure(figsize=(12, 6), facecolor="white")
    for panel, title in enumerate(("Prepared input trees", "After common-territory growth" if args.triplets else "After paired growth")):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(faces, facecolors="#acb5b8", edgecolors="none",
                                           alpha=.045, zsort="average"))
        colors = [(0, "#bd3028"), (1, "#2868ae")]
        if args.triplets:
            colors.append((2, "#9966b4"))
        for system, color in colors:
            for provenance in (0, 1):
                if panel == 0 and provenance == 1:
                    continue
                segments = [[positions[int(row["node0"])], positions[int(row["node1"])]]
                            for row in vessels if int(row["system"]) == system
                            and int(row["provenance"]) == provenance]
                ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=.65,
                                                     alpha=.65 if provenance == 0 else .95))
        if panel:
            points = np.array([positions[int(terminals[int(pair["arterial"])]["node"])]
                               for pair in pairs.values()])
            ax.scatter(*points.T, s=2, color="#16886e", depthshade=False)
        ax.set_xlim(middle[0] - extent, middle[0] + extent)
        ax.set_ylim(middle[1] - extent, middle[1] + extent)
        ax.set_zlim(middle[2] - extent, middle[2] + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=21, azim=-65)
        ax.set_axis_off()
        ax.set_title(title, pad=-4)
        caption = (f"{imported:,} fixed input segments" if panel == 0 else
                   f"{generated[0]:,} new arterial + {generated[1]:,} new venous segments")
        if args.triplets and panel:
            caption += f" + {generated[2]:,} new portal segments"
        ax.text2D(.5, .02, caption, transform=ax.transAxes, ha="center", fontsize=10)
    fig.suptitle(f"IRCAD case 12 · {len(pairs):,} tissue {'triplets' if args.triplets else 'pairs'}", fontsize=15, y=.97)
    handles = [Line2D([0], [0], color="#bd3028", label="Arterial"),
                        Line2D([0], [0], color="#2868ae", label="Venous"),
                        Line2D([0], [0], marker="o", linestyle="", color="#16886e",
                               label="Tissue territories")]
    if args.triplets:
        handles.insert(2, Line2D([0], [0], color="#9966b4", label="Portal"))
    fig.legend(handles=handles, loc="lower center", ncol=4 if args.triplets else 3,
               frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=0, right=1, top=.92, bottom=.08, wspace=-.1)
    fig.savefig(args.output / "before-after.png", dpi=180, metadata={"Software": "ImmersX"})
    plt.close(fig)


def mesh_section(args, background):
    bounds = background.GetBounds()
    y = (bounds[2] + bounds[3]) / 2
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, y, 0)
    plane.SetNormal(0, 1, 0)
    cut = vtk.vtkCutter()
    cut.SetInputData(background)
    cut.SetCutFunction(plane)
    # Preserve quadrilateral sections of tetrahedra, without artificial diagonals.
    cut.GenerateTrianglesOff()
    cut.Update()
    section = cut.GetOutput()
    polygons = [np.array([section.GetPoint(section.GetCell(i).GetPointId(j))
                          for j in range(section.GetCell(i).GetNumberOfPoints())])[:, [0, 2]] * 1000
                for i in range(section.GetNumberOfCells()) if section.GetCell(i).GetCellDimension() == 2]
    if not polygons:
        raise RuntimeError("The selected plane does not intersect the native volume")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.add_collection(PolyCollection(polygons, facecolors="#edf2f7", edgecolors="#435c75",
                                     linewidths=.4))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"Native tissue volume · {background.GetNumberOfCells():,} volume cells\n"
                 f"Section at y = {1000 * y:.1f} mm")
    fig.savefig(args.output / "volume-section.png", dpi=180, metadata={"Software": "ImmersX"})
    plt.close(fig)
    return y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prepared IRCAD input directory")
    parser.add_argument("result", type=Path, help="Native liver_vascular output directory")
    parser.add_argument("output", type=Path, help="Figure output directory")
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--triplets", type=int, default=0)
    parser.add_argument("--portal-separation", type=float, default=.003)
    parser.add_argument("--portal-flow-fraction", type=float)
    parser.add_argument("--separation", type=float, default=.003)
    parser.add_argument("--viscosity", type=float, default=.004)
    parser.add_argument("--source-revision", default="unrecorded",
                        help="Source revision used to build the simulation binary")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report = args.output / "verification.json"
    check(args)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    background = volume_mesh(args.result / "liver_0.pvtu")
    overview(args, background)
    y = mesh_section(args, background)
    sources = {"input/" + name: args.input / name for name in
               ("liver.stl", "arterial.vtk", "venous.vtk", "provenance.json")}
    if args.triplets:
        sources["input/portal.vtk"] = args.input / "portal.vtk"
    sources.update({"result/" + p.name: p for p in args.result.glob("*.csv")})
    pvtu = args.result / "liver_0.pvtu"
    sources["result/" + pvtu.name] = pvtu
    for piece in ET.parse(pvtu).findall(".//Piece"):
        name = piece.attrib["Source"]
        sources["result/" + name] = args.result / name
    manifest = dict(
        source_revision=args.source_revision, dataset="3D-IRCADb-1 case 12",
        attribution="Luc Soler et al., IRCAD technical report, 2010",
        dataset_url="https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/",
        dataset_license="https://creativecommons.org/licenses/by-nc-nd/4.0/",
        verification=json.loads(args.report.read_text()),
        rendering=dict(length_unit="mm", elevation_degrees=21, azimuth_degrees=-65,
                       line_widths="display widths, not physical diameters", section_y_m=y),
        software=dict(matplotlib=matplotlib.__version__, numpy=np.__version__, vtk=vtk.vtkVersion.GetVTKVersion()),
        sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(sources.items())},
        figures={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("*.png"))})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered liver figures in {args.output}")


if __name__ == "__main__":
    main()
