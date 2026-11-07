"""Reproduce the saved arterial inspection figures without running CCO."""

from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import vtk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D


if len(sys.argv) != 3:
    raise SystemExit("Usage: python plot.py EXTRACTED_INSPECTION_ARCHIVE NEW_OUTPUT_DIR")
data = Path(__file__).resolve().parent
archive, output = map(Path, sys.argv[1:3])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = json.loads((data / "provenance.json").read_text())
for name, expected in provenance["documentation_input_sha256"].items():
    assert sha(data / name) == expected, name
mesh_path = archive / "liver-generation.vtu"
assert sha(mesh_path) == provenance["generation_mesh_sha256"]
output.mkdir(exist_ok=False, parents=True)
vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)

segments = {"imported": [], "CCO": []}
with (data / "arterial-projection-segments.csv").open() as stream:
    records = list(csv.DictReader(stream))
assert len(records) == 8542
assert len({int(row["PhysicalVesselId"]) for row in records}) == 8542
for row in records:
    points = np.array([[float(row[axis + str(j) + "_m"])
                        for axis in "xyz"] for j in (0, 1)])
    assert np.isfinite(points).all()
    generated = int(row["Generated"])
    assert generated in (0, 1)
    segments["CCO" if generated else "imported"].append(points * 1000)
assert len(segments["imported"]) == 54
assert len(segments["CCO"]) == 8488

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(str(mesh_path))
reader.Update()
mesh = reader.GetOutput()
assert reader.GetErrorCode() == 0 and mesh.GetNumberOfCells() == 4245
assert all(mesh.GetCellType(i) == vtk.VTK_TETRA for i in range(4245))
surface_filter = vtk.vtkDataSetSurfaceFilter()
surface_filter.SetInputData(mesh)
surface_filter.Update()
surface = surface_filter.GetOutput()
triangles = [np.array([surface.GetPoint(surface.GetCell(i).GetPointId(j))
                       for j in range(3)]) * 1000
             for i in range(surface.GetNumberOfCells())]
body = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())]) * 1000

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none"})
colors = {"imported": "#526B7B", "CCO": "#BE3438"}
figure, axes = plt.subplots(1, 3, figsize=(15, 6.6))
figure.subplots_adjust(left=.055, right=.985, bottom=.24, top=.80, wspace=.25)
for axis, (u, v) in zip(axes, ((0, 1), (0, 2), (1, 2))):
    axis.add_collection(PolyCollection([t[:, [u, v]] for t in triangles],
                                      facecolors="#F0F3F5", edgecolors="none",
                                      zorder=1))
    for key in ("imported", "CCO"):
        axis.add_collection(LineCollection(
            [segment[:, [u, v]] for segment in segments[key]],
            colors=colors[key], linewidths=.55,
            zorder=2 if key == "imported" else 3))
    axis.set_xlim(body[:, u].min() - 6, body[:, u].max() + 6)
    axis.set_ylim(body[:, v].min() - 6, body[:, v].max() + 6)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("xyz"[u] + " (mm)")
    axis.set_ylabel("xyz"[v] + " (mm)")
    axis.set_title("xyz"[u].upper() + "xyz"[v].upper() + " projection")
    axis.spines[["top", "right"]].set_visible(False)
figure.suptitle("Anatomical liver CCO: completed arterial stage", fontsize=18, y=.965)
figure.text(.5, .90, "4,245 generated arterial terminals · 8,542 arterial segments",
            ha="center", fontsize=12)
figure.legend(handles=[
    Line2D([0], [0], color=colors["imported"], lw=1,
           label="Imported artery: 54 segments"),
    Line2D([0], [0], color=colors["CCO"], lw=1,
           label="New arterial CCO: 8,488 segments")],
    loc="lower center", bbox_to_anchor=(.5, .135), ncol=2, frameon=False)
figure.text(.5, .085,
            "Arterial stage only; no completed companion networks or coupled pressure/flow/displacement solution.",
            ha="center", color="#53606A")
figure.text(.5, .04,
            "Constant line width: actual Radius values are not shown. Background: 4,245-cell generation mesh.",
            ha="center", fontsize=9, color="#64717D")
outputs = []
for extension in ("png", "svg"):
    path = output / ("arterial-growth-projections." + extension)
    figure.savefig(path, dpi=220, facecolor="white")
    outputs.append(path)
plt.close(figure)

coverage = json.loads((data / "arterial-coverage.json").read_text())
table = np.genfromtxt(data / "arterial-coverage-cells.csv", delimiter=",", names=True)
assert len(table) == 4245
upper = (table["centroid_nearest_terminal_distance_m"] +
         table["centroid_radius_to_furthest_vertex_m"]) / table["H_m"]
order = np.argsort(upper)
cumulative = 100 * np.cumsum(table["volume_m3"][order]) / np.sum(table["volume_m3"])
histogram = coverage["occupancy_histogram"]
x = np.array(sorted(map(int, histogram)))
y = np.array([histogram[str(i)] for i in x])
assert y.sum() == 4245 and np.dot(x, y) == 4245
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
figure.subplots_adjust(left=.08, right=.97, bottom=.24, top=.80, wspace=.30)
axes[0].bar(x, y, color="#526B7B", width=.7)
for value, count in zip(x, y):
    axes[0].text(value, count + 22, str(count), ha="center", fontsize=9)
axes[0].set(xticks=x, xlabel="Generated arterial terminals located per cell",
            ylabel="Background cells", ylim=(0, max(y) * 1.15),
            title="Occupancy: global mean 1.00, local maximum 5")
axes[1].step(upper[order], cumulative, where="post", color="#BE3438", linewidth=1.4)
axes[1].axvline(2, color="#526B7B", linestyle="--", linewidth=1,
               label="2H distance contract")
axes[1].set(xlabel="Sufficient whole-cell distance upper bound / H",
            ylabel="Cumulative background volume (%)", xlim=(0, 2.1),
            ylim=(0, 103), title="Arterial-only coarse-mesh coverage")
axes[1].legend(frameon=False, loc="lower right")
axes[1].text(.04, .83, "All cells certified within 2H\n100% of this coarse volume",
             transform=axes[1].transAxes, fontsize=10)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
figure.suptitle("A4245: independent geometric coverage and occupancy", fontsize=16, y=.95)
figure.text(.5, .10,
            "Bound: d(nearest terminal, centroid) + max centroid-to-vertex distance. H: longest tetrahedron edge.",
            ha="center", fontsize=9, color="#53606A")
figure.text(.5, .045,
            "No native quadrature replay, triple-system/finer-mesh coverage, perfusion or inf-sup conclusion.",
            ha="center", fontsize=9, color="#53606A")
for extension in ("png", "svg"):
    path = output / ("arterial-coverage-occupancy." + extension)
    figure.savefig(path, dpi=220, facecolor="white")
    outputs.append(path)
plt.close(figure)
for path in outputs:
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
report = {
    "new_CCO_or_PDE_runs": 0,
    "script_sha256": sha(Path(__file__)),
    "data_sha256": {name: sha(data / name)
                    for name in provenance["documentation_input_sha256"]},
    "generation_mesh_sha256": sha(mesh_path),
    "output_sha256": {path.name: sha(path) for path in outputs},
    "matplotlib": matplotlib.__version__, "numpy": np.__version__,
    "vtk": vtk.vtkVersion.GetVTKVersion(),
}
(output / "render-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
