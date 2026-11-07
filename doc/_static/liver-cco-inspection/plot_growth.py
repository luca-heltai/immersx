"""Plot an already validated inspection checkpoint, without running CCO."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import vtk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

source, mesh_path, output = map(Path, sys.argv[1:4])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source) == '29ebf47afc93a649836ba65d47541196d4439eab870523c80ca79ad9aa6e7649'
assert sha(mesh_path) == '494b3b278f16af6c35a188076cde9df8a95733f00eb8952d69d08d363c287854'
report = dict(status='complete_partial_checkpoint',vessels=6979,identity=[0,3073,4245])
output.mkdir(exist_ok=False, parents=True)
vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(str(source))
reader.Update()
grid = reader.GetOutput()
assert reader.GetErrorCode() == 0 and grid.GetNumberOfCells() == report['vessels']
segments = {key: [] for key in ('input', 'arterial', 'venous', 'portal')}
roles = ('arterial', 'venous', 'portal')
for i in range(grid.GetNumberOfCells()):
    cell = grid.GetCell(i)
    points = np.array([grid.GetPoint(cell.GetPointId(j)) for j in range(2)]) * 1000
    assert np.isfinite(points).all()
    role = int(grid.GetCellData().GetArray('System').GetValue(i))
    generated = int(grid.GetCellData().GetArray('Generated').GetValue(i))
    segments[roles[role] if generated else 'input'].append(points)
mesh_reader = vtk.vtkXMLUnstructuredGridReader()
mesh_reader.SetFileName(str(mesh_path))
mesh_reader.Update()
mesh = mesh_reader.GetOutput()
assert mesh_reader.GetErrorCode() == 0 and mesh.GetNumberOfCells() == 4245
assert all(mesh.GetCellType(i) == vtk.VTK_TETRA for i in range(mesh.GetNumberOfCells()))
mesh_output = output / 'liver-generation.vtu'
mesh_writer = vtk.vtkXMLUnstructuredGridWriter()
mesh_writer.SetFileName(str(mesh_output))
mesh_writer.SetInputData(mesh)
assert mesh_writer.Write() == 1 and mesh_writer.GetErrorCode() == 0
surface_filter = vtk.vtkDataSetSurfaceFilter()
surface_filter.SetInputData(mesh)
surface_filter.Update()
surface = surface_filter.GetOutput()
triangles = [np.array([surface.GetPoint(surface.GetCell(i).GetPointId(j))
                       for j in range(3)]) * 1000
             for i in range(surface.GetNumberOfCells())]
body = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())]) * 1000
colors = dict(input='#8C969F', arterial='#C73339', venous='#2465A3', portal='#168649')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
figure, axes = plt.subplots(1, 3, figsize=(15, 6.6))
figure.subplots_adjust(left=.055, right=.985, bottom=.24, top=.80, wspace=.25)
for axis, (u, v) in zip(axes, ((0, 1), (0, 2), (1, 2))):
    axis.add_collection(PolyCollection([t[:, [u, v]] for t in triangles],
                                      facecolors='#F1F4F7', edgecolors='none', zorder=1))
    for key, values in segments.items():
        if values:
            axis.add_collection(LineCollection([s[:, [u, v]] for s in values],
                                colors=colors[key], linewidths=.85,
                                zorder=2 if key == 'input' else 3))
    axis.set_xlim(body[:, u].min() - 6, body[:, u].max() + 6)
    axis.set_ylim(body[:, v].min() - 6, body[:, v].max() + 6)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('xyz'[u] + ' (mm)')
    axis.set_ylabel('xyz'[v] + ' (mm)')
    axis.set_title('xyz'[u].upper() + 'xyz'[v].upper() + ' projection')
    axis.spines[['top', 'right']].set_visible(False)
role, count, requested = report['identity']
figure.suptitle('Liver CCO growth checkpoint', fontsize=18, y=.965)
figure.text(.5, .90, f'Active system: {roles[role]}; mutable leaves: {count:,} / requested {requested:,}',
            ha='center', fontsize=12)
legend = [Line2D([0], [0], color=colors[key], lw=1,
                 label=f'{"Anatomical input" if key == "input" else "New " + key + " CCO"}: {len(values):,} edges')
          for key, values in segments.items() if values]
figure.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .135),
              ncol=len(legend), frameon=False)
figure.text(.5, .085, 'Partial arterial growth; no coupled pressure, flow or displacement solution.',
            ha='center', color='#53606A')
figure.text(.5, .04, f'Constant line width does not encode radius. Liver silhouette: {mesh.GetNumberOfCells():,} generation cells.',
            ha='center', fontsize=9, color='#64717D')
paths = [output / ('arterial-growth-projections.' + extension) for extension in ('png', 'svg')]
for path in paths:
    figure.savefig(path, dpi=220, facecolor='white')
    if path.suffix == '.svg':
        path.write_text(chr(10).join(line.rstrip() for line in path.read_text().splitlines()) + chr(10))
plt.close(figure)
provenance = dict(partial_checkpoint=True, coupled_solution=False,
                  identity=report['identity'], segments={k: len(v) for k, v in segments.items()},
                  source_sha256={str(p): sha(p) for p in (source, mesh_path)},
                  script_sha256=sha(Path(__file__)),
                  output_sha256={p.name: sha(p) for p in paths + [mesh_output]},
                  visual_review='pending')
(output / 'figure-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(provenance, indent=2))
