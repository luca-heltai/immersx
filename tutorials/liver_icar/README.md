# Coupled IRCAD12 liver profiles

This example couples an IRCAD12 liver volume to separate arterial, portal and
hepatic venous networks through the vessel wall. The solid exterior uses
`traction + 1e5*u = 0`, with no essential displacement boundary. The physical
vascular archive is reused at every scale; its numerical intervals are rebuilt
against the selected tissue mesh.

The initial archive has three supplied interior feeder stubs and five accepted
CCO extensions, joined through one microvascular compartment. It has **no CCO
bifurcations** and does not resolve whole-organ vascular coverage. It is a small
starting case, not a patient-specific physiological validation.

## Run from release assets

Build `metric_flow_x_elastodynamics` with compatible MetricFlowX, SUNDIALS and
MPI libraries. Extract the liver release bundle into an asset directory with
this layout:

```text
assets/
  meshes/small.msh
  meshes/medium.msh
  meshes/large.msh
  network/manifest
  network/part-*.txt
  inputs/inlet_ha.csv
  inputs/liver.stl
  inputs/arterial.vtk
  inputs/venous.vtk
  inputs/portal.vtk
```

Keep the bundle's attribution, input provenance, report and checksums alongside
these files. Its generated anatomical meshes are distributed with permission
for this case; retain the original IRCAD attribution and licence.

From the repository root, prepare a short serial run without executing it:

```bash
python3 tutorials/liver_icar/run_case.py \
  --build-dir /absolute/path/to/build \
  --assets-dir /absolute/path/to/assets \
  --output /absolute/path/to/new-run \
  --profile tiny --duration .02 --prepare-only
```

Review `common.prm`, `parameters.prm` and `run.json` in the new run directory.
Omit `--prepare-only` to execute the same command in a new directory. The runner
refuses an existing output directory, validates CSV time coverage, records
input checksums and writes the solver output to `run.log`. It resolves every
runtime path absolutely, so it can also be launched from another directory.
The default asset directory is `<build>/data/liver_icar/assets`.

| Profile | Solid / native flow degree | Solid displacement + velocity DoFs | Numerical 1D cells | Coupled solver |
| --- | --- | ---: | ---: | --- |
| `tiny` | P1 / DG0 | 19,818 | 9 | Serial UMFPACK |
| `small` | P2 / DG1 | 132,672 | 9 | MUMPS, four ranks by default |
| `medium` | P2 / DG1 | 992,034 | 23 | Iterative; coupled run unvalidated |
| `large` | P2 / DG1 | 10,206,420 | 52 | Iterative; coupled run unvalidated |

The tiny profile has 19,881 total coupled DoFs. The other solid counts exclude
native flow, traces, capacitors and multipliers; read the actual coupled count
from the executable. Medium and large are preparation profiles, not completed
large simulations or a demonstrated scalable solver. All profiles default to
the recorded 7.2–8 s interval; use `--duration` for a shorter startup check.
Tiny requires one rank. `--ranks` overrides the other profiles' default.

The native archive redistribution and discretization checks passed at four
ranks on all three meshes. Across every crossed background cell, numerical
edge length divided by cell diameter lies in 0.727–1.775, 0.717–1.562 and
0.744–1.304 respectively. There are no short or exterior intervals. These
geometric checks do not prove a positive discrete inf-sup constant. Do not
refine the 1D grid independently of its background.

The independent surface remeshes use target edge lengths 0.012, 0.0045 and
0.0018 m. Their volumes are 1.5154, 1.5273 and 1.5293 litres. They provide
problem-size levels; the slightly different discrete domains must be accounted
for before interpreting a comparison as convergence.

## Physical data and numerical controls

The arterial CSV sums the eight `_ORG11_` liver terminal flows from the
physiological MOX-driven full ADAVN version-3 run. It excludes the cystic artery
and retains SI units and physical times. There is no periodic repetition or
extrapolation. Only this flow is imposed at the arterial root; its reported
upstream pressure is used as a constitutive reference and diagnostic.

Portal pressure is an assumed 8 mmHg reservoir (1066.57909932 Pa). The venous
reservoir is the selected MOX IVC reference (629.59739556 Pa), rather than the
peripheral hepatic venous entrance. A 69.5769% portal fraction and total design
flow 2.0836421871998312e-5 m3/s determine the generated reference compartments.
The transient portal flow and venous discharge are solved, not prescribed from
those design totals. Root reservoir series resistance is 1e5 Pa s/m3.

| Quantity | Value | Interpretation |
| --- | ---: | --- |
| Blood density / viscosity | 1060 kg/m3 / 0.00371 Pa s | Reference modelling values |
| Root radii, artery / hepatic vein / portal | 2.2 / 6.5 / 4.9 mm | Literature surrogates on synthetic interior stubs |
| Vessel Young modulus / thickness-to-radius ratio | 400000 Pa / 0.1 | Declared uniform wall model |
| Solid Young modulus / Poisson ratio | 6000 Pa / 0.49 | Declared homogeneous tissue surrogate |
| Native solid viscosity coefficient | 4.2 Pa s | Conditional Kelvin–Voigt mapping described below |
| Exterior spring coefficient | 1e5 Pa/m | Assumed external tissue support |

Pearson and coauthors report median shear viscosity 2.1 Pa s for healthy
volunteers at 50 Hz. The native damping term is
`eta * integral(epsilon(v):epsilon(test_v))`; matching the shear law
`2*eta_shear*epsilon(v)` therefore gives the native input `eta = 4.2 Pa s`.
Using that frequency-dependent measurement at cardiac time scales is a model
assumption, not a fit to this liver. No Rayleigh damping is added. Retain an
undamped control with `--solid-viscosity 0` when assessing this assumption.

The common file contains the materials, boundary data and time controls; the
four short profile files select the mesh, FE degree, solver and output paths.
The IDA correction `use_y_dot` holds the initial derivative at zero and solves
for a coupled static starting state. The first requested step is 0.001 s; IDA
may reduce it. Maximum BDF order is two and output spacing is 0.01 s.

Tolerances use displacement, velocity, pressure and native area units. The
flow field reuses MetricFlowX's componentwise tolerance vector. For a tenfold
tighter control, use `--tolerance-scale .1` in a separate output directory;
this scales every absolute tolerance and the common relative tolerance.
Changing only the relative tolerance can leave absolute-tolerance-dominated
components almost unchanged. Shared IDA statistics and the final timing table
are the relevant solver diagnostics; native standalone IDA CSVs are disabled.

## Regenerate the volume and physical archive

Follow the existing [IRCAD preparation tutorial](../../doc/tutorials/liver-vascularization.md)
with a new preparation directory and `--with-portal`. It downloads and checks
case 12, converts millimetres to metres and writes `liver.stl`. Use a fresh
asset root for regeneration; do not mix it with an extracted release bundle.
After setting `BUILD` to an absolute build directory:

```bash
export BUILD=/absolute/path/to/build
export ASSETS="$BUILD/data/liver_icar/assets"
micromamba run -n ircad-prep python scripts/prepare_ircad_liver.py \
  "$BUILD/ircad12-prepared" --with-portal
mkdir -p "$ASSETS/inputs"
cp "$BUILD/ircad12-prepared/liver.stl" "$ASSETS/inputs/liver.stl"
python3 - "$ASSETS/inputs" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
seeds = {
    'arterial': ((.100, .160, .200), (.120, .160, .200), .0022),
    'venous': ((.150, .150, .200), (.150, .170, .200), .0065),
    'portal': ((.100, .185, .200), (.120, .185, .200), .0049),
}
for role, (a, b, radius) in seeds.items():
    points = '\n'.join(' '.join(map(str, point)) for point in (a, b))
    (root / f'{role}.vtk').write_text(
        '# vtk DataFile Version 3.0\nIRCAD12 interior feeder stub\nASCII\n'
        'DATASET UNSTRUCTURED_GRID\nPOINTS 2 double\n' + points + '\n'
        'CELLS 1 3\n2 0 1\nCELL_TYPES 1\n3\nPOINT_DATA 2\n'
        'SCALARS Radius double 1\nLOOKUP_TABLE default\n'
        f'{radius}\n{radius}\n'
        'SCALARS Root int 1\nLOOKUP_TABLE default\n1\n0\n'
        'SCALARS Grow int 1\nLOOKUP_TABLE default\n0\n1\n')
PY
```

The stubs are interior modelling choices, not image-extracted centreline trees.
The tissue surface remains the prepared IRCAD surface. CMake configures all
`.prm.in` files under `data/liver_icar`; use their generated build-tree copies:

```bash
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DMetricFlowX_DIR=/absolute/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build "$BUILD" --target liver_vascular metric_flow_x_elastodynamics -j 4
mpirun -np 4 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/generate.prm"
micromamba run -n ircad-prep python scripts/check_liver_vascular.py "$ASSETS/inputs" \
  "$BUILD/test_output/liver_icar/generation" --triplets 1 \
  --separation .015 --portal-separation .015 \
  --portal-flow-fraction .69576933619234205 --viscosity .00371
```

The generator requires a new output directory and exports `liver.msh`, the
physical `network/` archive and diagnostics. For the selected volume levels:

```bash
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/mesh-small.prm"
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/mesh-medium.prm"
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/mesh-large.prm"
mkdir "$ASSETS/meshes"
cp "$BUILD/test_output/liver_icar/mesh-small/liver.msh" "$ASSETS/meshes/small.msh"
cp "$BUILD/test_output/liver_icar/mesh-medium/liver.msh" "$ASSETS/meshes/medium.msh"
cp "$BUILD/test_output/liver_icar/mesh-large/liver.msh" "$ASSETS/meshes/large.msh"
cp -a "$BUILD/test_output/liver_icar/generation/network" "$ASSETS/network"
```

Retain the exact `inputs/inlet_ha.csv` and its source extraction manifest from
the release. The coupled importer redistributes and rediscretizes this same
physical network against each chosen background; it does not regrow it.

## Sources

- Soler L and coauthors, *3D image reconstruction for comparison of algorithm
  database: A patient specific anatomical and medical image database*, IRCAD
  technical report, 2010. [3D-IRCADb-1 dataset](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/),
  original [CC BY-NC-ND 4.0 licence](https://creativecommons.org/licenses/by-nc-nd/4.0/).
- Tithof J, Pruett TL, Rao JS, *Lumped parameter liver simulation to predict
  acute haemodynamic alterations following partial resections*, Journal of the
  Royal Society Interface 20, 20230444 (2023).
  [DOI: 10.1098/rsif.2023.0444](https://doi.org/10.1098/rsif.2023.0444).
- Pearson A and coauthors, *Vibration-controlled transient elastography for
  noninvasive evaluation of liver steatosis*, Medical Physics 49, 1507–1521
  (2022). [DOI: 10.1002/mp.15484](https://doi.org/10.1002/mp.15484), Table 2.
- Moireau P and coauthors, *External tissue support and fluid–structure
  simulation in blood flows*, Biomechanics and Modeling in Mechanobiology 11,
  1–18 (2012). [DOI: 10.1007/s10237-011-0289-z](https://doi.org/10.1007/s10237-011-0289-z).
  This supports the spring formulation, not the chosen coefficient.
- Müller LO, Watanabe SM, Toro EF, Feijóo RA, Blanco PJ, *An anatomically
  detailed arterial-venous network model. Cerebral and coronary circulation*,
  Frontiers in Physiology 14, 1162391 (2023).
  [DOI: 10.3389/fphys.2023.1162391](https://doi.org/10.3389/fphys.2023.1162391).
  MOX-driven numerical input provenance and limitations are tracked in
  [deal.II-X issue 42](https://github.com/dealii-X/dealii-X/issues/42) and the
  [liver integration issue](https://github.com/dealii-X/dealii-X/issues/29).
- Schreiner W, Buxbaum PF, *Computer-optimization of vascular trees*, IEEE
  Transactions on Biomedical Engineering 40, 482–491 (1993).
  [DOI: 10.1109/10.243413](https://doi.org/10.1109/10.243413).
