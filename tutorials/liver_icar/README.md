# Coupled IRCAD12 liver profiles

This example couples an IRCAD12 liver volume to separate arterial, portal and
hepatic venous networks through the vessel wall. The solid exterior uses
`traction + 1e5*u = 0`, with no essential displacement boundary. The physical
vascular archive is reused at every scale; its numerical intervals are rebuilt
against the selected tissue mesh.

The archive has 14 physical vessels: three supplied interior feeder stubs and
11 generated vessels. Three genuine CCO bifurcations, one per vascular role,
connect six tissue leaves through two shared microvascular compartments. Every
tissue leaf belongs to a triplet. This small starting case does not resolve
whole-organ vascular coverage or establish patient-specific physiological validation.

## Run from release assets

The [research release](https://github.com/luca-heltai/immersx/releases/tag/liver-icar-start-2026-11-05)
contains `liver-icar-case.tar.gz`, three generated liver grids, the physical
CCO archive, parameters, scripts, provenance and numerical results. After
extraction, use `liver-icar/assets` as `--assets-dir`. The separate
[report](https://github.com/luca-heltai/immersx/releases/download/liver-icar-start-2026-11-05/liver-icar-report.md)
records every literature reference, measured result and modelling assumption.

Use MetricFlowX revision `129cf1107258670457acabfeb288adfea768a3bc`
([PR #87](https://github.com/luca-heltai/metric-flow-x/pull/87)) or a revision
containing its two physical corrections. Older builds can have wrong reversed-root
RCR signs and missing other-terminal external pressures even when they compile.

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

| Profile | Solid / native flow degree | Solid displacement + velocity DoFs | Numerical 1D cells | Coupled solver and default ranks | Default duration / output period |
| --- | --- | ---: | ---: | --- | ---: |
| `tiny` | P1 / DG0 | 19,818 | 26 | UMFPACK, 1 rank | 0.8 s / 10 ms |
| `small` | P2 / DG1 | 132,672 | 26 | MUMPS, 4 ranks | 0.8 s / 10 ms |
| `medium` | P2 / DG1 | 992,034 | 55 | Iterative, 8 ranks | 2 ms / 2 ms |
| `large` | P2 / DG1 | 10,206,420 | 128 | Iterative, 112 ranks | 2 ms / 2 ms |

The measured total coupled DoFs are 20,000, 132,932, 992,584 and
10,207,700 respectively. The table counts only solid displacement and velocity.
The [scaling tracker](https://github.com/luca-heltai/immersx/issues/343)
records the tested revisions, MPI counts, resource bounds and numerical audits.
The 2 ms startup and conservation checks with C++ runtime `b4d85cb6` used
these solver configurations: tiny iterative on 4 MPI ranks, small MUMPS on
4 ranks, medium iterative on 8 ranks, and large iterative on 112 ranks.
The tiny test manually overrode the serial UMFPACK recipe and selected flow
inverse limit 2048 and Schur materialization limit 512. The serial UMFPACK
runner default is a separate configuration. The complete 0.8 s small beat
is an earlier reference using runtime `a92bee56`.
The medium case has also passed the 20 ms transient with the public
configuration on 8 ranks, using a 2 ms output period and 11 saved times.
That recorded run took 6056.93 s, compared with 703.78 s for the 2 ms startup
check. The default remains 2 ms to keep the initial run shorter. These are
observed wall times, not a performance guarantee. A full beat on medium or
large has not been validated. These observations do not establish mesh
convergence, a uniform inf-sup bound or physiological validation.

Omitting `--duration` selects the interval in the table, starting at 7.2 s.
The medium and large defaults are short startup checks. Consistent
initialization can still dominate their wall time. Tiny and small retain a
0.8 s default; the earlier complete small beat remains a separate reference.
Use `--duration .002` for a short check at either smaller size.

An explicit `--duration` overrides every profile default and must lie in
`(0, 0.8]` s. `--output-period` selects a finite positive output interval in
seconds. It defaults to 0.01 s for tiny/small and 0.002 s for medium/large.
Both time increments must advance the start time. The rendered output
interval is the smaller of the requested period and the run duration;
`run.json` records both `requested_output_period_s` and `output_period_s`.
Because IDA stops at each output, match the recorded period when reproducing
a numerical run. Use `--profile medium --duration .02` for the audited
extended check; its default 2 ms period saves 11 times, including both endpoints.

For example, `--profile large --duration .8 --output-period .01` requests the
whole recorded 7.2–8 s interval with 81 saved times. This full large interval
has not been validated. The runner neither repeats nor extrapolates the input
CSV. Its metadata records the requested duration and
`current_runtime_audited_duration_s` separately.
`unvalidated_scale_profile` flags medium or large requests beyond their audited
intervals; tiny and small retain their existing full-interval workflow. The
validation scope still identifies the earlier reference for a full beat.
Changed assets, physical controls or tolerances require new audits. Every new
run needs its own numerical checks.

Tiny requires one rank. `--ranks` overrides the other profiles' defaults and
must match the available memory and CPU allocation. The runner selects time
intervals and execution resources; the physical parameters, tolerance values,
meshes and three problem-size levels remain unchanged.

The native archive redistribution and discretization checks passed at four
ranks on all three meshes. Across every crossed background cell, numerical
edge length divided by cell diameter lies in 0.708–1.962, 0.635–1.768 and
0.782–1.307 respectively. There are no short or exterior intervals. These
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
may reduce it. Maximum BDF order is two. The runner selects output spacing
from the profile or `--output-period`; the common template uses 0.01 s.
The MUMPS profiles request 100% extra workspace relative to the native
estimate; this changes allocation rather than the physical parameters.
Newton uses convergence coefficient `1e-4`, separate from LTE tolerances,
to control nonlinear algebraic residuals in the mixed physical units.
The liver recipe stops IDA at every output time. Saved states are evaluated
at internal step endpoints, with algebraic residuals controlled by Newton
convergence. Dense interpolation between steps need not satisfy those equations.

Tolerances use displacement, velocity, pressure and native area units. The
flow field reuses MetricFlowX's componentwise tolerance vector. For a tenfold
tighter control, use `--tolerance-scale .1` in a separate output directory;
this scales every absolute tolerance and the common relative tolerance.
Changing only the relative tolerance can leave absolute-tolerance-dominated
components almost unchanged. Shared IDA statistics and the final timing table
are the relevant solver diagnostics; native standalone IDA CSVs are disabled.

## Bounded coupled inverses

Medium and large select these controls under
`MetricFlowSystem<1, 3>/IDA parameters/Linear solver`:

```text
set Maximum replicated flow inverse DoFs = 2048
set Maximum materialized Schur DoFs = 512
```

The first selects a replicated LAPACK LU inverse for a flow field no larger
than the limit. The flow fields have 440 and 1024 DoFs; this does not replicate
the tissue matrix. Above the limit the application retains its default ILU.
Zero retains ILU at every size. The native standalone `Use direct solver`
option does not select the coupled field inverse.

The second samples the small Schur operator once per Jacobian snapshot and
reuses its dense column matrix in the inner GMRES. The limits apply to the
110/256 multiplier DoFs, not the roughly one/ten million total DoFs. A zero
limit keeps matrix-free inner actions, as does a field above the limit.
Storage is quadratic in the selected small-field size on every MPI rank.
The sampled Schur operator still uses approximate tissue inverses; it is not
the exact Schur complement of the full coupled Jacobian.

GMRES retains its metric and tolerance. Every materialized inner solve checks
its residual again with the original operator in that metric, including
transpose solves. If sampling roundoff leaves this check above target, it
performs up to three residual corrections within the original inner iteration
budget, with the same target based on the original right-hand side. Failure
still throws. The outer coupled true-residual check also remains active. Snapshot time and derivative weight describe the retained
linearization, which IDA may reuse during chord Newton iterations.

In C++, configure the execution adapter before initialization:

```cpp
using FlowVector = MetricFlowX::VectorType;
adapter.set_schur_materialization_limit(512);
auto flow = ImmersX::metric_flow_x(flow_problem);
flow.preconditioner_factory = [](const auto &matrix, const auto &prototype) {
  return ImmersX::make_replicated_direct_preconditioner<FlowVector>(
    matrix, prototype, 2048);
};
const auto fields = adapter.add(flow, "blood-flow");
```

The direct helper requires a nonempty square MPI field block within its
explicit bound and matching prototype layout. It owns the LU factors,
supports `Tvmult`, and is collective even on ranks with no owned flow rows.
Callers choose their fallback before selecting this strict helper. Leaving
the descriptor factory empty preserves ordinary `metric_flow_x(problem)`
registration and its ILU default.

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
    'venous': ((.150, .140, .180), (.130, .140, .180), .0065),
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
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/mesh-small.prm"
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/mesh-medium.prm"
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/mesh-large.prm"
mkdir "$ASSETS/meshes"
cp "$BUILD/test_output/liver_icar/mesh-small/liver.msh" "$ASSETS/meshes/small.msh"
cp "$BUILD/test_output/liver_icar/mesh-medium/liver.msh" "$ASSETS/meshes/medium.msh"
cp "$BUILD/test_output/liver_icar/mesh-large/liver.msh" "$ASSETS/meshes/large.msh"
mpirun -np 1 "$BUILD/liver_vascular" "$BUILD/data/liver_icar/generate.prm"
micromamba run -n ircad-prep python scripts/check_liver_vascular.py "$ASSETS/inputs" \
  "$BUILD/test_output/liver_icar/generation" --triplets 2 \
  --separation .015 --portal-separation .015 \
  --portal-flow-fraction .69576933619234205 --viscosity .00371
cp -a "$BUILD/test_output/liver_icar/generation/network" "$ASSETS/network"
```

The mesh-only profiles generate each volume from the prepared STL. Physical
network generation then reads the exact small mesh. It uses a 35 mm startup
extension, a 10 mm minimum physical branch length, 32 candidate vessels,
ten local optimization steps, random seed 23 and a 5,000-sample insertion
budget. The accepted two-triplet archive was generated on one rank. The
four-rank trial with the same seed exhausted its sampling budget; use the
serial command above for reproduction. The saved archive was successfully
redistributed and rediscretized on four ranks at all three mesh sizes.

Every generation output directory must be new. The physical generator exports
`liver.msh`, the `network/` archive and diagnostics. The three bifurcation nodes
have degree three and generated provenance, as do their incident edges. There
are no unused tissue leaves. The minimum physical branch length is 10.028 mm;
the independent sampled wall margin is at least 5.769 mm.

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
