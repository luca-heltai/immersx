# Vascularize a liver from a surface and two 1D trees

This tutorial starts with a closed liver surface and two rooted 1D vessel
meshes. `Domain<3,3>` meshes the enclosed tissue with the Gmsh API. The
`liver_vascular` application then extends the supplied arterial and venous
leaves toward common tissue territories. The supplied nodes, vessels and
radii remain fixed during growth.

```{mermaid}
flowchart LR
  S[Closed liver surface] --> M[Gmsh volume mesh]
  A[Arterial 1D tree] --> G[Paired vascular growth]
  V[Venous 1D tree] --> G
  M --> G
  G --> O[Vascularized tissue and physical archive]
```

The worked example uses **3D-IRCADb-1, case 12**. The
[IRCAD download](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)
contains organ and vessel **surface** meshes, rather than ready-made
centerlines. A preparation script extracts approximate centerlines and writes
ordinary 1D VTK meshes. If you already have suitable 1D meshes, start at
[Describe the input trees](#describe-the-input-trees).

This is a geometric growth example. The pressures, flow and extension settings
below are illustrative, not a patient calibration. The example uses hepatic
arterial supply and a retained hepatic venous tree. The published two-network run below omits portal inflow. The
{ref}`three-network extension <liver-portal-inflow>` adds that role through a common
microvascular compartment; these examples do not establish physiological liver perfusion.

```{figure} ../_static/liver-vascularization/before-after.png
:name: liver-paired-growth
:alt: Prepared arterial and venous trees beside the completed network with 1,000 tissue territories in the same liver volume
:width: 100%

Prepared input trees (left) and the verified 1,000-pair result (right), using
the same camera and spatial scale. Red is arterial and blue is venous; green
points mark arterial endpoints of the common tissue territories. The gray
envelope is the native remeshed tissue. Line widths are for display; physical
radii are stored in the output fields. Anatomical source: IRCAD case 12,
Soler and coauthors (2010).
```

## Build the application

Use deal.II with its Gmsh API, VTK and simplex support. The standalone growth
application does not require the optional `1dsolver` library. Run the commands
below from the repository root; `build-release` is the example build directory.

```bash
cmake -S . -B build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_COUPLED_PROBLEMS=OFF
cmake --build build-release --target liver_vascular -j 2
```

Set `DEAL_II_DIR` if CMake does not find your deal.II installation. Select
an MPI compiler wrapper with `-DCMAKE_CXX_COMPILER=mpicxx` when required by
your installation. A Debug build names the executable `liver_vascular_debug`.

## Prepare the IRCAD inputs

The preparation step needs Python, VTK, NumPy and VMTK. The tested environment
used Python 3.10, VMTK 1.5.0, VTK 9.2.6 and NumPy 2.2.6. For example, prepare
an isolated environment with conda or micromamba:

```bash
micromamba create -y -n ircad-prep -c conda-forge \
  python=3.10 vmtk=1.5.0 vtk=9.2.6 numpy=2.2.6
micromamba run -n ircad-prep python scripts/prepare_ircad_liver.py \
  build-release/data/ircad-case12
```

The output directory must be new. The script downloads only the case's
`MESHES_VTK.zip`, checks the three source-file SHA-256 hashes, and keeps the
original files in `raw/`. To reuse an existing archive, add
`--mesh-archive "$MESH_ARCHIVE"`, where `MESH_ARCHIVE` names your local copy.

The script writes:

| File | Purpose |
| --- | --- |
| `liver.stl` | Closed tissue surface, in metres |
| `arterial.vtk` | Rooted arterial 1D mesh with radius and growth-leaf fields |
| `venous.vtk` | Rooted venous 1D mesh with the same fields |
| `artery.stl`, `venacava.stl` | Retained vessel surfaces for inspection |
| `provenance.json` | Source hashes, units, versions and preparation measurements |

The source coordinates are in millimetres. All prepared coordinates and radii
are converted to metres. The script selects the largest liver component and
cancels two oppositely oriented copies of the same triangle that form a
zero-volume fin. It rejects any remaining open or nonmanifold liver edges.
The raw source files are unchanged.

VMTK's approximate network extraction identifies tubular paths and radii.
The script keeps the largest connected intrahepatic component of each vessel
system, checks that it is a tree, and selects its widest leaf as the root.
Disconnected components are listed in `provenance.json`; they are not added as
obstacles in this example. Inspect these choices before adapting the recipe to
another anatomy.

Two interior leaves per system are marked for extension. Dense samples near
these leaves can put several numerical intervals within one vessel diameter.
Treating each interval as a separate physical vessel then blocks a new branch
against an older interval of the same tube. The preparation script removes
only degree-two samples from the selected tails until the incoming interval
is sufficiently long. It preserves the leaf and bifurcation positions and
records the maximum geometric deviation. For the tested inputs, this removes
eight arterial samples and seven venous samples, with maximum deviations of
about 0.817 mm and 1.341 mm respectively.

The resulting input has **176 arterial and 1,084 venous line cells**. These are
the fixed inputs to the growth calculation, not a claim of exact centerline
reconstruction from the medical images.

The [IRCAD data](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)
use the [CC BY-NC-ND 4.0 licence](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Keep downloaded meshes, extracted centerlines and physical archives outside
the repository. The tutorial includes raster illustrations of the numerical
example, with source attribution. Cite Luc Soler and coauthors, *3D image reconstruction for
comparison of algorithm database: A patient specific anatomical and medical
image database*, IRCAD technical report, 2010.

(describe-the-input-trees)=
## Describe the input trees

Each input is a legacy VTK line mesh with three scalar point-data arrays:

| Array | Meaning |
| --- | --- |
| `Radius` | Positive vessel radius in metres |
| `Root` | One nonzero leaf: the arterial inlet or venous outlet |
| `Grow` | Nonzero on the non-root leaves to extend |

The mesh must be connected and acyclic. Line-cell direction need not already
follow the flow: the application orients edges from the root toward tissue.
Venous geometric orientation is also root-to-tissue; its physical flow is in
the opposite direction. Radii interpolate between the two endpoints of each
line cell. Viscosity comes from the parameter file.

The application assigns physical node ids from the arterial point indices,
then the venous point indices offset by the number of arterial points. These
are physical graph identities, not solver block numbers. For the supplied
prepared case, the arterial root is node **83** and the venous root is node
**177**. The reference pressure entries below refer to these roots. If you
change the input meshes, update the pressure keys to match their `Root` fields.

## Mesh the tissue and grow both systems

CMake generates the runnable file
`build-release/data/vascularization/liver.prm` from the checked-in
template. Its central settings are shown below. Paths in this example are
relative to the parameter file:

```text
subsection Liver
  set Arterial grid = ../ircad-case12/arterial.vtk
  set Venous grid = ../ircad-case12/venous.vtk
  set Output directory = ../../test_output/liver-vascularization
  subsection Domain
    set Grid generator = ../ircad-case12/liver.stl
    set Triangulation type = fullydistributed
    subsection Surface mesh
      set Input = auto
      set Target edge length = .012
      set Feature angle = 60
      set Volume cells = tetrahedra
    end
  end
  subsection Vascularization
    subsection Growth
      set Seed length = .003
      set Minimum vessel length = .0002
      set Candidate vessels = 16
      set Local optimization steps = 5
    end
    subsection Sampling
      set Terminal count = 1000
      set Random seed = 23
    end
    subsection Physiology
      set Dynamic viscosity = .004
      set Total new-tissue reference flow = 2e-9
      set Reference pressures = 83:12000,177:2000
      set Minimum radius = .00005
    end
    subsection Venous network
      set Enabled = true
      set Terminal separation = .003
    end
  end
end
```

Use the generated file for the complete tested controls, including the
candidate, optimization and geometry budgets. All lengths here are in metres,
pressures in pascals and flow in cubic metres per second.

`Domain<3,3>` first distinguishes an existing volume from a surface. A surface
is read into `Triangulation<2,3>`; reading it into `Triangulation<3,3>` does not
create a volume. The separate meshing step classifies the surface into Gmsh
patches, parametrizes them, creates a closed surface loop and generates the
volume. It then transfers the tetrahedra to deal.II before partitioning.
See [surface domains](../how-to/surface-domains) for the supported input readers
and every surface-meshing parameter.

For this surface, a feature angle of 60 degrees allows Gmsh to build valid
patch parametrizations; 40 degrees failed in the tested installation. With a
12 mm target edge length the tested mesh had **13,351 tetrahedra** and a volume
of about **1.515 litres**. The cleaned surface encloses about 1.530 litres, so
the coarse remeshing changes the enclosed volume by about 0.94%.

```{figure} ../_static/liver-vascularization/volume-section.png
:name: liver-volume-section
:alt: Plane section through the interior of the native liver volume, showing the intersections with tetrahedral cells and axes in millimetres
:width: 85%

Section at y = 163.2 mm through the exported volume of 13,351 tetrahedra.
Polygon edges follow the intersections of the plane with the volume cells.
The plot uses the actual native mesh and converts coordinates to millimetres
for display.
```

The arterial generator starts at the marked input leaves and assigns new
common tissue territories. The venous generator subsequently extends the
marked venous leaves to these territories, with distinct endpoints separated
by 3 mm. Each accepted pair receives a positive microvascular resistance.
Both growth stages check their new vessel envelopes against the tissue
boundary and both vessel systems. Input geometry remains fixed.

The separation must accommodate the endpoint radii. The initial 1 mm trial
failed venous startup on this case; 3 mm passed the same geometry checks.
Increasing a finite search budget does not make a geometrically blocked
connection admissible.

The recipe requests 1,000 common tissue territories, each with one new
arterial and one new venous terminal. For a short trial, set `Terminal count`
to `32` and pass `--pairs 32` to the checker below.

Run the full recipe in Release on 32 MPI processes, as in the verified
1,000-pair calculation:

```bash
mpirun -np 32 build-release/liver_vascular \
  build-release/data/vascularization/liver.prm
```

For a short two-process trial, set `Terminal count = 32` and choose a new
output directory in a copy of the parameter file, then run:

```bash
mpirun -np 2 build-release/liver_vascular \
  build-release/data/vascularization/liver-mpi.prm
```

The application resolves relative paths against its parameter file. It can be
launched from an unrelated directory. Each run requires a new output directory;
an unsuccessful result is not archived as an accepted vascular network.

For hexahedra, change `Volume cells` to `hexahedra`. Gmsh still generates the
tetrahedral volume first, then Gmsh subdivides each tetrahedron into four
conforming hexahedra. The tested liver consequently had **53,404 hexahedra**
with the same enclosed volume. This is an unstructured hexahedral mesh, not a
structured organ-aligned grid. Regenerate and verify the vascular result on
the selected background; changing its cells can change sampled growth paths.

## Inspect and verify the result

Open `liver_0.pvtu` and `vessels_0.pvtu` in ParaView. Give the liver a low opacity
and colour the vessel field `System`: zero is arterial and one is venous.
`Generated` distinguishes fixed imported segments (zero) from new segments
(one). `Radius` is the radius at the numerical interval midpoint, repeated at its
patch endpoints. Use it as the point scalar for a varying-radius Tube filter.

The output also contains:

- `network/`: a native distributed physical archive, including territories and
  terminal pairs;
- `nodes-*.csv` and `vessels-*.csv`: each owned physical record once, with stable
  integer identifiers, coordinates, radii and provenance;
- `terminals-*.csv` and `pairs-*.csv`: common territories, demand and bed
  resistance;
- `summary.csv`: global background, vessel and pair counts and vessel volume
  fraction.

If growth returns an invalid result, the application still writes
`invalid-result/arterial.pvtu`, `portal.pvtu` and `venous.pvtu`,
their rank-local VTU pieces, and `liver_0.pvtu` before exiting with failure.
Open the three PVTU files directly in ParaView. These diagnostic files preserve
crossings, coincident endpoints, collapsed edges and the original radius at
each vessel endpoint. Separate VTK points retain distinct radius traces at a
shared physical node; `PhysicalNodeId` records its identity.

`DiagnosticOnly=1` and `ValidationFailed=1` mark the vessel cells.
`diagnostic.json` declares that this is not an adoptable network archive;
no `network/` archive is produced for the rejected result. A role without
vessels has an empty dataset. Edges with missing or nonfinite endpoint
coordinates, or an unknown vascular system, cannot be drawn. The files
`omitted-*.csv` list their physical IDs, and `diagnostic.json` gives the total
omitted count. This output does not repair
the network or change the validation result.

Check the physical result independently of the generator:

```bash
micromamba run -n ircad-prep python scripts/check_liver_vascular.py \
  build-release/data/ircad-case12 \
  build-release/test_output/liver-vascularization \
  --pairs 1000 --separation .003 \
  --report build-release/test_output/liver-vascularization/verification.json
```

The checker compares all supplied nodes, ports, edges and endpoint radii with
the exported physical records. It requires growth in both systems, unique
arterial–venous pairings, equal paired demand, positive bed resistances and the
requested separation. It also samples each new vessel against the exported
volume. This sample check supplements the generator's continuous geometry
certification; it does not replace it.

The verified 32-rank Release run produced **1,000 tissue pairs and 5,256
physical vessels**: the 1,260 fixed input segments plus 1,998 new arterial and
1,998 new venous segments. The independent check preserved all input nodes,
ports, edges and radii and found a positive sampled wall margin for every new
vessel. Numerical subdivision of these physical vessels is a separate step
and can produce more 1D cells.

The dense recipe uses 16 candidate vessels per insertion; eight candidates
exhausted the finite sampling budget on this case. MPI partitions change the
sampled growth paths, so another process count can need different search
controls. The smaller 32-pair case was also verified on two and four ranks.

## Monitor growth and retain the latest diagnostic state

For a fresh generation, enable monitoring in the same parameter file:

```text
subsection Liver
  subsection Growth monitoring
    set Progress interval = 60
    set Checkpoint interval = 900
    set Maximum generation wall time = 18000
  end
end
```

All three values are seconds and default to zero, which disables that option.
These parameters require the application built with the monitoring API; the
frozen `c88` campaign runtime does not accept them. Monitoring is not supported
with `Companion checkpoint archive` continuation.

`progress.csv` records elapsed generation time, active system, requested and
accepted terminal counts, and completed batch count. Counts describe the active
native forest, not the combined three-role network. Records are flushed on a
role change and at the requested interval. The first safe point follows
bootstrap; later safe points follow completed accepted or rejected batches.
There is no progress notification from the middle of a batch.

`latest-checkpoint/` contains `arterial.pvtu`, `portal.pvtu`, `venous.pvtu`,
their rank-local pieces, a native `network/` archive, and `checkpoint.json`.
The files contain the last accepted state, never a staged candidate. Until a
companion role starts growing,
its VTK contains its original imported network. The manifest distinguishes
`original_input` from `accepted_growth` and reports each role's accepted growth
count separately. Imported physical IDs, coordinates and endpoint radii are
preserved. Generated hydraulic fields are not final.

The VTK files remain diagnostic, incomplete, non-adoptable and non-resumable,
including a snapshot taken just before successful generation finishes. Their
lengths and geometry do not certify admissibility or an inf-sup bound.
The separate native `network/` archive preserves the exact accepted Node,
Vessel and Terminal records, including IDs, territory labels, integrated demands,
endpoint radii, port normals and extra physical properties. It contains no
completed compartments. Unlike the VTK view, the native archive retains
unstarted roles exactly as represented by the generator.

The top-level diagnostic flags describe the VTK exports. The manifest's
`native_checkpoint` object names the separate native archive and reports
whether the current phase supports companion continuation. Arterial growth
cannot yet be resumed. Once arterial growth finishes, portal growth can be
continued; venous continuation also requires completed portal growth when
that role is enabled. Keep the original inputs and generation parameters and
set `Companion checkpoint archive` to `latest-checkpoint/network` and
`Companion continuation phase` to the reported phase. The native continuation
validator still checks the physical archive and original attachments before
accepting it. This starts a new search epoch, preserving accepted physical
records rather than the previous random generator or candidate history.
Neither the native checkpoint nor its diagnostic exports are final coupled
simulation input. The application produces the ordinary final archive only after all
required generation and validation work succeeds.

Publication writes a temporary `checkpoint-staging/`, then replaces
`latest-checkpoint/` and removes the previous snapshot. Storage therefore holds
one completed latest snapshot in steady state and at most two full snapshots
during replacement. A hard kill can leave staging or previous output; preserve
those files for inspection. A subsequent invocation refuses to overwrite them.
The rename sequence does not guarantee that a reader sees a `latest-checkpoint`
directory at every instant.

The maximum generation wall time requests a cooperative stop. It is checked at
the safe points above, then writes a checkpoint and finishes reconstructing the
last accepted state. The application saves the three `invalid-result/*.pvtu`
files and background mesh, creates no final top-level `network/`
archive, and returns exit status 2. A batch, bootstrap, checkpoint write or final
reconstruction can
extend past the requested time. Leave a measured allowance before any external
hard timeout; this parameter is not a hard wall-time bound. Observer or output
errors return failure and retain diagnostic files already written.

Applications using the library can provide the same optional controls:

```cpp
using namespace ImmersX::Vascularization;
GenerationControl control;
control.poll = [](const GenerationProgress &progress) {
  return GenerationAction{/*checkpoint=*/false,
                          /*stop=*/progress.accepted_terminals >= 100};
};
auto result = generate(background, mapping, initial, parameters, comm, control);
```

Supply or omit each callback consistently on every rank. `poll` runs
synchronously and must not perform collectives. Requests from all ranks are
combined: any rank may request a checkpoint or stop. An optional `checkpoint`
callback receives an owned, materialized `NetworkData` by value on every rank;
its final Boolean argument marks a stop. That callback may perform collective
output if all ranks follow the same sequence. It must finish before returning
and must label the snapshot as incomplete diagnostic data. A stop returns
`controlled_stop=true` with invalid validation and prevents further calibration
trials. Empty controls preserve the ordinary generation path without extra
collectives or accepted-state materializations. Materializing a requested
snapshot traverses the current network and adds output cost.

## Reproduce the illustrations

The published figures come from the checked 32-rank run. Their
[manifest](../_static/liver-vascularization/manifest.json) records the simulation
source revision, input and output SHA-256 hashes, software versions, camera,
section plane and independent verification results. The anatomical source is
3D-IRCADb-1 case 12, credited to Soler and coauthors (2010) above.

Install Matplotlib in the preparation environment, then render the same views
from your own accepted result:

```bash
micromamba run -n ircad-prep python -m pip install matplotlib==3.10.9
micromamba run -n ircad-prep python scripts/render_liver_figures.py \
  build-release/data/ircad-case12 \
  build-release/test_output/liver-vascularization \
  build-release/liver-figures --pairs 1000 \
  --source-revision "$(git rev-parse HEAD)"
```

Pass the revision used to build the simulation binary. The renderer reruns
the physical-data checks before plotting and writes two PNGs, a verification
report and a manifest. It merges exactly coincident DataOut patch vertices
before extracting the tissue exterior. The section preserves quadrilateral
cuts of tetrahedra without introducing display diagonals.

A successful growth calculation demonstrates a paired geometric network on
this tissue mesh. It does not establish physiological accuracy, resolve
capillaries, or guarantee that every point of the liver is adequately served.
Use the existing [coverage and calibration controls](../how-to/vascular-generation)
when imposing a service distance or a volume objective.

(liver-unchecked-growth)=
## Defer geometric admissibility during three-network growth

To generate arterial, portal and hepatic venous forests before geometric
repair, disable the optional CCO admissibility gates:

```text
subsection Liver
  subsection Vascularization
    subsection Growth
      set Enforce geometry = false
    end
  end
end
```

The default is `true`. The disabled gates include wall and vessel clearance,
junction-angle heuristics, and configured lower bounds on scaled vessel
length and radius. Positive finite physical lengths, radii, viscosity and
resistance, graph ownership and topology, Murray consistency and the native
reference-flow checks remain required. Tissue sampling still uses the liver
volume. This option does not repair geometry or certify nonintersection.

Use the anatomical three-tree inputs and remap the reference-pressure keys
to their actual imported roots. The earlier coupled release contains only
three feeder stubs; reusing that archive does not generate a whole-organ
vasculature. Keep every new physical archive with the exact input trees,
root identifiers and parameters that produced it.

The restarted scale campaign grows a separate arterial, portal and hepatic
venous family for each background. Its initial common-terminal target is one
per global background cell and per system; a completed family is not reused
on the other campaign backgrounds. This is a global count target, not a
claim that every cell contains exactly one terminal. Coverage remains a
separate objective:
the maximum service distance is `2` with `Length scale = local mesh size`.
The detailed settings and the distinction between total count and per-cell
occupancy are described below.

A successful `liver_vascular` run writes three physical inspection datasets:

```text
physical-vtk/arterial.pvtu
physical-vtk/portal.pvtu
physical-vtk/venous.pvtu
```

Open each PVTU with its referenced VTU pieces present. These files preserve
physical vessel endpoint traces; they accompany the accepted `network/`
archive. On failure, the same three roles are written under
`invalid-result/` and must not be treated as an accepted archive. Before
generation starts, `initial-input/` saves the three imported networks.
If generation throws or a run stops before returning a network, these input
snapshots remain available; they contain no intermediate grown state.

The new profile targets are approximately 100k, 1M and 10M total coupled DoFs.
Count displacement, velocity, native flow and multiplier blocks after
rediscretizing all three networks on each final background. Historical scale
labels from the 14-vessel archive are not counts for the new forests. The
[tracking issue](https://github.com/luca-heltai/immersx/issues/343) and
[implementation PR](https://github.com/luca-heltai/immersx/pull/375) distinguish
prepared backgrounds, successful hydraulic generation and converged coupled
results.

The [new forest-family release](https://github.com/luca-heltai/immersx/releases/tag/liver-h-scaled-2026-11-10)
contains successful small and medium native generations, with 2316 and 13351
background tetrahedra and the same number of newly generated terminals **per
vascular role**, respectively. Each family has its own three forests; the
medium case does not reuse the small graph or the historical 2048-territory
archive. Independent rigid hydraulic audits passed for both. The continuous
`2H` coverage proof reported for the small family applies to its own
2316-cell generation background; it is not a medium-family coverage result.

### Rigid flow distribution for the new families

The small and medium archives also contain solved rigid-reference flows.
For each physical vessel, the plotted speed is
$|Q_e|/(\pi\min(r_{e,0},r_{e,1})^2)$: the largest endpoint speed for its
linear radius taper. These are physical-vessel statistics, rather than
statistics over the rediscretized numerical intervals. The median values are:

| Family | Arterial (m/s) | Portal (m/s) | Venous (m/s) |
| --- | ---: | ---: | ---: |
| Small, 2316 terminals per role | 0.04073 | 0.02063 | 0.01576 |
| Medium, 13351 terminals per role | 0.03173 | 0.01611 | 0.01247 |

Every recorded speed is below 1 m/s in these two rigid references. Small
terminal radii therefore do not imply large velocities when their solved
local flows are also small. This observation is not a physiological velocity
criterion or a validation of the coupled model.

```{figure} ../_static/liver-icar/new-family-perfusion/speed-cdf.svg
:alt: Count-weighted cumulative speed distributions for the three physical vascular roles in the small and medium rigid references.

Each physical vessel has equal weight. Both panels use the same logarithmic
speed scale. No transient or coupled flow solution is included.
```

To separate unequal territory sizes from unequal perfusion, define

$$
\eta_{i,s} =
\frac{q_{i,s}/\sum_j q_{j,s}}{d_{i,s}/\sum_j d_{j,s}},
$$

where $s$ denotes the vascular role, $q_{i,s}$ is its solved terminal flow and
$d_{i,s}>0$ is the recorded territory demand. The demand comes from the model;
it is not a measured perfusion field. A value of one means that the territory
receives the same fraction of role flow as its fraction of recorded demand.
The diagnostic does not impose that value on the resistance-driven solution.

```{figure} ../_static/liver-icar/new-family-perfusion/perfusion-cdf.svg
:alt: Territory-demand-weighted cumulative distributions of terminal flow share divided by recorded demand share, for all three roles and both new families.

Each terminal is weighted by its fraction of recorded role demand. The
vertical line marks proportional supply. This weighting differs from the
equal-vessel weighting in the speed plot.
```

The CSV flows were checked against the reference writer and independently
against $p_0-p_1=R_eQ_e$ on every vessel; the maximum absolute discrepancy was
below $1.2\times10^{-12}$ Pa. These checks describe the rigid network and its
specified supply totals. They do not establish patient calibration, stable
1D–3D coupling, or a converged transient cycle. In particular, the coupled
portal-pressure reservoir described below is a different boundary closure.

The [CDF records](../_static/liver-icar/new-family-perfusion/cdf.csv),
[numerical summary](../_static/liver-icar/new-family-perfusion/summary.json)
and [provenance](../_static/liver-icar/new-family-perfusion/provenance.json)
retain definitions, archive checksums and hashes of the source CSV partitions.
The bundled CDF samples use 1001 common evaluation points for each role,
family and metric. Regenerate both figures from those records without running
a solver or downloading the anatomical grids:

```bash
python3 -m pip install -r doc/_static/liver-icar/new-family-perfusion/requirements.txt
python3 doc/_static/liver-icar/new-family-perfusion/plot.py \
  --output-dir /tmp/liver-reference-figures
```

### Measured setup for the new forest families

Earlier complete `Setup only = true` runs on 11 November 2026 measured the
following spaces. Both use P2 solid displacement and velocity, degree-one
native flow, and the automatic degree-one multiplier space. No state
initialization, IDA correction or time integration was executed.

| Family | Background tetrahedra | Terminals per role | Numerical 1D intervals | Displacement | Velocity | Native flow | Multiplier | Total DoFs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 2316 | 2316 | 14787 | 12984 | 12984 | 118296 | 29574 | 173838 |
| Medium | 13351 | 13351 | 83026 | 66336 | 66336 | 664208 | 166052 | 962932 |

Background counts come from the complete archived tetrahedral meshes, not
rank-local active-plus-ghost counts. The global 1D counts sum locally owned
cells across MPI ranks and agree with unique `(vessel, begin, end)` records
in all `mesh_scale-*.csv` partitions. The four block sizes sum to each reported
total. Setup completed in 75.24 s with 8 MPI ranks and 267.51 s with 16 ranks.
These timings measure setup, not solver convergence or integration throughput.

```{figure} ../_static/liver-icar/new-family-setups/block-composition.svg
:alt: Four measured coupled DoF blocks for separately generated small and medium liver forests.

Measured block composition with degree-one native flow and multiplier. These
are the new 2316- and 13351-terminal-per-role families, separate from the
historical fixed-network measurements below.
```

Later runs selected degree-two native flow and the automatic degree-two
multiplier, retaining P2 solid fields, the same physical archives and
backgrounds, and disabled co-refinement. The measured DG2 blocks are now:

| DG2 family | Native intervals | Displacement | Velocity | Native flow | Multiplier | Total DoFs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 14787 | 12984 | 12984 | 147870 | 44361 | 218199 |
| Medium | 83026 | 66336 | 66336 | 830260 | 249078 | 1212010 |

The small setup completed on 8 MPI ranks. An earlier 8-rank medium setup
reached its 900 s limit before reporting the final blocks; at that historical
snapshot, 1212010 was only the formula forecast
$132672+13\times83026$. The later 32-rank medium transient attempt measured
all four blocks above. Passing setup or initial correction is distinct from
completing the requested transient; its failure is recorded below. The
37897-DoF diagnostic remains a separate network.

The lower configured interval bound is $L\ge0.5H_{\max}$ and the upper bound
is $L\le2H_{\min}$. Here $H$ is the **diameter** of a background cell,
not its volume to the power $1/3$. The minimum and maximum are taken over
native overlap cells. The native short flag tests
$L<0.5H_{\max}(1-10^{-12})$.

| Family | Short / all intervals | Short / intervals with overlap | No-overlap intervals | Intervals with an exterior vertex | Median $L/H_{\max}$ over overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 13286 / 14787 (89.85%) | 13286 / 13388 (99.24%) | 1399 | 1389 | 0.11507 |
| Medium | 77675 / 83026 (93.56%) | 77675 / 78376 (99.11%) | 4650 | 4626 | 0.12902 |

The no-overlap and exterior-vertex counters describe different tests and must
not be substituted for one another. No-overlap records have zero recorded
$H$ and are excluded from the ratios. A positive overlap flag does not prove
that the entire interval or its lifted wall lies inside the tissue.

```{figure} ../_static/liver-icar/new-family-setups/interval-distribution.svg
:alt: Logarithmic histograms of interval length divided by maximum overlapping tissue cell diameter, with the lower bound at 0.5.

Each curve contains only intervals with native overlap and sums to 100% over
its logarithmic bins. The shaded region is below the configured lower bound.
The percentages among all intervals and among overlap intervals have different
denominators, as shown in the table.
```

Most numerical intervals therefore do not meet the configured lower
interval bound. The two overlap median ratios are comparable relative to the
background diameter, but neither these ratios nor successful setup establish
an inf-sup bound. No such bound or constant was measured. The short coupled
transients below are separate observations. At this measurement snapshot,
the separate 111250-terminal-per-role large generation was still running; no completed large generation or measured
coupled DoF count is included here.

The historical degree-one setup executable is an immutable runtime built from
`c88c5fd8` with the shared immutable WeakTerm/target-snapshot header change later integrated in
[PR #380](https://github.com/luca-heltai/immersx/pull/380), plus setup trace
markers. Its executable hash is
`b7c998cf7d551f638c554cc534550d4a8e44f50a613e3eb4fd75d2bae446d5f3`;
the shared library hash and patch hashes are in the linked provenance.
These measurements do not assert binary or numerical equivalence with current
`master`. Those historical setup parameters retain an arterial flow input
and a portal pressure reservoir at 1133.2402930275 Pa, whereas the rigid hydraulic reference
prescribes both supply totals. The boundary models differ.

Download the [small native family](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/liver-h-scaled-small-2316.tar.gz)
and [medium native family](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/liver-h-scaled-medium-13351.tar.gz)
for the archived networks, original radii, parameters and hydraulic evidence.
The medium archive SHA-256 is
`adaee8358d001663aa68a178e2dc046566fb190b654f70f7502752b7fcdfbefe`.
Original generation parameter files retain cluster-specific absolute paths;
relocate input, include and output paths before using a copy. The archived
setup parameters also belong to the frozen runtime. Current checked-in
examples use `Thickness` and `Cross section`; do not assume archived
`Constant thickness` or other older parameter names remain accepted.
The portable short-transient package below supplies current parameter names
and a separately verified public-source small replay; it does not relabel
these older setup measurements.

The [count CSV](../_static/liver-icar/new-family-setups/block-counts.csv),
[histogram CSV](../_static/liver-icar/new-family-setups/interval-histograms.csv)
and [provenance](../_static/liver-icar/new-family-setups/provenance.json)
retain the plotted numbers, native partition hashes and source measurements.
Regenerate both figures without running a solver, from the repository root:

```bash
python3 -m pip install -r doc/_static/liver-icar/new-family-setups/requirements.txt
python3 doc/_static/liver-icar/new-family-setups/plot.py \
  --output-dir /tmp/liver-setup-figures
```

The plotting script also accepts `--raw-runs SMALL_SETUP_DIR MEDIUM_SETUP_DIR`
to rebuild its count and histogram CSVs from complete native setup records;
see its module documentation. The histograms retain all overlap intervals in
80 logarithmic bins and do not clip tails. This evidence follows
[issue #343](https://github.com/luca-heltai/immersx/issues/343) and
[issue #378](https://github.com/luca-heltai/immersx/issues/378); neither issue
is closed by these measurements.

### Coupled initialization diagnostics

The setup counts above measure spaces. The initialization study uses a
separate diagnostic network with 64 common terminal territories per role and
2776 background tetrahedra.
It is not the newly generated small family. With degree-two native area and
velocity and a degree-two multiplier, it has 37897 total coupled DoFs.

Degree two matters for tapered vessels: a linearly interpolated reference
radius has a quadratic reference area. The measured plain reference rest has
zero solid displacement, velocity and wall multiplier, zero native velocities,
and cell and trace areas equal to their reference values. Its wall-constraint
residual was checked before starting the continuation. A trace-corrected
initialization and a rigid hydraulic seed are different starting states.

These earlier controlled diagnostics did not produce an accepted timestep:

| Diagnostic | Verified result | Limitation |
| --- | --- | --- |
| Rigid hydraulic and reaction-pressure seeds | Native fields and wall-pressure projection can be inspected before IDA | Standard static IC failed |
| Portal reservoir fitted to the aggregate rigid-flow target | The rigid graph reproduces the requested portal inflow proxy | Changing this boundary alone did not resolve static IC |
| State-transmission `HLL_HDG` flux | A directional coupled-Jacobian check passes | Standard static IC failed |
| Normal-corrected HLL | Independent normal-reversal tests and a directional coupled-Jacobian check pass | Standard static IC failed |
| Affine continuation from the measured DG2 reference rest | Several artificial-load stages pass fresh static-correction checks | The bounded run stopped before reaching the original equations |

The HLL orientation defect and its independent tests are described in
[MetricFlowX PR #92](https://github.com/luca-heltai/metric-flow-x/pull/92).
Normal-orientation tests and directional Jacobian checks address specific
numerical contracts. The successful run below additionally changes interior
transmission; the earlier checks alone did not establish its startup behavior.

The continuation uses fixed physical time, zero state derivative and

$$
F_s(y) = F(y) - (1-s) F(y_{\mathrm{rest}}), \qquad 0 \leq s \leq 1.
$$

Intermediate stages contain artificial forcing. Only a fresh check of the
original residual at $s=1$, followed by the original IDA initialization and
time integration, can produce an accepted simulation. A reduced scaled
residual or a small regularized pseudo-step is not used as that certificate.
Directional finite differences check selected perturbations; they do not
prove a Jacobian is correct at every state.

The portal target used here is a sum of splanchnic arterial terminal flows
from the MOX input. It is a proxy, not a resolved portal-vein measurement.
The resistance-driven rigid reference and the coupled pressure-reservoir
boundary must therefore be checked separately. Neither is patient calibration.

Diagnostic VTK files distinguish the initialized seed, the last certified
artificial stage, the last attempted stage and the restored last-valid
candidate. They retain unaccepted-state labels. An initialized seed exported
after an IDA failure is not IDA's last internal Newton iterate. Keep each VTM
beside its three role-specific VTU files when opening it in ParaView.

[Tracking issue #378](https://github.com/luca-heltai/immersx/issues/378)
records the experiment outcomes. The return-code interpretation follows the
[SUNDIALS 6.7 IDA documentation](https://sundials.readthedocs.io/en/v6.7.0/ida/Usage/index.html):
`IDA_LINESEARCH_FAIL` reports failure of the initial-condition line search;
it does not establish that a physical equilibrium does not exist.

### Accepted 50 ms coupled diagnostic

A later controlled run selected `Interior transmission = flux` with
`HLL_HDG`, using the native implementation in
[MetricFlowX PR #93](https://github.com/luca-heltai/metric-flow-x/pull/93).
This option transmits physical flow and total head across ordinary interior
faces; `state` remains the native default. Rebuild both the native package and
its C++ consumers when updating this SDK, because its class header changed.
The public selection is:

```text
subsection MetricFlowSystem<1, 3>
  set Interior transmission = flux
end
```

In this recorded diagnostic, all 11 affine stages converged. At $s=1$, the
artificial forcing was zero and a fresh original-equation static check gave a
correction WRMS of $1.0102081\times10^{-6}$. Standard IDA initial-condition
correction then passed, followed by ten accepted steps from 7.2 to 7.25 s.
No error-test or nonlinear-convergence failures were reported. This result
covers the 37897-DoF, 64-territory diagnostic; cardiac cycles and integration
on the separately grown small, medium and large forests remain separate
validation tasks.

```{figure} ../_static/liver-icar/diagnostic64-startup/continuation-correction.svg
:alt: Recorded stage and original-equation correction WRMS values at eleven affine loads, meeting at the successful original-equation stage s equal to one.

Reported Newton-correction norms at the end of each converged stage. The
blue curve concerns the artificially forced $F_s$ equations; the orange curve
checks the original equations. Both use the tenfold startup relaxation of the
base relative and componentwise absolute tolerances. A small artificial-stage
correction does not certify the original equations before
$s=1$. WRMS is a solver norm, not a physical solution-error estimate.
```

```{figure} ../_static/liver-icar/diagnostic64-startup/accepted-ida-counts.svg
:alt: Actual cumulative IDA step, Newton, Jacobian and linear-solve counters at six accepted output times over fifty milliseconds.

Actual IDA counters at the six saved output times; lines join those records.
The final counters are 10 steps, 24 Newton iterations, 7 Jacobian setups and
26 linear solves, excluding the separate affine continuation. The reported
BDF order at positive output times was one, with maximum order configured as
two. The trace comes from the output-statistics log records; the configured
`ida_steps.csv` in this run contains only its header.
```

A separate follow-up with ordinary startup and no affine continuation also
reached 7.25 s in ten accepted steps, with zero reported error-test or
nonlinear-convergence failures. Its cumulative counters were 32 Newton
iterations, 7 Jacobian setups and 34 linear solves, including initial
correction work. No independent original-equation static certificate was
recorded for that follow-up; the preceding certificate belongs to the affine
run.

The [continuation records](../_static/liver-icar/diagnostic64-startup/continuation.csv),
[IDA output records](../_static/liver-icar/diagnostic64-startup/ida-output.csv)
and [summary and provenance](../_static/liver-icar/diagnostic64-startup/summary.json)
contain scalar solver data and source hashes. Reproduce both figures without
running the solver or loading anatomical data:

```bash
python3 -m pip install -r doc/_static/liver-icar/diagnostic64-startup/requirements.txt
python3 doc/_static/liver-icar/diagnostic64-startup/plot.py \
  --output-dir /tmp/liver-startup-figures
```

If you have the separately supplied diagnostic output bundle, open the
following relative paths in ParaView. Keep each PVD, PVTU or VTM beside all its
referenced files; copying only an index file omits its data.

| Entry point | Contents and scope |
| --- | --- |
| `accepted-trajectory/flow/flow.pvd` | All three networks at six accepted times from 7.2 to 7.25 s; each PVTU references four rank pieces |
| `accepted-trajectory/solid/solid.pvd` | Solid displacement and velocity at the same six accepted times |
| `accepted-trajectory/flow/pressure_jump.pvd` | Accepted wall-pressure output |
| `accepted-final-roles-t7.25/accepted-all-roles-t7.25.vtm` | Separate arterial, venous and portal VTUs from the actual accepted final native output |
| `affine-initial/vtk/candidate-all-roles.vtm` | Unaccepted reference-rest seed at 7.2 s |
| `affine-last-converged-stage/vtk/candidate-all-roles.vtm` | Certified $s=1$ static candidate at 7.2 s, before IDA integration |

The final role VTUs retain every original native point-data array, coordinate
and cell connection. Their roles follow the exact native-material to physical
vessel-ID transfer, checked against all 541 cell identities and per-vessel
cell counts; no geometric matching was used. Native material IDs and physical
vessel IDs are distinct. Added role, time and source-index metadata identify
the extraction. The original native patch sampling and precision are retained.
For radius-aware display, see [Variable Radius Tube](../how-to/paraview-area-tube.md).

The last-valid, last-converged and last-attempted static snapshots coincide in
this successful run. Each retains conservative unaccepted labels and a
separate original-static-certificate flag. These static candidates precede
IDA and must be kept distinct from the accepted trajectory and final-time
role VTUs. The scalar plotting records are included here; the anatomical and
VTK result files remain outside the repository.

### Public short transient on the independently grown small family

A controlled replay using public ImmersX
[`637c9473`](https://github.com/luca-heltai/immersx/tree/637c947313c3a1ebcc6681396bb26fd30a7430bd)
and MetricFlowX
[`67ac3925`](https://github.com/luca-heltai/metric-flow-x/tree/67ac392251baac522955cd0cdb2cbd44ffec4375)
completed standard initialization and 7.2–7.25 s integration on the actual
small family: 13968 physical vessels, 14787 native intervals and 218199 coupled
DoFs. The application and its own core were rebuilt coherently against that
native SDK. Thirty-two focused Release/Debug one- and two-rank numerical
checks passed before the eight-rank replay.

The final IDA counters were 10 accepted steps, 29 Newton iterations, 7 Jacobian
setups and 31 linear solves, with zero reported error-test or nonlinear
convergence failures. All six saved IDA counter rows matched the frozen
reference. Sampled output coordinates and connectivity matched exactly;
recorded field differences were small but not all arrays were bit-identical.
This comparison does not certify every FE or matrix coefficient. No independent
original-equation static certificate was recorded for this case with ordinary
initialization. The certificate and figures above belong to the separate
affine diagnostic.

The medium family now has a measured total of 1212010 coupled DoFs. Its earlier
frozen-runtime trial passed initial correction and saved accepted outputs at
7.2 and 7.21 s, then failed the retained-Jacobian MUMPS backsolve residual check.
The requested 7.25 s endpoint was not reached; a later internal state is not
an accepted output. At this report snapshot the public-runtime retry was
still running, so no successful public medium trajectory is claimed.
The large family remains unavailable as a completed three-role archive.

Download the
[portable family input package](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/liver-independent-family-coupled-inputs-public-small50ms-20261111-v2.tar.gz)
and its
[checksum file](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/SHA256SUMS-coupled-inputs-public-small50ms-20261111-v2).
The archive SHA-256 is
`c710ae01e2e6424292ffa97db843d8377d7db78de020ce1fd0f477d1eab9e76d`.
Its README links the existing family archives, original waveform dependency,
public source revisions and coupled build instructions. It contains parameters,
a path-rendering helper and aggregate verification metadata; it does not
redistribute anatomical inputs, raw field tables or the waveform.

For current ImmersX, remove `set Number of refinement cycles = 1` from
only the `Elastodynamics` subsection in `profiles/small-common.prm.in` and
`profiles/medium-common.prm.in` before running the helper. This solid parameter
was removed in PR #406. Keep the identically named entry under
`MetricFlowSystem<1, 3>`, which remains valid. The archive and checksum above
refer to the original, unmodified download.

After verifying and extracting it, prepare paths without starting a solver:

```bash
python3 liver-independent-family-coupled-inputs/prepare_profile.py \
  --profile small \
  --assets-dir /absolute/data/liver-h-scaled-small-2316 \
  --inflow-csv /absolute/data/inlet_ha.csv \
  --build-dir /absolute/immersx/build \
  --output /absolute/new/small-50ms
```

The helper verifies the tissue mesh and every native archive partition, rejects
an existing output directory and prints the MPI command. It preserves physical
and solver settings while rendering input, include and output paths. Run the
printed command on an allocated compute node. The medium profile remains a
recorded diagnostic input; selecting it does not imply that its pending replay
has passed. The large profile is rejected until a complete archive exists.
These profiles use the original HA series with period zero over 50 ms; they
are not repeated-cycle inputs.

The transient pressure reservoirs differ from the rigid reference's prescribed
supply totals. Portal reservoir proposals are 1732.08 Pa (12.99 mmHg) for the
small family and 1817.18 Pa (13.63 mmHg) for the medium family, compared with
the 8.5 mmHg baseline. Role-wise wall-reference pressures remain unchanged.
These values are rigid-model mean-flow proposals, not a calibration of healthy
portal pressure or of the compliant periodic response. A difference between
model reservoirs is not the clinical hepatic venous pressure gradient (HVPG),
which uses wedged and free hepatic-vein measurements; see
[Baveno VII, §1.7](https://doi.org/10.1016/j.jhep.2021.12.022).

### Ten-cycle diagnostic and remaining solid drift

A separate 64-territory, 37897-DoF diagnostic completed ten forcing cycles
from 7.2 to 15.2 s with period 0.8 s. Its final two cycles used the tight field
tolerances. Early relative and absolute tolerance factors were 100; the factors
held through cycle seven and decreased during cycle eight. This differs from
the tenfold relaxation of the earlier affine-correction plot. The run completed
1545 accepted steps with ten recovered error-test failures and zero nonlinear
convergence failures. It used a read-only output observer; it is not the public
small-family replay above.

For repeated forcing this diagnostic used an explicitly artificial continuous
periodic closure of the original HA samples. With $T=t_1-t_0$ and
$\Delta=q(t_1)-q(t_0)$, it used

$$
q_{\mathrm{closed}}(t)=q(t)-\Delta\left(\frac{t-t_0}{T}-\frac12\right).
$$

The sample times and trapezoidal cycle mean are preserved, while the two
endpoint values become equal. This is a declared modelling assumption, not
the original MOX waveform or an inferred patient measurement.

A predeclared one-percent candidate criterion passed for the observed native
DG2 coefficients, wall multiplier and root-flow cycle comparisons. That
restricted result does not establish periodicity of the whole coupled system.
The solid output still changed between cycles nine and ten:

```{figure} ../_static/liver-icar/diagnostic64-solid-drift/solid-drift.svg
:alt: Relative change of saved solid displacement and velocity samples from cycle nine to ten, shown at the cycle endpoints and over eighty paired waveform phases.

Normalized solid drift in the separate 64-territory ten-cycle diagnostic.
Endpoint differences are 11.66% for displacement and 30.45% for velocity;
concatenating 80 phase pairs gives 4.918% and 14.880%, respectively. These
sample-based comparisons do not support a periodicity claim for the complete
coupled state,
even though the native/multiplier candidate criterion passed.
```

For each field the ratio is
$\lVert y_{10}-y_9\rVert_2/\lVert y_9\rVert_2$, without a denominator floor.
Endpoints are the actual saved states at 14.4 and 15.2 s. The waveform metric
concatenates 80 saved phase pairs with $13.6<t_9\le14.4$ and
$t_{10}=t_9+0.8$ before taking the norms. Both entries whose literal PVD time
is 13.6 are excluded; filenames are selected from the PVD rather than inferred
from an index formula.

The solid arrays are Float32 DataOut point samples, including repeated cell
vertices: 11104 points and 33312 scalar components per vector snapshot.
These Euclidean sample norms are not quadrature- or mass-weighted FE norms,
all solid coefficients, an IDA local-error estimate or a residual certificate.
Neither endpoint reference norm is below $a_{\mathrm{tol}}\sqrt N$, with
$N=33312$, displacement $a_{\mathrm{tol}}=10^{-7}$ m and velocity
$a_{\mathrm{tol}}=10^{-5}$ m/s. Six individual velocity phases have small
reference norms, so large individual
phase ratios need that qualification; the plotted waveform ratio aggregates
all 80 phases.

The [four dimensionless plotting values](../_static/liver-icar/diagnostic64-solid-drift/data.csv)
and [definitions and source hashes](../_static/liver-icar/diagnostic64-solid-drift/summary.json)
contain no raw field samples. Reproduce the figure without a solver or
anatomical data:

```bash
python3 -m pip install -r doc/_static/liver-icar/diagnostic64-solid-drift/requirements.txt
python3 doc/_static/liver-icar/diagnostic64-solid-drift/plot.py \
  --output-dir /tmp/liver-solid-drift
```

### Check hydraulic capacity before scaling

The first anatomical 64-territory probe retained 835 input segments and grew
382 CCO segments. With distinct portal and venous endpoint offsets, all three
forests grew but the native common-compartment check rejected the requested
pressures and supply split. Disabling geometric checks did not make that
hydraulic state admissible.

An independent rigid Poiseuille calculation gives each role an ideal
equipressure manifold at its generated leaves. It omits all microvascular
and other-role losses. At the prescribed mean flows, its minimum possible
pressure drops are:

| Role | Mean flow (mL/s) | Minimum drop (Pa) | Available per-role head (Pa) |
| --- | ---: | ---: | ---: |
| Hepatic artery | 6.3391 | 5917.37 | 9434.94 |
| Portal vein | 14.4973 | 634.92 | 436.98 |
| Hepatic veins | 20.8364 | 781.56 | 436.98 |

The portal and hepatic venous comparisons assume forward flow in every
portal and venous compartment arm. Each role is tested separately, so a pass
would be necessary rather than sufficient. The failed portal bound already
limits its optimistic flow to about 9.98 mL/s. These are bounds for this rigid
generated graph, not measured physiological values or a conclusion about a
compliant coupled model.

```{figure} ../_static/liver-unchecked-growth/rigid-capacity-bounds.png
:alt: Optimistic pressure-drop bounds for anatomical inputs and a rejected 64-territory CCO result, split between retained and generated vessels
:width: 100%

Necessary rigid pressure-drop bounds at the prescribed mean flows. The stacked
bars partition the grown-network optimum between retained and generated
vessels using the same currents. Dashed lines show the available per-role
head; panel scales differ. This witness failed native hydraulic validation.
```

The [numeric table](../_static/liver-unchecked-growth/data.csv),
[plot script](../_static/liver-unchecked-growth/script.py) and
[provenance](../_static/liver-unchecked-growth/provenance.json) retain this
diagnostic. The plot script reads the two adjacent audit JSON files and
`source-records.json`; it performs no new hydraulic solve. Preserve these
failed outputs separately from successful scale databases.

### Compare radii at matched terminals

A separate 64-territory native reference succeeds with an explicitly assumed
portal reservoir of 8.5 mmHg, rather than the preceding 8 mmHg. The MOX-derived
mean flows stay unchanged. The arterial, hepatic venous and portal input
prefixes use radius cutoffs of 1.5, 3 and 2 mm respectively. The retained
coordinates and radii are copied exactly from the anatomical inputs; the
discarded distal data remain in the original preparation. These choices are
model inputs, not a calibration of the IRCAD subject.

That reference has 456 physical vessels and 64 common compartments. All three
arm flows and resistances are positive; independent owned-CSV checks give
maximum node KCL defect $2.58\times10^{-16}$ m³/s and minimum common-pressure
gap 26.20 Pa. This is a successful rigid reference, not an accepted coupled
tissue solution.

Compare corresponding leaf endpoints before judging vessel sizes from the
display. The matched terminal radii in this reference are:

| Role | Minimum radius (mm) | Maximum radius (mm) |
| --- | ---: | ---: |
| Hepatic artery | 0.142 | 0.708 |
| Portal vein | 0.323 | 1.340 |
| Hepatic veins | 0.439 | 1.945 |

```{figure} ../_static/liver-terminal-radii/terminal-radius-flow.png
:alt: Radius and absolute reference flow at 192 matched leaf endpoints, grouped into arterial, portal and hepatic venous systems
:width: 100%

Exact radius and flow traces at the 64 matched terminals of each role.
Trunk and root values are excluded; radii at a junction are not averaged.
These relationships reflect the generated model and its Murray normalization,
rather than independent anatomical validation.
```

The [matched records](../_static/liver-terminal-radii/data.csv),
[renderer](../_static/liver-terminal-radii/script.py) and
[provenance](../_static/liver-terminal-radii/provenance.json) retain the physical
identifiers and endpoint selection. The renderer can replay the figure from
the CSV without generating or solving a network.

### Prescribe total supply and solve for local perfusion

The default `Reference flow distribution = tissue demand` requires each new
terminal to realize its demand-weighted design flow. To prescribe only the
total supply and let physical resistances determine local perfusion, use
`network resistance` under `Vascularization/Physiology`. This mode requires
explicit positive microvascular resistances and a positive total reference
flow. Growth still uses demand-weighted design flows for its objective; they
are no longer constraints on the final flow in each territory.

For a three-system steady reference, prescribe arterial and portal root
inflows and anchor the venous root pressure. Remove the two supply-root
pressure conditions: a root cannot receive both pressure and flow.
For the proximal input described above, the root Node IDs are arterial 6,
venous 7 and portal 47:

```text
subsection Physiology
  set Reference flow distribution = network resistance
  set Total new-tissue reference flow = 2.0836421871998312e-5
  set Reference pressures = 7:629.59739556
  set Unpaired reference inflows = 6:6.339078457494437e-6,47:1.4497343414503875e-5
end
subsection Portal network
  set Flow fraction = .69576933619234205
  # Supply three positive resistances in arterial, portal, venous order.
  set Compartment resistances = 1e12,1e11,1e11
end
```

The resistances above illustrate the parameter contract, rather than a fitted
liver model. They apply per territory in Pa s/m³ and must be chosen for the
number of represented compartments. There is no automatic rescaling by demand.
For a triplet, the native solve determines the common pressure and three arm
flows from these resistances, enforcing local Kirchhoff balance. Generation
checks the summed new-territory arterial, portal and venous flows against
$(1-f_P)Q, f_P Q, Q$, respectively. Retained beds remain active but do not
contribute to these new-territory totals.

Positive finite resistances, positive supply and drainage in every compartment,
physical node balance and declared vessel orientation remain required.
For unequal territories, `Scale compartment resistances by demand = true`
under `Portal network` multiplies every supplied arm resistance by
$\bar d/d_i$, where $d_i$ is the territory demand and $\bar d$ is its
mean over **new** territories. The supplied values then represent one
mean-size territory; equal demands reproduce the uniform per-bed model.
Retained beds are unchanged. With uniform tissue-demand density, this makes
conductance proportional to territory volume. It does not prescribe local
flow: the pressure drops remain part of the network solution. The option
requires `network resistance` and three supplied positive arm resistances.
The default is `false`.

The two inlet pressures are outputs: matching the total MOX flows alone does
not calibrate those pressures or establish patient-specific physiology.
The archived transient consumer currently uses an arterial inflow and portal
and venous pressure reservoirs; loading this archive does not automatically
impose the same portal inflow in that consumer.

### Resistance-driven reference results

With the new policy, both dense CCO forests pass the native rigid reference
solve and an independent CSV audit. All three arm flows and resistances are
positive. The maximum relative defect in the prescribed global flows is
$3.5\times10^{-14}$ across the two cases; node balance, bed balance and
pressure-work versus dissipation are checked independently.

| Common territories | Physical vessels, all roles | Arterial root pressure (Pa) | Portal root pressure (Pa) | Resistance model |
| --- | --- | --- | --- | --- |
| 512 | 3144 | 11933.97 | 1431.61 | Demand-scaled arms |
| 2048 | 12360 | 12226.22 | 1736.42 | Demand-scaled arms, portal coefficient ×2 |

Both prescribe arterial supply $6.3390784575\times10^{-6}$ m³/s and portal
supply $1.4497343415\times10^{-5}$ m³/s, with venous pressure 629.5974 Pa.
The 512 and 2048 physical graphs, radii and IDs are unchanged from their
earlier rejected attempts: the reference closure changes. The 2048 case
needs the stated portal-arm sensitivity to retain positive perfusion.
It is therefore a comparison of model variants, rather than an isolated
terminal-count or convergence study. These are rigid references, not
patient-calibrated or coupled transient results.

```{figure} ../_static/liver-reference-perfusion/demand-versus-perfusion.png
:alt: Demand fraction versus actual arterial, portal and venous flow fractions
:width: 100%

Each point is a matched common territory. The dashed line is proportional
perfusion. Local flows follow the resistance graph and can differ from the
demand-weighted design flows.
```

```{figure} ../_static/liver-reference-perfusion/portal-fraction-distribution.png
:alt: Histograms of local portal fractions in the two accepted rigid references
:width: 100%

Local portal supply fractions vary around the prescribed global fraction.
The 2048 case uses twice the portal resistance coefficient of the 512 case.
```

The [matched numeric records](../_static/liver-reference-perfusion/perfusion-distribution.csv),
[renderer](../_static/liver-reference-perfusion/plot_perfusion.py) and
[provenance](../_static/liver-reference-perfusion/provenance.json) retain the
territory/terminal IDs, actual flows and three arm resistances. The renderer
uses only the bundled CSV.

(liver-resistance-space-sizes)=
### Historical fixed-network setup diagnostics

These earlier diagnostics reuse the accepted 2048-territory physical archive
unchanged on three tissue backgrounds. They do not form the per-background
CCO campaign above: that campaign requires a separately grown three-system
family whose terminal target scales with its own background cell count.
The solid uses P2 displacement and velocity; the native flow and multiplier
use degree one. The measured block sizes are:

| Tissue tetrahedra | Displacement + velocity DoFs | Native flow DoFs | Multiplier DoFs | Total DoFs | MPI ranks |
| --- | --- | --- | --- | --- | --- |
| 2776 | 30864 | 105648 | 26412 | 162924 | 8 |
| 111250 | 992034 | 111912 | 27978 | 1131924 | 32 |
| 1222373 | 10206420 | 151280 | 37820 | 10395520 | 112 |

These are completed `Setup only` runs. They construct the coupled spaces and
stop before initializing the coupled state or invoking IDA. Setup wall times
are 100.85, 531.37 and 669.11 seconds, respectively; these are not
time-integration or solver benchmarks. The transient consumer still uses
arterial inflow and portal/venous pressure reservoirs, whereas the rigid
reference prescribes both supply totals. The coupled boundary model and
initial state still need validation.

The configured interval band is $0.5H_{\max}\le L\le2H_{\min}$, where
$H$ is the background cell diameter over native overlaps. It is not met by
all intervals:

| Tissue tetrahedra | Median $L/H_{\max}$ | Intervals below $0.5H_{\max}$ among intervals with overlap |
| --- | --- | --- |
| 2776 | 0.13082 | 11676 / 11835 (98.66%) |
| 111250 | 0.41242 | 7279 / 12576 (57.88%) |
| 1222373 | 0.82864 | 4228 / 17262 (24.49%) |

No interval exceeds the upper limit. Intervals without native overlap have
zero recorded $H$ and are excluded from these ratios. An overlap flag does
not mean an entire interval is inside the tissue. These measurements do not
establish an inf-sup condition; short physical branches remain a numerical
issue for the coupled solve.

Coverage is a separate measurement. On the **2776-cell generation background**,
each tetrahedron has, for each vascular role, a terminal whose ball of radius
$2H$ contains all four vertices with a floating-point margin. Convexity then
covers the entire affine tetrahedron. This gives a 100% volume lower bound
for joint terminal coverage of this 1.4554 L background. It does not measure
coverage relative to the finer solver backgrounds.

The [historical reference and fixed-network setup release](https://github.com/luca-heltai/immersx/releases/tag/liver-resistance-reference-2026-11-10)
contains the accepted 512- and 2048-territory reference examples, the three
background meshes, parameters, numeric tables, independent audits and
portable preparation scripts. Its setup package retains the fixed-network
block counts and per-interval mesh-scale records; it does not supply three
new forest families generated at one terminal per background cell. The
archived files remain unchanged. Its README gives commands for preparing a
new output directory; it does not start a simulation automatically.

### Earlier demand-constrained inspection attempts

Three denser inspection sets have completed all three forests on the same
2,776-cell background:

| Inspection set | Generated leaf terminals per role | Physical vessels, all roles | Native hydraulic result |
| --- | ---: | ---: | --- |
| 512 | 512 / 512 / 512 | 3,144 | Rejected common-compartment pressures |
| 2048 | 2,048 / 2,048 / 2,048 | 12,360 | Rejected common-compartment pressures |
| 2499 | 2,499 / 2,499 / 2,499 | 15,066 | Rejected common-compartment pressures |

The counts come from physical node identities and graph degree, not the
requested target. All three retain 73 anatomical input segments and use the 8.5
mmHg portal sensitivity. Their sampling, quadrature and local optimization
controls differ; their wall times are not an isolated performance comparison.
They are inspection data, not successful coupled scale profiles.

For each attempt, open the three `generated/invalid-result/*.pvtu` files
with their VTU pieces. The `Radius` point array retains separate physical
endpoint traces. The campaign records in the tracking issue identify the
parameter files, numerical tables, checksums and failed checks. A complete
growth count does not override a failed hydraulic validation.

(liver-terminal-density)=
## Set terminal density and coverage

Generate a coarse tissue volume first and record its global active-cell count.
The [mesh-only workflow](#save-the-volume-and-reuse-the-physical-network) lets
you inspect that grid before starting CCO. A Gmsh target edge length does not
determine the exact number of cells.

To start with one new tissue terminal per global background cell in each
enabled network, use:

```text
subsection Liver
  subsection Vascularization
    subsection Sampling
      set Terminal count = 1
      set Terminal count multiplier factor = background cell count
    end
  end
end
```

The multiplier uses the global cell count once; it is not a count per MPI rank.
For a grid of 4,245 cells, the initial target is 4,245 common tissue territories,
with one new arterial, portal and venous terminal per territory when all three
roles are enabled. Imported ports and retained input ends are separate records.

Allow modest flexibility through `Objectives/Minimum new terminals` and
`Objectives/Maximum new terminals`. These bounds are absolute counts, not
additional multipliers. For the 4,245-cell example, bounds of 3,820 and 4,670
allow approximately ten percent variation. Recompute them when the background
changes. `Calibration attempt budget` limits the number of trials; exhausting
it does not prove that the requested geometry or physiology is impossible.

With `Length scale = local mesh size`, set `Objectives/Maximum service distance`
to `2` for a maximum tissue-to-terminal distance of twice the diameter of the
tissue cell containing the evaluation point. For the affine tetrahedra below,
this diameter $H_K$ is the longest cell edge. It is not a minimum spacing
between terminals. `Objectives/Minimum tissue coverage = .95` requests at
least 95 percent volume-weighted quadrature coverage. With portal inflow
enabled, this criterion uses the points served by all three systems, rather
than the average of their separate coverages.

Changing `Length scale` also changes the units of length-valued growth controls,
including seed length, minimum vessel length, clearance, minimum radius and
terminal separation. Re-express those controls in mesh units and inspect
their effective physical lengths. For example, a seed length of `.5` then means half a local cell
diameter. Imported coordinates and radii, pressures, flow rates and exported
physical data retain their SI units. See the
[generation controls](../how-to/vascular-generation) for the complete list.

One terminal per global cell is a total-count target, not an occupancy cap.
Inspect the number of generated terminals in each cell, separately for each
role, together with the service-distance coverage. Cells can contain zero or
several terminals while satisfying a distance criterion. Geometric coverage
does not establish perfusion or a feasible reference-pressure solve.

After growth and after every change of the tissue grid, rediscretize the
physical networks and inspect the numerical 1D interval lengths against the
local background diameter. The discretization target and lower/upper length
factors control those numerical intervals; they do not change the physical
terminal count. Subdivision preserves the endpoints of each physical vessel;
it does not merge distinct physical vessels across a degree-two node. If a
physical segment is already shorter than the intended lower interval-length
bound, further subdivision cannot repair that bound. Review the background
resolution and the physical input preparation instead.

Record the actual coupled block DoFs and interval-length
diagnostics before increasing the problem size. Comparable mesh lengths are
an important check, but do not by themselves establish a discrete inf-sup
condition.

### Inspect the completed arterial stage

The anatomical V21 calculation reached its requested **4,245 generated arterial
terminals on 4,245 tetrahedra**. Its arterial stage contains 8,542 physical
segments: 54 retained arterial input segments and 8,488 generated CCO segments.
The full snapshot preserves all 835 retained input vessels and their endpoint
coordinates and radius traces. The input-tail preparation uses a 0.34 mm radius
threshold at existing points; this is a resolution sensitivity, not a clinical
cutoff.

The parent calculation stopped at its six-hour limit while portal growth had
reached 3,140 terminals; hepatic venous growth had not started. The arterial
finish snapshot is therefore an inspection result, not a completed three-network
archive. It contains no coupled time steps, pressure/flow/displacement solution,
or restart state.

```{figure} ../_static/liver-cco-inspection/A4245/arterial-growth-projections.png
:alt: XY, XZ and YZ projections of the completed 4,245-terminal artery, with imported segments in blue-gray and generated CCO segments in red
:width: 100%

Completed arterial growth in the coarse IRCAD case 12 generation volume.
Blue-gray lines are the 54 retained arterial segments; red lines are the 8,488
generated segments. Coordinates are in millimetres. Constant line width makes
the branches visible and does not encode physical radius.
```

One terminal per global cell is a count target, not one terminal in every cell.
The measured arterial occupancy is:

| Generated arterial terminals in a cell | Cells |
| ---: | ---: |
| 0 | 1,515 |
| 1 | 1,606 |
| 2 | 802 |
| 3 | 263 |
| 4 | 49 |
| 5 | 10 |

All 4,245 terminal coordinates were located inside this background. The global
mean is one terminal per cell, with a local maximum of five. A cell containing
no terminal can still be served by a nearby terminal.

An independent geometric check uses the nearest generated arterial terminal
distance $d_K$ from the cell centroid, the largest centroid-to-vertex distance
$r_K$, and the longest edge $H_K$. The guarded sufficient condition
$d_K+r_K\leq 2H_K$ holds for **every tetrahedron**, certifying 100% of this coarse
volume within the requested maximum distance of $2H_K$. This follows from
convexity of a straight tetrahedron and the 1-Lipschitz nearest-terminal distance.
It also implies the distance predicate at its quadrature points; it does not
replay the native quadrature or distributed routing. The requested minimum was
95%. This arterial-only result establishes neither portal/venous joint coverage,
coverage on a finer anatomical volume, perfusion, nor an inf-sup condition.

```{figure} ../_static/liver-cco-inspection/A4245/arterial-coverage-occupancy.png
:alt: Arterial terminal occupancy from zero to five per cell and the volume-weighted sufficient whole-cell distance bound relative to the 2H maximum
:width: 100%

Measured occupancy and conservative whole-cell coverage for A4245. The right
panel weights cells by volume and compares the sufficient distance bound with
the original maximum-distance contract, $2H_K$. These are geometric measurements
on the coarse arterial background, not pressure or perfusion results.
```

Download the [A4245 inspection archive](https://github.com/luca-heltai/immersx/releases/download/liver-icar-cco-inspection-2026-11-07/liver-cco-A4245-v21-20261107.tar.gz)
and extract it into a new directory. The stable download URL becomes available
when the inspection release is published; while it remains a draft, use the
authenticated draft assets listed in the
[tracking record](https://github.com/luca-heltai/immersx/issues/343).
The archive SHA-256 is
`b54d531ab31baf4466dcd1b335ecf55f5c6806eeccd5b9c1351bd8033d04f5c0`.
Verify its internal `SHA256SUMS` before opening these files in ParaView:

- `artery.vtu`: completed arterial physical vessels with native `Radius`,
  `Generated` and physical IDs;
- `liver-generation.vtu`: the 4,245-tetrahedron generation volume;
- `arterial-terminals.vtp`: the generated arterial endpoints;
- `network-context.vtu`: the arterial result together with the retained portal
  and venous inputs, without completed companion growth.

These names are relative to the extracted inspection archive. Keep the duplicated
endpoint traces: incident vessels can have different native radii at a shared
physical node.
Use `Radius` for a varying-radius Tube filter rather than interpreting the
constant-width projections as a lumen image. These physical segments have not
yet been rediscretized and checked against the final coupled background.

The [per-cell measurements](../_static/liver-cco-inspection/A4245/arterial-coverage-cells.csv),
[plotted segments](../_static/liver-cco-inspection/A4245/arterial-projection-segments.csv),
[coverage record](../_static/liver-cco-inspection/A4245/arterial-coverage.json) and
[figure provenance](../_static/liver-cco-inspection/A4245/provenance.json) retain
the figure inputs and scope. The accompanying
[plot script](../_static/liver-cco-inspection/A4245/plot.py) reproduces the views
from these tables and the archived `liver-generation.vtu`; see its
[instructions](../_static/liver-cco-inspection/A4245/README.txt).

### Historical A3073 checkpoint

The [anatomical case tracking record](https://github.com/luca-heltai/immersx/issues/343)
includes a saved arterial checkpoint with 3,073 generated tissue terminals on
4,245 tetrahedra. It retains 835 input physical vessels and adds 6,144 arterial
physical vessels. Portal and hepatic venous growth are incomplete at this
checkpoint. The initial total-count target is 4,245 terminals per enabled role.
The recorded input preparation trims distal tail samples at existing points
using a 0.34 mm radius threshold. This is a resolution sensitivity, not a
clinical cutoff; retained coordinates and radius traces are unchanged.

```{figure} ../_static/liver-cco-inspection/arterial-growth-projections.png
:alt: Three projections of the IRCAD liver with gray input vessels and red arterial branches generated by CCO
:width: 100%

Partial arterial CCO growth from the prepared IRCAD case 12 trees. Gray lines
are retained input vessels; red lines are generated arterial vessels. Constant
line widths make the fine branches visible and do not encode physical radius.
```

Locate generated terminal endpoints in the volume cells and measure distance
coverage as well as occupancy. In this snapshot, 2,266 cells contain generated
arterial terminals, with at most four per cell. A cell with no terminal can
still be served by a nearby one.

For an independent geometric check, let $d_K$ be the distance from a cell's
centroid to the nearest generated arterial terminal, and let $r_K$ be the
largest centroid-to-vertex distance. Then $d_K+r_K\leq 2h_K$ is sufficient to
serve the entire straight tetrahedron. The plot uses floating-point guards
and cell-volume weights. This bound covers the full generation volume at the
3,073-terminal checkpoint. It uses only generated arterial terminals and is
separate from the native order-two quadrature and three-network coverage test.

```{figure} ../_static/liver-cco-inspection/arterial-coverage-and-cell-counts.png
:alt: Arterial geometric coverage during CCO growth and the terminal occupancy histogram at 3,073 terminals
:width: 100%

Coverage and cell occupancy of saved arterial growth snapshots. The sufficient
whole-cell bound reaches full volume coverage while many cells contain no
terminal. These are geometric measurements, not pressure or perfusion results.
```

### Check interval lengths before coupling

Rediscretizing the historical A3073 physical checkpoint on the original
tetrahedra produces 6,987 numerical intervals, including seven outside the tissue.
Of the 6,980 inside intervals, 6,948 are below the configured lower bound
$L/h_{\max}=0.5$; none exceed $L/h_{\min}=2$. The generated arterial subset
alone has 6,124 lower-bound failures. Good terminal coverage therefore does not
satisfy the numerical interval-length requirement.

The following diagnostics retain the physical network and subdivide the
original tetrahedra into four conforming hexahedra each before local background
refinement. They use the existing
[local co-refinement workflow](../how-to/local-co-refinement.md) and native
vascular rediscretization. The refinement level limits the background passes;
remaining lower-bound failures are reported rather than suppressed.

| Background | Cells | Solid displacement + velocity DoFs | Inside intervals below $0.5h_{\max}$ |
| --- | ---: | ---: | ---: |
| Original tetrahedra | 4,245 | Not measured here | 6,948 / 6,980 |
| Hexahedra, local level 1 | 111,193 | 811,926 | 3,735 / 7,111 |
| Hexahedra, local level 2 | 321,487 | 2,710,380 | 632 / 7,954 |

The solid counts use vector-valued Q1 displacement and velocity before
eliminating hanging-node constraints. They exclude native flow and multiplier
unknowns. These measurements are not completed coupled scale profiles or an
inf-sup verification. Native interval lengths must be checked again after any
further background or network change.

```{figure} ../_static/liver-cco-inspection/interval-length-comparison.png
:alt: Native interval-length distributions relative to crossed-cell diameter for the original and locally refined backgrounds
:width: 100%

Native inside-interval diagnostics for the same partial physical network.
The lower threshold is 0.5; the upper threshold uses the minimum diameter of
crossed cells and is checked separately. The seven exterior intervals are
excluded from this distribution.
```

The prescribed rigid reference is an additional gate. An offline Poiseuille
check uses the saved radii and current native demand allocation. At arterial
mean flow $6.33907845749\times10^{-6}$ m³/s, 2,405 of these 3,073 arterial
terminals fall below even the ideal hepatic venous outlet pressure of
629.597 Pa. Actual venous path losses would raise the required terminal
pressure in that model. The prescribed rigid reference fails this necessary
check. No accepted coupled solution is associated with this checkpoint; the
check does not establish infeasibility of an elastic transient problem.

For ParaView inspection, the diagnostic bundle uses these paths:

- `geometry/vascular-physical-A3073.vtu`: physical vessels and per-endpoint radius traces;
- `geometry/liver-background.vtu`: the generation volume;
- `geometry/vascular-native-intervals-A3073.vtu`: native intervals with
  `Inside`, `LengthOverHmax` and `LengthOverHmin` arrays;
- `diagnostics/coverage/coverage-diagnostics.vtu`: background-cell occupancy
  and geometric coverage diagnostics.

Select `Inside=1` before plotting interval ratios. Retain separate physical
endpoint IDs and radius traces at junctions. The original 32-rank checkpoint
pieces are inspection data; they do not contain all state needed for a
reproducible CCO restart. The tracking record lists the draft inspection assets
and their SHA-256 checksums. The
[coverage CSV](../_static/liver-cco-inspection/geometric-coverage.csv),
[interval summary](../_static/liver-cco-inspection/interval-summary.csv) and
[figure provenance](../_static/liver-cco-inspection/provenance.json) retain the
measurements used here.


(liver-portal-inflow)=
## Add portal inflow

The optional portal network supplies the same territories as the artery; the
venous tree drains their combined flow. It is not a partition of the liver into
arterial and portal territories.

```{image} ../_static/liver-vascularization/common-compartment.svg
:alt: Arterial and portal inflows meet at one microvascular pressure and drain through the vein
:width: 90%
```

Prepare a new input directory with the third tree:

```bash
micromamba run -n ircad-prep python scripts/prepare_ircad_liver.py \
  build-release/data/ircad-case12-portal --with-portal
```

This additionally reads `MESHES_VTK/portalvein.vtk`, a surface in the IRCAD
archive, and extracts `portal.vtk` with the same VMTK procedure. The portal
surface is not a usable 1D mesh. The preparation manifest records the source
digest, retained component, root and selected leaves. The checked recipe yields
739 portal nodes and two growth leaves. Arterial and venous coordinates, radii and physical identifiers stay the same.
With `--with-portal`, a selected growth endpoint inside another tree's
conservative capsule envelope is disabled in the new input's `Grow` array and
recorded in the manifest. The recipe disables arterial endpoint 11, which
intersects the portal envelope; arterial endpoint 10 remains available.
Preparation fails if no selected endpoint remains for any role. The generator
still certifies every proposed vessel continuously. Arterial and venous trees
are imported before the portal tree so their physical identifiers stay the same. The portal root is then physical node **1262** for
this recipe. Recompute these root ids if the preparation settings change.

Copy the generated parameter file for a separate run:

```bash
cp build-release/data/vascularization/liver.prm \
  build-release/data/vascularization/liver-portal.prm
```

Use the new prepared directory for all three line meshes and the liver surface.
Edit the copied file's existing subsections with these settings. The application
default remains 1,000 territories per network; this shorter trial requests 32:

```text
subsection Liver
  set Arterial grid = ../ircad-case12-portal/arterial.vtk
  set Venous grid = ../ircad-case12-portal/venous.vtk
  set Portal grid = ../ircad-case12-portal/portal.vtk
  set Output directory = ../../test_output/liver-portal
  subsection Domain
    set Grid generator = ../ircad-case12-portal/liver.stl
  end
  subsection Vascularization
    subsection Sampling
      set Terminal count = 32
    end
    subsection Physiology
      set Reference pressures = 83:12000,177:2000,1262:5000
    end
    subsection Portal network
      set Enabled = true
      set Terminal separation = .004
      set Flow fraction = .7
    end
  end
end
```

Save the parameter file in `build-release/data/vascularization`, as in the
two-network example. All paths are relative to that file. Keep the venous network enabled.
The total new-tissue flow now denotes venous drainage: with the illustrative
fraction `0.7`, portal inflow supplies 70% and arterial inflow 30%. These values
and root pressures are modeling inputs. Leave `Compartment resistances` at its
three zero defaults to derive positive arms using a common pressure between the
venous terminal pressure and both supplying terminal pressures. The final solve
checks each requested supply and drainage, including `Q_v = Q_a + Q_p`.
See {ref}`three-network reference flow <vascular-three-compartments>`
for the equations and supplied-resistance alternative.

The CSV output includes `triplets-*.csv` with terminal identities, three arm
resistances, actual supply and drainage flows and the common pressure. In VTK,
`System` values remain `0=arterial`, `1=venous`, with `2=portal`. The physical
archive retains all three roles and their common compartments. Existing
arterial–venous parameter files retain their original behavior.

Run the copied parameter file:

```bash
mpirun -np 4 build-release/liver_vascular \
  build-release/data/vascularization/liver-portal.prm
```

The verified four-rank Release run with deal.II 9.7.1 produced **32 triplets,
2,185 physical vessels and 11,047 tissue cells**. It retained 2,001 input nodes
and 1,998 input segments, and added 63 arterial, 62 portal and 62 venous
segments. The independent checker preserved all input coordinates, ports and
radii, verified the requested supply split and found a positive sampled wall
margin for every new vessel.

This 32-territory trial provides a shorter starting point. A different count or
MPI partition can need different search settings; fewer territories do not
guarantee an admissible initial connection for every selected attachment.
The published 1,000-pair run and its figures above remain the arterial–venous
reference. To check a three-network run, pass `--triplets N`
instead of `--pairs N`, together with the portal separation:

```bash
micromamba run -n ircad-prep python scripts/check_liver_vascular.py \
  build-release/data/ircad-case12-portal build-release/test_output/liver-portal \
  --triplets 32 --separation .003 --portal-separation .004 \
  --portal-flow-fraction .7
```

The figure renderer accepts the same three options and colors portal vessels
purple. Geometry certification, demand matching and positive perfusion are
required before writing an accepted result; the finite growth budget can fail
for an incompatible geometry or search configuration.


## Continue companion growth from physical data

The native `resume_companions` entry accepts the original anatomical network,
a physical checkpoint, the unchanged generation parameters and an explicit
active phase (`System::portal` or `System::venous`). The arterial phase must
already be complete. Portal growth must also be complete when starting the
venous phase. Existing generated companion leaves count toward the absolute
number of completed arterial territories; the requested count is not an
additional number of leaves.

The original nodes and vessels, their physical properties, untouched input
terminals, and completed supply forests are retained. The checkpoint must
identify each generated terminal's original common territory and demand.
Completed forests undergo the native geometry certificate using their stored
radii before growth is skipped. Invalid attachments, changed anatomical data,
changed demands, inconsistent root radii and duplicate territory claims are
rejected collectively. The ordinary complete-bed and reference validation
still runs after all enabled roles are complete.

For `liver_vascular`, keep the original tree files and Grow flags and set:

```text
subsection Liver
  set Companion checkpoint archive = /absolute/path/to/physical-checkpoint
  set Companion continuation phase = portal
end
```

This directory must be a native network-only physical archive: preserve all
Node, Vessel and Terminal IDs and metadata, including territory labels, but do
not register unfinished pairs, triplets or compartments as completed beds.
The diagnostic PVTU/CSV exports alone are not such an archive. An external
converter must restore and verify missing physical metadata against the
original input before writing one with the native archive writer. A partial
checkpoint is an intermediate growth state, not a coupled simulation input.

Continuation starts a new finite search epoch. It does not restore the old
random generator or rejected-candidate history and does not promise the same
trajectory as an uninterrupted run. Count limits and coverage targets remain
in force; the entry does not rerun the terminal-count calibration search or
relax incomplete common-territory matching.

(save-the-volume-and-reuse-the-physical-network)=
## Save the volume and reuse the physical network

To save the exact generated volume before distributed partitioning, set:

```text
subsection Liver
  set Write volume mesh = true
end
```

The new output directory then contains `liver.msh` as well as the physical
`network/` archive. Set `Only generate mesh = true` to stop after writing the
volume, for example when preparing several tissue resolutions. This mode also
requires `Write volume mesh = true`. It does not generate or validate a vascular
network. The ordinary growth path remains the default.

With a compatible MetricFlowX package, `metric_flow_x_elastodynamics` can load
the physical archive and subdivide its vessels against its final tissue mesh.
The imported tissue mesh can be `liver.msh`; select a fully distributed Domain
for tetrahedra. Configure the ordinary solid, IDA, inflow and lift parameters,
then add the following explicit vascular data:

```text
subsection MetricFlowX coupling
  subsection Archived network
    set Directory = /absolute/path/to/network
    set Young moduli = 400000,400000,400000
    set Wall thickness ratios = .1,.1,.1
    set Reference pressures = 10064.5,629.6,1066.6
    set Portal pressure = 1066.6
    set Venous pressure = 629.6
    set Root resistance = 100000
  end
end
```

The three material entries are in **arterial, venous, portal** order. Young
moduli and reference pressures are in Pa; wall thickness is the given ratio
times the mean vessel radius. Root resistance is in Pa s/m³. The numbers above
illustrate the parameter contract, not a patient calibration. The arterial
root retains the native inflow function or CSV. Portal and venous roots use
separate pressure reservoirs with the stated positive series resistance.
This requires a backend with orientation-correct RCR conditions at both vessel
ends. The saved common microvascular compartments connect the tissue leaves;
they do not acquire extra independent outlet resistances. An archive with
unconnected tissue leaves is rejected by default. In particular, the anatomical
growth example above retains unused input leaves. To declare selected ends
closed, set `Sealed input terminal ids` under the same `Archived network`
subsection to their comma-separated physical **Terminal IDs**. Every remaining
tissue leaf must belong to a saved bed, and a bed terminal cannot also be
sealed. This needs the native sealed-endpoint API from
[MetricFlowX #88](https://github.com/luca-heltai/metric-flow-x/pull/88) and a
matching rebuild. A sealed end has exactly zero flow and adds no reservoir;
it is an explicit model assumption and must not count as a perfused terminal.

Set `Setup only = true` under `MetricFlowX coupling` to construct the
coupled spaces and report their block DoFs, then stop before state
initialization or IDA. This measures the configured discretization without
claiming a converged reference, initial condition or timestep. The default is
`false`.

For fine physical vessels, a compatible backend exposes
`Minimum positive area` under `MetricFlowSystem<1, 3>`, in m².
This option requires
[MetricFlowX #90](https://github.com/luca-heltai/metric-flow-x/pull/90).
The physical-network importer uses that same threshold and preserves the
supplied radii. Older backends retain their fixed $10^{-10}$ m² threshold.
Choose a smaller value only after checking the actual minimum reference area;
the importer rejects reference areas at or below the configured threshold.

The coupled output reports each block's DoFs and the total. Per-rank
`mesh_scale-*.csv` files retain each numerical interval's length and the
minimum/maximum diameter of crossed tissue cells. Short-cell exceptions are
reported explicitly. Terminal CSV diagnostics use actual boundary traces,
including the coupled external pressure. Mass diagnostics compare net outward
flow with volume change between accepted output times; the first record has no
previous interval.

For an equilibrium start, IDA's `Correction type at initial time = use_y_dot`
uses a zero state derivative to solve the complete coupled residual at the
initial time. This equilibrates the chosen boundary data; it does not establish
a periodic coupled state. The coupled application constrains both cell and trace
area coefficients to remain positive before evaluating a Newton trial.
Nonfinite residuals are reported collectively as recoverable callback errors.
Equilibrium correction can still stop when the selected SUNDIALS mode cannot
recover from a failed residual evaluation; see the
[IDA initial-condition implementation](https://github.com/LLNL/sundials/blob/v6.7.0/src/idas/idas_ic.c).
For exterior tissue support, configure the solid's
`Robin boundary ids`, `Functions/Robin coefficient` and
`Functions/Robin boundary` as described in the
[Robin boundary reference](../reference/robin-boundaries.md).


Coupled error control can use different absolute tolerances for quantities
with different units. Set `Displacement absolute tolerance` (m),
`Velocity absolute tolerance` (m/s), and `Wall pressure absolute tolerance`
(Pa) under `MetricFlowX coupling`. Their zero defaults retain the common IDA
absolute tolerance. `Use native flow tolerances = true` reuses MetricFlowX's
`Area absolute tolerance fraction`, `Velocity absolute tolerance`, and
`Capacitor pressure absolute tolerance` for its mixed area/velocity/pressure vector.
This delegates component numbering and reference-area scaling to MetricFlowX.
Choose and verify these tolerances against the spatial and temporal errors of
the intended case.

The historical
[two-territory coupled pilot](https://github.com/luca-heltai/immersx/releases/tag/liver-icar-start-2026-11-05)
uses an anatomical liver volume with three supplied interior feeder stubs and
11 generated vessels. Its vascular graph does not extend the prepared IRCAD
vascular trees or resolve whole-organ coverage. The larger pilot profiles
remain preparation cases; their transient solver performance is unvalidated.
Use the anatomical tracking record above for the full-tree growth work.

The [coupled IRCAD12 liver tutorial](liver-icar-coupled.md) provides a separate
two-territory arterial/portal/venous case, three tissue resolutions, physical
parameters, numerical audits, reproducible result figures and ParaView paths.
Its recorded results include the small complete interval, the medium 20 ms
check and the large 2 ms startup. These windows do not establish physiological
validation, mesh convergence or a uniform inf-sup bound.

For an archived liver case, `Stop at output times = true` under
`/MetricFlowX coupling/` ends an internal IDA step at each saved time. The
default remains dense output. Internal stops retain the BDF history and avoid
interpolation defects in nonlinear hydraulic constraints. Control Newton
convergence separately through the IDA `Nonlinear solver` subsection.

## Diagnosing rejected growth candidates

The growth statistics CSV includes `candidate_length_rejections`,
`candidate_geometry_rejections`, `candidate_boundary_rejections`, and
`candidate_no_feasible_start`. Length and geometry counters count rejected
junction trials, including optimization trials; they are not rejected terminal
counts. Boundary failures are a subset of geometry failures in the candidate
search. A candidate with no feasible midpoint or centroid start increments
`candidate_no_feasible_start` once. Counters accumulate local work and are
reduced with the other growth statistics; they do not change acceptance gates.
They do not yet distinguish own-system and cross-system collisions or provide
a spatial rejection map.
