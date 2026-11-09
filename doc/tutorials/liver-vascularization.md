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
coordinates, or an unknown vascular system, cannot be drawn: `omitted-*.csv` lists their physical IDs and
`diagnostic.json` gives the total omitted count. This output does not repair
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

For an archived liver case, `Stop at output times = true` under
`/MetricFlowX coupling/` ends an internal IDA step at each saved time. The
default remains dense output. Internal stops retain the BDF history and avoid
interpolation defects in nonlinear hydraulic constraints. Control Newton
convergence separately through the IDA `Nonlinear solver` subsection.
