# Local co-refinement of two domains

`LocalCoRefinement<d1, d2, spacedim>` adjusts two meshes before application
finite-element spaces are constructed. The first domain is **space**, the second
is **embedded**. These are roles, not restrictions on intrinsic dimension: for
example, both `<2, 1, 2>` and `<1, 2, 2>` are supported.

The utility retains imported local fields while refining and partitions
`fullydistributed` meshes only after all adjustments have finished. It does not
transfer an application's solution or refine an already finalized domain.

Uniform triangle and tetrahedron meshes use matching reference-cell mappings.
Set their Domain backend to `fullydistributed`; the serial preparation mesh
can be refined before distribution. Simplex preparation disables deal.II's
hypercube smoothing rules. Imported point and cell fields follow refinement.

## Ordinary usage

This example assumes that `line.vtk` contains an array named `thickness`.

```cpp
#include <immersx/grid/local_co_refinement.h>

ImmersX::DomainParameters<2> space_parameters("/Space/Domain/");
ImmersX::DomainParameters<1, 2> embedded_parameters("/Embedded/Domain/");
embedded_parameters.name_of_grid = "/absolute/path/to/line.vtk";
embedded_parameters.read_vtk_fields = true;

ImmersX::RefinementParameters<1> refinement;
// Parse the parameter file after constructing the parameter objects.
refinement.refinement_strategy = "both";
refinement.lower_factor = 0.5;
refinement.upper_factor = 2.0;

ImmersX::Domain<2> space(space_parameters);
ImmersX::Domain<1, 2> embedded(embedded_parameters);
const auto report =
  ImmersX::LocalCoRefinement<2, 1, 2>(refinement).run(space, embedded);

// Both meshes are finalized now. Attach application FE spaces and access
// imported fields only after run().
const auto thickness = embedded.field("thickness");
```

The domains and the utility keep references to their parameter objects; those
objects must outlive them. Both domains must have congruent MPI communicators.
Call `run()` collectively, with the same parameters on every rank.

Do **not** call `make_grid()` before `run()`: `make_grid()` prepares and finalizes
immediately. An already finalized domain is rejected rather than overwritten.
To modify the meshes before adjustment, explicitly call
`space.prepare_grid()` and `embedded.prepare_grid(true)`, modify their
`refinement_triangulation()` objects, then call `run()`. The `true` keeps the
embedded preparation mesh serial even when its final backend is `distributed`.
Do not attach application DoFHandlers to this temporary preparation mesh.

The [Coral registry](coral-backend.md) provides the same workflow as a
`Co-refine domains` operation on two owning domain handles, with a separate
report output and explicit downstream FE-space construction.

## Using a mesh owned by a Problem

`Domain` can also observe an existing parallel triangulation:

```cpp
// Construct the Problem before the Domain: it owns the triangulation.
Problem problem;
ImmersX::Domain<2> domain(domain_parameters, problem.triangulation());
```

The Problem and the parameter object must outlive `domain`. Domain stores an
`ObserverPointer` to the mesh. Its usual constructor also creates a
`unique_ptr` owning the mesh; the external-mesh constructor leaves that owner
empty and takes the MPI communicator from the supplied triangulation.

A populated external mesh is not generated, scaled, or refined by the
constructor. With `Read fields = true`, the constructor imports the configured
VTK fields onto that mesh, which must match the input mesh. Access a property
with `domain.field("a0")`. On a `parallel::distributed::Triangulation`, retained
fields follow refinement and coarsening through triangulation signals without
rereading the file. Attach Domain before those mesh changes. The owner remains
responsible for transferring its own solution.

For an empty external mesh, use `make_grid()` or the preparation/finalization
workflow above. This also supports `fullydistributed` meshes, for which the
existing property transfer occurs on the serial preparation mesh before
partitioning. Observing a mesh does not add adaptive-refinement support to
`fullydistributed` triangulations.

If the owner has already read a serial coarse mesh, call
`domain.prepare_grid(coarse_mesh)`. Domain uses the supplied coarse geometry
and preserves its material and boundary ids. It imports the configured
properties before refinement and uses the same `finalize_grid()` transfer.
The final observed mesh must still be empty.

## Partition the embedded mesh with the background

Set the partitioning policy in the same refinement parameter group:

```text
subsection Local refinement parameters
  set Embedded partitioning policy = immersed
end
```

The default is `default`, which keeps each Domain's ordinary partitioning.
With `immersed`, co-refinement finalizes the background first, then uses
`ImmersedRepartitioner`, also used by `ReducedPoisson`, to assign each embedded
cell to the rank owning the background cell containing its center. If the center
belongs to cells on several ranks, the lowest rank is selected. This aligns
ownership at cell centers; an embedded cell may still cross a partition boundary.
The embedded mesh must lie inside the background mesh.

This option requires a full-dimensional background (`d1 == spacedim`) and a
`fullydistributed` embedded Domain. One-dimensional Domains always use this
backend. For higher dimensions, set `Triangulation type = fullydistributed` in
the embedded Domain's parameter group. Both supported background backends work.
Unsupported dimension/backend combinations are rejected before mesh preparation.

Imported point and cell fields are transferred after the final ownership is
established, so the option does not require reading the input again. It applies
only to `LocalCoRefinement::run()`, before application FE spaces exist; the raw
`adjust_grids()` compatibility entry point only adjusts refinement.

For a custom deal.II repartitioning policy, call `Domain::prepare_grid()`, adjust
the preparation mesh, then call `Domain::finalize_grid(policy)`. This overload
requires a fullydistributed Domain. It evaluates the policy collectively once
and transfers retained input fields onto that final partition. Neither the
Domain nor its triangulation retains a reference to the policy.

## What happens, in order

1. **Initial refinement.** Each empty domain reads or generates its mesh,
   imports the requested VTK fields, applies scaling, and applies its own
   `Initial refinement`. Explicitly prepared domains skip this preparation.
2. **Pre-refinement.** Globally refine space and embedded by their respective
   pre-refinement cycle counts.
3. **Local adjustment.** Construct two Boost rtrees through deal.II's
   `GridTools::Cache`, query intersecting cell bounding boxes, and repeatedly
   refine according to the selected strategy. The caches update after mesh
   changes; queries never reuse cell sizes from before a refinement.
4. **Space post-refinement.** For each requested cycle, refine every space cell
   whose bounding box intersects an embedded cell bounding box, then repeat
   the entire local adjustment. This is **local**, not global refinement of
   space. It runs regardless of the strategy.
5. **Embedded post-refinement.** Globally refine embedded by its requested
   cycle count, then repeat the entire local adjustment once more. This also
   runs regardless of the strategy.
6. **Finalization.** Finalize space first, then partition embedded with the
   selected policy and redistribute its retained input fields onto the final
   mesh.

Thus `space` does not mean embedded is untouched: embedded's initial,
pre-refinement, and post-refinement cycles still apply. Likewise, `embedded`
still allows the specified space pre/post cycles. Strategy controls only the
ratio-driven local adjustment.

## The size criterion

For each intersecting bounding-box pair, let `h_space` and `h_embedded` be the
**diagonals of the axis-aligned cell bounding boxes**. They are not cell volumes,
edge lengths, or a measure of the portion of an embedded cell inside space.
The same open interval applies to every strategy:

```text
lower_factor * h_space < h_embedded < upper_factor * h_space
```

| Strategy | Ratio-driven refinement |
| --- | --- |
| `space` | Refine space when `h_embedded <= lower_factor * h_space`. Never refine embedded to correct the opposite bound. |
| `embedded` | Refine embedded when `h_embedded >= upper_factor * h_space`. Never refine space to correct the opposite bound. |
| `both` | Execute a space marking/refinement pass, query the updated trees, then execute an embedded marking/refinement pass. Repeat the two passes until neither marks any cell. |

Equality triggers refinement. Cells are never coarsened. A pair of boxes that
touches at an edge or vertex counts as intersecting. Bounding boxes can also
intersect when the actual cells do not, especially for oblique segments. All
such pairs participate in the criterion; this is intentionally not an exact
cell-intersection algorithm.

The two factors keep the same meaning when the strategy changes. If the size
ratio were instead written with inverted roles, its equivalent bounds would
be `1 / upper_factor` and `1 / lower_factor`, not the original factors.

### Limits and the report

Factors must be finite and satisfy `0 < lower_factor < upper_factor`.
Prefer `upper_factor > 2 * lower_factor` for dyadic isotropic refinement. A
narrower interval may miss every attainable ratio and make `both` alternate
refinements until the level cap is reached. A wider interval still does not
guarantee that every pair can satisfy the bounds: one cell may intersect
several cells with incompatible sizes.

`Max refinement level` limits **only ratio-driven local adjustment**. It does
not limit `Initial refinement`, global pre-refinement, local space
post-refinement, or global embedded post-refinement. Those requested cycles
can raise cells above the cap; the following ratio-driven adjustment then
cannot refine those cells further.

`run()` returns a `LocalCoRefinementReport` with global counts:

- `intersecting_pairs`: all final intersecting bounding-box pairs;
- `lower_violations`: pairs at or below the lower bound;
- `upper_violations`: pairs at or above the upper bound;
- `satisfied()`: both violation counts are zero.

Remaining violations are **not a fatal error**. The utility prints a final
summary and returns their counts. They can result from the chosen single-mesh
strategy, the level cap, or incompatible intersecting sizes. Zero intersecting
pairs gives a satisfied report with zero pairs; it does not prove that the
embedded mesh lies inside the space domain. The report describes the final
meshes after post-refinement and its subsequent corrections, not just the
first local adjustment.

## Parameters

The parameter group is `Local refinement parameters`, under the path of the
owning parameter configuration. For existing tensor-product coupling files it
remains under `Tensor product coupling`.

```text
subsection Local refinement parameters
  set Refinement strategy             = both
  set Lower factor                    = 0.5
  set Upper factor                    = 2
  set Max refinement level            = 10
  set Space pre-refinement cycles     = 0
  set Embedded pre-refinement cycles  = 0
  set Space post-refinement cycles    = 0
  set Embedded post-refinement cycles = 0
  set Refinement factor               = 0
end
```

These are the defaults except that the default strategy is `space`.

`Refinement factor` is a deprecated compatibility input for the old single-mesh
configuration. Its default is now zero, which disables it. When positive:

- with `space`, it replaces `Lower factor` by `1 / Refinement factor`;
- with `embedded`, it replaces `Upper factor` by `Refinement factor`;
- with `both`, it is rejected: specify the two bounds explicitly.

The other bound remains unchanged and the resulting interval must be valid.
The new criterion is strict, so equality at the replaced bound now triggers
refinement too. New files should leave this compatibility input at zero.
The zero-dimensional point-cloud workflow retains its separate, unchanged
`RefinementParameters<0>` and its original `Refinement factor` semantics;
there is no second mesh to co-refine there.

## Fields, ids, and parallel execution

Input point fields are represented by `FE_Q(1)` and interpolated onto children.
Input cell fields use `FE_DGQ(0)` and retain their parent's value on each child,
including jumps between adjacent cells. All catalog entries and vector
components are retained. Material, boundary, and manifold ids follow deal.II
refinement and mesh copying; manifold objects are cloned onto the final mesh.

The imported fields must be enabled in **each** domain whose data should be
retained. Boundary and material id selections are independent of field
retention. The original input file is never reread onto the refined final mesh.
There is no averaging of different cell-property values during finalization.

The embedded preparation mesh is replicated serially on every MPI rank. When
space is distributed, ranks query their locally owned space cells and union
embedded refinement flags across ranks before refining. Synchronizing only a
stopping condition would produce different embedded meshes on different ranks;
the utility synchronizes the actual cell flags too.

For a `fullydistributed` domain, mesh construction and **all** refinement stages
remain serial and replicated until finalization. This also applies to
one-dimensional domains, for which `Domain` uses the fullydistributed backend.
It requires memory for the complete preparation mesh on each rank.

A serial preparation mesh destined for `parallel::distributed` is finalized by
reconstructing its coarse mesh and replaying its hierarchy: deal.II's p4est
backend cannot directly copy a refined serial triangulation. Fields are
redistributed only after the matching final hierarchy exists.

Existing raw-triangulation coupling code calls `adjust_grids()`, which delegates
to the same adjustment algorithm. Its transfer hooks must already be attached;
it cannot adjust an already finalized fullydistributed space mesh. New
application code should use the domain-level utility and its delayed
finalization rather than that compatibility entry point.

## Use `both` in elasticity and reduced Poisson

The current `elasticity` (with `Coupling type = TensorProduct`) and
`reduced_poisson` applications both use `ReducedCoupling`. Its preprocessing
hook runs the same adjustment algorithm before application DoFs and particle
search structures are built. The representative mesh remains serial until
all refinement passes finish; its imported properties are transferred on each
pass and then redistributed with the final mesh. No separate implementation
of `both` is needed in either solver.

In both application parameter files, select:

```text
subsection Tensor product coupling
  subsection Local refinement parameters
    set Refinement strategy = both
    set Lower factor = 0.5
    set Upper factor = 2
    set Refinement factor = 0
    set Max refinement level = 6
  end
end
```

Set the deprecated `Refinement factor` to zero when adapting an old file that
explicitly sets it to one. These settings control initial co-refinement, not
the later physics-driven `Refinement and remeshing / Strategy`. The current
raw-triangulation application path requires a distributed background mesh;
for elasticity, select `Triangulation type = distributed`. Delaying a
fullydistributed background mesh requires the domain-level API above.
The fullydistributed representative mesh is already partitioned after
co-refinement in both applications. Zero-dimensional point inputs have no
second mesh and do not support `both`.

Two executable examples use the same input with two unequal segments, an
affine point field `height`, and a discontinuous cell field `radius`. They use
`Thickness = radius` and nonzero reduced data derived from `height`:

```bash
cmake --build build-debug --target elasticity_debug reduced_poisson_debug -j
build-debug/elasticity_debug \
  build-debug/gtests/parameters/elasticity_co_refinement_both.prm
build-debug/reduced_poisson_debug \
  build-debug/gtests/parameters/reduced_poisson_co_refinement_both.prm
```

`LocalCoRefinement.BOTH_ElasticityApplication` and
`LocalCoRefinement.BOTH_ReducedPoissonApplication` load these generated input
files and exercise setup, assembly, and a nonzero solve in serial and MPI.
They require refinement of both meshes, not just acceptance of the parameter.
The elasticity test also checks the final diameter bounds and the imported
point/cell values independently of the refinement algorithm.

## Examples from a test

Every image below is an output of
`LocalCoRefinement.DocumentationExamples` in
`gtests/local_co_refinement_01.cc`. The test constructs the same random binary
tree geometry with `std::mt19937` seed **258**, runs the utility for each
configuration, compares its report with an independent all-pairs oracle, and
writes final meshes in VTK, images in SVG, and counts in `summary.csv`.
The figures are not made by a separate Python refinement implementation.

The box is a `Triangulation<2, 2>` of `[0, 1]^2`. The green branches form the
geometry of a `Triangulation<1, 2>` with 30 initial segments. deal.II's
codimension-one orientation checks reject the triple junctions of this tree.
The example therefore gives geometrically coincident junction endpoints
**distinct vertex indices**. It tests geometric co-refinement, not continuity
or flow conservation at a degree-three junction. The test constructs that mesh
directly in the embedded domain's serial preparation phase, without passing
its coincident endpoints through a mesh reader's cleanup step.

Gray lines show space-cell boundaries; green circles show embedded vertices.
Both domain `Initial refinement` counts are zero. Both final backends are
fullydistributed. The examples use the following explicit settings:

| Example | Strategy | Lower / upper | Pre space / embedded | Post space / embedded | Cap |
| --- | --- | --- | --- | --- | --- |
| Space | `space` | 0.5 / 2 | 1 / 0 | 0 / 0 | 7 |
| Embedded | `embedded` | 0.5 / 2 | 4 / 0 | 0 / 0 | 7 |
| Both | `both` | 0.5 / 2 | 2 / 0 | 0 / 0 | 7 |
| Wider interval | `both` | 0.2 / 3 | 2 / 0 | 0 / 0 | 7 |
| Pre/post cycles | `both` | 0.5 / 2 | 2 / 1 | 1 / 1 | 7 |
| Capped | `both` | 0.5 / 2 | 0 / 2 | 0 / 0 | 1 |

### Space and embedded

```{figure} ../_static/local-co-refinement/space.svg
:alt: Space-only adjustment around the random tree
:width: 65%

Space: local refinement concentrates around branches. Embedded has no
ratio-driven refinement, so upper-bound violations remain where its segments
intersect small space cells.
```

```{figure} ../_static/local-co-refinement/embedded.svg
:alt: Embedded-only adjustment inside a uniformly refined box
:width: 65%

Embedded: the four space pre-refinement cycles stay uniform. Long branches
are split where required by the upper bound. Lower-bound violations remain
near short segments because space cannot be corrected by this strategy.
```

### Both and a wider interval

```{figure} ../_static/local-co-refinement/both.svg
:alt: Alternating adjustment of the box and tree
:width: 65%

Both: space passes run before embedded passes, and each embedded pass sees
the updated space sizes.
```

```{figure} ../_static/local-co-refinement/both-wide.svg
:alt: Both strategy with relaxed diameter bounds
:width: 65%

Wider interval: the same input and pre-refinement with lower 0.2 and upper 3
permit a larger difference between the two mesh sizes.
```

### Pre/post cycles and a limiting cap

```{figure} ../_static/local-co-refinement/both-post.svg
:alt: Both strategy with extra pre- and post-refinement cycles
:width: 65%

Pre/post cycles: embedded receives one global pre pass; space receives one
local post pass; embedded receives one global post pass. Ratio corrections
run after the post stages too.
```

```{figure} ../_static/local-co-refinement/both-capped.svg
:alt: Coarse box left with unresolved violations at refinement level one
:width: 65%

Capped: embedded pre-refinement produces small segments but space cannot
refine beyond level one in ratio-driven passes. The report retains the
unsatisfied lower-bound pairs instead of claiming success.
```

The exact cell and violation counts are published directly from the test:

```{csv-table}
:file: ../_static/local-co-refinement/summary.csv
:header-rows: 1
```

### Regenerate the published figures

From the repository root:

```bash
cmake --build build-debug --target gtests_debug -j
GTEST_FILTER='LocalCoRefinement.DocumentationExamples' \
  "$PWD/build-debug/gtests/gtests_debug"
python3 scripts/update_local_co_refinement_figures.py \
  build-debug/test_output/local-co-refinement/examples
./scripts/build_doc.sh
```

The test writes only to `TEST_OUTPUT_DIR` in the build tree. The update script
copies just its six SVG figures and its CSV into the documentation assets;
the VTK meshes remain in the build tree for inspection. To check that published
assets still match fresh test outputs without modifying them, add `--check` to
the update-script command. Run the test before publishing figures; do not use
outputs from a failed run.
