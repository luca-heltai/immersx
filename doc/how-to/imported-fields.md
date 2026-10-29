# Imported finite-element fields

`ImportedFiniteElementFields` stores finite-element coefficients supplied by an
external file or producer. It owns the imported FE space and coefficient
vectors; application code consumes the data as ordinary `Field` objects and
dependency-free frozen observables.

```cpp
auto imported = std::make_shared<
  ImmersX::ImportedFiniteElementFields<1, 3>>(
    vtk_filename, poisson.triangulation());

const auto path = imported->field("path_length");
const auto frozen_path = ImmersX::frozen(
  path.field(), imported->coefficients("path_length"));
```

The imported space is separate from the execution adapter. It owns its
DoFHandler, finite element, constraints, locally relevant indices, and
coefficients, but it does not own a Problem, solver vector, or execution block.
`FiniteElementSpaceView` remains a non-owning view, so the same FE-space vocabulary is
used for native and imported data.

On quad/hex meshes, PointData uses continuous `FE_Q(1)` elements and CellData
uses discontinuous `FE_DGQ(0)` elements. On triangle/tetrahedron meshes, the
corresponding elements are `FE_SimplexP(1)` and `FE_SimplexDGP(0)`. The mesh
must contain a single reference-cell type. Initial refinement transfers both
point and cell fields before a fully distributed Domain is finalized.
Imported coefficients are transferred across supported distributed refinement through deal.II's `SolutionTransfer`. Consumers must
use the triangulation associated with the imported space; interpolation or
remote point evaluation is not implicit.

The resulting frozen observable has no active dependencies, but it supports
the same deal.II FE operations as an active observable:

```cpp
auto frozen_path_gradient = ImmersX::gradient(frozen_path);
```

For symbolic combinations, compose the typed observables and the existing
symbolic evaluator at the application boundary. The public model remains
`Field` → `Observable` → `weak_term`; imported storage is an implementation
detail of the data source, not a second representation hierarchy.
