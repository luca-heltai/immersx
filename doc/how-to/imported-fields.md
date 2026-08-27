# Imported finite-element fields

An imported field is a finite-element field whose coefficients originate
outside the current state vector. VTK PointData is represented with a
continuous `FE_Q` field and VTK CellData with a discontinuous `FE_DGQ` field.
Both are stored on a separate DoFHandler associated with the Problem mesh.

```cpp
auto imported = std::make_shared<
  ImmersX::ImportedFiniteElementFields<1, 3>>(
    vtk_filename, poisson.triangulation());
poisson.set_imported_fields(imported);

const auto path_length = imported->field("path_distance");
```

`field(name, component)` returns a scalar component view without copying the
coefficient vector. The view exposes its finite-element representation and
frozen coefficients for value or gradient sampling. The target triangulation
must already describe the same mesh as the VTK file; interpolation or remote
point evaluation between different meshes is deliberately not implicit.

The imported storage is immutable from the application’s point of view and can
be shared by multiple consumers with
`std::shared_ptr<const ImportedFiniteElementFields<...>>`. On a
`parallel::distributed::Triangulation`, the imported DoFHandler and
coefficients are automatically transferred across refinement and coarsening
through deal.II’s `SolutionTransfer`. Existing `FieldView` objects therefore
remain valid and observe the current mesh and coefficient vector after the
operation; the coefficient-vector object itself is retained so shared
consumers continue to refer to the same storage. The transfer is synchronized
by RAII-managed triangulation signal connections.

Refinement support is intentionally limited to
`parallel::distributed::Triangulation`. Fully distributed triangulations do
not currently implement refinement, and imported fields report this as an
unsupported operation rather than attempting a partial transfer. A refinement
callback must also complete before another refinement callback begins; this
guards against stale or overlapping transfer state.

The object is bound to exactly one `Triangulation`: sharing one imported
instance requires all consumers to use that same triangulation object. Problems
with distinct triangulations create distinct imported-field instances, even
when their meshes are geometrically equivalent.

Each `FieldView` retains the complete shared imported storage, including the
DoFHandler, finite element, index sets, constraints, catalog, and coefficient
vector. It is therefore safe to keep and sample a view after the parent
`ImportedFiniteElementFields` handle has been destroyed. The generic storage is
usable by any Problem; VTK material or geometric data is only one use case.

For example, both Poisson and Elasticity expose the same attachment contract:

```cpp
poisson.set_imported_fields(imported);
elasticity.set_imported_fields(imported);

const auto path_length = imported->field("path_length");
const auto lambda      = imported->field("lambda");
```

The view’s `representation()` is reconstructed from the current imported
DoFHandler, index sets, and constraints. Rebuild any retained sampling plan or
expression representation after mesh refinement; those objects intentionally
capture the sampling geometry at construction time.

PointData and CellData continue to use the existing VTK finite-element choice:
continuous `FE_Q` for PointData and discontinuous `FE_DGQ` for CellData. Field
metadata remains available through `catalog()`.

## Expressions

Expression bindings distinguish a live state source from frozen coefficients:

```cpp
const auto source = ImmersX::FiniteElementRepresentation<1, 3>(
  poisson.triangulation(), poisson.dof_handler(),
  poisson.locally_owned_dofs(), poisson.locally_relevant_dofs(),
  poisson.constraints());
const auto u = ImmersX::state_field(source, solution_field);
const auto path = ImmersX::frozen_field(path_length);

const auto pressure = ImmersX::make_fe_expression(
  source,
  {ImmersX::gradient(u, "grad_u_z", 2),
   ImmersX::value(path, "path_length")},
  "alpha * (grad_u_z + path_length)",
  {{"alpha", 2.0}});
```

Only `state_field` bindings appear in `dependencies()` and contribute to
`linearize()`. A frozen binding is evaluated from its shared coefficient
storage, including gradients, and contributes no Jacobian term.
