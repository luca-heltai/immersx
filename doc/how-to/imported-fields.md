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

The imported storage is immutable after construction and can be shared by
multiple consumers with `std::shared_ptr<const ImportedFiniteElementFields<...>>`.
The generic storage is usable by any Problem; VTK material or geometric data is
only one use case.
