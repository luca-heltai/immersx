# Configure reduced coupling

Reduced coupling represents a lower-dimensional geometry in a bulk finite-
element problem. The reduced-coupling section controls the representative
grid, thickness, transverse cross-section basis, quadrature, and particle
search. The [Reduced Poisson tutorial](../tutorials/reduced-poisson) introduces
these choices through a minimal cylinder; use this page when changing the
configuration.

The important distinction is between:

- the reduced or representative dimension, where coefficients are stored;
- the represented physical support, where the lifted field is evaluated; and
- the ambient `space dimension`, where the support is embedded.

Imported point and cell fields can be named in expressions. A variable radius,
for example, is a field in the input mesh and is selected by the `Input file
fields` option before it is used by `Thickness`. Distributed point ownership
and search are handled by the particle-coupling infrastructure.

For joint refinement of the bulk and representative meshes in elasticity and
reduced Poisson, see [local co-refinement](local-co-refinement.md).

## Simplex bulk preparation

`ReducedPoisson` prepares fully distributed bulk meshes on a replicated serial
mesh. Its line and point-cloud refinement settings apply before partitioning
and before particle ownership is constructed. deal.II particle location on
simplex meshes requires conforming vertex neighborhoods, so the application
calls `GridTools::remove_hanging_nodes` with isotropic refinement before
finalizing the Domain. This conformity pass can refine cells outside the
requested neighborhood. Use one solve/refinement cycle with this backend.

A custom Problem using `ReducedCoupling<0, ...>` can set
`prepare_point_cloud_background` to prepare its Domain after point positions
and thicknesses have been read. The callback receives those two arrays and
must finalize the background collectively before returning. Positive-dimensional
representatives use the existing `preprocess_serial_triangulation` hook.

When assembling a custom coupled problem with prescribed bulk values, assemble
the reduced datum first and pass it to the four-argument matrix overload:

```cpp
coupling.assemble_reduced_rhs(reduced_rhs);
coupling.assemble_coupling_matrix(matrix, bulk_dofs, bulk_constraints, reduced_rhs);
```

The matrix has background rows and representative columns. The overload
subtracts its uncondensed transpose action on the prescribed background values
from `reduced_rhs`, then condenses the coupling matrix. The three-argument
overload assembles only the matrix.
