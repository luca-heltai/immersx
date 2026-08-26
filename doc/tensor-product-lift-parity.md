# Tensor-product lift parity

This page tracks the numerical and capability parity between the reference
`TensorProductSpace` / `TensorProductRepresentation` path and the modern
parameterized `TensorProductLift` / `TensorProductLiftRepresentation` path.

## Reference implementation policy

`TensorProductSpace` is retained unchanged as the reference path. It will only
be removed after `ReducedPoisson` and `Elasticity` achieve equivalent results
and configuration capabilities on the new path.

The migration matrix below records the current status of every capability
that must reach parity before the reference path can be considered
replaceable. Rows are marked `parity verified`, `partial`, or `unsupported`.

## Migration matrix

| Capability | TensorProductSpace / reference path | New Representation / lift path | Status |
| --- | --- | --- | --- |
| Representative quadrature | `Quadrature type`, `Number of quadrature points`, `Number of quadrature repetitions` on the representative domain | `Representative quadrature` subsection of `TensorProductLiftParameters`; both paths share `detail::make_tensor_product_quadrature` | parity verified |
| Cross-section mesh | `ReferenceCrossSection` (`hyper_ball`, `hyper_sphere`, `hyper_cube`, refinement) | Same `ReferenceCrossSection` class | parity verified |
| Cross-section quadrature | Section `Quadrature type`, `Number of points`, `Number of repetitions` | Same section parameters | parity verified |
| Selected modes | Section `Selected indices` / `Selected modes` | Section `Selected indices` / `Selected modes`; both paths share `detail::transform_representative_point` | parity verified |
| Modal algebraic coefficients | `FESystem(FE_Q, n_selected_basis)` with one independent (representative DoF, mode) coefficient per mode, in deal.II `FESystem` order | Modal source representation exposed through `make_modal_lift`; each slot keeps its own algebraic index | parity verified |
| Constant thickness | Numeric `Thickness` | Numeric `Thickness` | parity verified |
| Symbolic thickness | Imported reduced-grid field expressions (`Thickness = radius`, time-dependent expressions) through `SymbolicFieldEvaluator` | Source-side `evaluate_thickness(point, time, properties)` provider seam; clear error when no provider is installed | partial |
| Imported fields | VTK/legacy input with `FieldCatalog`, `InputFieldBinding`, property interpolation | Not supported by design: the lift never reads VTK and never stores imported fields; sources resolve their own properties | unsupported |
| Distributed source ownership | Repartitioned fully distributed reduced mesh | Source representation owns the distributed mesh; transpose compression is distributed-correct | partial (serial parity tests only) |
| 0D source | Point-cloud / particle representative domains | `TensorProductLiftSupport` accepts `reduced_dim == 0` but the modern path has no 0D test or application yet | partial |
| 1D source | VTK line input (`one_cylinder.vtk`, `mstree_*.vtk`) | Source FE representation on the same 1D embedded line | parity verified |
| Scalar values | Scalar `n_components == 1` | Scalar `n_components == 1` | parity verified |
| Vector values | `n_components > 1` supported | `n_components` parameter exists; no multi-component test on the modern path | partial |
| ParticleCoupling | `ParticleCoupling` search/transfer machinery | Representation-driven lifted quadrature; no particle-based output parity test | partial |
| Coupling matrix | `ReducedCoupling::assemble_coupling_matrix` | Direct assembly from `locally_owned_quadrature_points` of the modal lift | parity verified (single and multi-mode action and transpose-action) |
| Mass matrix | `ReducedCoupling::assemble_coupling_mass_matrix` | No equivalent on the modern path | unsupported |
| Reduced RHS | `ReducedCoupling::assemble_reduced_rhs` | No equivalent on the modern path | unsupported |
| ReducedPoisson | Reference implementation | Not migrated in this PR | unsupported |
| Elasticity pressure coupling | `ElasticityTensorProductCoupling*` tests | `app_coupled_poisson_elasticity` with `pressure.lift(pressure_lift)`; legacy Elasticity itself not migrated | partial |

## Verified parity tests

The following GoogleTests pin the current parity:

- `TensorProductLiftParity.Mode0SingleLineCell` — identical mode-0 geometry,
  weights, total measure, mode values, and algebraic space.
- `TensorProductLiftParity.MultiModeSingleLineCell` — modal DoF count,
  (representative DoF, mode) to algebraic index mapping, lifted
  tensor-product basis values, points, and weights for modes `{0, 1}`.
- `TensorProductLiftParity.ReducedCouplingActionSingleMode` and
  `TensorProductLiftParity.ReducedCouplingActionMultiMode` — the legacy
  `ReducedCoupling` matrix and the modern modal-path coupling matrix have the
  same action `C x` and transpose-action `C^T y`.
- `PressureRepresentation.PointEvaluationDuality` and
  `PressureRepresentation.MPI_PointEvaluationDuality` — the FE point
  evaluation is an exact adjoint pair, including the additive variants.

## Physical versus modal lifting semantics

The lift distinguishes two algebraic meanings and never silently conflates
them:

- **Physical source (CASE A)**: the source is one coefficient set (for
  example a pressure field lifted as constant across the cross-section).
  Only mode 0 is meaningful as a coefficient; requesting more modes on a
  scalar source fails with a clear error.
- **Modal source (CASE B)**: the source field itself owns independent
  coefficients indexed by (representative DoF, mode), for example an RLM
  multiplier `lambda_{i,m}`. The modal source exposes one algebraic slot per
  (local DoF, mode) in deal.II `FESystem` order, and
  `TensorProductLiftRepresentation` built through `make_modal_lift` keeps
  those slots as distinct unknowns, mapping coefficients to
  `N_i(s) phi_m(xi)`.

A lifted physical observable never fabricates new Field storage: CASE B
requires the owning multiplier Field to provide the modal coefficient space.

## Known gaps before ReducedPoisson migration

- Symbolic thickness on the modern path requires a source that owns the
  referenced properties; the app-level Poisson source does not currently
  expose such properties.
- The modern path has no mass-matrix or reduced-RHS assembly yet.
- Distributed multi-rank parity tests for the modal path are not yet present.
- `ReducedPoisson` and the legacy `Elasticity` tensor-product coupling have not
  been migrated; `TensorProductSpace` remains the reference implementation.
