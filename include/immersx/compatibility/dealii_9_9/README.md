# deal.II 9.9 compatibility

## SUNDIALS IDA

`ida.h`, `ida.templates.h`, `n_vector.h`, `n_vector.templates.h`, and
`utilities.h` are snapshots of
[deal.II PR #20260](https://github.com/dealii/dealii/pull/20260),
commit `607797ef9f3df61760000e5c07430567e6ddd955`. The original paths are
`include/deal.II/sundials/{ida.h,n_vector.h,n_vector.templates.h,utilities.h}` and
`source/sundials/ida.cc`. Upstream SPDX and copyright notices are retained.
This is an open-PR snapshot, not a claim that these APIs shipped in 9.9.

Local packaging changes:

- Move declarations into `ImmersX::Compatibility::SUNDIALS` and use unique
  include guards, so installed IDA/KINSOL wrappers can coexist with this code.
- Include the IDA implementation as `ida.templates.h`, without upstream
  explicit instantiations, and make the two non-template N_Vector functions
  inline. Applications can instantiate their own supported vector types.
- Use local N_Vector and callback-exception helpers. Copying `utilities.h`
  avoids introducing `dealii::SUNDIALS::Utilities`, which shadows
  `dealii::Utilities` in installed deal.II 9.7.1 N_Vector templates. Qualify
  the new minimum-quotient MPI reduction explicitly for the same reason.
- Replace `AssertIDA(status)` with the equivalent namespace-local
  `Assert(status >= 0, ExcIDAError(status))`.
- Include IDA/IDAS through the upstream header's existing configuration guard
  rather than unconditionally including IDAS in the implementation.
- Apply the repository formatter.

`IDAAdapter` and `IDAParameters::additional_data()` use this
implementation. The installed
KINSOL implementation is unchanged. Solver controls, constraints, setup hooks,
and accessors follow the pinned upstream implementation and SUNDIALS version
guards. Direct `AdditionalData` defaults follow upstream (including the
`1e-15` minimum step); ImmersX parameter defaults remain unchanged.
See `doc/developer/design/time-residual-sundials.md` for adapter usage.

## DoFHandlerCoupling

`dof_handler_coupling.h` is copied from
[deal.II PR #18888](https://github.com/dealii/dealii/pull/18888), commit
`c061c24bd41b4eb9fb62d1625dc0b818a54bc2bd` (`step-80-new`), file
`include/deal.II/non_matching/dof_handler_coupling.h`. The upstream SPDX license
and copyright notice are retained. This is a snapshot of an open upstream PR,
not an assertion that its API has shipped in deal.II 9.9.

The `create_nitsche_restriction_matrix` assembly loop is unchanged. It sums
`penalty * shape_value(i, reference_q) * shape_value(j, reference_q) * JxW`
only for matching FE components and scatters through `AffineConstraints`.
The benchmark compares it against a separately written explicit particle loop
before using it as the native oracle for ImmersX.

Local backport adjustments:

- Include dependencies used by the header explicitly (FEValues, FullMatrix,
  distributed triangulation and ObserverPointer).
- In the quadrature-particle generator, use the base triangulations rather
  than requiring two `parallel::distributed::Triangulation` casts. Count locally
  owned cells by traversal. This supports serial immersed lines, for which
  p4est triangulations are not available, without changing the weight or basis
  evaluation used by the penalty matrix.
- Request and evaluate normals only for codimension one. Codimension-two
  particle properties retain zero normal placeholders; the penalty matrix
  uses only the physical quadrature weight, not normals.
- Apply the repository's C++ formatter.

The other upstream coupling methods are retained but are not advertised as
verified backports. In particular, interpolation retains upstream distributed
mesh assumptions. The tests exercise only the penalty matrix and quadrature
particle access with primitive scalar/vector Q1 elements, including multiple
MPI ranks. The background is p4est-distributed; integration cells are partitioned
by `parallel::shared::Triangulation`, including 1D geometry. The caller must
`compress(VectorOperation::add)` after the upstream matrix assembly loop; the
kernel itself does not perform that collective.

The declaration is in `dealii::NonMatching`, with the upstream include guard.
Do not include a different version of `DoFHandlerCoupling` in the same translation
unit. Installed applications should choose either this pinned compatibility
header or the version supplied by their deal.II installation, not both.
