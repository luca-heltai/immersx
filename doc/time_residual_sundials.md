# Time residual and SUNDIALS integration

The time-residual path now uses the canonical Field/State core. Semantic
contributors receive `ImmersX::EvaluationContext` and add rows through
`ImmersX::ResidualAccumulator`; Jacobian contributors use
`ImmersX::LinearizationContext` and the same semantic field accessors.
The generic composer for `F(t,y,ydot)=0` is
`ImmersX::SemiDiscreteModel<VectorType>`, which is also suitable for steady
evaluations when no state derivative is supplied.

`ImmersX::FieldId` identifies state and residual rows. Independent timelines
are identified separately by `ImmersX::HistoryGroupId`, so several fields may
share one history grid. The semantic core is now used by real distributed
Elastodynamics and unsteady-Stokes contributors.

The production-intended IDA instantiation is

```cpp
SundialsIDAResidualAdapter<LA::MPI::Vector, LA::MPI::BlockVector>
```

Its adapter-local `BlockFieldLayout<FieldVectorType, GlobalBlockVectorType>`
maps one IDA block directly to one semantic Field. Binding a block to a
`StateView` or `ResidualAccumulator` therefore creates no scalar gather or
scatter. The mapping is an execution/storage choice: `FieldId` is not a
global block number, and native Problem block ordering remains local to the
Problem. The older `ImmersX::detail::MonolithicFieldLayout` remains only for
serial synthetic compatibility tests.

## Local deal.II/SUNDIALS contracts

The local build uses deal.II 9.8.0-rc1 with `DEAL_II_WITH_SUNDIALS` enabled and
SUNDIALS 6.7.0.  The installed deal.II wrappers expose `SUNDIALS::IDA`,
`SUNDIALS::KINSOL`, and `SUNDIALS::ARKode`/`ARKStepper`.

IDA exposes the first-order residual contract directly:

```text
residual(t, y, ydot, residual)
setup_jacobian(t, y, ydot, alpha)
solve_with_jacobian(rhs, dst, tolerance)
differential_components() -> IndexSet
```

The Jacobian required by IDA is

```text
J = dF/dy + alpha * dF/dydot.
```

The differential-component callback is optional in deal.II and defaults to a
complete `IndexSet`; a DAE adapter must provide the locally owned differential
indices when algebraic variables are present.  The wrapper also accepts a
vector reinitialization callback and optional restart/output callbacks.

KINSOL exposes the time-independent nonlinear callbacks

```text
residual(current_u, residual)
setup_jacobian(current_u, current_residual)
solve_with_jacobian(rhs, dst, tolerance)
```

along with vector reinitialization, scaling, and optional custom setup.  It is
therefore compatible with the same residual contributor and action types after
the time terms have been removed or frozen by the caller.

ARKode uses an `ARKStepper` with additive callbacks

```text
explicit_function(t, y, fE)
implicit_function(t, y, fI)
mass_times_vector(t, v, Mv)
jacobian_times_vector(v, Jv, t, y, fI)
solve_linearized_system(op, preconditioner, x, rhs, tolerance)
```

The linearized operator supplied to the custom solve path represents
`M - gamma*J`.  The wrapper also provides setup and preconditioner callbacks.
The installed deal.II wrapper does not define a dedicated `MRIStep` C++ class,
but its `ARKodeStepper` interface is explicitly extensible; the underlying
SUNDIALS installation provides `arkode/arkode_mristep.h`, including
`MRIStepCreate`, inner-stepper callbacks, pre/post-inner hooks, and
`MRIStepSetJacTimes`.  A future MRIStep adapter should therefore be a thin
stepper wrapper using the same history query rather than a second residual
model.

## Mapping and Jacobian actions

`ImmersX::JacobianAction<VectorType>` is the universal `vmult` capability. Existing
deal.II matrices and block matrices can be wrapped with
`ImmersX::JacobianAction::from_matrix()` or `from_linear_operator()`. These wrappers
retain the native operator as an optional capability but do not attempt to
convert a generic matrix-free action back into a sparse matrix.  A
backend-specific Trilinos/PETSc `LinearOperator` payload can be passed to the
templated `from_linear_operator()` overload; the action erases that payload at
the contributor boundary.  The matrix wrapper is non-owning, matching
deal.II's `linear_operator(matrix)` convention; the problem that owns the
matrix must outlive the action.

`SundialsIDAResidualAdapter` only translates the deal.II IDA callbacks. The
residual model supplies physics, the semantic model supplies field-wise `Jv`,
and a caller-supplied linear solve policy consumes the block-vector
`ImmersX::JacobianAction`. The differential mask is built from each
`FieldDescriptor::TimeRole` and the adapter-local block ownership/index sets.

The distributed tests exercise two MPI ranks with one block per Field:

```text
Elastodynamics: solid.displacement (differential), solid.velocity (differential)
Stokes:         fluid.velocity (differential), fluid.pressure (algebraic)
```

The Elastodynamics contributor evaluates `M*d_dot-M*v` and
`M*v_dot+K*d+D*v-f(t)` from external state views. The Stokes contributor uses
`rho*M_u*u_dot + C_uu*u + C_up*p - rho*f(t)` and `C_pu*u`, where
`C_up=continuous_operator.block(0,1)` and
`C_pu=continuous_operator.block(1,0)` preserve the current weak-form signs
(both pressure/divergence blocks are negative when `B` denotes the positive
divergence coupling). The pressure metric block retained by the native
Navier--Stokes preconditioner is not a time derivative term and is excluded
from the semantic residual and differential mask. The tests compare residual
and Jacobian actions with the native distributed operators and run short IDA
solves. Their small linear solve policy uses FGMRES with an identity
preconditioner; this is a validation strategy, not a performance claim.

The current real IDA gate is unsteady Stokes
(`include_convective_term=false`). A fully implicit convection Jacobian and an
ARKode/IMEX execution path remain follow-up work.
