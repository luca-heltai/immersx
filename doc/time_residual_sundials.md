# Time residual and SUNDIALS integration prototype

The time-residual path now uses the canonical Field/State core. Semantic
contributors receive `ImmersX::EvaluationContext` and add rows through
`ImmersX::ResidualAccumulator`; Jacobian contributors use
`ImmersX::LinearizationContext` and the same semantic field accessors.
The generic composer for `F(t,y,ydot)=0` is
`ImmersX::SemiDiscreteModel<VectorType>`, which is also suitable for steady
evaluations when no state derivative is supplied.

`ImmersX::FieldId` identifies state and residual rows. Independent timelines
are identified separately by `ImmersX::HistoryGroupId`, so several fields may
share one history grid. IDA's monolithic vector is an adapter concern handled
by the internal `ImmersX::detail::MonolithicFieldLayout`, while native Problem
block numbers remain local to the public `ImmersX::NativeFieldLayout`.

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

## Prototype mapping

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

`SundialsIDAResidualAdapter` only translates the deal.II IDA callbacks.  The
residual model supplies physics, the semantic model supplies field-wise `Jv`,
and a caller-supplied linear solve policy consumes the gathered monolithic
`ImmersX::JacobianAction`. The synthetic tests use GMRES on the matrix-free
`LinearOperator` view and validate the differential/algebraic mask and the
resulting DAE solution, including a mixed native-block Problem and an
auxiliary algebraic multiplier field.
