# Time residual and SUNDIALS integration

The time-residual path uses the canonical Field/State core. Semantic
contributors receive `ImmersX::EvaluationContext` and register residual
`dealii::PackagedOperation`s plus separate `dealii::LinearOperator` blocks for
`dF/dy` and `dF/dydot`. The execution adapters share an internal semantic
model for `F(t,y,ydot)=0`; its residual operations are applied immediately and
are not retained after evaluation.

`ImmersX::FieldId` identifies state and residual rows. Independent timelines
are identified separately by `ImmersX::HistoryGroupId`, so several fields may
share one history grid. The semantic core is now used by real distributed
Elastodynamics and unsteady-Stokes contributors.

The public IDA composition API is

```cpp
IDAAdapter<LA::MPI::Vector, LA::MPI::BlockVector> ida(data,
                                                       MPI_COMM_WORLD,
                                                       linear_solve);
auto matrix = ida.add(matrix_problem, "matrix");
auto fiber  = ida.add(fiber_problem, "fiber");
auto coupling = ida.add(interaction, "fiber_coupling",
                        matrix.velocity, fiber.velocity);
auto state     = ida.make_state();
auto state_dot = ida.make_state();
```

The adapter privately maps one automatically assigned IDA block directly to
each semantic Field. Binding a block to a `StateView` therefore creates no
scalar gather or scatter. The mapping is an execution/storage choice:
`FieldId` is not a global block number, and native Problem block ordering
remains local to the Problem. The adapter derives the IDA differential mask by
offsetting and unioning each field's semantic `differential_components` mask.
A mask has the field vector's global size and may describe a complete, empty,
or mixed set of components.

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

along with vector reinitialization, scaling, and optional custom setup.  It can
use the same residual contributor and action types after the time terms have
been removed or frozen by the caller.

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

## Mapping and Jacobian operators

Contributors use deal.II's `LinearOperator` directly. Backend-specific
payloads are erased only at the execution boundary by constructing a standard
payload-free `LinearOperator` whose callbacks capture the native operator.
The matrix wrapper is non-owning, matching deal.II's
`linear_operator(matrix)` convention; the Problem that owns the matrix must
outlive the contributor.

`IDAAdapter` translates the callbacks and constructs each global block as
`dF/dy + alpha*dF/dydot`. The caller-supplied linear solve policy consumes the
resulting standard deal.II `LinearOperator`; contributors never receive IDA's
`alpha`.

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

The current real IDA gates include unsteady Stokes
(`include_convective_term=false`) and a distributed five-field fiber model:
two Elastodynamics contributors and a vector multiplier Interaction provide
four differential fields plus one algebraic multiplier field. The five-field
test uses a short ramp-forced IDA integration with FGMRES and an identity
preconditioner; this validates composition and callback semantics, not
scalability or production preconditioning. A fully implicit convection
Jacobian and an ARKode/IMEX execution path remain follow-up work.

The mixed-field DAE test uses one four-component semantic field with only its
first two components marked differential. It checks the mask received by IDA,
the residual `F = y + ydot`, and the corresponding `dF/dy + alpha*dF/dydot`
Jacobian action.
