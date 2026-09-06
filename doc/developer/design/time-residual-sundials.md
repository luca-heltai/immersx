# Residual execution with IDA and KINSOL

The semantic execution model exposes a residual and its two state
derivatives. A contributor adds residual operations to the shared
`SemiDiscreteModel` and registers separate operators for
`dF/dy` and `dF/dydot`. The execution adapter evaluates these operations for
the supplied state; the evaluation context and its state views are not stored
after the evaluation.

## IDA

`IDAAdapter` connects the semantic model to deal.II's SUNDIALS IDA wrapper.
The application registers contributors before solving, creates a state and a
state derivative, and supplies accepted-state output when native data must be
updated:

```cpp
TimeParameters time_parameters;
IDAAdapter<FieldVector, GlobalVector> ida(time_parameters, MPI_COMM_WORLD);
auto fields = ida.add(problem, "solid");

ida.set_output_step(
  [&problem, &ida, fields](double time,
                           const GlobalVector &state,
                           const GlobalVector &state_dot,
                           unsigned int step) {
    problem.accept_state(
      ida.field(state, fields.fields().displacement),
      ida.field(state, fields.fields().velocity),
      time,
      step);
    (void)state_dot;
  });

auto state = ida.make_state();
auto state_dot = ida.make_state();
ida.solve(state, state_dot);
```

The application fills the initial state and derivative before `solve()`. IDA
receives a differential-component mask built from each field descriptor. A
field can be fully differential, fully algebraic, or mixed by component.

IDA asks for the solver Jacobian

```{math}
J = \frac{\partial F}{\partial y}
  + \alpha\frac{\partial F}{\partial \dot y}.
```

`alpha` belongs to IDA and is applied inside `IDAAdapter`. Contributors and
Problems provide the two derivative operators separately. The adapter can
use its built-in GMRES/FGMRES path or an application-supplied linear solve
callback.

`TimeParameters` owns the initial and final times, step policy, output
frequency, IDA running parameters, error tolerances, differential/algebraic
error handling, and initial-condition correction settings. It converts these
values to deal.II's IDA `AdditionalData` through `ida_parameters()`.

The optional
`set_compute_consistent_initial_conditions(time, state, state_dot)` callback
can update both vectors when the correction type is `use_y_diff`. The optional
`set_solver_should_restart()` callback can transfer state after an application
change and request an IDA restart. These callbacks operate on execution
vectors; updating Problem-owned matrices, constraints, and accepted state is
the application's responsibility.

## KINSOL

`KINSOLAdapter` connects the same semantic contributor model to deal.II's
SUNDIALS KINSOL wrapper for a steady nonlinear problem:

```cpp
KINSOLAdapterParameters parameters;
KINSOLAdapter<FieldVector, GlobalVector> kinsol(parameters, MPI_COMM_WORLD);
auto fields = kinsol.add(problem, "steady");
auto state = kinsol.make_state();
kinsol.solve(state);
```

KINSOL requires `F(y)=0`; contributors with derivative terms are rejected.
The adapter prepares the semantic Jacobian for each nonlinear solve and uses
either the configured linear solve callback or its built-in GMRES path.
`KINSOLAdapterParameters` selects `newton` or `linesearch` and sets the
nonlinear stopping and Jacobian-setup controls.

Both adapters use deal.II `LinearOperator` objects for Jacobian actions. A
native matrix can be exposed through `SemidiscreteBuilder::matrix_operator()`;
the Problem that owns the matrix must outlive the contributor.
