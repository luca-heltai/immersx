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
TimeIntervalParameters time_interval;
IDAParameters ida_parameters;
IDAAdapter<FieldVector, GlobalVector> ida(time_interval,
                                         ida_parameters,
                                         MPI_COMM_WORLD);
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

`TimeIntervalParameters` owns the initial and final times and the physical
output interval. `IDAParameters` owns IDA's adaptive-step controls, error
tolerances, differential/algebraic error handling, and initial-condition
correction settings. The adapter combines the two into deal.II's IDA
`AdditionalData`; the generic fixed-step controls are not used by IDA.

The adapter uses the pinned compatibility implementation of
[deal.II PR #20260](https://github.com/dealii/dealii/pull/20260), not the
installed IDA wrapper. Its type is
`ImmersX::Compatibility::SUNDIALS::IDA<GlobalVector>`. Existing parameter-file
settings and defaults are unchanged. To set the additional upstream controls,
copy the assembled configuration before calling `solver()` or `solve()`:

```cpp
auto controls = ida.additional_data();
controls.maximum_step_size = 0.01;
controls.maximum_number_of_steps = 2000;
controls.linear_tolerance_factor = 0.05;
ida.set_additional_data(controls);
```

Both `IDAParameters` and the compatibility
`AdditionalData::add_parameters()` expose all backported solver controls.
The adapter keeps the simulation interval in `TimeIntervalParameters`.
The `Running parameters` subsection includes minimum/maximum step sizes,
the internal-step limit per output, and error/convergence failure limits.
The `Nonlinear solver`, `Linear solver`, `Step adaptivity`, and
`Initial condition correction parameters` subsections expose the remaining
Newton, linear solve, step-growth, and initial-correction controls.
Adaptivity controls require SUNDIALS 6.2 or newer; the constraint-failure limit
requires 7.6 or newer.

After registering contributors, `ida.solver()` exposes the constraint and
expert setup callbacks. For inequality constraints, create an execution vector
with `ida.make_state()`, fill it through `ida.field()`, and return it from
`ida.solver().get_constraint_vector`. Entries encode unconstrained (0),
nonnegative/nonpositive (+1/-1), or strictly positive/negative (+2/-2) values.
Keep the vector alive through the solve and resize it after any layout change.
An all-zero constraint vector means no constraints.

`ida.solver().custom_setup` runs on every reset, after the wrapper installs its
callbacks and before initial-condition correction. `get_ida_memory()` permits
IDA queries; it is null before the first reset. Both that pointer and
`get_sun_context()` (SUNDIALS 6+) are invalidated by reset. Do not replace
wrapper-owned user data or callbacks.

The optional
`set_compute_consistent_initial_conditions(time, state, state_dot)` callback
can update both vectors when the correction type is `use_y_diff`. The optional
`set_solver_should_restart()` callback can transfer state after an application
change and request an IDA restart. These callbacks operate on execution
vectors; updating Problem-owned matrices, constraints, and accepted state is
the application's responsibility.

### Local absolute tolerances and startup ramp

The default error control uses one absolute tolerance and one relative
tolerance. Assign an absolute tolerance to a semantic Field before solver
initialization when its physical scale differs from the other Fields:

```cpp
ida.set_absolute_tolerance(fields.fields().displacement, 1.e-8);
ida.set_local_tolerances(fields.fields().velocity,
  [&problem](FieldVector &absolute) {
    problem.fill_velocity_absolute_tolerances(absolute);
  });
```

`fill_velocity_absolute_tolerances` is an application callback in this example.
It writes positive, finite absolute tolerances on locally owned entries. It
may use reference geometry or physical coefficients owned by the Problem.
The adapter calls it on each initialization and restart, with the current
Field layout. Fields without an override use the scalar absolute tolerance.
Relative tolerance remains scalar. Do not use execution block indices to
assign physical tolerances.

The optional startup ramp relaxes the configured tolerances, holds them for a
specified duration, then tightens them geometrically. The following example
uses the standard `/IDA/` subsection:

```text
subsection Time interval
  set Initial time = 0
  set Final time = 1
  set Output time interval = 0.01
end
subsection IDA
  set Print current statistics = true
  subsection Error control
    set Absolute error tolerance = 1e-8
    set Relative error tolerance = 1e-6
  end
  subsection Running parameters
    set Maximum step size = 0.01
    set Maximum number of steps = 2000
  end
  subsection Tolerance ramp
    set Enabled = true
    set Initial relative factor = 100
    set Initial absolute factor = 10
    set Hold duration = 0.05
    set End time = 0.2
  end
end
```

`Hold duration` is measured from the simulation's initial time; `End time`
is an absolute physical time. The end must be after the hold, no later than
the final time, and on the output grid configured by `TimeIntervalParameters`.
Both initial factors must be finite and at least one. Disabled ramps leave
scalar error control unchanged unless Field tolerances are configured.

The inverse error weights are

```{math}
w_i(t,y) = \frac{1}{s_a(t)\,a_i+s_r(t)\,r\,|y_i|},
```

where $a_i$ is the Field-local or scalar absolute tolerance, $r$ is the scalar
relative tolerance, and $s_a,s_r$ are the ramp multipliers. Weights use IDA's
internal time. Output values can be interpolated at earlier times, so output
time and internal time need not coincide.

Changing weights does not restart IDA or discard its BDF history. The adapter
sets a stop at the ramp end, then restores the final stop time, including
when no application output callback is installed. After a genuine application
restart the ramp continues on the original physical timeline.

The adapter installs its weight and setup callbacks when local tolerances or
the ramp are enabled. Use `IDAAdapter::solve()` and keep these callbacks
intact; replacing them through `solver()` bypasses adapter bookkeeping.
No wrapper-owned userdata is replaced. Callback exceptions are captured before
returning through the SUNDIALS C interface and rethrown in C++.

### Statistics and failure handling

`ida.statistics()` returns internal steps, residual evaluations, nonlinear
iterations, Jacobian setups, linear solve calls, error/convergence failures,
restart count, weight evaluations, current order, last step size, internal
time, and current ramp multipliers. Work counters accumulate across
adapter-managed restarts. Statistics are also maintained without application
output. `Print current statistics` prints progress at every output callback;
its default is false.

`solve()` returns the accumulated internal step count. Output callback counts
are different: IDA can interpolate several output states from its internal
history. The first callback uses the configured initial time, including when
it is nonzero. Native setup and integration failures throw in both Debug and
Release builds. Empty MPI ranks participate in weight validation and vector
minimum reductions.

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
