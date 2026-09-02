# Semidiscrete contributors

Application authors compose standalone Problems and Interactions through an
execution adapter. The adapter owns storage, execution blocks, DAE metadata,
and solver policy. The public composition vocabulary is deliberately small:
add a Problem, observe a quantity, lift it to a user-described support, couple
it to another Problem, and solve.

## Application authors

The returned value is a semantic Problem handle. It keeps the contributor's
Field identifiers and a non-owning view of the adapter; it does not duplicate
Problem state or execution storage. A typical coupled workflow is:

```{code-block} cpp
IDAParameters<GlobalVector> ida_parameters;
IDAAdapter<FieldVector, GlobalVector> ida(ida_parameters, MPI_COMM_WORLD);
auto fluid = ida.add(flow_problem, "fluid");
auto wall  = ida.add(elastic_problem, "wall");

auto pressure = fluid.observe(Pressure{});
auto wall_pressure = pressure.lift(VesselSurface{});
ida.couple(wall_pressure, wall, Traction{});

auto state     = ida.make_state();
auto state_dot = ida.make_state();
ida.solve(state, state_dot);
```

`Pressure`, `VesselSurface`, and `Traction` are small descriptors supplied by
the relevant physics modules. ImmersX core does not need to know what those
quantities mean. The module interprets them and creates the appropriate
observable, lifting, and interaction implementation.

For an affine steady problem, use `LinearAdapter`:

```{code-block} cpp
using Adapter = LinearAdapter<LA::MPI::Vector, LA::MPI::BlockVector>;
LinearAdapterParameters linear_parameters;
Adapter linear(linear_parameters, MPI_COMM_WORLD);
auto bulk = linear.add(bulk_problem, "bulk");
auto state = linear.make_state();
linear.solve(state);
bulk_problem.set_solution(linear.field(state, bulk.fields().solution));
bulk_problem.output_results();
```

The execution adapter owns the coupled solver state, while each Problem owns
its accepted physical state and native output. After a successful solve,
applications transfer each physical Field back to its owning Problem before
calling that Problem's output method. An Interaction owns any auxiliary Fields
it introduces, such as Lagrange multipliers, and provides their native output;
the execution adapter does not own visualization.

For example, a composed Poisson/elasticity solve keeps the two physical output
paths explicit:

```{code-block} cpp
auto state = adapter.make_state();
adapter.solve(state);

poisson_problem.set_solution(
  adapter.field(state, poisson.fields().solution));
elasticity_problem.set_solution(
  adapter.field(state, elastic.fields().displacement));

poisson_problem.output_results();
elasticity_problem.output_results(0);
```

For a Lagrange-multiplier Interaction, the accepted-state handoff is explicit
and uses the semantic multiplier Field returned by the Interaction contributor:

```{code-block} cpp
auto state = adapter.make_state();
adapter.solve(state);

poisson_problem.set_solution(
  adapter.field(state, poisson.fields().solution));
interaction.set_multiplier(
  adapter.field(state, continuity.fields().multiplier));

poisson_problem.output_results();
interaction.output_results(output_directory, "multiplier", 0);
```

The scalar and vector Lagrange-multiplier Interactions place this auxiliary
field on the second representation's finite-element space. Problems output
Problem-owned fields; Interactions output Interaction-owned fields. No
execution block number is part of this handoff.

The standard adapters select their iterative/direct policy internally. An
expert application may still pass a custom solve callback when it needs full
control of the linear solve.

For a direct contributor-level workflow, Field identifiers are available from
the handle's `fields()` view:

```{code-block} cpp
auto bulk = linear.add(bulk_problem, "bulk");
auto embedded = linear.add(embedded_problem, "embedded");
auto continuity = linear.add(interaction, "continuity",
                             bulk.fields().solution,
                             embedded.fields().solution);
```

The returned `continuity.fields().multiplier` is an algebraic semantic Field.
The application never names its execution block.

## Observables, lifting, and coupling

Application code does not construct observable or coupling implementation
objects. Physics modules expose descriptors and interpret them through the
following customization shapes:

```{code-block} cpp
template <typename ProblemHandle>
auto make_representation(const ProblemHandle &problem, Pressure)
{
  return make_pressure_quantity(problem.fields().pressure);
}

template <typename Quantity>
auto make_lift(const Quantity &quantity, VesselSurface surface)
{
  return make_surface_quantity(quantity, surface);
}

template <typename Quantity, typename ProblemHandle>
auto make_interaction(const Quantity &quantity,
                      const ProblemHandle &wall,
                      Traction traction)
{
  return make_traction_interaction(quantity,
                                   wall.fields().force,
                                   traction);
}
```

These functions are ordinary templates found by ADL. A descriptor may instead
provide a `create(...)` member. No `ObservableBase`, factory, registry, or
inheritance hierarchy is required.

The implementation returned by `make_representation` is a reusable observable
view derived from one or more Fields. It owns no execution block. For example,
physics code may expose a scaled quantity:

```{code-block} cpp
auto pressure = fluid.observe(Pressure{}).scaled(2.);
```

When the sampling geometry is selected by a later lift, a physics module can
return a deferred finite-element expression. It describes the source FE view,
semantic bindings, expression text, and constants, but does not create a
quadrature or retained sampling plan:

```{code-block} cpp
const Pressure descriptor;
auto pressure = make_fe_expression(
  source_view,
  {value(problem.fields().solution, "A")},
  "factor*A",
  {{"factor", descriptor.factor}});
auto lifted_pressure = pressure.lift(pressure_lift);
```

`pressure.lift(pressure_lift)` obtains the representative quadrature from the
lift, samples the expression through the existing
`make_expression_representation` path, and then composes the existing value
transfer. Applications do not provide `UpdateFlags`,
`RetainedSamplingPlan`, or symbolic-kernel objects. For an explicit sampling
conversion, use `sample(expression, quadrature)`.

The following details are useful when implementing a descriptor, but are not
needed by application authors. A representation has a source Field dependency
and a physical evaluation domain. `RepresentationDomain` records dimensions
and a geometry identity; an `EvaluationRequest` supplies points for one
evaluation. `QuantitySpace<ValueType>` carries the typed value and domain.

It does not own a mesh, DoFHandler, quadrature, or target Problem. An
`EvaluationRequest` supplies the points and optional evaluation-policy
identity for one call. Thus the same cylindrical Representation can be
evaluated at several point sets without changing its domain or source
dependency:

```{code-block} cpp
RepresentationDomain surface_domain(2, 3, "cylindrical-surface");
EvaluationRequest request_a(surface_points_a, "surface-points-a");
EvaluationRequest request_b(surface_points_b, "surface-points-b");
auto values_a = surface_quantity.evaluate(context, request_a);
auto values_b = surface_quantity.evaluate(context, request_b);
```

The physical value space is described by `QuantitySpace<ValueType>`. It
combines the typed quantity value with its `RepresentationDomain`, while
owning no mesh, DoFHandler, Problem, residual, or execution storage. A
Representation therefore has a state type and a quantity type; its derivative
is the map $dq/dy$ from state perturbations to quantity perturbations. Those
types are allowed to differ even when the identity and scaled examples happen
to use the same algebraic vector type.

Later representations can select components, evaluate other nonlinear
observables, or provide geometry-dependent lifting without changing the
Problem/Field execution storage.

A geometry lifting remains a Representation. The geometry owns only the map
between target points and source parameters; it does not know vectors or
evaluation point sets. `ValueTransfer` owns the algebraic interpolation, and
the lifting composes the transfer with the source Representation:

```{code-block} cpp
ParametricGeometryMap cylinder_map(
  source_parameters, surface_domain,
  [](const dealii::Point<3> &point) { return point[0]; });
EvaluationRequest surface_request(surface_points, "surface-points");
auto surface_quantity =
  lift(line_quantity, cylinder_map, target_prototype, surface_request);
```

`surface_quantity.source()` is the same Field as `line_quantity.source()`,
while `surface_quantity.domain()` is the cylindrical surface domain. Its
geometry map is independent of its vector-valued transfer. Evaluation and
linearization apply the requested transfer and then the source Representation;
no Field, execution block, or Problem is created.

`ValueTransfer` maps quantity values to quantity values. It is distinct from a
`CouplingOperator`, which applies the weak form from quantity values into a
residual/test space. The latter is the boundary where integration and target
residual storage enter.

For a mixed Field, a component view selects an `IndexSet` in the existing Field
vector. The view and its Representation remain non-owning; evaluating them
does not create a compacted vector or a new execution block.

## CouplingOperator boundary

A `CouplingOperator` maps evaluated quantity values into a target residual
space. It owns the target-space action and its reinitialization, but it does
not know the source Field or Representation:

```{code-block} cpp
CouplingSpace<Vector> target_space(target_prototype);
CouplingOperator<Vector, Vector> coupling(weak_form, target_space);
```

An Interaction composes the two derivatives when registering a term:

$$
\frac{dR_{target}}{dy_{source}} =
\frac{dR_{target}}{dq}\frac{dq}{dy_{source}}.
$$

The Interaction names the target row and registers the residual; weak-form
actions stay in the CouplingOperator, while Representation supplies the
quantity and its source linearization. The quantity and target vector types
are separate so a later lifting can connect different spaces.

## Contributor authors

A Problem contributor declares semantic Fields and contributes residual,
`dF/dy`, and `dF/dydot` terms. A contributor is selected by the adapter through
an ADL customization point such as:

```{code-block} cpp
template <typename Builder>
Fields contribute(Builder &builder, const MyProblem &problem)
{
  auto temperature = builder.differential_field("temperature", owned, relevant);
  auto mass = payload_free(linear_operator<Vector, Vector>(problem.mass()));
  auto stiffness =
    payload_free(linear_operator<Vector, Vector>(problem.stiffness()));

  builder.term(temperature, "heat")
    .residual([temperature, mass, stiffness](const auto &ctx) {
      return mass * ctx.derivative(temperature) +
             stiffness * ctx.state(temperature);
    })
    .state(temperature, stiffness)
    .derivative(temperature, mass);
  return {temperature};
}
```

The scoped builder supplies the contributor prefix. The term handle keeps one
residual and its two Jacobian parts under one semantic name, so term selection
cannot retain an inconsistent linearization. `PackagedOperation`s are applied
immediately and must not retain temporary evaluation state.

The general `builder.field(name, owned, relevant, differential_components)`
form accepts any differential-component `IndexSet` with the field vector's
global size. Use `differential_field()` for a fully differential field and
`algebraic_field()` for a field with no differential components. A mixed field
uses the general form; the execution adapter consumes the mask without knowing
how its components are organized.

Interactions add terms because systems are related. A Lagrange-multiplier
Interaction contributes participant multiplier forces and the constraint row:

```text
R_first  = C lambda
R_second = -Q^T lambda
R_lambda = C^T first - Q second
```

Scalar and vector interactions translate their shared `ConstraintEquation`
through the same generic semantic mechanism. The multiplier is algebraic and
contributors never receive IDA's `alpha`.

A non-constraint Interaction can read an observable and add a load directly to
an existing Problem-owned row. Application code uses the same descriptor API:

```{code-block} cpp
auto pressure = fluid.observe(Pressure{});
auto wall_pressure = pressure.lift(VesselSurface{});
adapter.couple(wall_pressure, solid, Traction{});
```

The physics descriptor implementation contributes both the load residual and
its `dF/dy` operator. The adapter still owns the global block layout; the
interaction only names the semantic target row and the observable-to-force
operator.

## Adapter distinction and lifetime

`LinearAdapter` requires an affine steady residual and rejects derivative
terms. `IDAAdapter` evaluates the canonical residual

$$F(t,y,\dot y)=0$$

and alone forms

$$J=dF/dy+\alpha dF/d\dot y.$$

State-dependent Jacobian factories are evaluated with stable snapshots held by
the prepared global operator. Problems and Interactions captured by a
contributor must outlive the adapter. Future KINSOL and ARKode adapters can
reuse the same contributor contract.
