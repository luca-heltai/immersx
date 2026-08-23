# Semidiscrete contributors

Application authors compose standalone Problems and Interactions through an
execution adapter. The adapter owns storage, execution blocks, DAE metadata,
and solver policy.

## Application authors

Transient elastodynamics uses direct Problem and Interaction additions:

```{code-block} cpp
IDAAdapter<FieldVector, GlobalVector> ida(data, MPI_COMM_WORLD, solve);
auto matrix = ida.add(matrix_problem, "matrix");
auto fiber  = ida.add(fiber_problem, "fiber");
auto coupling = ida.add(interaction, "fiber_coupling",
                        matrix.velocity, fiber.velocity);

auto state     = ida.make_state();
auto state_dot = ida.make_state();
ida.solve(state, state_dot);
```

For an affine steady problem, use `LinearAdapter`:

```{code-block} cpp
using Adapter = LinearAdapter<LA::MPI::Vector, LA::MPI::BlockVector>;
Adapter linear(MPI_COMM_WORLD, solve_global_operator);
auto bulk = linear.add(bulk_problem, "bulk");
auto state = linear.make_state();
linear.solve(state);
bulk_problem.set_solution(linear.field(state, bulk.solution));
```

Two standalone Poisson Problems and a scalar continuity Interaction use the
same API:

```{code-block} cpp
auto bulk = linear.add(bulk_problem, "bulk");
auto embedded = linear.add(embedded_problem, "embedded");
auto continuity = linear.add(interaction, "continuity",
                             bulk.solution, embedded.solution);
```

The returned `continuity.multiplier` is an algebraic semantic Field. The
application never names its execution block.

## What is a Representation?

A `Field` is a semantic state row owned by a Problem. A `Representation` is a
lightweight observable or lifting derived from one or more Fields. It owns no
execution block and is not added to an adapter as a Problem. The minimal
identity representation is obtained from an adapter with:

```{code-block} cpp
auto poisson = adapter.add(problem, "poisson");
auto temperature = adapter.observe(poisson.solution);
```

During evaluation it returns the source Field state, and its linearization is
the identity operator. Every Representation exposes both `source()` for Field
dependency identity and `domain()` for the physical evaluation domain. These
are deliberately distinct: a Field is not a geometric support. For example,
a scaled observable evaluates $q=2u$ and linearizes to $2I$:

```{code-block} cpp
auto pressure = adapter.observe(fluid.solution).scaled(2.);
```

`RepresentationDomain` records only dimensions and a geometry identity. It
does not own a mesh, DoFHandler, quadrature, or target Problem. An
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

A non-constraint Interaction can read a Representation and add a load directly
to an existing Problem-owned row. For example, a pressure representation can
drive an elasticity force Field without introducing a multiplier or another
execution block:

```{code-block} cpp
auto pressure = adapter.observe(fluid.pressure);
auto traction = PressureLoadInteraction<Vector>(pressure,
                                                solid.force,
                                                pressure_to_force);
adapter.add(traction, "pressure_traction");
```

The interaction contributes both the load residual and its `dF/dy` operator.
The adapter still owns the global block layout; the interaction only names the
semantic target row and the representation-to-force operator.

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
