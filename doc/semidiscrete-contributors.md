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

## Contributor authors

A Problem contributor declares semantic Fields and contributes residual,
`dF/dy`, and `dF/dydot` terms. A contributor is selected by the adapter through
an ADL customization point such as:

```{code-block} cpp
template <typename Builder>
Fields contribute(Builder &builder, const MyProblem &problem)
{
  auto temperature = builder.field("temperature", TimeRole::differential,
                                  owned, relevant);
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
