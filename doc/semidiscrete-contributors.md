# Semidiscrete contributors

ImmersX contributors describe the semantic equations of a problem while the
execution adapter owns storage, block numbering, DAE metadata, and solver
policy. The current IDA adapter composes contributors directly:

```{code-block} cpp
using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;

IDAAdapter<FieldVector, GlobalVector> ida(data, MPI_COMM_WORLD, solve);
auto matrix = ida.add(elastodynamics(matrix_problem), "matrix");
auto fiber  = ida.add(elastodynamics(fiber_problem), "fiber");
ida.add(vector_lagrange_multiplier(interaction,
                                   matrix.velocity,
                                   fiber.velocity),
        "fiber_coupling");

GlobalVector state;
GlobalVector state_dot;
ida.reinit(state);
ida.reinit(state_dot);
ida.solve(state, state_dot);
```

The adapter assigns one private execution block to each registered Field in
registration order. A Field's `TimeRole` supplies IDA's differential mask;
applications do not provide block numbers or a second DAE description.

## Writing a Problem contributor

A Problem contributor has four responsibilities:

1. declare its semantic Fields;
2. provide residual `PackagedOperation`s;
3. provide `dF/dy` `LinearOperator`s; and
4. provide `dF/dydot` `LinearOperator`s.

For the heat equation

$$M\dot T + KT - f(t)=0,$$

the contributor declares `temperature` as a differential Field, registers a
residual equivalent to `M*Tdot + K*T - f`, and registers `K` and `M` as the
state and derivative operators. Deal.II expressions can be used directly:

```{code-block} cpp
auto temperature = builder.add_field("heat", "temperature",
                                     TimeRole::differential,
                                     owned, relevant);
auto M = payload_free(linear_operator<Vector, Vector>(mass_matrix));
auto K = payload_free(linear_operator<Vector, Vector>(stiffness_matrix));

builder.add_residual(temperature, "heat",
  [temperature, M, K, &problem](const auto &ctx) {
    auto forcing = problem.forcing(ctx.time());
    return M * ctx.state_derivative()->field(temperature, ctx.time()) +
           K * ctx.state().field(temperature, ctx.time()) - forcing;
  });
builder.add_state_operator(temperature, temperature, "diffusion", K);
builder.add_derivative_operator(temperature, temperature, "mass", M);
```

The `PackagedOperation` returned by a residual factory is applied immediately
by the model. It must not retain temporary vectors or operators.

## Writing an Interaction contributor

An Interaction consumes participant Fields, may declare auxiliary Fields, and
contributes residual rows plus `dF/dy` and `dF/dydot` blocks in exactly the
same way as a Problem. The vector multiplier interaction is the canonical
example. With `C` coupling the first velocity to `lambda` and `Q` pairing the
second velocity with `lambda`, it contributes

```text
R_first  = C lambda
R_second = -Q^T lambda
R_lambda = C^T first - Q second
```

and the four corresponding state blocks. `lambda` is declared by the
Interaction as an algebraic Field; no solver-specific alpha is passed to it.

## Lifetime and state rules

- Problem and Interaction objects captured by contributors must outlive the
  execution adapter using them.
- PackagedOperation results are ephemeral and are applied immediately.
- Vectors and operators referenced by a PackagedOperation must stay alive until
  `apply()` or `apply_add()` completes.
- Residual/Jacobian evaluation is not acceptance of a new physical state.
- A Problem must not update accepted state or history merely because IDA asks
  for a residual at a candidate Newton state.
- Derived caches are fine when they behave logically const from the solver's
  point of view.

IDA alone forms the solver Jacobian:

$$J_{IDA}=\frac{dF}{dy}+\alpha\frac{dF}{d\dot y}.$$

Contributors therefore remain solver-neutral and can be reused by future
execution adapters.
