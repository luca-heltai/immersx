# Fiber-reinforced elastodynamics

This tutorial composes two independent, nonmatching, full-dimensional
elasticity Problems:

```text
matrix: ElastodynamicsSolver<dim,dim>
fiber:  ElastodynamicsSolver<dim,dim>
```

The canonical input is
`tutorials/fiber_reinforced_elastodynamics/parameters.prm.in`:

```{literalinclude} ../../tutorials/fiber_reinforced_elastodynamics/parameters.prm.in
:language: ini
```

Run the 2D input with:

```bash
./build/fiber_reinforced_elastodynamics_debug \
  build/tutorials/fiber_reinforced_elastodynamics/parameters.prm
```

The matrix occupies `[-1,1]^2`. The fiber is an independently meshed thin
rectangle inside it. Both spaces are full-dimensional vector FE spaces; the
fiber is embedded geometrically, not represented by a reduced finite-element
dimension. The fiber coefficients are an additive excess contribution in the
fiber region.

The application exposes matrix and fiber velocity fields and creates an
independent vector multiplier field. The constraint is assembled from
`weak_term(value(field), test(lambda))` terms. Its transpose reactions enforce
velocity compatibility on the two spaces. The five semantic fields are

```text
matrix.displacement  differential
matrix.velocity      differential
fiber.displacement   differential
fiber.velocity       differential
coupling.lambda      algebraic
```

The application registers these contributors with `IDAAdapter`, accepts the
displacement and velocity fields after accepted steps, and writes the two
Problem outputs plus the multiplier on its own FE space. The application
supports `2/2` and `3/3` dimension selections.
