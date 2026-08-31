# Building a coupled problem

This tutorial builds one steady coupled system from two existing Problems:

```text
PoissonProblem<1,3> -- p=2u --> cylindrical surface -- traction --> ElasticStaticProblem<3,3>
```

The executable is `coupled_poisson_elasticity`. It is a deliberately small
vertical slice showing how application-specific physics can extend the public
composition API without adding a global coupled-system class.

The canonical input is
`tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in`:

```{literalinclude} ../../tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in
:language: ini
```

## Application vocabulary

The example defines three descriptors next to the application:

- `Pressure` names the observable `p=2u` of the Poisson solution;
- `CylinderSurface` names a fixed radius cylinder parameterized by axial
  coordinate `s` and angle `theta`;
- `Traction` names the one-way weak load
  \(\int_\Gamma p\,n\cdot v\,\mathrm{d}\Gamma\) on elasticity.

The executable’s composition is correspondingly short:

```cpp
const auto poisson = adapter.add(poisson_problem);
const auto elastic = adapter.add(elasticity_problem);
const auto pressure =
  poisson.observe(Pressure{}).lift(surface);
adapter.couple(pressure, elastic, traction);

auto state = adapter.make_state();
adapter.solve(state);

poisson_problem.set_solution(
  adapter.field(state, poisson.fields().solution));
elasticity_problem.set_solution(
  adapter.field(state, elastic.fields().displacement));

poisson_problem.output_results();
elasticity_problem.output_results(0);
```

The adapter owns the coupled solver state, but each Problem owns its accepted
physical state and native output. The explicit handoff above is therefore the
acceptance/output boundary. The Poisson and elasticity parameter sections may
use different output directories, or the same directory with distinct
`Output name` values; the application does not assemble filenames itself.

The application descriptor implementations hide the geometry map, value
transfer, finite-element point evaluation, and weak-form operator. They are
not part of the public application workflow and are not new library-level
composition objects.

## Running it

Build the project and run the generated smoke-test parameters:

```bash
mpirun -np 1 ./build/coupled_poisson_elasticity_debug \
  build/tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm
```

The application prints and records checks for the coupled residual, the
pressure scaling, and the traction balance. The regression test launches the
same executable from an unrelated working directory and checks those values.
Its canonical input is the one shown above.

## Scope of this slice

The cylinder is fixed and the coupling is one-way. There are no multipliers,
nonlinear elasticity terms, ALE/metric-flow terms, FSI time integration, or
new solver infrastructure. The example is intended to establish the public
composition workflow before a future metric-flow-x tutorial expands the
geometry and execution model.

Next, continue to [fiber-reinforced elastodynamics](fiber-reinforced-elastodynamics)
for a distributed composed example.
