# Coupled Poisson–elasticity

This application composes a 1D-in-3D Poisson Problem with a 3D static
elasticity Problem. The Poisson solution is multiplied by two, lifted from
the representative cylinder, converted to a normal traction, and assembled
against the elasticity displacement test field:

```text
PoissonSolver<1,3> -> value -> lift -> normal traction
                                      -> test(displacement)
ElasticStaticProblem<3,3> ------------^
```

The executable is `coupled_poisson_elasticity`, from
`apps/app_coupled_poisson_elasticity.cc`. Its input is
`tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in`:

```{literalinclude} ../../tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in
:language: ini
```

The application creates `FESpaceView`s for the two native DoFHandlers, names
the solution fields, constructs the lift and the surface normal, and adds a
`weak_term` to a `LinearAdapter`. The adapter owns the execution vector and
the coupled solve. Each Problem receives its accepted solution before native
output is written.

Run the configured input with:

```bash
mpirun -np 1 ./build/coupled_poisson_elasticity_debug \
  build/tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm
```

The executable writes the two Problem outputs and a diagnostics file below
the configured output directory. It checks the coupled residual, the pressure
factor, and the traction balance. The same input is used by the application
integration test.
