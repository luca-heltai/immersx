# Tutorials

Tutorials are a learning path through real ImmersX executables. Each runnable
example names its canonical input, includes that input from the repository, and
is exercised by the application smoke tests.

## Main path

1. [Poisson](poisson) — your first scalar finite-element simulation.
2. [Static elasticity](elasticity) — vector-valued finite elements, material
   data, and boundary conditions.
3. [Elastodynamics](elastodynamics) — time-dependent displacement and velocity
   fields.
4. [Reduced Poisson](reduced-poisson) — a lower-dimensional coupling geometry.
5. [Coupled Poisson–elasticity](coupled-poisson-elasticity) — observe, lift,
   couple, and solve.
6. [Fiber-reinforced elastodynamics](fiber-reinforced-elastodynamics) — an
   advanced full-order distributed coupling workflow.

[Navier–Stokes](navier-stokes) is an independent fluid tutorial. The
application reference lists other specialized and legacy executables.

```{toctree}
:maxdepth: 1

poisson
elasticity
elastodynamics
reduced-poisson
coupled-poisson-elasticity
fiber-reinforced-elastodynamics
navier-stokes
```
