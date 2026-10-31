# Tutorials

These tutorials use the parameter files under `tutorials/` and the
executables listed in the [application reference](../reference/applications).
They cover scalar and vector problems, time integration, reduced geometry,
weak coupling, and a two-way vessel-wall interaction.

[Vascular generation](vascular-generation) provides four native cube and sphere
examples, an arterial–venous case, and a physical-graph rediscretization control.

The introductory sequence is:

1. [Poisson](poisson) — a scalar finite-element solve.
2. [Static elasticity](elasticity) — a vector-valued finite-element solve.
3. [Elastodynamics](elastodynamics) — displacement and velocity with IDA.
4. [Reduced Poisson](reduced-poisson) — an immersed lower-dimensional space.
5. [Coupled Poisson](coupled-poisson) — two Poisson Problems and a multiplier.
6. [Coupled Poisson–elasticity](coupled-poisson-elasticity) — a lifted load
   from a 1D Problem onto 3D elasticity.
7. [Fiber-reinforced elastodynamics](fiber-reinforced-elastodynamics) — two
   nonmatching Problems and a multiplier constraint.
8. [Fiber-reinforced harmonic elasticity](fiber-reinforced-frequency) — a frequency
   sweep with reduced or full-dimensional reinforcement and independent or shared
   displacements.

For the elastostatic–flow project, start with the
[MetricFlowX coupling onboarding guide](metric-flow-x-elastostatic-onboarding).
It follows independent runs, file exchange, partitioned iteration, and a
monolithic `IDAAdapter` formulation, with a work split for two collaborators.

[Navier–Stokes](navier-stokes) documents the independent incompressible-flow
application. The [MetricFlowX and elastodynamics tutorial](metric-flow-x-elastodynamics)
shows the parameter-driven two-way vessel-wall coupling. The
[MetricFlowX and elastodynamics coupling formulation](metric-flow-x-elastodynamics-coupling)
introduces the coupled equations. The
[one-vessel manufactured solution](metric-flow-x-elastodynamics-mms) gives
piecewise analytical fields, an inlet that supplies the changing volume, and
parameter files for the corrected residual equations.

```{toctree}
:maxdepth: 1

vascular-generation
poisson
elasticity
elastodynamics
reduced-poisson
coupled-poisson
coupled-poisson-elasticity
fiber-reinforced-elastodynamics
navier-stokes
metric-flow-x-elastodynamics
metric-flow-x-elastodynamics-coupling
```
