# Navier–Stokes

This tutorial introduces the standalone `NavierStokesSolver` application. It
is the full-dimensional fluid Problem intended to become the fluid component
of a future FSI workflow. It is deliberately independent of elasticity,
immersed coupling, and the semantic residual architecture.

The runnable example is:

- `tutorials/navier_stokes/transient_2d.prm.in`

The implementation was extracted from the fluid portion of deal.II step-80.
The reference fluid/manufactured-solution test is the revision
`c061c24bd41b4eb9fb62d1625dc0b818a54bc2bd`; solid, elasticity, immersed, and
coupling branches are not part of this application.

## The problem

On a full-dimensional domain $\Omega$, the solver advances velocity $u$ and
pressure $p$ for the incompressible system

```{math}
\begin{aligned}
\rho\left(\partial_t u + (u\cdot\nabla)u\right)
  - \nabla\cdot\left(2\nu\,\varepsilon(u)\right) + \nabla p &= \rho f,
\\
\nabla\cdot u &= 0,
\end{aligned}
```

where

```{math}
\varepsilon(u) = \tfrac12\left(\nabla u + \nabla u^T\right).
```

The implementation currently supports `dim = spacedim = 2` and
`dim = spacedim = 3`. Surface or embedded Navier–Stokes problems are not
supported by this solver.

## Finite elements and native blocks

The default example uses a Taylor–Hood-like pair:

- `FE_Q(2)` for the velocity;
- `FE_Q(1)` for the pressure.

Component-wise DoF renumbering gives one native two-block system:

```text
[ A   B^T ] [ u ] = [ velocity RHS ]
[ B     0 ] [ p ]   [ pressure RHS ]
```

The pressure-pressure block has no physical pressure mass term. The constant
pressure nullspace is removed by constraining one pressure DoF to zero. This
normalization fixes the pressure gauge without adding a multiplier block.

## Time discretization

Each step uses backward Euler for the mass and viscous terms. Given the
accepted state $u^n$, the assembled velocity equation contains

```{math}
\frac{\rho}{\Delta t}(u^{n+1},v)
 + 2\nu(\varepsilon(u^{n+1}),\varepsilon(v))
 - (p^{n+1},\nabla\cdot v),
```

and the incompressibility equation contributes

```{math}
(\nabla\cdot u^{n+1},q).
```

When `Include convective term = true`, the convective velocity is lagged at
the accepted previous state and moved to the right-hand side:

```{math}
-\rho\left((u^n\cdot\nabla)u^n,v\right).
```

Setting the option to `false` gives the unsteady Stokes limit. No nonlinear
Newton iteration, ALE motion, or multirate time integration is introduced by
this tutorial.

## Solver lifecycle

`NavierStokesSolver::run()` follows the same application-oriented structure as
`PoissonSolver`:

1. `make_grid()` creates the distributed triangulation;
2. `setup_fe()` builds the mixed velocity-pressure finite element;
3. `setup_system()` creates the native block matrix, mass matrix, vectors, and
   constraints;
4. `advance_one_timestep()` updates the parsed-function time, refreshes
   time-dependent velocity constraints, assembles, solves, and accepts one
   state;
5. `output_results()` writes velocity, pressure, and subdomain data at the
   configured frequency.

The linear solve uses FGMRES on the indefinite block system. The velocity
block and pressure mass approximation have separate AMG preconditioners.
The distributed triangulation and block vectors work with MPI from the
beginning.

## Running the tutorial

Build the project, then run the example from the repository root:

```bash
CCACHE_DIR=/Users/heltai/.ccache cmake --build build --target navier_stokes_debug -j2
./build/navier_stokes_debug build/tutorials/navier_stokes/transient_2d.prm
```

The Release executable is named `./build/navier_stokes`. The example writes a
time series under `build/test_output/tutorial-output/navier-stokes-2d` and
emits output every second time step. Set `Output frequency = 0` to disable
visualization output. For a
parallel run, use two MPI ranks:

```bash
mpirun -np 2 ./build/navier_stokes_debug build/tutorials/navier_stokes/transient_2d.prm
```

The application synchronizes the ranks before releasing the solver resources,
so the same lifecycle is valid for serial and MPI execution.

The application reads `dimension` and `space dimension` before constructing
the statically typed solver. The parameter-file name does not select the
dimension:

```text
dimension = 2, space dimension = 2  -> NavierStokesSolver<2>
dimension = 3, space dimension = 3  -> NavierStokesSolver<3>
```

## Parameter-file choices

The top-level `Navier-Stokes` subsection contains:

- `Finite element spaces`: velocity and pressure polynomial degrees;
- `Grid generation`: generator, arguments, and distributed triangulation;
- `Physical properties`: `Density`, `Viscosity`, and the explicit-convection
  toggle;
- `Time stepping`: initial/final time and either a fixed or number-of-steps
  policy;
- `Right hand side`, `Dirichlet boundary conditions`, and `Initial condition`:
  parsed vector functions with `dim + 1` components, where the last component
  is pressure and is ignored for velocity data;
- `Solver`: outer FGMRES and inner block-solver controls.

Velocity Dirichlet data is applied on the boundary ids listed in
`Dirichlet boundary ids`. The parsed functions are updated to the current
time before each step, so nonzero time-dependent velocity data is handled in
the constraints for that step.

The complete input file used by this tutorial is:

```{literalinclude} ../../tutorials/navier_stokes/transient_2d.prm.in
:language: bash
```

## Manufactured-solution validation

The step-80-inspired MMS regression is kept with the tests rather than making
the application depend on a verification-only workflow:

```bash
mkdir -p build/test_directory
(cd build/test_directory && \
  mpirun -np 2 ../gtests/gtests_debug \
  --gtest_filter='NavierStokes.Step80ManufacturedSolutionConvergence-*.MPI_*')
```

The test uses `ParsedConvergenceTable` and the parameter file
`gtests/parameters/navier_stokes_step80_mms_2d.prm`. It reports the velocity
L2 error and computes rates with respect to DoFs. With Q2 velocity, the
observed rates are approximately three.

## Source files

The application is implemented in:

- `apps/app_navier_stokes.cc` — MPI initialization and 2D/3D dispatch;
- `include/immersx/physics/navier_stokes.h` — parameter and solver interfaces;
- `source/navier_stokes.cc` — mesh setup, mixed assembly, block solve,
  time-stepping, and output;
- `gtests/navier_stokes_01.cc` — setup and transient Stokes/Navier–Stokes
  coverage;
- `gtests/navier_stokes_step80_mms_01.cc` — manufactured-solution convergence;
- `tutorials/navier_stokes/transient_2d.prm.in` — the runnable tutorial input.

This Problem is intentionally shaped like `PoissonSolver` so it can later be
adapted to ImmersX's semantic Field, Representation, and residual interfaces.
Those future adapters are outside the scope of this tutorial.
