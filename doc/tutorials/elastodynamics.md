# Elastodynamics

This tutorial adds time dependence to the static elasticity problem. It
introduces displacement and velocity as state fields and advances them with
the public `IDAAdapter` execution path. IDA solves the canonical residual
`F(t, y, ydot) = 0`; the application accepts the resulting state before native
output is written.

The executable is `elastodynamics`, from `apps/app_elastodynamics.cc`. The
canonical 2D input is `tutorials/elastodynamics/strong_dirichlet.prm.in`:

```{literalinclude} ../../tutorials/elastodynamics/strong_dirichlet.prm.in
:language: ini
```

The application reads `dimension` and `space dimension` from the file. This
input selects the full-dimensional 2D solver; the executable also supports the
full-dimensional 3D instantiation. The input uses one time step and a
manufactured displacement so it stays a small, deterministic smoke example.

Run it with:

```bash
cmake --build build -j
./build/elastodynamics_debug \
  build/tutorials/elastodynamics/strong_dirichlet.prm
```

Output is written below `build/test_output/tutorial-output/elastodynamics-strong`.
The same input is exercised by `AppExecutables.TutorialElastodynamics`.

The physical model exposes separate mass, stiffness, and damping operators.
The application registers the Problem directly with `IDAAdapter`, marks both
displacement and velocity as differential Fields, and installs an accepted
output callback. `initial_acceleration()` supplies a physically consistent
initial derivative; `accept_state()` updates the Problem only after IDA has
accepted or interpolated a state. Solver-neutral residual and
execution-adapter concepts are described in
[Architecture concepts](../concepts/architecture).
The larger convergence cases remain in `tutorials/elastodynamics/` and are
listed in the [verification reference](../reference/verification).

Next, [Reduced Poisson](reduced-poisson) introduces a lower-dimensional
geometry and coupling.
