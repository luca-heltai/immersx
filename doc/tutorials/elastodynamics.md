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

`ElastodynamicsSolver` also runs on simplex meshes (triangles in 2D,
tetrahedra in 3D). Read the mesh from a file and select the fullydistributed
triangulation, since p4est holds only quadrilaterals and hexahedra:

```ini
subsection Elastodynamics
  subsection Domain
    set Grid generator = liver.msh
    set Triangulation type = fullydistributed
  end
end
```

On simplex meshes the solver uses `FE_SimplexP`, `QGaussSimplex`, and a linear
`MappingFE`; meshes that mix simplex and hypercube cells are rejected.
`Initial refinement` refines the mesh before it is distributed. The
fullydistributed triangulation cannot be refined afterwards, so
`Number of refinement cycles` must stay at 1. The
`ParsedConvergenceTable` error integration supports only hypercube cells:
`run()` skips the error table on simplex meshes and `compute_error()` throws.
`gtests/elastodynamics_simplex_01.cc` runs a 48-tetrahedron cube from
`data/tests/simplex_cube.msh.in` with nonzero body force and Dirichlet data,
in serial and on two MPI ranks. The generated
`gtests/parameters/elastodynamics_simplex.prm` also checks quadratic spatial
displacement and cubic time dependence with the native trapezoidal method.

The `elastodynamics_adapter` executable accepts the same uniform simplex
meshes through `Elastodynamics/Domain`. Set
`Triangulation type = fullydistributed` and use one refinement cycle.
Its IDA solve uses simplex operators and skips the convergence table.
The generated test input
`gtests/parameters/elastodynamics_adapter_simplex.prm` supplies nonzero body
force and time-dependent displacement and velocity boundary values.

Next, [Reduced Poisson](reduced-poisson) introduces a lower-dimensional
geometry and coupling.
