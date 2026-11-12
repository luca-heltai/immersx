# Elastodynamics

This tutorial adds time dependence to the static elasticity problem. It
introduces displacement and velocity as state fields. The `elastodynamics`
executable uses fixed-step backward Euler or trapezoidal integration;
`elastodynamics_adapter` uses `IDAAdapter` to solve the residual
`F(t, y, ydot) = 0`.

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
The adapter application registers the Problem directly with `IDAAdapter`,
marks both displacement and velocity as differential Fields, and installs an accepted
output callback. `initial_acceleration()` supplies a physically consistent
initial derivative; `accept_state()` updates the Problem only after IDA has
accepted or interpolated a state. Solver-neutral residual and
execution-adapter concepts are described in
[Architecture concepts](../concepts/architecture).
The larger convergence cases remain in `tutorials/elastodynamics/` and are
listed in the [verification reference](../reference/verification).

Materials are selected under `Elastodynamics/Material properties`. Map mesh
material ids to named subsections with `Material tags by material id`, for
example `0: default, 1: stiff`. Unmapped ids use `default`; several ids may
share a tag. Each material accepts `Density`, `Lame mu`, `Lame lambda`,
`Rayleigh alpha`, `Rayleigh beta`, and `Viscosity eta`.

The damping bilinear form on each cell is

$$
C(v,w) = \alpha\rho (v,w)
       + \beta\bigl(2\mu (\varepsilon(v),\varepsilon(w))
                     + \lambda (\nabla\cdot v,\nabla\cdot w)\bigr)
       + \eta (\varepsilon(v),\varepsilon(w)).
$$

Density and shear modulus must be finite and positive. The remaining
coefficients must be finite and non-negative. The viscosity convention is
the same as in `ElasticityProblem`: there is no factor of two before
`Viscosity eta`. To migrate an older elastodynamics input, move the
`Material` entries to `Material properties/default` and replace
`Damping shear = s` with `Viscosity eta = 2*s`. The former `Damping bulk`
parameter has no independent counterpart in this schema; `Rayleigh beta`
scales the entire elastic stiffness.

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

## Kinematic rows and linear solvers

Both executables use `D = diag(M)` in the kinematic equation
`D (u_dot - v) = 0`; the momentum equation retains the consistent mass `M`.
The two kinematic blocks have diagonal sparsity. The standalone solver
keeps these rows diagonal on independent DoFs when hanging nodes are present
and restores the constrained displacement and velocity after the solve.
It uses FGMRES with the second-order block preconditioner.

The adapter defaults to the iterative solver. To select a direct solver, add:

```ini
subsection Elastodynamics
  subsection IDA
    subsection Linear solver
      set Type = umfpack
    end
  end
end
```

`umfpack` requires one MPI rank; `mumps` supports multiple ranks when deal.II
has MUMPS enabled. Both direct solvers eliminate the kinematic rows before
factorization and recover the eliminated field by back-substitution. This applies
to the consistent initial condition with zero derivative weight. The diagonal
kinematic block is inverted entrywise. Other systems without the required
kinematic structure retain the monolithic factorization.

Next, [Reduced Poisson](reduced-poisson) introduces a lower-dimensional
geometry and coupling.

## Robin tractions

A displacement spring on the boundary satisfies

$$
\sigma(u,v)n + \beta_R(x,t)u = g(x,t), \qquad \beta_R(x,t)\geq 0,
$$

where $v=\dot u$ and $\sigma(u,v)$ includes the elastic and viscous stresses
of the material model above. The scalar Robin coefficient $\beta_R$ is
independent of the material's Rayleigh beta. It multiplies displacement; it
does not introduce boundary damping proportional to velocity.

For a colored unit square, this example prescribes displacement on three
faces and a time-dependent Robin traction on the right face:

```ini
subsection Elastodynamics
  set Dirichlet boundary ids = 0, 2, 3
  set Neumann boundary ids =
  set Robin boundary ids = 1
  subsection Functions
    subsection Robin coefficient
      set Function expression = 2+x+t
    end
    subsection Robin boundary
      set Function expression = sin(t); 0
    end
  end
end
```

Set `Domain/Grid generator arguments = 0: 1: true` for that face numbering.
The vector load has one component per ambient spatial coordinate. Coefficients
must remain finite and nonnegative, loads must be finite, and Robin ids must
be disjoint from displacement and Neumann ids.

Both the native fixed-step solver and the IDA adapter evaluate the coefficient
and load at the requested physical time. In the native trapezoidal method,
the previous-time elastic force uses the previous coefficient. Time-varying
loads can be combined with a constant boundary spring. A Robin update leaves
the mass and material damping operators unchanged.

The same entries are available in the matrix and fiber `Elastodynamics`
subsections of `fiber_reinforced_elastodynamics`, and in the solid subsection
of `metric_flow_x_elastodynamics`. Hypercube and simplex face integration use
the same reference-cell choice as the volume operators.
