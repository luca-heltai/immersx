# Poisson

This tutorial introduces the `PoissonSolver` application. It follows the
familiar deal.II step-6 sequence while retaining the parameter parsing,
distributed linear algebra, boundary-data handling, mesh import, output,
convergence, timing, and refinement conventions used by ImmersX.

The example files are:

- `tutorials/poisson/poisson_2d.prm`
- `tutorials/poisson/poisson_12d.prm`

The implementation is deliberately a small scalar discretization that can be
used on its own or as the scalar finite-element component of a later solver
composition.

## The problem

The solver discretizes

```{math}
-\Delta_\Omega u = f \qquad \text{in } \Omega,
```

with strong Dirichlet data on the boundary ids listed in
`Dirichlet boundary ids`:

```{math}
u = u_D \qquad \text{on } \partial\Omega_D.
```

The mesh has topological dimension `dim` and is embedded in
`spacedim` dimensions. Thus `PoissonSolver<2>` is the ordinary planar Poisson
solver, while `PoissonSolver<1,2>` and `PoissonSolver<1,3>` discretize the
corresponding one-dimensional Laplace--Beltrami operator on an embedded curve.
The right-hand side and boundary functions are evaluated in the embedding
coordinates.

The valid template pairs are:

| Class | Mesh | Typical interpretation |
| --- | --- | --- |
| `PoissonSolver<1>` | 1D in 1D | interval Poisson problem |
| `PoissonSolver<1, 2>` | 1D in 2D | curve Laplace--Beltrami problem |
| `PoissonSolver<1, 3>` | 1D in 3D | space-curve Laplace--Beltrami problem |
| `PoissonSolver<2>` | 2D in 2D | planar Poisson problem |
| `PoissonSolver<2, 3>` | 2D in 3D | surface Laplace--Beltrami problem |
| `PoissonSolver<3>` | 3D in 3D | volumetric Poisson problem |

## Implementation lifecycle

`PoissonSolver::run()` executes the following sequence:

1. `make_grid()` generates a mesh from a parsed grid name and arguments. If
   generation fails, the same names are interpreted as an input grid and CAD
   description by the repository grid reader.
2. `setup_fe()` creates `FE_Q<dim, spacedim>` and a matching cell quadrature.
3. `setup_system()` distributes the scalar degrees of freedom, creates
   hanging-node and Dirichlet constraints, builds the distributed sparsity
   pattern, and allocates ordinary `LA::MPI::Vector` objects for the solution,
   right-hand side, and locally relevant solution.
4. `assemble_system()` forms the scalar stiffness matrix and load vector from
   the gradient bilinear form

   ```{math}
   a(u,v) = \int_\Omega \nabla u\cdot\nabla v\,\mathrm{d}x,
   \qquad
   \ell(v) = \int_\Omega f v\,\mathrm{d}x.
   ```

5. `solve()` uses conjugate gradients with the repository's AMG
   preconditioner.
6. `output_results()` writes one VTU file per cycle and maintains a PVD
   collection together with a subdomain field.
7. `refine_grid()` estimates the cell error, marks cells according to the
   selected strategy, transfers the solution, and rebuilds the system.

The public methods are intentionally close to step 6, so an embedding solver
can use the discretization lifecycle without inheriting any application logic.

## Running the tutorial

Build the project, then run the 2D example from the repository root:

```bash
./build/poisson[_debug] tutorials/poisson/poisson_2d.prm
```

The sample uses a square represented by `hyper_cube`, a constant source
`f=1`, homogeneous Dirichlet data, three global refinement cycles, and the
distributed triangulation backend. Results are written under
`./output/poisson_2d`.

The `poisson_12d.prm` file is a one-cycle smoke example for a curve embedded in
the plane. It deliberately leaves `Triangulation type = distributed` in the
file: `dim=1` forces the fully distributed backend automatically.

The application selects the template pair from the input filename. The
recognized suffixes are:

- `1d` for `PoissonSolver<1>`;
- `12d` for `PoissonSolver<1,2>`;
- `13d` for `PoissonSolver<1,3>`;
- `23d` for `PoissonSolver<2,3>`;
- `3d` for `PoissonSolver<3>`;
- any other name for `PoissonSolver<2>`.

For example, a curve input can be run as
`./build/poisson_debug curve_12d.prm`.

## Triangulation backends

The `Grid generation/Triangulation type` parameter accepts `distributed` or
`fullydistributed`.

- `distributed` is the default for two- and three-dimensional meshes. It
  supports the usual adaptive refinement and solution transfer path.
- `fullydistributed` builds a serial mesh first, copies it into deal.II's
  fully distributed triangulation, and is useful when the coarse mesh itself
  must be distributed. It can be selected for any valid template pair.
- For `dim=1`, the fully distributed backend is selected automatically because
  deal.II does not implement its p4est-backed distributed 1D triangulation.

Fully distributed meshes are immutable after `copy_triangulation()` in the
deal.II version used by this project. Therefore a fully distributed Poisson
run should use one initial mesh (`Number of refinement cycles = 1`); adaptive
refinement and solution transfer remain available on the ordinary distributed
backend.

## Parameter-file choices

The top-level `Poisson` subsection contains the common controls:

- `FE degree`, `Initial refinement`, and `Dirichlet boundary ids` define the
  finite-element space and strong boundary constraints;
- `Right hand side` and `Dirichlet boundary conditions` are parsed functions;
- `Grid generation` chooses generated or imported geometry and the parallel
  triangulation backend;
- `Refinement and remeshing` controls global or indicator-based refinement;
- `Solver/Control` configures the CG stopping criterion;
- `Error` controls the optional exact-solution convergence table.

For an imported mesh, set `Grid generator` to the input filename and provide
the repository's grid/CAD argument string in `Grid generator arguments`. The
same fallback is used for generated names that are not recognized by deal.II.

## Source files

The application is implemented in:

- `apps/app_poisson.cc` — MPI initialization, parameter loading, and dimension
  dispatch;
- `include/poisson.h` — parameter and solver interfaces;
- `source/poisson.cc` — mesh setup, FE assembly, solve, output, and transfer;
- `gtests/poisson_01.cc` — construction coverage for every valid dimension pair
  and a one-cycle embedded solve.
