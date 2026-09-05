# Application reference

The executable names below are produced from `apps/app_*.cc`. Dimension
selection is read from the parameter file where applicable; the filename never
dispatches a template instantiation.

| Executable | Entry point and main problem | Supported dimensions | Model / dependencies | Canonical example and status |
| --- | --- | --- | --- | --- |
| `poisson` | `apps/app_poisson.cc`; `PoissonSolver` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | `LinearAdapter` with iterative/block-diagonal policy; MPI | `tutorials/poisson/poisson_2d.prm.in`; introductory production path |
| `elastic_static` | `apps/app_elastic_static.cc`; `ElasticStaticProblem` | `2/2`, `3/3` | `LinearAdapter`; MPI | `tutorials/elastic_static/elastic_static.prm.in`; production static path |
| `elastodynamics` | `apps/app_elastodynamics.cc`; `ElastodynamicsSolver` | `2/2`, `3/3` | `IDAAdapter` with displacement/velocity differential Fields; MPI | `tutorials/elastodynamics/strong_dirichlet.prm.in`; tutorial path |
| `reduced_poisson` | `apps/app_reduced_poisson.cc`; `ReducedPoisson` | 2D and 3D bulk cases with reduced dimension `0` or `1` and configured cross-section dimensions | Requires deal.II VTK support; MPI | `tutorials/reduced_poisson/single_cylinder_3d.prm.in`; specialized reduced path |
| `coupled_poisson_elasticity` | `apps/app_coupled_poisson_elasticity.cc`; fixed `PoissonSolver<1,3>` plus `ElasticStaticProblem<3,3>` | fixed 1D/3D-to-3D composition | `LinearAdapter`; MPI | `tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in`; composition tutorial |
| `coupled_poisson` | `apps/app_coupled_poisson.cc`; bulk `PoissonSolver<2>` plus embedded `PoissonSolver<1,2>` and generic multiplier Constraint | fixed 2D/1D-to-2D composition | `LinearAdapter`; MPI | `tutorials/coupled_poisson/coupled_poisson.prm.in`; semantic composition tutorial |
| `fiber_reinforced_elastodynamics` | `apps/app_fiber_reinforced_elastodynamics.cc`; matrix and fiber `Elastodynamics` Problems | `2/2`, `3/3` | Five-field `IDAAdapter` with generic multiplier Constraint; MPI | `tutorials/fiber_reinforced_elastodynamics/parameters.prm.in`; advanced tutorial |
| `metric_flow_x_elastodynamics` | `apps/app_metric_flow_x_elastodynamics.cc`; MetricFlowX plus 3D elastodynamics and `MetricFlowXVesselWallConstraint` | fixed 1D/3D-to-3D composition | `IDAAdapter` with two-way vessel-wall pressure feedback; requires MetricFlowX and deal.II SUNDIALS; MPI | `gtests/parameters/metric_flow_x_elastodynamics.prm.in`; vertical integration path |
| `navier_stokes` | `apps/app_navier_stokes.cc`; `NavierStokesSolver` | `2/2`, `3/3` | `IDAAdapter` with differential velocity and algebraic pressure Fields; MPI | `tutorials/navier_stokes/transient_2d.prm.in`; independent tutorial |
| `laplacian` | `apps/app_laplacian.cc`; legacy `PoissonProblem` | `2/2`, `2/3`, `3/3` | Older immersed scalar path; MPI | `gtests/parameters/laplacian_simple.prm.in`; legacy/special-purpose |
| `elasticity` | `apps/app_elasticity.cc`; legacy `ElasticityProblem` | `2/2`, `2/3`, `3/3` | Immersed elasticity and verification workflows; MPI | `tutorials/elasticity/`; legacy/verification path |
| `coupled_elasticity` | `apps/app_coupled_elasticity.cc`; legacy 3D elasticity plus 1D hemodynamics | 3D tissue with `lib1dsolver` | Optional `lib1dsolver`, `ENABLE_COUPLED_PROBLEMS`; MPI | `COUPLED_PROBLEM.md`; legacy specialized workflow |
| `pseudocoupling1D` | `apps/app_pseudocoupling1D.cc`; pseudo-coupled tissue/1D model | 3D tissue with `lib1dsolver` | Optional `lib1dsolver`, `ENABLE_COUPLED_PROBLEMS`; MPI | No canonical CI input; legacy/special-purpose |

Debug builds append `_debug` to the executable name. All applications accept
an explicit parameter path; the [How-to guides](../how-to/index) explain
configuration and working-directory-safe execution.
