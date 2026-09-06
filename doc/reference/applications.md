# Application reference

Each source file `apps/app_*.cc` produces an executable with the `app_` prefix
removed. Debug targets append `_debug`. Applications read an explicit
parameter-file path; when a file supports dimension dispatch, the `dimension`
and `space dimension` entries select the template instantiation.

| Executable | Entry point and problem | Dimensions or composition | Dependencies and input |
| --- | --- | --- | --- |
| `poisson` | `apps/app_poisson.cc`; `PoissonSolver` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | MPI; `tutorials/poisson/poisson_2d.prm.in` |
| `poisson_adapter` | `apps/app_poisson_adapter.cc`; `PoissonSolver` through `LinearAdapter` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | MPI; `gtests/parameters/poisson_simple_2d.prm.in` |
| `elastic_static` | `apps/app_elastic_static.cc`; `ElasticStaticProblem` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | MPI; `tutorials/elastic_static/elastic_static.prm.in` |
| `elastic_static_adapter` | `apps/app_elastic_static_adapter.cc`; `ElasticStaticProblem` through `LinearAdapter` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | MPI; `gtests/parameters/elastic_static.prm.in` |
| `elastodynamics` | `apps/app_elastodynamics.cc`; `ElastodynamicsSolver` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | MPI; `tutorials/elastodynamics/strong_dirichlet.prm.in` |
| `elastodynamics_adapter` | `apps/app_elastodynamics_adapter.cc`; `ElastodynamicsSolver` through `IDAAdapter` | `1/1`, `1/2`, `1/3`, `2/2`, `2/3`, `3/3` | MPI and deal.II SUNDIALS; `gtests/parameters/dynamic_purely_elastic.prm.in` |
| `reduced_poisson` | `apps/app_reduced_poisson.cc`; `ReducedPoisson` | `2/2` and `3/3` with reduced dimension `0` or `1`; cross-section dimensions are checked by the application | MPI and deal.II VTK; `tutorials/reduced_poisson/single_cylinder_3d.prm.in` |
| `coupled_poisson` | `apps/app_coupled_poisson.cc`; 2D bulk and 1D embedded `PoissonSolver` with a multiplier `Constraint` | Fixed 2D bulk plus 1D-in-2D embedded field | MPI; `tutorials/coupled_poisson/coupled_poisson.prm.in` |
| `coupled_poisson_elasticity` | `apps/app_coupled_poisson_elasticity.cc`; 1D-in-3D `PoissonSolver` and 3D `ElasticStaticProblem` | Fixed 1D-in-3D pressure lift coupled to 3D elasticity | MPI; `tutorials/coupled_poisson_elasticity/coupled_poisson_elasticity.prm.in` |
| `fiber_reinforced_elastodynamics` | `apps/app_fiber_reinforced_elastodynamics.cc`; matrix and fiber `Elastodynamics` Problems with a multiplier constraint | `2/2`, `3/3` | MPI; `tutorials/fiber_reinforced_elastodynamics/parameters.prm.in` |
| `metric_flow_x` | `apps/app_metric_flow_x.cc`; MetricFlowX `BloodFlowSystem<1,3>` through `IDAAdapter` | Fixed 1D-in-3D | MPI, deal.II SUNDIALS, and MetricFlowX; `gtests/parameters/metric_flow_x.prm.in` |
| `metric_flow_x_elastodynamics` | `apps/app_metric_flow_x_elastodynamics.cc`; MetricFlowX and 3D `Elastodynamics` with `MetricFlowXVesselWallConstraint` | Fixed 1D-in-3D flow and 3D wall | MPI, deal.II SUNDIALS, and MetricFlowX; `tutorials/metric_flow_x_elastodynamics/01_wall_kinematics.prm.in` |
| `navier_stokes` | `apps/app_navier_stokes.cc`; `NavierStokesSolver` | `2/2`, `3/3` | MPI; uses `IDAAdapter` with deal.II SUNDIALS and the native solver path without it; `tutorials/navier_stokes/transient_2d.prm.in` |
| `laplacian` | `apps/app_laplacian.cc`; `PoissonProblem` | `2/2`, `2/3`, `3/3` | MPI; `gtests/parameters/laplacian_simple.prm.in` |
| `elasticity` | `apps/app_elasticity.cc`; `ElasticityProblem` | `2/2`, `2/3`, `3/3` | MPI; `tutorials/elasticity/strong_dirichlet.prm` |
| `coupled_elasticity` | `apps/app_coupled_elasticity.cc`; 3D tissue and 1D hemodynamics | 3D tissue; command-line coupling controls | MPI, OpenMP, and `lib1dsolver`; enable with `ENABLE_COUPLED_PROBLEMS` |
| `pseudocoupling1D` | `apps/app_pseudocoupling1D.cc`; 3D tissue and 1D pseudo-coupling | 3D tissue; five positional arguments | MPI, OpenMP, and `lib1dsolver`; enable with `ENABLE_COUPLED_PROBLEMS` |

MetricFlowX executables are excluded from the build unless both MetricFlowX
and deal.II SUNDIALS are available. The two `lib1dsolver` executables are
built from their source files, but their functional code is enabled only when
`ENABLE_COUPLED_PROBLEMS` is on and the library is found with RTTI support.
All other executables are part of the normal application target set. The
[running applications guide](../how-to/running-applications) explains the
parameter-file command format.
