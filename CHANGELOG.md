# Changelog

All notable changes to ImmersX are documented in this file. Each entry
begins with a link to the pull request that introduced the change.

The file is included in the documentation and is published at
`doc/changes.md`. Add new entries under `[Unreleased]` using the
skeleton in `.github/CHANGELOG_TEMPLATE.md`.

## [Unreleased]

### Added

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Run forward Euler elasticity on uniform simplex meshes with P1 displacement elements and a positive row-sum lumped mass. Reject higher simplex degrees before assembly.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Run legacy elasticity on uniform simplex meshes with matching displacement elements, cell and boundary integration, boundary constraints, and stress output.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Solve frequency-domain elasticity on uniform triangle and tetrahedron meshes, with matching displacement elements, cell and face quadrature, boundary interpolation, stress integration, and output.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Run the IDA elastodynamics adapter on simplex meshes, skipping its unavailable convergence table. Verify nonzero time-dependent displacement and velocity on simplex and hypercube meshes.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Run the static elasticity adapter on simplex meshes and retain hypercube error tables.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Solve static elasticity on uniform simplex meshes with automatic simplex displacement elements and matching cell and boundary quadrature. The simplex lifecycle skips the unavailable error table.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Run the Poisson adapter on simplex meshes. Skip its unavailable simplex error table and provide locally relevant values for error integration on hypercube meshes in MPI.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Assemble weak terms, known sources, and multiplier metrics on uniform simplex meshes. Choose matching mappings for finite-element spaces, imported fields, particle search, immersed partitioning, and lifted triangular sheets. Reject mixed cell meshes in Domains and integration supports.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Allow Poisson to solve on uniform simplex meshes with fully distributed ownership, simplex finite elements and quadrature, and a linear geometry mapping.

- [PR #292](https://github.com/luca-heltai/immersx/pull/292) Allow the coupled MetricFlowX–elastodynamics application to partition vessels with the solid cell owners through the `immersed` policy, with serial and MPI RCR regression cases.
- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Allow ElastodynamicsSolver to run on simplex (triangle and tetrahedron) meshes with `FE_SimplexP`, simplex Gauss quadrature, and a linear `MappingFE`. Simplex meshes require the fullydistributed Domain triangulation. `run()` skips the error table on simplex meshes, and `compute_error()` throws there.
- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Report the solid boundary ids and the number of vessel cells with a vertex outside the solid mesh at the start of `metric_flow_x_elastodynamics`.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Select harmonic reinforcement dimensions and displacement formulation from parameters: reduced meshes use a normal volume lift, full-dimensional meshes need no lift, and either independent displacements with continuity multipliers or one shared displacement can be solved.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Integrate known volume sources and linear weak terms on independent physical volume supports, retaining nonzero essential offsets.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Lift scalar representative fields and geometric integration supports to volumes in codimension one and two, using the complete surface tangent space for normal sheet extrusion.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Solve harmonic matrix-plus-excess-fiber systems in 2D and 3D with independent meshes, real and imaginary displacement-continuity constraints, frequency sweeps, and displacement and multiplier output.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Compose harmonic elasticity Problems with `LinearAdapter` using separate real and imaginary displacement Fields, material-dependent damping, and phased body and Neumann loads.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Support Field-local absolute tolerances, a history-preserving startup tolerance ramp, and cumulative IDA progress statistics.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Expose all backported IDA solver controls through IDAParameters while preserving existing defaults.
- [PR #277](https://github.com/luca-heltai/immersx/pull/277) Allow local co-refinement to partition embedded cells with their background owners through the `immersed` policy, preserving imported fields on the final partition.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Allow Domain to observe an externally owned triangulation, retaining its imported fields without taking ownership of the mesh.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Allow serial and distributed coupled IDA solves with native deal.II SparseDirectMUMPS when DEAL_II_WITH_MUMPS is enabled, selected through the application parameter file.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Allow the coupled MetricFlowX–elastodynamics application to select a serial UMFPACK solve from its parameter file. Materialize and factorize each IDA Jacobian once, and reuse the factorization for subsequent right-hand sides.
- [PR #271](https://github.com/luca-heltai/immersx/pull/271) Show ImmersX/native deal.II overhead ratios by default in performance charts, with an equal-time reference and an absolute-timing option. Keep incompatible measurements separate and explain unavailable comparisons and single-sample histories.
- [PR #268](https://github.com/luca-heltai/immersx/pull/268) Backport deal.II's expanded SUNDIALS IDA controls, inequality constraints, N_Vector operations, and expert setup access. Use the backport in IDAAdapter and allow configuration before solver initialization.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Allow lifted Poisson modal data to use imported point/cell field aliases, nonlinear expressions, coordinates, time zero, and `pi`/`E`, evaluated from retained ghosted Domain data without another application FE space.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Expose the 3D scalar primitive execution path and modal lift/data, imported-radius, multiplier, coefficient-output, and lifted-output nodes in Coral, with a runnable component-level graph.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Add the compositional `LiftedPoisson` application with configurable bulk and modal FE spaces, imported radii, section-weighted modal constraints, Schur/AL solves, adaptive bulk refinement, and native/lifted output.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Add a component-level Coral static-elasticity graph with nonzero modal data, serial/MPI execution, and configured absolute parameter/output paths.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Add parameter-driven composable static elasticity in 2D and 3D, with independent vector modal constraints, section-weighted Schur/AL execution, pressure loads, imported radius and modal data, heterogeneous materials, boundary conditions, and refinement cycles. Legacy elasticity remains available.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Add amplitude-preserving selected vector lift modes, supplied physical Constraint metrics, and adapter-bound semantic output.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Expose field-preserving domain co-refinement, named refinement parameters, and global reports in the Coral registry, with a fullydistributed Poisson graph tested in serial and MPI.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Provide executable `both` co-refinement configurations for elasticity and reduced Poisson, with serial/MPI integration tests that refine both meshes and retain imported properties through a nonzero solve.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Add local co-refinement of two domains with space, embedded, and both strategies, preserving imported fields through refinement and delayed fullydistributed partitioning.
- [PR #257](https://github.com/luca-heltai/immersx/pull/257) Compare scalar and vector nonmatching penalty matrices with explicit deal.II particle assembly and step-80, including codimension-two and independent volume supports, in serial and MPI. Check distributed ownership and integration measure, and show separate one/two-rank performance history series.
- [PR #254](https://github.com/luca-heltai/immersx/pull/254) Allow parameter files to select the coupled Poisson multiplier, static elasticity displacement, and fiber velocity multiplier finite elements while preserving degree-based defaults.
- [PR #251](https://github.com/luca-heltai/immersx/pull/251) Add a performance history page with native comparisons, commit coverage, diagnostic regressions, and an archived benchmark pipeline that runs before documentation builds. Keep changed benchmark sources and explicit timing scopes in separate comparison series without relabelling historical data. Restrict official archive writes and Pages deployment to trusted master events, and verify batch provenance before writing.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Added a how-to guide for choosing weak-term integration supports.
- [PR #247](https://github.com/luca-heltai/immersx/pull/247) Added background-to-background weak terms over independent codimension-one supports, including nonlinear terms with frozen inputs.
- [PR #246](https://github.com/luca-heltai/immersx/pull/246) Added geometric quadrature supports for fixed tensor-product lifts and linear weak-term assembly on a distributed bulk mesh.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Added support-aware nonlinear weak terms over multiple finite element spaces and frozen inputs.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Added explicit embedded-mesh integration supports and native deal.II nonmatching mass assembly for value pairings.
- [PR #242](https://github.com/luca-heltai/immersx/pull/242) Added boundary integration supports and normal, conormal, and tangential expressions for linear weak terms.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Added cell supports for selecting the integration domain of linear weak terms.
- [PR #240](https://github.com/luca-heltai/immersx/pull/240) Added an opt-in benchmark for comparing matching mass assembly with deal.II.
- [PR #236](https://github.com/luca-heltai/immersx/pull/236) Add an explicit forward-Euler elasticity solver with a lumped mass matrix.
- [PR #235](https://github.com/luca-heltai/immersx/pull/235) Add a frequency-domain elasticity solver.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Add Coral operations for domain triangulations, VTK output, and output handlers.
- [PR #231](https://github.com/luca-heltai/immersx/pull/231) Added primitive static elasticity boundary conditions.
- [PR #229](https://github.com/luca-heltai/immersx/pull/229) Added Poisson primitive right-hand-side terms.
- [PR #228](https://github.com/luca-heltai/immersx/pull/228) Exposed compositional Coral boundary conditions.
- [PR #227](https://github.com/luca-heltai/immersx/pull/227) Exposed compositional Coral finite-element expressions.
- [PR #225](https://github.com/luca-heltai/immersx/pull/225) Added time-aware aggregable boundary conditions.
- [PR #224](https://github.com/luca-heltai/immersx/pull/224) Added solver-neutral known terms and boundary integration.
- [PR #220](https://github.com/luca-heltai/immersx/pull/220) Added the generic primitive scalar Poisson path.
- [PR #217](https://github.com/luca-heltai/immersx/pull/217) Added the solver-neutral `OutputHandler`.
- [PR #195](https://github.com/luca-heltai/immersx/pull/195) Added the KINSOL steady nonlinear adapter.
- [PR #193](https://github.com/luca-heltai/immersx/pull/193) Added nonlinear Observables and the MetricFlowX radial law.
- [PR #190](https://github.com/luca-heltai/immersx/pull/190) Implemented the tensor-product lifting backend and cleaned up transposes.
- [PR #184](https://github.com/luca-heltai/immersx/pull/184) Composed unified constraint sums.
- [PR #183](https://github.com/luca-heltai/immersx/pull/183) Supported scaled weak-term observables.
- [PR #178](https://github.com/luca-heltai/immersx/pull/178) Added the cached nonmatching weak-term backend.
- [PR #175](https://github.com/luca-heltai/immersx/pull/175) Added the finite-element space and Observable foundation API.
- [PR #174](https://github.com/luca-heltai/immersx/pull/174) Restored the standalone Poisson and ElasticStatic execution paths.
- [PR #173](https://github.com/luca-heltai/immersx/pull/173) Supported mixed-dimensional elastodynamics execution paths.
- [PR #168](https://github.com/luca-heltai/immersx/pull/168) Completed two-way MetricFlowX vessel-wall coupling.
- [PR #167](https://github.com/luca-heltai/immersx/pull/167) Added one-way MetricFlowX-Elastodynamics coupling.
- [PR #164](https://github.com/luca-heltai/immersx/pull/164) Added application-consistent initial conditions to the IDA adapter.
- [PR #161](https://github.com/luca-heltai/immersx/pull/161) Added optional MetricFlowX integration.
- [PR #132](https://github.com/luca-heltai/immersx/pull/132) Supported active and frozen finite-element expression sources.
- [PR #131](https://github.com/luca-heltai/immersx/pull/131) Added reusable imported finite-element fields.
- [PR #130](https://github.com/luca-heltai/immersx/pull/130) Coupled Poisson gradients and imported path fields to elasticity.
- [PR #125](https://github.com/luca-heltai/immersx/pull/125) Composed expression representations with the tensor-product lift.
- [PR #124](https://github.com/luca-heltai/immersx/pull/124) Inferred gradient sampling requirements for expressions.
- [PR #122](https://github.com/luca-heltai/immersx/pull/122) Added the scalar expression representation.
- [PR #121](https://github.com/luca-heltai/immersx/pull/121) Added retained finite-element sampling operators.
- [PR #120](https://github.com/luca-heltai/immersx/pull/120) Added the symbolic expression kernel.
- [PR #118](https://github.com/luca-heltai/immersx/pull/118) Added the parameterized tensor-product lift.
- [PR #116](https://github.com/luca-heltai/immersx/pull/116) Productionized the parameterized ElasticStatic problem.
- [PR #115](https://github.com/luca-heltai/immersx/pull/115) Added the coupled Poisson-to-elasticity vertical slice.
- [PR #107](https://github.com/luca-heltai/immersx/pull/107) Added geometry-aware Representation lifting.
- [PR #106](https://github.com/luca-heltai/immersx/pull/106) Added the weak `CouplingOperator` boundary.
- [PR #103](https://github.com/luca-heltai/immersx/pull/103) Added the standalone `ElasticStaticProblem`.
- [PR #102](https://github.com/luca-heltai/immersx/pull/102) Added the representation-driven load interaction.
- [PR #99](https://github.com/luca-heltai/immersx/pull/99) Added mixed Field component views.
- [PR #98](https://github.com/luca-heltai/immersx/pull/98) Added the minimal Field representation concept.
- [PR #94](https://github.com/luca-heltai/immersx/pull/94) Added deal.II semidiscrete contributors and automatic IDA composition.
- [PR #92](https://github.com/luca-heltai/immersx/pull/92) Added composable semidiscrete physics adapters.
- [PR #91](https://github.com/luca-heltai/immersx/pull/91) Added full-order fiber-reinforced elastodynamics.
- [PR #86](https://github.com/luca-heltai/immersx/pull/86) Added distributed IDA adapters for the standalone PDEs.
- [PR #85](https://github.com/luca-heltai/immersx/pull/85) Added the standalone transient Navier-Stokes solver.
- [PR #78](https://github.com/luca-heltai/immersx/pull/78) Added the representation-driven Poisson coupling prototype.
- [PR #75](https://github.com/luca-heltai/immersx/pull/75) Added the dimension-generic Poisson and Laplace-Beltrami solver.
- [PR #71](https://github.com/luca-heltai/immersx/pull/71) Allowed a zero reduced dimension in the tensor-product space.
- [PR #70](https://github.com/luca-heltai/immersx/pull/70) Made imported VTK fields usable in expressions.
- [PR #69](https://github.com/luca-heltai/immersx/pull/69) Added the VTK-backed reduced-coupling path to elasticity.
- [PR #62](https://github.com/luca-heltai/immersx/pull/62) Supported strong dynamic elasticity constraints.

### Changed

- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Interpret the harmonic reinforcement root dimension as the reinforcement mesh dimension and the space dimension as the matrix dimension. Use dimension one for a line reinforcement in 2D or 3D.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Share one frequency list between the harmonic matrix and fiber, and preserve the actual initial refinement when saving and reloading either supported parameter alias.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Support harmonic elasticity on embedded domains with `1 <= dim <= spacedim <= 3` and independent parameter subsections. Grid settings use `Domain`, while existing grid and refinement entries remain accepted.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Configure standalone and coupled MetricFlowX IDA integration, time bounds, output cadence, and tolerance ramps in the native `MetricFlowSystem<1, 3>/IDA parameters` section. Remove unused fixed-step controls from the coupled configuration.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Refresh eliminated Dirichlet contributions automatically when evaluating supported linear weak-term residuals, reusing the matrix while constraint entries stay fixed. Remove `with_lifted_affine_updates()` from the public API.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Write coupled MetricFlowX test and tutorial output in one directory per case, using `solid`, `flow`, and `pressure_jump` filenames.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Keep fiber IDA composition and state local to one solve, write the multiplier through semantic output, and move the uncoupled reference solve to validation tests.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Keep the Poisson-elasticity application focused on pressure traction and semantic output, with residual and traction checks in integration tests.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Simplify coupled Poisson setup and write its solution and multiplier fields directly through semantic output handlers.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Use fixed lower and upper diameter bounds for local mesh adjustment; the old refinement factor remains a deprecated single-mesh input.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Include matrix materialization in trace assembly timings and report synchronized maximum-rank MPI timings with explicit setup, action, and operation-count metadata.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Reuse trial and test basis values during linear assembly and input FEValues across cells with a shared mapping during nonlinear assembly.
- [PR #237](https://github.com/luca-heltai/immersx/pull/237) Replace custom MetricFlowX coupling with structural finite-element observables and cell integration.
- [PR #226](https://github.com/luca-heltai/immersx/pull/226) Associated boundary conditions with semantic fields.
- [PR #222](https://github.com/luca-heltai/immersx/pull/222) Returned typed semantic fields from the contributors.
- [PR #215](https://github.com/luca-heltai/immersx/pull/215) Replaced the 2D Coral facade with generic composition.
- [PR #214](https://github.com/luca-heltai/immersx/pull/214) Unified the Coral parameter-loading nodes.
- [PR #213](https://github.com/luca-heltai/immersx/pull/213) Introduced `Domain`-based mesh parameters and migrated the solvers.
- [PR #204](https://github.com/luca-heltai/immersx/pull/204) Refactored time parameter and integrator ownership.
- [PR #200](https://github.com/luca-heltai/immersx/pull/200) Added the deliberate application-coupling roadmap.
- [PR #199](https://github.com/luca-heltai/immersx/pull/199) Reduced memory use in elasticity.
- [PR #189](https://github.com/luca-heltai/immersx/pull/189) Used typed deal.II finite-element expressions for weak terms.
- [PR #188](https://github.com/luca-heltai/immersx/pull/188) Completed the unified fiber-constraint follow-up.
- [PR #187](https://github.com/luca-heltai/immersx/pull/187) Migrated the fiber driver to unified weak-term constraints.
- [PR #185](https://github.com/luca-heltai/immersx/pull/185) Moved the normal pressure load into `weak_term`.
- [PR #181](https://github.com/luca-heltai/immersx/pull/181) Unified Lagrange multiplier constraints over weak terms.
- [PR #176](https://github.com/luca-heltai/immersx/pull/176) Optimized weak-term assembly for simple finite-element paths.
- [PR #166](https://github.com/luca-heltai/immersx/pull/166) Refactored time parameter ownership around `TimeParameters`.
- [PR #162](https://github.com/luca-heltai/immersx/pull/162) Added parameter objects to the execution adapters.
- [PR #160](https://github.com/luca-heltai/immersx/pull/160) Modernized applications around the semantic execution adapters.
- [PR #159](https://github.com/luca-heltai/immersx/pull/159) Gave multiplier Interactions native output ownership.
- [PR #158](https://github.com/luca-heltai/immersx/pull/158) Wrote native output after composed solves.
- [PR #157](https://github.com/luca-heltai/immersx/pull/157) Simplified linear operator composition.
- [PR #152](https://github.com/luca-heltai/immersx/pull/152) Repaired the final semidiscrete composition stack.
- [PR #138](https://github.com/luca-heltai/immersx/pull/138) Hardened field transfer and pruned stale reduced-field APIs.
- [PR #128](https://github.com/luca-heltai/immersx/pull/128) Built pressure from deferred finite-element expressions.
- [PR #123](https://github.com/luca-heltai/immersx/pull/123) Supported semantic expression dependencies.
- [PR #117](https://github.com/luca-heltai/immersx/pull/117) Brought `ElasticStatic` to parity with elasticity.
- [PR #114](https://github.com/luca-heltai/immersx/pull/114) Simplified the public multiphysics composition API.
- [PR #113](https://github.com/luca-heltai/immersx/pull/113) Clarified Representation quantity spaces.
- [PR #109](https://github.com/luca-heltai/immersx/pull/109) Consolidated the Representation domain, evaluation, and transfer boundaries.
- [PR #108](https://github.com/luca-heltai/immersx/pull/108) Separated geometry maps from Representation lifting.
- [PR #105](https://github.com/luca-heltai/immersx/pull/105) Separated the Representation source and evaluation domain.
- [PR #104](https://github.com/luca-heltai/immersx/pull/104) Clarified Representation evaluation and linearization.
- [PR #100](https://github.com/luca-heltai/immersx/pull/100) Made `TensorProduct` a representation-first lifting.
- [PR #97](https://github.com/luca-heltai/immersx/pull/97) Replaced `TimeRole` with per-field differential component masks.
- [PR #83](https://github.com/luca-heltai/immersx/pull/83) Unified semantic residual fields and added mixed DAE integration.
- [PR #81](https://github.com/luca-heltai/immersx/pull/81) Dispatched applications from parameter dimensions.
- [PR #79](https://github.com/luca-heltai/immersx/pull/79) Stabilized the representation and constraint interaction architecture.
- [PR #76](https://github.com/luca-heltai/immersx/pull/76) Separated the reduced-space core from the VTK backend.
- [PR #73](https://github.com/luca-heltai/immersx/pull/73) Improved distributed tensor-product coupling.

### Fixed

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Retain all prescribed displacement components in the legacy elasticity weak Dirichlet stress terms. This restores consistency on simplex and hypercube meshes.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Identify the physical mesh reference cell collectively when MPI ranks own no cells. Ignore fully distributed artificial placeholders when selecting elements, quadrature, and search mappings; keep field views and geometry accessors local, and reject mixed physical meshes.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Import point and cell fields on simplex meshes with matching finite elements. Preserve those fields during initial refinement and use reference-cell mappings and smoothing rules during Domain co-refinement. Boundary recipes use the target field mapping; point-cloud refinement accepts serial preparation meshes and uses each cell's actual bounding box.

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Preserve multiplier mass contributions whose matrix rows belong to other MPI ranks.

- [PR #294](https://github.com/luca-heltai/immersx/pull/294) Avoid division by zero in small fiber multiplier Schur solves with deal.II 9.7, including configurable discontinuous multiplier spaces in serial and MPI.
- [PR #292](https://github.com/luca-heltai/immersx/pull/292) Evaluate lifted observables and nonlinear weak terms with each cell's active finite element and local DoF count, including hp terminal pressure fields. Preserve differentiated fields' execution numbering for compact and permuted layouts and check coupled Jacobians per residual block.
- [PR #292](https://github.com/luca-heltai/immersx/pull/292) Use MetricFlowX's native hp DoFHandler numbering and ghost indices for coupled area and velocity fields, including RCR capacitor pressures represented as a fifth FE component. Remove the conversion for the previous manually numbered capacitor layout.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Preserve saved Coral graphs by keeping known-source factory and execution-node keys unchanged when KnownTerm has a default integration support.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Apply harmonic 1D Dirichlet amplitudes only on physical boundary cells with deal.II 9.7, preserving nonzero complex boundary data across MPI partitions.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Retain essential source offsets in unreindexed nonmatching weak terms, including harmonic continuity between independently meshed full-dimensional materials.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Write harmonic displacement datasets inside the selected output directory even when its path has no trailing separator.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Use the correct Trilinos vector assembly mode in the harmonic complex-Arnoldi updates, which previously failed Debug assertions.
- [PR #289](https://github.com/luca-heltai/immersx/pull/289) Correct LU roundoff in LinearAdapter default direct solves with one residual correction using the cached matrix and factorization, including explicitly reused factorizations.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Report native IDA failures in Release, honor nonzero initial output times, and handle empty ranks, pooled block-layout changes, and callback failures during initialization.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Save coupled MetricFlowX fluid output with the accepted simulation times and write the initial state once in each output series. Check fluid, solid, and multiplier output times in the coupled and MMS application tests.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Support time-dependent Dirichlet data in the MetricFlowX elastodynamics coupling and use exact displacement and velocity boundary data in the cube MMS tutorials while retaining exact Neumann variants on both meshes.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Apply 1D Poisson Dirichlet data only to non-artificial cells with deal.II 9.7, preserving the pressure solution across MPI partitions.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Keep the Coral Poisson–elasticity workflow compatible with the simplified application interfaces and report traction balance from its displacement residual.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Handle exact Krylov breakdown in the default IDA GMRES path with modified Gram-Schmidt, including deal.II 9.7 builds.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Reduce coupled Jacobian setup costs by copying distributed matrices through their native backend, materializing each sum operand once, and avoiding repeated row extraction. Skip quadratic scans of empty blocks when converting the Jacobian to a monolithic matrix. Mark absent operator blocks as zero so block preconditioners skip unnecessary inverse applications.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Apply vessel-wall traction in solid momentum and native external-pressure feedback in flow, including boundary and interface equations, while preserving source-free continuity. Exchange reference-area ghost coefficients before constructing the distributed wall lift. Add an opt-in Schur correction for displacement-velocity coupling, retain small-load corrections, preserve callback errors, and stop on failed IDA steps in Release.
- [PR #264](https://github.com/luca-heltai/immersx/pull/264) Prevent the Navier–Stokes tutorial from timing out in Debug builds by reusing finite-element evaluations during nonlinear Jacobian assembly.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Use one lifted assembly for scalar and vector constraints, preserve constrained master sparsity, and allow explicit source Dirichlet-value updates while retaining collective static-data and constraint-structure guards. Keep scalar coefficient-output and vector coefficient-creation nodes distinct in the combined Coral registry.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Preserve affine Dirichlet contributions in lifted weak terms, assemble known loads and legacy coupling graphs across MPI owners, use relative inner Schur solves and check the true FGMRES residual, and reconstruct hanging-node values before FE output.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Fix vector lifted coupling on locally refined meshes, including nonzero affine boundary data.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Retain static inhomogeneous Dirichlet shifts in lifted weak constraints, update Observable dependencies on semantic registration, and assemble known loads into writable MPI vectors.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Use compiler-independent Coral report-operation keys so co-refinement graphs load on Linux and macOS.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Retain the reduced Poisson augmented-Lagrangian solution in the accepted solution vector as well as the output vector, so solution diagnostics reflect the nonzero solve.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Retain the reduced Poisson augmented-Lagrangian solution in the accepted solution vector as well as the output vector, so solution diagnostics reflect the nonzero solve.
- [PR #254](https://github.com/luca-heltai/immersx/pull/254) Register MetricFlowX time parameters before reading input and provide a mixed-flow preconditioner for coupled IDA solves. Exercise standalone and nonzero-inflow coupled applications in serial and MPI.
- [PR #253](https://github.com/luca-heltai/immersx/pull/253) Preserve every ambient vector component in native nonmatching assembly and give the fiber driver's matrix-only reference independent history.
- [PR #252](https://github.com/luca-heltai/immersx/pull/252) Select all ambient components of embedded vector fields and evolve the fiber driver's matrix-only backward-Euler reference from its own displacement and velocity history.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Use a straight cross-section in the lifted-support benchmark and preserve frozen-input mesh/mapping checks in the nonlinear input cache.
- [PR #247](https://github.com/luca-heltai/immersx/pull/247) Retain off-rank independent-support Jacobian entries and reject frozen inputs on incompatible background meshes or mappings.
- [PR #246](https://github.com/luca-heltai/immersx/pull/246) Request Jacobians for lifted geometry, preserve off-rank physical-support matrix entries, and route quadrature points collectively when a rank has no points.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Assemble nonlinear residuals into owned-only vectors, retain off-rank matching and lifted Jacobian entries, and reject frozen inputs on incompatible meshes or mappings.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Use ghosted imported inputs and owned-only outputs in distributed frozen-field tests.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Reject unsupported inhomogeneous source constraints in nonmatching and lifted linear pairings, including offsets changed after assembly, and reject inhomogeneous linear multiplier reactions.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Retain relevant rows in nonmatching coupling sparsity.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Preserve all ambient vector components in native nonmatching value pairings, including offset fields and hanging-node constraints.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Use particle-based assembly when deal.II's native nonmatching path cannot handle a distributed immersed mesh or a multi-rank background mesh.
- [PR #242](https://github.com/luca-heltai/immersx/pull/242) Include dynamic affine constraint contributions in explicit boundary supports and inferred boundary regions, including normal and conormal contractions.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Keep constrained residual and Jacobian rows condensed for reindexed mixed fields while preserving native diagonals for unreindexed square cell terms.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Preserve and dynamically refresh affine cell contributions in state, frozen-source, and multiplier residuals while retaining native constrained diagonal rows.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Derive locally relevant DoFs for FE-space views by default and retain off-rank matching matrix entries during distributed assembly.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Keep persisted Coral weak-term and continuity graphs readable across compilers.
- [PR #238](https://github.com/luca-heltai/immersx/pull/238) Avoid indexed writes to ghosted Trilinos vectors in `OutputHandler`.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Write the short Coral parameter values used by each graph run to a sibling file.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read point and cell fields from XML VTK mesh files.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Treat grid names with a file extension as mesh files even when the path is unresolved.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read XML VTK mesh files with VTK versions before 9.3.
- [PR #230](https://github.com/luca-heltai/immersx/pull/230) Fixed Coral first-run parameter initialization.
- [PR #223](https://github.com/luca-heltai/immersx/pull/223) Fixed the Coral CTest filter escaping.
- [PR #210](https://github.com/luca-heltai/immersx/pull/210) Fixed the Schur preconditioner breakdown on lucky termination.
- [PR #208](https://github.com/luca-heltai/immersx/pull/208) Fixed elastodynamics refinement and moving-boundary constraints.
- [PR #207](https://github.com/luca-heltai/immersx/pull/207) Fixed Newmark inhomogeneous boundary handling.
- [PR #205](https://github.com/luca-heltai/immersx/pull/205) Fixed the dynamic elasticity constraint instability.
- [PR #202](https://github.com/luca-heltai/immersx/pull/202) Fixed elastodynamics adapter constrained degrees of freedom.
- [PR #198](https://github.com/luca-heltai/immersx/pull/198) Fixed the inclusion catalog scalar read.
- [PR #191](https://github.com/luca-heltai/immersx/pull/191) Fixed sparse weak-term regression comparisons.
- [PR #180](https://github.com/luca-heltai/immersx/pull/180) Fixed the strict API documentation build.
- [PR #172](https://github.com/luca-heltai/immersx/pull/172) Fixed rotation-aware tensor-product vector modes.
- [PR #163](https://github.com/luca-heltai/immersx/pull/163) Fixed MetricFlowX IDA parameter construction.
- [PR #156](https://github.com/luca-heltai/immersx/pull/156) Fixed distributed Trilinos Schur preconditioner maps.
- [PR #155](https://github.com/luca-heltai/immersx/pull/155) Fixed the Fiber IDA initial solver lifetime.
- [PR #137](https://github.com/luca-heltai/immersx/pull/137) Fixed reduced-coupling assembly and deduplicated imported-field transfer.
- [PR #136](https://github.com/luca-heltai/immersx/pull/136) Fixed elasticity compilation without VTK.
- [PR #135](https://github.com/luca-heltai/immersx/pull/135) Fixed duplicate symbolic outputs in the tensor-product right-hand side.
- [PR #112](https://github.com/luca-heltai/immersx/pull/112) Fixed the matrix-free transfer test build.
- [PR #89](https://github.com/luca-heltai/immersx/pull/89) Fixed parallel elasticity assembly for the coupled case.
- [PR #74](https://github.com/luca-heltai/immersx/pull/74) Addressed a failing test.
- [PR #68](https://github.com/luca-heltai/immersx/pull/68) Removed compiler warnings with deal.II 9.8.
- [PR #67](https://github.com/luca-heltai/immersx/pull/67) Fixed the inclusion basis-function scaling.

### Removed

- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Remove the fiber driver's `matrix_only_displacement_difference()` diagnostic; validation evolves an independent dense reference instead.

- [PR #192](https://github.com/luca-heltai/immersx/pull/192) Purged the transitional representation architecture.
- [PR #186](https://github.com/luca-heltai/immersx/pull/186) Removed the obsolete representation load interaction.
- [PR #96](https://github.com/luca-heltai/immersx/pull/96) Removed the semidiscrete compatibility layers.
- [PR #93](https://github.com/luca-heltai/immersx/pull/93) Removed the obsolete monolithic compatibility layer.

### Documentation

- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Document simplex meshes in the elastodynamics tutorial and reading the solid mesh from a file in the MetricFlowX–elastodynamics tutorial.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Document Field-local IDA tolerances, startup ramps, progress statistics, and restart behavior.

- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Profile the one-vessel elastodynamics MMS, explain Jacobian materialization and nested solver costs, and compare iterative and serial UMFPACK timings, memory, and numerical results.

- [PR #273](https://github.com/luca-heltai/immersx/pull/273) Render the MetricFlowX coupling equations and elastodynamics state variables correctly, and check all generated pages for math rendering errors in CI.
- [PR #272](https://github.com/luca-heltai/immersx/pull/272) Add an English onboarding guide for elastostatic–MetricFlowX coupling, from independent applications and file exchange to a monolithic IDAAdapter solve.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Document lifted Poisson composition, modal coefficient ordering and measure, solver policy, adaptive output, Coral execution, and the tested scope of legacy parity.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Document composable static elasticity, its equations and extension points, generated application/Coral inputs, and unsupported legacy capabilities.

- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Document application-level `both` configuration, migration from the old refinement factor, and executable elasticity and reduced Poisson examples.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Explain the local co-refinement lifecycle and publish random-tree mesh comparisons generated by a GoogleTest, with commands to reproduce the figures.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Explain support-refinement error control, ghosted frozen inputs, and geometry snapshots for independent supports.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Document cell, boundary, independent immersed, and lifted weak-term supports, and distinguish historical reports from new measurements.
- [PR #248](https://github.com/luca-heltai/immersx/pull/248) Document reverse-chronological changelog ordering and checks for missing entries from merged PRs.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Allow pull requests without a related issue to omit the closing reference.
- [PR #196](https://github.com/luca-heltai/immersx/pull/196) Refreshed the documentation for the current architecture.
- [PR #95](https://github.com/luca-heltai/immersx/pull/95) Updated the agent instructions and the contributing guide.
- [PR #82](https://github.com/luca-heltai/immersx/pull/82) Revised the core architecture documentation.
- [PR #77](https://github.com/luca-heltai/immersx/pull/77) Documented the ImmersX core architecture.

### Tutorial

- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Add a frequency-sweep tutorial for a 1D fiber in a 3D matrix, with a 2D section lift and runnable independent and shared displacement inputs.

- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Explain independent and shared harmonic displacements, normal volume lifts, material units, and full-dimensional reinforcement, with runnable examples of both formulations.

- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Document harmonic fiber reinforcement, its phase convention and extension API, and add a runnable frequency-sweep tutorial checked in serial and MPI by CTest.

- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Use a single coarse vessel VTK for the cube MMS cases, with parameter-controlled refinement and reference area retained by Domain during flow-mesh refinement.

- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Add a one-vessel manufactured-solution tutorial with piecewise displacement, nonzero pressure jump, native pressure feedback, and volume supplied by the inlet. Keep centerline cells within a factor of two of the solid axial cell lengths and integrate the lift with four points per solid axial cell.

- [PR #234](https://github.com/luca-heltai/immersx/pull/234) Run the tutorial platform through noVNC with remote Slurm execution and selectable Debug or Release packages.
- [PR #233](https://github.com/luca-heltai/immersx/pull/233) Build multi-architecture Coral, visualizer, and platform images for the dealiiX tutorial stack.
- [PR #127](https://github.com/luca-heltai/immersx/pull/127) Updated the tutorial documentation.
- [PR #84](https://github.com/luca-heltai/immersx/pull/84) Added the standalone first-order elastodynamics solver and its tutorials.
- [PR #80](https://github.com/luca-heltai/immersx/pull/80) Added the random-particle ReducedPoisson tutorials.

### Build and packaging

- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Use the unified solution-transfer header so legacy applications build with deal.II 9.9.

- [PR #269](https://github.com/luca-heltai/immersx/pull/269) Link each ImmersX variant to the matching MetricFlowX and deal.II libraries in DebugRelease builds, retain compatible legacy single-variant packages, and check generated linkage in CTest.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Include set serialization explicitly for distributed static-elasticity material IDs so the application builds with deal.II 9.7.1 and Boost on Linux.

- [PR #219](https://github.com/luca-heltai/immersx/pull/219) Added the Coral Debug and Release CI jobs.
- [PR #211](https://github.com/luca-heltai/immersx/pull/211) Configured the shared ccache and VS Code path mapping.
- [PR #201](https://github.com/luca-heltai/immersx/pull/201) Restored DebugRelease single-config builds.
- [PR #194](https://github.com/luca-heltai/immersx/pull/194) Cancelled superseded GitHub Actions runs.
- [PR #88](https://github.com/luca-heltai/immersx/pull/88) Prepared ImmersX for packaging.
- [PR #87](https://github.com/luca-heltai/immersx/pull/87) Centralized build-tree test inputs and added the contributing guide.

### Testing and CI

- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Check native elastodynamics with quadratic displacement elements, nonzero forcing, and time-dependent boundary data on tetrahedra and hexahedra in serial and MPI.

- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Test ElastodynamicsSolver on a 48-tetrahedron Gmsh cube and a hexahedral cube with a nonzero linear-in-space, quadratic-in-time exact solution, in serial and on two MPI ranks.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Run harmonic reinforcement sweeps through parameter-file dispatch for all six dimension pairs, both formulations, and serial/two-rank MPI execution, checking the expected displacement and multiplier datasets. Verify owned affine offsets against a separately reconstructed finite-element field.

- [PR #282](https://github.com/luca-heltai/immersx/pull/282) Reduce mesh sizes in the 2D cross-section checks and mixed-boundary time-integrator regression without changing analytic values or tolerances. Keep all eight time steps and run the faster integrator regression in the quick serial/MPI groups. Synchronize MPI application tests before checking files written by rank zero.

- [PR #281](https://github.com/luca-heltai/immersx/pull/281) Keep standalone header and typed category checks while removing unused links and demonstration builds. Skip runtime search paths in configure-only package fixtures and run independent CI tests in two slots with shared output protected.

- [PR #281](https://github.com/luca-heltai/immersx/pull/281) Cancel superseded PR documentation previews while preserving official performance measurements.

- [PR #280](https://github.com/luca-heltai/immersx/pull/280) Preserve compiler cache entries between CI runs by using the restored cache directory and refreshing saved caches.

- [PR #274](https://github.com/luca-heltai/immersx/pull/274) Avoid deal.II deprecation warnings when building the lifted Poisson parity tests.

- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Bound Coral CI builds to two compiler processes to reduce peak memory when both lifted applications are enabled.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Compare lifted Poisson coupling entries, modal loads, and bulk/multiplier coefficients with legacy Schur and AL solves on cylinders, variable radii, and a network; independently check nonzero forcing and Dirichlet data with hanging nodes in serial and MPI. Use feasible primal/multiplier discretizations and check residuals in the prescribed-constraint roadmap examples.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Compare full static elasticity fields, modal loads and operator actions with native oracles in serial and MPI, and check generated parameter examples, output fields, cycle counts, and mesh refinement through CTest. Keep the ordinary category-overlap fixture targeted when new lifted suites are added.

- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Run co-refinement serial/MPI cases through CTest using environment-based filters, and check velocity boundary data collectively when a rank owns no prescribed boundary DoFs.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Retain original review benchmark reports and their immutable measured source provenance without relabelling them as repaired-head measurements.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Check boundary mass against MatrixCreator and piecewise FE trace convergence against a fitted one-dimensional mass matrix, while retaining distributed nonlinear and frozen-input regressions.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Compare lifted source-to-bulk assembly with an independent physical-quadrature reference using owned writable vectors, and retain historical Debug/Release benchmark records without relabeling their source.
- [PR #248](https://github.com/luca-heltai/immersx/pull/248) Run pull-request tests for stacked branches targeting any branch.
- [PR #247](https://github.com/luca-heltai/immersx/pull/247) Test replicated and fully distributed embedded support routing, scalar/vector nonlinear actions, transposes, and finite differences across partitions.
- [PR #246](https://github.com/luca-heltai/immersx/pull/246) Classify the compiled GoogleTest registry, reject missing or overlapping categories, discover tests without MPI initialization, and include shared tests in MPI runs.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Added a MeshWorker reference and a paired nonlinear residual/Jacobian benchmark for heterogeneous weak terms with frozen inputs.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Build and test Coral plugins against the `dealiix-tutorial` Coral branch.
- [PR #221](https://github.com/luca-heltai/immersx/pull/221) Added the issue #209 elastodynamics equivalence regression.
- [PR #182](https://github.com/luca-heltai/immersx/pull/182) Migrated the coupled regressions to unified constraints.
- [PR #177](https://github.com/luca-heltai/immersx/pull/177) Reorganized the tests into labeled CTest suites.
- [PR #169](https://github.com/luca-heltai/immersx/pull/169) Added the one-vessel two-way MetricFlowX-Elastodynamics MMS verification.
- [PR #90](https://github.com/luca-heltai/immersx/pull/90) Reduced the Debug testsuite runtime.

## [0.3] - 2026-05-11

### Added

- [PR #61](https://github.com/luca-heltai/immersx/pull/61) Added per-boundary-id Dirichlet and Neumann data with modulated parsed functions.
- [PR #60](https://github.com/luca-heltai/immersx/pull/60) Added fully distributed triangulation support to the elasticity problem.
- [PR #51](https://github.com/luca-heltai/immersx/pull/51) Added per-material-id material properties.

### Changed

- [PR #55](https://github.com/luca-heltai/immersx/pull/55) Grouped inclusions that fall in the same cell.
- [PR #54](https://github.com/luca-heltai/immersx/pull/54) Moved vessel-pressure computation and output into coupled elasticity.
- [PR #53](https://github.com/luca-heltai/immersx/pull/53) Cleaned up and refactored coupled elasticity.
- [PR #50](https://github.com/luca-heltai/immersx/pull/50) Incorporated the brain-dynamics changes contributed by Yashu.

### Fixed

- [PR #58](https://github.com/luca-heltai/immersx/pull/58) Fixed the reduced Poisson problem.

### Documentation

- [PR #64](https://github.com/luca-heltai/immersx/pull/64) Rebranded the project with a new logo and README.
- [PR #56](https://github.com/luca-heltai/immersx/pull/56) Full documentation pass over the missing classes.

### Build and packaging

- [PR #57](https://github.com/luca-heltai/immersx/pull/57) Updated the GitHub Pages workflow.

### Testing and CI

- [PR #59](https://github.com/luca-heltai/immersx/pull/59) Added the liver-benchmark mesh preparation files.

## [0.2] - 2026-01-29

### Added

- [PR #44](https://github.com/luca-heltai/immersx/pull/44) Added a `properties` member and a vertex- and cell-agnostic field transfer.
- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Added rigid-body mode extraction for AMG preconditioning.
- [PR #40](https://github.com/luca-heltai/immersx/pull/40) Added VTK utilities for reading cell and point data.
- [PR #36](https://github.com/luca-heltai/immersx/pull/36) Added local refinement support for deal.II inclusions.
- [PR #32](https://github.com/luca-heltai/immersx/pull/32) Added initial deal.II-based inclusion support.
- [PR #31](https://github.com/luca-heltai/immersx/pull/31) Added parameter-file-driven inclusion data and corrected pressure and stress output.
- [PR #27](https://github.com/luca-heltai/immersx/pull/27) Added the initial coupled elasticity problem.

### Changed

- [PR #48](https://github.com/luca-heltai/immersx/pull/48) Updated the FVM headers.
- [PR #25](https://github.com/luca-heltai/immersx/pull/25) Updated the Docker environment and the 1D solver library download.

### Fixed

- [PR #49](https://github.com/luca-heltai/immersx/pull/49) Fixed compiler warnings.
- [PR #47](https://github.com/luca-heltai/immersx/pull/47) Removed the duplicated mass matrix from elasticity.
- [PR #46](https://github.com/luca-heltai/immersx/pull/46) Added missing VTK compile flags.
- [PR #43](https://github.com/luca-heltai/immersx/pull/43) Fixed the scaling of the Reduced Poisson problem.
- [PR #30](https://github.com/luca-heltai/immersx/pull/30) Addressed failing tests.

### Testing and CI

- [PR #33](https://github.com/luca-heltai/immersx/pull/33) Added an inclusion-orientation regression test for the solution and multiplier norms.
- [PR #28](https://github.com/luca-heltai/immersx/pull/28) Added the Lorthois datasets.

## [0.1] - 2024-11-07

### Added

- [PR #22](https://github.com/luca-heltai/immersx/pull/22) Added a geometric multigrid matrix-free preconditioner for the elasticity block.
- [PR #20](https://github.com/luca-heltai/immersx/pull/20) Added a matrix-free `CouplingOperator` for the action and transpose of the restriction operator.
- [PR #19](https://github.com/luca-heltai/immersx/pull/19) Added a matrix-free bulk evaluator.
- [PR #18](https://github.com/luca-heltai/immersx/pull/18) Enabled distributed parallel runs.
- [PR #13](https://github.com/luca-heltai/immersx/pull/13) Added local elasticity assembly.
- [PR #11](https://github.com/luca-heltai/immersx/pull/11) Added Dirichlet-only elasticity boundary conditions.
- [PR #5](https://github.com/luca-heltai/immersx/pull/5) Added Neumann and Robin boundary conditions.

### Changed

- [PR #23](https://github.com/luca-heltai/immersx/pull/23) Updated the license and repository metadata.
- [PR #8](https://github.com/luca-heltai/immersx/pull/8) Split declarations and definitions into separate files.
- [PR #6](https://github.com/luca-heltai/immersx/pull/6) Moved the `ReferenceInclusion` struct into its own header.

### Fixed

- [PR #17](https://github.com/luca-heltai/immersx/pull/17) Provided the locally relevant row index set when building distributed sparsity patterns.
- [PR #12](https://github.com/luca-heltai/immersx/pull/12) Fixed rotation handling.

### Testing and CI

- [PR #14](https://github.com/luca-heltai/immersx/pull/14) Added rotation tests.
