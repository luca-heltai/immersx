# Changelog

All notable changes to ImmersX are documented in this file. Each entry
begins with a link to the pull request that introduced the change.

The file is included in the documentation and is published at
`doc/changes.md`. Add new entries under `[Unreleased]` using the
skeleton in `.github/CHANGELOG_TEMPLATE.md`.

## [Unreleased]

### Added

- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Add Coral operations for domain triangulations, VTK output, and output handlers.
- [PR #62](https://github.com/luca-heltai/immersx/pull/62) Supported strong dynamic elasticity constraints.
- [PR #69](https://github.com/luca-heltai/immersx/pull/69) Added the VTK-backed reduced-coupling path to elasticity.
- [PR #70](https://github.com/luca-heltai/immersx/pull/70) Made imported VTK fields usable in expressions.
- [PR #71](https://github.com/luca-heltai/immersx/pull/71) Allowed a zero reduced dimension in the tensor-product space.
- [PR #75](https://github.com/luca-heltai/immersx/pull/75) Added the dimension-generic Poisson and Laplace-Beltrami solver.
- [PR #78](https://github.com/luca-heltai/immersx/pull/78) Added the representation-driven Poisson coupling prototype.
- [PR #85](https://github.com/luca-heltai/immersx/pull/85) Added the standalone transient Navier-Stokes solver.
- [PR #86](https://github.com/luca-heltai/immersx/pull/86) Added distributed IDA adapters for the standalone PDEs.
- [PR #91](https://github.com/luca-heltai/immersx/pull/91) Added full-order fiber-reinforced elastodynamics.
- [PR #92](https://github.com/luca-heltai/immersx/pull/92) Added composable semidiscrete physics adapters.
- [PR #94](https://github.com/luca-heltai/immersx/pull/94) Added deal.II semidiscrete contributors and automatic IDA composition.
- [PR #98](https://github.com/luca-heltai/immersx/pull/98) Added the minimal Field representation concept.
- [PR #99](https://github.com/luca-heltai/immersx/pull/99) Added mixed Field component views.
- [PR #102](https://github.com/luca-heltai/immersx/pull/102) Added the representation-driven load interaction.
- [PR #103](https://github.com/luca-heltai/immersx/pull/103) Added the standalone `ElasticStaticProblem`.
- [PR #106](https://github.com/luca-heltai/immersx/pull/106) Added the weak `CouplingOperator` boundary.
- [PR #107](https://github.com/luca-heltai/immersx/pull/107) Added geometry-aware Representation lifting.
- [PR #115](https://github.com/luca-heltai/immersx/pull/115) Added the coupled Poisson-to-elasticity vertical slice.
- [PR #116](https://github.com/luca-heltai/immersx/pull/116) Productionized the parameterized ElasticStatic problem.
- [PR #118](https://github.com/luca-heltai/immersx/pull/118) Added the parameterized tensor-product lift.
- [PR #120](https://github.com/luca-heltai/immersx/pull/120) Added the symbolic expression kernel.
- [PR #121](https://github.com/luca-heltai/immersx/pull/121) Added retained finite-element sampling operators.
- [PR #122](https://github.com/luca-heltai/immersx/pull/122) Added the scalar expression representation.
- [PR #124](https://github.com/luca-heltai/immersx/pull/124) Inferred gradient sampling requirements for expressions.
- [PR #125](https://github.com/luca-heltai/immersx/pull/125) Composed expression representations with the tensor-product lift.
- [PR #130](https://github.com/luca-heltai/immersx/pull/130) Coupled Poisson gradients and imported path fields to elasticity.
- [PR #131](https://github.com/luca-heltai/immersx/pull/131) Added reusable imported finite-element fields.
- [PR #132](https://github.com/luca-heltai/immersx/pull/132) Supported active and frozen finite-element expression sources.
- [PR #161](https://github.com/luca-heltai/immersx/pull/161) Added optional MetricFlowX integration.
- [PR #164](https://github.com/luca-heltai/immersx/pull/164) Added application-consistent initial conditions to the IDA adapter.
- [PR #167](https://github.com/luca-heltai/immersx/pull/167) Added one-way MetricFlowX-Elastodynamics coupling.
- [PR #168](https://github.com/luca-heltai/immersx/pull/168) Completed two-way MetricFlowX vessel-wall coupling.
- [PR #173](https://github.com/luca-heltai/immersx/pull/173) Supported mixed-dimensional elastodynamics execution paths.
- [PR #174](https://github.com/luca-heltai/immersx/pull/174) Restored the standalone Poisson and ElasticStatic execution paths.
- [PR #175](https://github.com/luca-heltai/immersx/pull/175) Added the finite-element space and Observable foundation API.
- [PR #178](https://github.com/luca-heltai/immersx/pull/178) Added the cached nonmatching weak-term backend.
- [PR #183](https://github.com/luca-heltai/immersx/pull/183) Supported scaled weak-term observables.
- [PR #184](https://github.com/luca-heltai/immersx/pull/184) Composed unified constraint sums.
- [PR #190](https://github.com/luca-heltai/immersx/pull/190) Implemented the tensor-product lifting backend and cleaned up transposes.
- [PR #193](https://github.com/luca-heltai/immersx/pull/193) Added nonlinear Observables and the MetricFlowX radial law.
- [PR #195](https://github.com/luca-heltai/immersx/pull/195) Added the KINSOL steady nonlinear adapter.
- [PR #217](https://github.com/luca-heltai/immersx/pull/217) Added the solver-neutral `OutputHandler`.
- [PR #220](https://github.com/luca-heltai/immersx/pull/220) Added the generic primitive scalar Poisson path.
- [PR #224](https://github.com/luca-heltai/immersx/pull/224) Added solver-neutral known terms and boundary integration.
- [PR #225](https://github.com/luca-heltai/immersx/pull/225) Added time-aware aggregable boundary conditions.
- [PR #227](https://github.com/luca-heltai/immersx/pull/227) Exposed compositional Coral finite-element expressions.
- [PR #228](https://github.com/luca-heltai/immersx/pull/228) Exposed compositional Coral boundary conditions.
- [PR #229](https://github.com/luca-heltai/immersx/pull/229) Added Poisson primitive right-hand-side terms.
- [PR #231](https://github.com/luca-heltai/immersx/pull/231) Added primitive static elasticity boundary conditions.

### Changed

- [PR #73](https://github.com/luca-heltai/immersx/pull/73) Improved distributed tensor-product coupling.
- [PR #76](https://github.com/luca-heltai/immersx/pull/76) Separated the reduced-space core from the VTK backend.
- [PR #79](https://github.com/luca-heltai/immersx/pull/79) Stabilized the representation and constraint interaction architecture.
- [PR #81](https://github.com/luca-heltai/immersx/pull/81) Dispatched applications from parameter dimensions.
- [PR #83](https://github.com/luca-heltai/immersx/pull/83) Unified semantic residual fields and added mixed DAE integration.
- [PR #97](https://github.com/luca-heltai/immersx/pull/97) Replaced `TimeRole` with per-field differential component masks.
- [PR #100](https://github.com/luca-heltai/immersx/pull/100) Made `TensorProduct` a representation-first lifting.
- [PR #104](https://github.com/luca-heltai/immersx/pull/104) Clarified Representation evaluation and linearization.
- [PR #105](https://github.com/luca-heltai/immersx/pull/105) Separated the Representation source and evaluation domain.
- [PR #108](https://github.com/luca-heltai/immersx/pull/108) Separated geometry maps from Representation lifting.
- [PR #109](https://github.com/luca-heltai/immersx/pull/109) Consolidated the Representation domain, evaluation, and transfer boundaries.
- [PR #113](https://github.com/luca-heltai/immersx/pull/113) Clarified Representation quantity spaces.
- [PR #114](https://github.com/luca-heltai/immersx/pull/114) Simplified the public multiphysics composition API.
- [PR #117](https://github.com/luca-heltai/immersx/pull/117) Brought `ElasticStatic` to parity with elasticity.
- [PR #123](https://github.com/luca-heltai/immersx/pull/123) Supported semantic expression dependencies.
- [PR #128](https://github.com/luca-heltai/immersx/pull/128) Built pressure from deferred finite-element expressions.
- [PR #138](https://github.com/luca-heltai/immersx/pull/138) Hardened field transfer and pruned stale reduced-field APIs.
- [PR #152](https://github.com/luca-heltai/immersx/pull/152) Repaired the final semidiscrete composition stack.
- [PR #157](https://github.com/luca-heltai/immersx/pull/157) Simplified linear operator composition.
- [PR #158](https://github.com/luca-heltai/immersx/pull/158) Wrote native output after composed solves.
- [PR #159](https://github.com/luca-heltai/immersx/pull/159) Gave multiplier Interactions native output ownership.
- [PR #160](https://github.com/luca-heltai/immersx/pull/160) Modernized applications around the semantic execution adapters.
- [PR #162](https://github.com/luca-heltai/immersx/pull/162) Added parameter objects to the execution adapters.
- [PR #166](https://github.com/luca-heltai/immersx/pull/166) Refactored time parameter ownership around `TimeParameters`.
- [PR #176](https://github.com/luca-heltai/immersx/pull/176) Optimized weak-term assembly for simple finite-element paths.
- [PR #181](https://github.com/luca-heltai/immersx/pull/181) Unified Lagrange multiplier constraints over weak terms.
- [PR #185](https://github.com/luca-heltai/immersx/pull/185) Moved the normal pressure load into `weak_term`.
- [PR #187](https://github.com/luca-heltai/immersx/pull/187) Migrated the fiber driver to unified weak-term constraints.
- [PR #188](https://github.com/luca-heltai/immersx/pull/188) Completed the unified fiber-constraint follow-up.
- [PR #189](https://github.com/luca-heltai/immersx/pull/189) Used typed deal.II finite-element expressions for weak terms.
- [PR #199](https://github.com/luca-heltai/immersx/pull/199) Reduced memory use in elasticity.
- [PR #200](https://github.com/luca-heltai/immersx/pull/200) Added the deliberate application-coupling roadmap.
- [PR #204](https://github.com/luca-heltai/immersx/pull/204) Refactored time parameter and integrator ownership.
- [PR #213](https://github.com/luca-heltai/immersx/pull/213) Introduced `Domain`-based mesh parameters and migrated the solvers.
- [PR #214](https://github.com/luca-heltai/immersx/pull/214) Unified the Coral parameter-loading nodes.
- [PR #215](https://github.com/luca-heltai/immersx/pull/215) Replaced the 2D Coral facade with generic composition.
- [PR #222](https://github.com/luca-heltai/immersx/pull/222) Returned typed semantic fields from the contributors.
- [PR #226](https://github.com/luca-heltai/immersx/pull/226) Associated boundary conditions with semantic fields.

### Fixed

- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Write the short Coral parameter values used by each graph run to a sibling file.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read point and cell fields from XML VTK mesh files.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Treat grid names with a file extension as mesh files even when the path is unresolved.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read XML VTK mesh files with VTK versions before 9.3.
- [PR #67](https://github.com/luca-heltai/immersx/pull/67) Fixed the inclusion basis-function scaling.
- [PR #68](https://github.com/luca-heltai/immersx/pull/68) Removed compiler warnings with deal.II 9.8.
- [PR #74](https://github.com/luca-heltai/immersx/pull/74) Addressed a failing test.
- [PR #89](https://github.com/luca-heltai/immersx/pull/89) Fixed parallel elasticity assembly for the coupled case.
- [PR #112](https://github.com/luca-heltai/immersx/pull/112) Fixed the matrix-free transfer test build.
- [PR #135](https://github.com/luca-heltai/immersx/pull/135) Fixed duplicate symbolic outputs in the tensor-product right-hand side.
- [PR #136](https://github.com/luca-heltai/immersx/pull/136) Fixed elasticity compilation without VTK.
- [PR #137](https://github.com/luca-heltai/immersx/pull/137) Fixed reduced-coupling assembly and deduplicated imported-field transfer.
- [PR #155](https://github.com/luca-heltai/immersx/pull/155) Fixed the Fiber IDA initial solver lifetime.
- [PR #156](https://github.com/luca-heltai/immersx/pull/156) Fixed distributed Trilinos Schur preconditioner maps.
- [PR #163](https://github.com/luca-heltai/immersx/pull/163) Fixed MetricFlowX IDA parameter construction.
- [PR #172](https://github.com/luca-heltai/immersx/pull/172) Fixed rotation-aware tensor-product vector modes.
- [PR #180](https://github.com/luca-heltai/immersx/pull/180) Fixed the strict API documentation build.
- [PR #191](https://github.com/luca-heltai/immersx/pull/191) Fixed sparse weak-term regression comparisons.
- [PR #198](https://github.com/luca-heltai/immersx/pull/198) Fixed the inclusion catalog scalar read.
- [PR #202](https://github.com/luca-heltai/immersx/pull/202) Fixed elastodynamics adapter constrained degrees of freedom.
- [PR #205](https://github.com/luca-heltai/immersx/pull/205) Fixed the dynamic elasticity constraint instability.
- [PR #207](https://github.com/luca-heltai/immersx/pull/207) Fixed Newmark inhomogeneous boundary handling.
- [PR #208](https://github.com/luca-heltai/immersx/pull/208) Fixed elastodynamics refinement and moving-boundary constraints.
- [PR #210](https://github.com/luca-heltai/immersx/pull/210) Fixed the Schur preconditioner breakdown on lucky termination.
- [PR #223](https://github.com/luca-heltai/immersx/pull/223) Fixed the Coral CTest filter escaping.
- [PR #230](https://github.com/luca-heltai/immersx/pull/230) Fixed Coral first-run parameter initialization.

### Removed

- [PR #93](https://github.com/luca-heltai/immersx/pull/93) Removed the obsolete monolithic compatibility layer.
- [PR #96](https://github.com/luca-heltai/immersx/pull/96) Removed the semidiscrete compatibility layers.
- [PR #186](https://github.com/luca-heltai/immersx/pull/186) Removed the obsolete representation load interaction.
- [PR #192](https://github.com/luca-heltai/immersx/pull/192) Purged the transitional representation architecture.

### Documentation

- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Allow pull requests without a related issue to omit the closing reference.
- [PR #77](https://github.com/luca-heltai/immersx/pull/77) Documented the ImmersX core architecture.
- [PR #82](https://github.com/luca-heltai/immersx/pull/82) Revised the core architecture documentation.
- [PR #95](https://github.com/luca-heltai/immersx/pull/95) Updated the agent instructions and the contributing guide.
- [PR #196](https://github.com/luca-heltai/immersx/pull/196) Refreshed the documentation for the current architecture.

### Tutorial

- [PR #233](https://github.com/luca-heltai/immersx/pull/233) Build multi-architecture Coral, visualizer, and platform images for the dealiiX tutorial stack.
- [PR #N](https://github.com/luca-heltai/immersx/pull/N) Run the tutorial platform through noVNC with remote Slurm execution and selectable Debug or Release packages.
- [PR #80](https://github.com/luca-heltai/immersx/pull/80) Added the random-particle ReducedPoisson tutorials.
- [PR #84](https://github.com/luca-heltai/immersx/pull/84) Added the standalone first-order elastodynamics solver and its tutorials.
- [PR #127](https://github.com/luca-heltai/immersx/pull/127) Updated the tutorial documentation.

### Build and packaging

- [PR #87](https://github.com/luca-heltai/immersx/pull/87) Centralized build-tree test inputs and added the contributing guide.
- [PR #88](https://github.com/luca-heltai/immersx/pull/88) Prepared ImmersX for packaging.
- [PR #194](https://github.com/luca-heltai/immersx/pull/194) Cancelled superseded GitHub Actions runs.
- [PR #201](https://github.com/luca-heltai/immersx/pull/201) Restored DebugRelease single-config builds.
- [PR #211](https://github.com/luca-heltai/immersx/pull/211) Configured the shared ccache and VS Code path mapping.
- [PR #219](https://github.com/luca-heltai/immersx/pull/219) Added the Coral Debug and Release CI jobs.

### Testing and CI

- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Build and test Coral plugins against the `dealiix-tutorial` Coral branch.
- [PR #90](https://github.com/luca-heltai/immersx/pull/90) Reduced the Debug testsuite runtime.
- [PR #169](https://github.com/luca-heltai/immersx/pull/169) Added the one-vessel two-way MetricFlowX-Elastodynamics MMS verification.
- [PR #177](https://github.com/luca-heltai/immersx/pull/177) Reorganized the tests into labeled CTest suites.
- [PR #182](https://github.com/luca-heltai/immersx/pull/182) Migrated the coupled regressions to unified constraints.
- [PR #221](https://github.com/luca-heltai/immersx/pull/221) Added the issue #209 elastodynamics equivalence regression.

## [0.3] - 2026-05-11

### Added

- [PR #51](https://github.com/luca-heltai/immersx/pull/51) Added per-material-id material properties.
- [PR #60](https://github.com/luca-heltai/immersx/pull/60) Added fully distributed triangulation support to the elasticity problem.
- [PR #61](https://github.com/luca-heltai/immersx/pull/61) Added per-boundary-id Dirichlet and Neumann data with modulated parsed functions.

### Changed

- [PR #50](https://github.com/luca-heltai/immersx/pull/50) Incorporated the brain-dynamics changes contributed by Yashu.
- [PR #53](https://github.com/luca-heltai/immersx/pull/53) Cleaned up and refactored coupled elasticity.
- [PR #54](https://github.com/luca-heltai/immersx/pull/54) Moved vessel-pressure computation and output into coupled elasticity.
- [PR #55](https://github.com/luca-heltai/immersx/pull/55) Grouped inclusions that fall in the same cell.

### Fixed

- [PR #58](https://github.com/luca-heltai/immersx/pull/58) Fixed the reduced Poisson problem.

### Documentation

- [PR #56](https://github.com/luca-heltai/immersx/pull/56) Full documentation pass over the missing classes.
- [PR #64](https://github.com/luca-heltai/immersx/pull/64) Rebranded the project with a new logo and README.

### Build and packaging

- [PR #57](https://github.com/luca-heltai/immersx/pull/57) Updated the GitHub Pages workflow.

### Testing and CI

- [PR #59](https://github.com/luca-heltai/immersx/pull/59) Added the liver-benchmark mesh preparation files.

## [0.2] - 2026-01-29

### Added

- [PR #27](https://github.com/luca-heltai/immersx/pull/27) Added the initial coupled elasticity problem.
- [PR #31](https://github.com/luca-heltai/immersx/pull/31) Added parameter-file-driven inclusion data and corrected pressure and stress output.
- [PR #32](https://github.com/luca-heltai/immersx/pull/32) Added initial deal.II-based inclusion support.
- [PR #36](https://github.com/luca-heltai/immersx/pull/36) Added local refinement support for deal.II inclusions.
- [PR #40](https://github.com/luca-heltai/immersx/pull/40) Added VTK utilities for reading cell and point data.
- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Added rigid-body mode extraction for AMG preconditioning.
- [PR #44](https://github.com/luca-heltai/immersx/pull/44) Added a `properties` member and a vertex- and cell-agnostic field transfer.

### Changed

- [PR #25](https://github.com/luca-heltai/immersx/pull/25) Updated the Docker environment and the 1D solver library download.
- [PR #48](https://github.com/luca-heltai/immersx/pull/48) Updated the FVM headers.

### Fixed

- [PR #30](https://github.com/luca-heltai/immersx/pull/30) Addressed failing tests.
- [PR #43](https://github.com/luca-heltai/immersx/pull/43) Fixed the scaling of the Reduced Poisson problem.
- [PR #46](https://github.com/luca-heltai/immersx/pull/46) Added missing VTK compile flags.
- [PR #47](https://github.com/luca-heltai/immersx/pull/47) Removed the duplicated mass matrix from elasticity.
- [PR #49](https://github.com/luca-heltai/immersx/pull/49) Fixed compiler warnings.

### Testing and CI

- [PR #28](https://github.com/luca-heltai/immersx/pull/28) Added the Lorthois datasets.
- [PR #33](https://github.com/luca-heltai/immersx/pull/33) Added an inclusion-orientation regression test for the solution and multiplier norms.

## [0.1] - 2024-11-07

### Added

- [PR #5](https://github.com/luca-heltai/immersx/pull/5) Added Neumann and Robin boundary conditions.
- [PR #11](https://github.com/luca-heltai/immersx/pull/11) Added Dirichlet-only elasticity boundary conditions.
- [PR #13](https://github.com/luca-heltai/immersx/pull/13) Added local elasticity assembly.
- [PR #18](https://github.com/luca-heltai/immersx/pull/18) Enabled distributed parallel runs.
- [PR #19](https://github.com/luca-heltai/immersx/pull/19) Added a matrix-free bulk evaluator.
- [PR #20](https://github.com/luca-heltai/immersx/pull/20) Added a matrix-free `CouplingOperator` for the action and transpose of the restriction operator.
- [PR #22](https://github.com/luca-heltai/immersx/pull/22) Added a geometric multigrid matrix-free preconditioner for the elasticity block.

### Changed

- [PR #6](https://github.com/luca-heltai/immersx/pull/6) Moved the `ReferenceInclusion` struct into its own header.
- [PR #8](https://github.com/luca-heltai/immersx/pull/8) Split declarations and definitions into separate files.
- [PR #23](https://github.com/luca-heltai/immersx/pull/23) Updated the license and repository metadata.

### Fixed

- [PR #12](https://github.com/luca-heltai/immersx/pull/12) Fixed rotation handling.
- [PR #17](https://github.com/luca-heltai/immersx/pull/17) Provided the locally relevant row index set when building distributed sparsity patterns.

### Testing and CI

- [PR #14](https://github.com/luca-heltai/immersx/pull/14) Added rotation tests.
