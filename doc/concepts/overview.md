# Overview

ImmersX is organized around embedded and mixed-dimensional simulation
workflows: bulk finite element problems, immersed lower-dimensional geometry,
and coupling operators that avoid conforming background meshes.

The main C++ components are:

- `ElasticityProblem` in `include/immersx/physics/elasticity.h` for bulk elasticity with optional transient integration and immersed coupling.
- `PoissonProblem` in `include/immersx/physics/laplacian.h` for scalar immersed Poisson/Laplacian problems.
- `ReducedPoisson`, `ReducedCoupling`, and `TensorProductSpace` in `include/immersx/physics/reduced_poisson.h`, `include/immersx/coupling/reduced_coupling.h`, and `include/immersx/coupling/tensor_product_space.h` for reduced-order coupling workflows.
- `Inclusions` in `include/immersx/coupling/inclusions.h` for immersed geometry, quadrature data, and reduced basis metadata.
- `ReferenceCrossSection` in `include/immersx/coupling/reference_cross_section.h` for reference reduced-basis and quadrature construction.
- `ParticleCoupling` in `include/immersx/coupling/particle_coupling.h` for particle insertion and distributed ownership mapping.

The current classes are being evolved toward a semantic architecture in which
a Problem may own multiple Fields, Problems and Interactions contribute
additively to a semi-discrete residual $F(t,y,\dot y)=0$, and execution
adapters provide the steady, DAE, IMEX, multirate, or partitioned policy. This
common Field/residual vocabulary is a validated development direction, not yet
a claim that those names are public API on `master`; see
{doc}`../developer/design/architecture-status` for the implementation status
boundary.

The mathematical and algorithmic background of the repository is described in {cite:p}`HeltaiZunino-2023-a`.

The repository also contains benchmark inputs, exploratory notebooks, and coupled 3D/1D assets used by specialized workflows.
