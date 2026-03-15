# Overview

The main C++ components are:

- `ElasticityProblem` in `include/elasticity.h` for bulk elasticity with optional transient integration and immersed coupling.
- `PoissonProblem` in `include/laplacian.h` for scalar immersed Poisson/Laplacian problems.
- `ReducedPoisson`, `ReducedCoupling`, and `TensorProductSpace` in `include/reduced_poisson.h`, `include/reduced_coupling.h`, and `include/tensor_product_space.h` for reduced-order coupling workflows.
- `Inclusions` in `include/inclusions.h` for immersed geometry, quadrature data, and reduced basis metadata.
- `ReferenceCrossSection` in `include/reference_cross_section.h` for reference reduced-basis and quadrature construction.
- `ParticleCoupling` in `include/particle_coupling.h` for particle insertion and distributed ownership mapping.

The mathematical and algorithmic background of the repository is described in {cite:p}`HeltaiZunino-2023-a`.

The repository also contains benchmark inputs, exploratory notebooks, and coupled 3D/1D assets used by specialized workflows.
