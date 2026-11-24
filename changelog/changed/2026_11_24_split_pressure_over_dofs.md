PR: pending

- `CoupledElasticityProblem::split_pressure_over_inclusions` is renamed `split_pressure_over_dofs`: it interpolates the pressure at the support points of the immersed geometry, the inclusions with point coupling and the degrees of freedom along the reduced grid with tensor-product coupling, to the cells of the 1D model.
