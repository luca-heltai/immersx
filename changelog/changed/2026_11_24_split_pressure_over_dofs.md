PR: #513

- `CoupledElasticityProblem::split_pressure_over_inclusions` is renamed `split_pressure_over_dofs`: it interpolates the pressure at the support points of the immersed geometry, the inclusions with point coupling and the degrees of freedom along the reduced grid with tensor-product coupling, to the cells of the 1D model.
- `Inclusions::update_inclusions_data` with per-cell samples accepts a vessel with a single inclusion and gives it the mean of the samples, as `split_pressure_over_dofs` returns a single value for it. Before, it threw.
