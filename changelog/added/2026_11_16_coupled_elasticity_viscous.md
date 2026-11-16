PR: pending

- `coupled_elasticity` advances a damped elastodynamic tissue with one Newmark
  step per exchange when the parameter file describes a dynamic run, and
  solves a static problem otherwise.
- In dynamic runs, `VesselElasticityProblem::run_timestep(t)` takes a Newmark
  step that imposes the wall displacement with the multiplier. The first step
  starts at the static equilibrium of the current wall data.
- `update_vessel_radius_data` and `split_pressure_over_inclusions` support
  tensor-product coupling, so 1D wall displacements reach the tissue and
  `coupled_elasticity` mode 1 exchanges pointwise pressures along the reduced
  grid.
