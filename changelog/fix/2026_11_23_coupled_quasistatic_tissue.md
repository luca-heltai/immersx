PR: pending

- `VesselElasticityProblem::run_timestep0()`, and so `coupled_elasticity`,
  rejects a quasi-static tissue (`Density` 0 with `Initial time` before
  `Final time`). That mode applied the 1D wall displacement as a pressure, and
  with point coupling multiplied it by the inclusion `Boundary data` (0 by
  default), so coupled runs returned a wrong solution without an error.
