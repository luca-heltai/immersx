PR: #498

- `VesselElasticityProblem`, and so `coupled_elasticity`, imposes the wall
  displacement in a quasi-static tissue (`Density` 0 with `Initial time`
  before `Final time`). That mode applied the 1D wall displacement as a
  pressure, and with point coupling multiplied it by the inclusion
  `Boundary data` (0 by default), so coupled runs returned a wrong solution
  without an error. Each step is now a static solve with the boundary
  conditions and the wall data evaluated at the step time.
- Point-coupling wall data given by the inclusion `Boundary data` expression
  are evaluated at the solve time in `coupled_elasticity`. Before, they
  stayed at time 0.
- Without the 1D input file, `coupled_elasticity` steps a quasi-static tissue
  from `Initial time` to `Final time`.
