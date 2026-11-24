PR: #449

- Wall-constrained Newmark steps now weight the constraint row of the
  saddle point system like the momentum row, by `1 / (beta dt^2)`. Before,
  FGMRES could stop after one iteration on the momentum residual alone, with
  the wall constraint unmet; the multiplier then oscillated from step to step
  with an amplitude growing like `1 / dt^2`. Static solves are unchanged.
- The inner CG solve of the augmented block in wall-constrained solves, static
  and Newmark, is now preconditioned with the AMG of the stiffness or Newmark
  matrix. Without a preconditioner its iterations doubled with each mesh
  refinement (about 900 per solve at 620k dofs, 47 with AMG).
- Static solves of `VesselElasticityProblem` with tensor-product coupling now
  evaluate a time-dependent `Reduced right hand side` at the solve time. Before,
  the expression was always evaluated at `t = 0`, so a static
  `coupled_elasticity` run without a 1D input imposed zero wall data.
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
- `ElasticityProblem::solve_quasistatic()` checked the point-coupling
  restriction the wrong way round: it rejected `Prescribed vessel pressure =
  true` and accepted the displacement constraint, which it then solved as a
  pressure. With point coupling it now requires a prescribed vessel pressure,
  as the Newmark solve does.
- The AMG preconditioners of the elasticity stiffness matrix and of the
  Newmark matrix now use the rigid-body modes as near-null space, also with
  tensor-product coupling and wherever they are rebuilt (Newmark equilibrium
  start, Robin coefficient changes).
- `Solvers/Reduced mass` now defaults to `Tolerance = 0`, so the mass solves
  of the augmented Lagrangian stop on `Reduction` alone. With the absolute
  default `1e-10` they stopped at step 0 for small wall data and returned a
  wrong multiplier; the displacement was not affected.
