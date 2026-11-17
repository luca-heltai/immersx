PR: pending

- Wall-constrained Newmark steps now weight the constraint row of the
  saddle point system like the momentum row, by `1 / (beta dt^2)`. Before,
  FGMRES could stop after one iteration on the momentum residual alone, with
  the wall constraint unmet; the multiplier then oscillated from step to step
  with an amplitude growing like `1 / dt^2`. Static solves are unchanged.
