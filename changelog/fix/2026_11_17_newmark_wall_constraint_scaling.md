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
