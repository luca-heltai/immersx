PR: #449

- In `coupled_elasticity`, static and quasi-static steps after the first
  keep the stiffness matrix and its AMG preconditioner and rebuild only the
  load, including the lifting of time-dependent Dirichlet values. Runs with
  Robin boundary faces still reassemble the matrix at every step.
- `Solvers/Reduced mass` now defaults to `Tolerance = 0`, so the mass solves
  of the augmented Lagrangian stop on `Reduction` alone, in every application
  that uses them. With the absolute default `1e-10` they stopped at step 0 for
  small wall data and returned a wrong multiplier in Newmark steps; the
  displacement was not affected. Parameter files that set `Tolerance` in
  `subsection Reduced mass` keep their value; the others, for example
  `prms/elasticity/circle_one_hole.prm`, now stop on `Reduction` alone.
