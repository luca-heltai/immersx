PR: #503

- In `coupled_elasticity`, static and quasi-static steps after the first
  keep the stiffness matrix and its AMG preconditioner and rebuild only the
  load, including the lifting of time-dependent Dirichlet values. Runs with
  Robin boundary faces still reassemble the matrix at every step.
