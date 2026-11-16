PR: pending

- Added `gtests/elasticity_boundary_stress_01.cc`, which checks the boundary
  tractions and normal displacement fluxes of `ElasticityProblem` against
  exact Dirichlet solutions: a uniform strain in 2D and 3D with Q1 and a
  quadratic field with body force in 2D with Q2. The tests run in serial and
  with MPI.
