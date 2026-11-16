PR: pending

- Added unit tests of `ElasticityProblem` to `gtests/elasticity_03.cc`: body
  force and Neumann loads against exact integrals, time modulation of the
  forcing terms, affine solutions with hanging nodes and with weak Dirichlet
  (Nitsche) data, time-mode inference, Rayleigh and Kelvin-Voigt damping, the
  Newmark operator and Newmark value constraints.
- `BOTH_ForcingTermsNeumannLoadMatchesStaticAssembly` fails until issue #454
  is fixed: static and time-dependent runs apply different Neumann loads.
- Added tests of `make_vessel_index_of_a_cell()` for tensor-product coupling
  to `gtests/coupled_elasticity_03.cc`.
