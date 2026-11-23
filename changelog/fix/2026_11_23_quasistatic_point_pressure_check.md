PR: pending

- `ElasticityProblem::solve_quasistatic()` checked the point-coupling
  restriction the wrong way round: it rejected `Prescribed vessel pressure =
  true` and accepted the displacement constraint, which it then solved as a
  pressure. With point coupling it now requires a prescribed vessel pressure,
  as the Newmark solve does.
