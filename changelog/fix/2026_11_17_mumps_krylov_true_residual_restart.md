PR: #469

- Restart MUMPS Krylov refinement after projected convergence when the original Jacobian defect remains too large, using the remaining iteration budget and rejecting corrections that make no progress. The raw residual tolerance is unchanged.
