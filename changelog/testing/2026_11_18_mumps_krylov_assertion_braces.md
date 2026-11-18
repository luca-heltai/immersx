PR: #470

- Build the conditional MUMPS Krylov and IDA regression assertions with dangling-else warnings treated as errors by explicitly bracing their conditions. Numerical expectations and solver behavior are unchanged.
