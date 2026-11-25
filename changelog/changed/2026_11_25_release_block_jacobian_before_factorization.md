PR: #517

- Direct Jacobian solves release the assembled block matrix after second-order condensation (or monolithic materialization) and before factorization, reducing peak memory without retaining per-cell matrices.
