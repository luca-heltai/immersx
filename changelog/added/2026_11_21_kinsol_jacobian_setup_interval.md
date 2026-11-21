PR: pending

- Add a `Maximum Jacobian setup interval` option to coupled steady KINSOL initialization. `1` preserves per-iteration setup; positive multiples of `5` allow reuse of the current Jacobian and factorization.
