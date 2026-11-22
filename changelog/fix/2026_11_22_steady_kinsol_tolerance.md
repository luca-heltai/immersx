PR: pending

- Use the configured relative Krylov tolerance for steady KINSOL solves instead of IDA's native error-weight target, which can demand subnormal residuals in a different norm.
