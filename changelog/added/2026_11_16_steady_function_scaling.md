PR: #447

- Add `IDAAdapter::steady_function_scaling()` to compute full steady residual row scales without setting up a factorization or preconditioner, independently of the selected linear solver (Trilinos backend).
