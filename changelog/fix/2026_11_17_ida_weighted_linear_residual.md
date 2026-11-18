PR: #468

- Use native IDA error weights for built-in iterative solves, and independently require both the native weighted residual target and the configured raw relative residual bound without assembling scaled matrices. Direct solves, configured callbacks and steady initialization retain their existing paths.

- If native tolerance conversion becomes indeterminate at extreme valid weights, accept an unscaled correction only after an exact entrywise full-J residual check and the unchanged raw gate.

- Size native error-weight buffers from the current callback RHS after a resize restart, preserving the existing restart and consistent-initial-condition behavior. Nonlinear-correction diagnostics use the same current-map rule.
