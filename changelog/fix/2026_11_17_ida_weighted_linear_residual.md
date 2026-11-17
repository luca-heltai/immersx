PR: pending

- Use native IDA error weights for built-in iterative solves, and independently require both the native weighted residual target and the configured raw relative residual bound without assembling scaled matrices. Direct solves, configured callbacks and steady initialization retain their existing paths.
