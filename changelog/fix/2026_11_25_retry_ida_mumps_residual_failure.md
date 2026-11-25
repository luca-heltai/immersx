PR: #518

- Treat a failed MUMPS retained-Jacobian residual check as a recoverable IDA callback error, allowing IDA to reject and retry the transient step while keeping the strict residual check.
