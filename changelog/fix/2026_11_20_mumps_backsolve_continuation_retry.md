PR: #484

- Reject a steady-continuation trial and reduce its eta step when the strict retained-Jacobian MUMPS residual check fails consistently across MPI ranks; preserve the residual acceptance threshold and propagate unrelated failures.
