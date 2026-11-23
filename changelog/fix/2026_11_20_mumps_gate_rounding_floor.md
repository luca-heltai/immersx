PR: #496

- The IDA MUMPS residual check no longer rejects solves whose defect is at the rounding level of evaluating `J x - b`. It accepts `||J x - b||_2 <= max(tau ||b||_2, gamma (||J|| ||x||_2 + ||b||_2))`, where `tau` is `MUMPS/Relative tolerance`, `gamma = n eps / (1 - n eps)` for the longest row of `J`, and `||J||` is bounded from the materialized block norms. A small right-hand side, such as the first step from rest under a small inflow, previously failed at a relative defect of order `1e-8`. A zero correction is still rejected, and the failure message reports both targets.
