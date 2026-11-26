PR: #525

- Keep the standalone elastodynamics time-integration matrix unallocated during setup and coupled IDA evaluations; assemble it on the first standalone time step.
