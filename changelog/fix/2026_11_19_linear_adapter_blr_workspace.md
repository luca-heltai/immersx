PR: #0

- Pass the configured workspace relaxation to the LinearAdapter MUMPS-BLR
  factorization. Keep the native default for -1 and reject invalid or
  rank-dependent values before entering MUMPS.
