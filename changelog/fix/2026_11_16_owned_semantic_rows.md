PR: #441

- Steady scaling and direct-factorization diagonal detection safely traverse
  noncontiguous MPI ownership between semantic fields without constructing
  sparse iterators for unowned rows.
