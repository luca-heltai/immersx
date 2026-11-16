PR: #441

- Steady scaling and direct-factorization diagonal detection safely traverse
  noncontiguous MPI ownership between semantic fields without constructing
  sparse iterators for unowned rows.
- Second-order condensation now extracts only the known entries of each
  owned row, preserving noncontiguous per-field ownership without constructing
  an accessor in a non-owned gap.
  Its constrained flags use the backend-owned vector and a ReadWriteVector
  import, avoiding the contiguous-partition requirement of a separate
  LA::distributed::Vector when the actual field partition has gaps.
