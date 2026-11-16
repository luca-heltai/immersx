PR: #445

- Support disjoint MPI ownership in sparse diagonal detection and second-order
  condensation. Read owned rows without constructing accessors for intervening
  unowned rows, and import constrained flags with vectors that support the actual
  field partition.
