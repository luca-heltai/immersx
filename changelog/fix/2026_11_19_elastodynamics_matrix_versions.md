PR: #0

- Reuse immutable global mass, diagonal mass, damping, and constrained kinematic
  matrices across coupled elastodynamics Jacobian snapshots. Version stiffness by
  spatial and Robin assembly revisions so retained Jacobians keep their original
  matrix after boundary updates, without caching local cell matrices or retaining
  an unbounded history.
