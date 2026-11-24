PR: #507

- The AMG preconditioners of the elasticity stiffness matrix and of the
  Newmark matrix now use the rigid-body modes as near-null space, also with
  tensor-product coupling and wherever they are rebuilt (Newmark equilibrium
  start, Robin coefficient changes). The modes were previously released before
  the AMG read them.
