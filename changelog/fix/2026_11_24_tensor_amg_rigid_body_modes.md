PR: #507

- The AMG preconditioner of the elasticity stiffness matrix now uses the
  rigid-body modes as near-null space also with tensor-product coupling, and
  also when it is rebuilt for the equilibrium start of a Newmark run or after a
  Robin coefficient changes. The modes were previously released before the AMG
  read them.
