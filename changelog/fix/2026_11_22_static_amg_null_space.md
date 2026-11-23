PR: #493

- The static elasticity solve no longer fails with NaN in the first CG step on meshes with more than 2000 DoFs, for example with normal-flux end faces on a locally refined mesh or in parallel. The AMG preconditioner read the rigid-body null space after it had been freed (#458).
