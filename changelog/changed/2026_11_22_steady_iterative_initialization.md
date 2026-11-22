PR: #489

- Allow coupled steady initialization to use direct MUMPS/UMFPACK, MUMPS BLR FGMRES, or the Schur preconditioner. Reuse the assembled block Jacobian for steady residual scaling and MUMPS BLR setup.
