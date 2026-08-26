# Choose boundary conditions

The scalar and elasticity applications accept boundary identifiers from the
generated mesh. Strong Dirichlet data are supplied through the corresponding
parsed-function subsection. Elasticity also has weak Dirichlet, Neumann, and
normal-flux paths; use the dedicated verification inputs when comparing those
formulations.

For a reproducible workflow:

1. choose boundary identifiers from the grid generator or imported mesh;
2. set the matching boundary-id list in the application section;
3. define the function with the required number of components;
4. run from an explicit parameter-file path and inspect the configured output
   directory.

The [static elasticity tutorial](../tutorials/elasticity) uses one simple
canonical case. Its verification cases remain available in the
`tutorials/elasticity/` asset directory.
