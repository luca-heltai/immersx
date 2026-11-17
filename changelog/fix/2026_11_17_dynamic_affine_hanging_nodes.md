PR: pending

- Dynamic elasticity runs now add the affine contribution of the boundary
  data also on cells around hanging nodes that inherit boundary data from
  their masters. Before, only cells with a Dirichlet or normal-flux face got
  it, and locally refined meshes near the boundary had wrong displacements
  next to the hanging nodes (#466).
