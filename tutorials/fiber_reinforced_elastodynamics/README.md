# Full-order fiber-reinforced elastodynamics

Run the tutorial from the repository root with:

```text
./build/fiber_reinforced_elastodynamics_debug \
  build/tutorials/fiber_reinforced_elastodynamics/parameters.prm
```

The matrix occupies `[-1,1]^2`. The fiber is an independently meshed thin
rectangle inside the matrix. Both meshes are full-dimensional 2D `FE_Q` vector
spaces; “embedded” describes the geometry, not the finite-element dimension.

The matrix material fills the whole background domain. The fiber Problem is an
additive/excess contribution in the fiber region, with its own positive density
and elastic coefficients. It is not a second complete solid material silently
added on top of the matrix.

The distributed vector Lagrange multiplier enforces velocity-level continuity.
The initial displacement is required to satisfy the same pairing constraint.
The output contains matrix and fiber displacement/velocity files and the
multiplier on the fiber mesh.
