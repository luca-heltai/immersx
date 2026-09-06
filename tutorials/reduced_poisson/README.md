# Reduced Poisson inputs

The inputs in this directory cover point clouds in 2D and 3D, a hyperspherical
cross-section case, a vascular tree, and cylinder examples. The
`random_particles_*.prm` files use VTK point data fields named `radius` and
`rhs`; the reduced coupling uses them as thickness and reduced right-hand-side
data.

The point-cloud inputs are run with:

```bash
./build/reduced_poisson_debug \
  tutorials/reduced_poisson/random_particles_2d.prm
```

The `cross section dimension` setting controls the intrinsic cross-section
dimension independently of the bulk space dimension. The local-refinement
settings distinguish bulk mesh refinement from point-scale refinement.
