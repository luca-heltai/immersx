# Random ReducedPoisson examples

Generate the point clouds from the repository root:

```bash
python notebooks/generate_particles.py \\
  --dimensions 2 \\
  --output-file notebooks/random_particles_2d.vtu

python notebooks/generate_particles.py \\
  --dimensions 3 \\
  --output-file notebooks/random_particles_3d.vtu
```

The generator writes one `VTK_VERTEX` per particle and stores the random
`radius` and `rhs` values as point data. Particles are separated by at least
the sum of their radii plus the requested clearance, and every particle is
inside the coordinate box with the same boundary margin.

Run the two examples from the repository root:

```bash
./build/reduced_poisson_debug \\
  tutorials/reduced_poisson/random_particles_2d.prm

./build/reduced_poisson_debug \\
  tutorials/reduced_poisson/random_particles_3d.prm
```

The top-level `cross section dimension` parameter selects the intrinsic
dimension of the point cross-section independently of the embedding space.
For example, use `2` for disks in 3D and `3` for spheres in 3D. The `rhs`
point field is used as the reduced right-hand side and `radius` is used as the
thickness. For a zero-dimensional representative domain, the local refinement
subsection exposes the bulk `Space pre/post-refinement` cycles plus the
point-scale `Refinement factor` and `Max refinement level`; the point cloud
itself is not refined.
