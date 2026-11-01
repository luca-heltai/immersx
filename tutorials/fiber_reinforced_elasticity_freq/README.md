# Harmonic reinforcement

CMake configures these inputs into the build tree:

- `parameters.prm`: 1D reinforcement in 2D, with a 1D lift.
- `shared.prm`: the same geometry with one shared displacement.
- `full_dimension.prm`: 2D reinforcement in 2D, without a lift.
- `fiber_3d.prm`: 1D reinforcement in 3D, with a 2D section lift.
- `fiber_3d_shared.prm`: the same 3D geometry with one shared displacement.
- `sheet.prm`: 2D reinforcement in 3D, with a 1D normal lift.

```bash
cmake --build build-debug --target fiber_reinforced_elasticity_freq_debug -j 4
mpirun -np 2 build-debug/fiber_reinforced_elasticity_freq_debug \
  build-debug/tutorials/fiber_reinforced_elasticity_freq/parameters.prm
```

The root dimensions select the reinforcement geometry; the matrix is always
full-dimensional. `Formulation = independent` uses two displacement fields and
continuity multipliers. `Formulation = shared` integrates the excess material
on the same physical reinforcement volume using only the matrix displacement.
Both sweep 0, 0.1, and 0.2 Hz and write results below the build output tree.

See `doc/tutorials/fiber-reinforced-frequency.md` for equations, material units,
phase conventions, API usage, and current limits.

See `doc/tutorials/fiber-in-3d-frequency.md` for the 3D matrix example.
