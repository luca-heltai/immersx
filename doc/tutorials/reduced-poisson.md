# Reduced Poisson

This tutorial introduces the first mixed-dimensional workflow: a scalar bulk
Poisson problem coupled to a reduced multiplier space supported on an immersed
geometry. The example is intentionally one straight cylinder with one
transverse mode.

The executable is `reduced_poisson`, implemented by
`apps/app_reduced_poisson.cc`. It requires deal.II with VTK support. The
canonical input is `tutorials/reduced_poisson/single_cylinder_3d.prm.in`:

```{literalinclude} ../../tutorials/reduced_poisson/single_cylinder_3d.prm.in
:language: ini
```

## What is reduced

The bulk problem lives in 3D, while the representative geometry is a 1D
centerline. A reference cross section is swept along that centerline to form
the represented interface. The multiplier is expanded in the selected
cross-section basis, so `Selected indices = 0` is the smallest useful case.

The root entries explicitly select `dimension`, `space dimension`, and
`reduced dimension`. As with the other applications, the filename has no role
in dimension dispatch.

## Run it

```bash
cmake --build build -j
./build/reduced_poisson_debug \
  build/tutorials/reduced_poisson/single_cylinder_3d.prm
```

The generated parameter file points to the configured cylinder mesh under
`build/data/tests/one_cylinder.vtk` and writes output below
`build/test_output/tutorial-output/reduced-poisson-single-cylinder`. The same
input is exercised by the application smoke test.

## Explore the coupling

After the first run, try the canonical variable-radius and multimode inputs
under `tutorials/reduced_poisson/`. The [reduced-coupling how-to](../how-to/reduced-coupling)
explains imported fields, thickness, modes, quadrature, and distributed point
search. The [mathematical background](../concepts/mathematical-background)
explains the reduced Lagrange-multiplier formulation.

Continue with [Coupled Poisson–elasticity](coupled-poisson-elasticity) to see a
public composition workflow that observes and lifts a field before coupling
it to another Problem.
