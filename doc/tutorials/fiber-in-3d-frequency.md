# A 1D fiber in a 3D matrix under harmonic traction

This example couples a reinforcement mesh of type `<1,3>` to a matrix of type
`<3,3>`. Set `dimension = 1` and `space dimension = 3` at the root of the
parameter file. The matrix always uses the ambient dimension.

## Geometry and loading

The matrix is the cube `[-1,1]^3`. A straight fiber runs along the x axis from
`(-0.65,0,0)` to `(0.65,0,0)`. Its two normal directions define a square
section. With `Lift/Thickness = 0.1`, each section side has length 0.2:
the thickness parameter is the half-width. The lifted reinforcement volume is
`1.3 * 0.2 * 0.2 = 0.052`, entirely inside the matrix.

The matrix mesh has two initial global refinements. The fiber has one, giving
two line cells. Matrix and fiber meshes use distributed and fully distributed
triangulations, respectively. Every displacement has three components.

Boundary id 0 clamps the matrix face at `x = -1`; boundary id 1 applies a
traction `(1,0,0)` at `x = 1`, with a phase of 15 degrees. The other faces
have zero natural traction. The fiber has no imposed boundary data. The sweep
uses 0, 0.1, and 0.2 Hz, including a static reference and two damped harmonic
solutions.

The matrix has Lamé coefficients `mu = lambda = 1` and density 1. The fiber
adds `mu = lambda = 8` and density 0.2 per unit physical reinforcement volume.
These are excess coefficients: the matrix material remains present in the
lifted volume. Do not multiply these coefficients by the section area.

## Configure and run

From the repository root:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=mpicxx -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_COUPLED_PROBLEMS=OFF -DIMMERSX_BUILD_CORAL_PLUGINS=OFF
cmake --build build-debug --target fiber_reinforced_elasticity_freq_debug -j 4

mpirun -np 2 build-debug/fiber_reinforced_elasticity_freq_debug \
  build-debug/tutorials/fiber_reinforced_elasticity_freq/fiber_3d.prm

mpirun -np 2 build-debug/fiber_reinforced_elasticity_freq_debug \
  build-debug/tutorials/fiber_reinforced_elasticity_freq/fiber_3d_shared.prm
```

Replace `/path/to/deal.II` with the installed package location. CMake generates
the inputs and their absolute output paths in the build tree. For Release,
configure `CMAKE_BUILD_TYPE=Release` and use the target and executable
`fiber_reinforced_elasticity_freq`. A serial run uses the same inputs without
`mpirun -np 2`.

## Choose the displacement formulation

`fiber_3d.prm` selects `Formulation = independent`. Matrix and fiber have
separate real and imaginary displacement fields. The fiber displacement is
constant across each lifted section, and vector multipliers enforce continuity
against the matrix in the weak section-average sense. The degree-zero
multiplier space gives one three-component multiplier per line cell for each
complex part. The direct solver handles the complete saddle-point system.

`fiber_3d_shared.prm` selects `Formulation = shared`. Only the matrix
displacement is solved for. The fiber supplies geometry and excess material
coefficients, integrated with the matrix's full ambient gradients at points
throughout the lifted volume. The fiber boundary-id sets must remain empty.

The two formulations use the same physical reinforcement volume, but have
different kinematics. Their displacement fields need not coincide. Neither
input introduces independent beam rotations or higher section modes.

The complete independent input is:

```{literalinclude} ../../tutorials/fiber_reinforced_elasticity_freq/fiber_3d.prm.in
:language: ini
```

The shared input changes the formulation and output directory:

```{literalinclude} ../../tutorials/fiber_reinforced_elasticity_freq/fiber_3d_shared.prm.in
:language: ini
```

## Inspect the results

Outputs are written below the build output tree in
`tutorial-output/fiber-reinforced-elasticity-frequency-fiber-3d` and
`tutorial-output/fiber-reinforced-elasticity-frequency-fiber-3d-shared`.
Each directory contains `frequencies.csv` and matrix datasets for all three
frequencies. Open the matrix PVTU files in ParaView to inspect the real and
imaginary displacements, amplitudes, and phases. The independent run also
writes fiber and multiplier datasets. The fiber output is a line mesh; it
does not display a resolved square volume mesh.

For applications using the C++ driver, `reinforcement_measure()` should return
0.052; `residual_norm()` and `continuity_norm()` report algebraic residuals
after each solve. The command-line application writes the datasets and frequency
metadata but does not print these diagnostics.

At zero frequency the damping terms vanish, while the prescribed traction
phase is retained in the complex static load. The nonzero frequencies include
both the traction phase and material damping.
A small algebraic residual verifies the linear solve; it does not establish
mesh or quadrature convergence.

For convergence studies, refine the matrix and fiber separately and increase
the representative and section quadrature resolution. All lifted points must
remain inside the matrix. This example uses a straight fiber and a constant
section mode. See [harmonic reinforcement](fiber-reinforced-frequency.md) for
the equations, phase convention, quadrature controls, and current limits.
