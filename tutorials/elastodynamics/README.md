# Elastodynamics inputs

The files in this directory are standalone `elastodynamics` inputs for
strongly constrained 2D manufactured-solution and damping cases:

- `strong_dirichlet.prm.in`;
- `neumann.prm`;
- `kelvin_voigt.prm`.

The application imposes essential displacement and velocity data. The
`neumann.prm` filename identifies the manufactured case, but it does not make
the application assemble a Neumann boundary operator. The `.prm.in` input is
configured into the build tree before execution.
