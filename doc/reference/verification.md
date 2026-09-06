# Verification inputs

The repository keeps manufactured-solution and convergence inputs alongside
the tutorial assets so they remain runnable without becoming part of the
introductory learning path.

- `tutorials/elasticity/` contains immersed-elasticity static and dynamic
  verification cases.
- `tutorials/elastodynamics/` contains standalone time-dependent convergence
  cases.
- `gtests/parameters/` contains small configured inputs used by focused tests.

The one-vessel two-way verification cases are documented in the four pages
under `doc/tutorials/metric-flow-x-elastodynamics-*.md`. Their independent
analytical gates are implemented in
`gtests/metric_flow_x_elastodynamics_mms_01.cc`; they verify the nonlinear
Area-to-radius map, the radial traction-jump coefficient, mass conservation,
pressure-work normalization, and exact solid gradients by centered finite
differences. The registered gate also verifies the assembled
TensorProduct-lift coupling, multiplier metric, pressure sign, and full
coupled Jacobian. The opt-in spatial, temporal, and combined studies are
diagnostics only; the full-system linear solver does not complete on the finest
coupled levels.

Use the [application reference](applications) to choose the executable and the
[testing guide](../developer/testing) to run a regression or GoogleTest.
