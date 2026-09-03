# Verification inputs

The repository keeps manufactured-solution and convergence inputs alongside
the tutorial assets so they remain runnable without becoming part of the
introductory learning path.

- `tutorials/elasticity/` contains legacy immersed-elasticity static and
  dynamic verification cases.
- `tutorials/elastodynamics/` contains standalone time-dependent convergence
  cases.
- `gtests/parameters/` contains small configured inputs used by focused tests.

The one-vessel two-way verification path is documented in the four progressive
pages under `doc/tutorials/metric-flow-x-elastodynamics-*.md`. Its independent
analytical gates are implemented in
`gtests/metric_flow_x_elastodynamics_mms_01.cc`; they verify the nonlinear
Area-to-radius map, the radial traction-jump coefficient, mass conservation,
and pressure-work normalization.

Use the [application reference](applications) to choose the executable and the
[testing guide](../developer/testing) to run a regression or GoogleTest.
