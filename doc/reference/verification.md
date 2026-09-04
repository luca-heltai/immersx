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
pressure-work normalization, and exact solid gradients by centered finite
differences. The production gate also verifies the actual assembled
TensorProduct-lift coupling, multiplier metric, pressure sign, and full
coupled Jacobian. The opt-in spatial, temporal, and combined studies are
diagnostics only; they do not claim asymptotic convergence while the production
full-system linear solver remains the limiting blocker on fine coupled levels.

Use the [application reference](applications) to choose the executable and the
[testing guide](../developer/testing) to run a regression or GoogleTest.
