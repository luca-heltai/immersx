# Verification inputs

The repository keeps manufactured-solution and convergence inputs alongside
the tutorial assets so they remain runnable without becoming part of the
introductory learning path.

- `tutorials/elasticity/` contains legacy immersed-elasticity static and
  dynamic verification cases.
- `tutorials/elastodynamics/` contains standalone time-dependent convergence
  cases.
- `gtests/parameters/` contains small configured inputs used by focused tests.

Use the [application reference](applications) to choose the executable and the
[testing guide](../developer/testing) to run a regression or GoogleTest.
