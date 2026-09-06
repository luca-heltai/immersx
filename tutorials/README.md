# Tutorial assets

This directory contains the parameter files used by the documentation and
application tests. Inputs ending in `.prm.in` are configured by CMake into the
build tree with the same relative path and the final `.in` suffix removed.
They may use `@TEST_DATA_DIR@` and `@TEST_OUTPUT_DIR@`.

The asset directories cover Poisson, static elasticity, elastodynamics,
reduced Poisson, coupled Poisson, coupled Poisson–elasticity, fiber-reinforced
elastodynamics, Navier–Stokes, and MetricFlowX elastodynamics. Additional
elasticity inputs contain boundary-condition and verification cases.
