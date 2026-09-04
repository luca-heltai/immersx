# One-vessel two-way verification

This family follows the production `metric_flow_x_elastodynamics` composition
from geometry and wall kinematics to the analytical static and transient
manufactured states.
The `.prm.in` files are configured by CMake; run them with the generated files
under `build-*/tutorials/metric_flow_x_elastodynamics/`.

The analytical formulas used by the tutorials and the independent
kinematic/composition gates live in the shared C++ verification helper
`gtests/metric_flow_x_elastodynamics_mms.h` and its tests.

Tutorials 01–04 set `MMS case` to `kinematics`, `static_equilibrium`, `spatial`,
and `transient` in their `MetricFlowX elastodynamics tutorial` input
subsection. Tutorials 01–03 evaluate documented stationary cases through the
assembled residual; Tutorial 04 is configured for final time `0.1` and an
actual IDA step.

The actual stationary, temporal, and combined diagnostic studies are MPI
GoogleTests in
`gtests/metric_flow_x_elastodynamics_mms_driver.cc`; generated CSV files stay
under the build-tree test-output directory. The registered test is a
verification baseline, not an asymptotic convergence claim. Its known blocker
is the production full-system linear solver at fine coupled levels.
