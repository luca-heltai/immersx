# One-vessel two-way verification

This family follows the production `metric_flow_x_elastodynamics` composition
from geometry and wall kinematics to the analytical static and transient
manufactured states.
The `.prm.in` files are configured by CMake; run them with the generated files
under `build-*/tutorials/metric_flow_x_elastodynamics/`.

The analytical formulas and the independent kinematic/composition gates live in
`gtests/metric_flow_x_elastodynamics_mms.h` and
`gtests/metric_flow_x_elastodynamics_mms_01.cc`.

The actual four-level studies are MPI GoogleTests in
`gtests/metric_flow_x_elastodynamics_mms_driver.cc`; generated CSV files stay
under the build-tree test-output directory.
