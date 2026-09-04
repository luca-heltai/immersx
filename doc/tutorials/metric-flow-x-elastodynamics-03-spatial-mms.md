# Tutorial 03: spatial manufactured vessel-wall coupling

The spatial manufactured state uses `k=2*pi/L` and

$$
A^*(s)=A_0+a\sin(ks),\qquad
d(s)=\sqrt{A^*(s)/\pi}-R_0,
\qquad \lambda^*(s)=-K d(s).
$$

The nonlinear radius map is retained exactly. Since the radial profile is
piecewise, the continuous solid source is obtained from `-div sigma(u*)`, not
from a discrete residual. The flow mass equation remains source free. The
source formulas and the native multiplier-to-pressure convention are captured
in the shared analytical gate; the checked-in input is the generated
production-path baseline for this tutorial step. The actual two-way IDA
verification driver also reports area, flow velocity, solid displacement,
solid H1, velocity, multiplier-metric, and constraint errors in physical norms.

Use the configured production input:

```bash
build-metric-flow-x-debug/metric_flow_x_elastodynamics_debug \
  build-metric-flow-x-debug/tutorials/metric_flow_x_elastodynamics/03_spatial_mms.prm
```

The independent formulas and two-way residual composition are exercised with:

```bash
ctest --test-dir build-metric-flow-x-debug -V \
  -R '^metric_flow_x_elastodynamics_mms_verification$'
```

Runtime files belong below the build-tree test-output directory.

For the opt-in stationary spatial diagnostic, run:

```bash
cd build-metric-flow-x-debug/gtests
IMMERSX_RUN_MMS_STUDIES=1 mpirun -np 2 ./metric_flow_x_elastodynamics_mms_verification_debug \
  --gtest_filter=OneVesselMMSDriver.MPI_ActualFourLevelSpatialStudy
```

The CSV table is written below
`build-metric-flow-x-debug/test_output/metric-flow-x-elastodynamics-mms/`.
It is diagnostic rather than an asymptotic FE-convergence claim: the
production full-system linear solve currently blocks on the finest coupled
levels. The stationary MMS uses a genuinely time-independent two-way problem,
so this limitation is solver scalability rather than a transient-study
substitution.
