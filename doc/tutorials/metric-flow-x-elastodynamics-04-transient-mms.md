# Tutorial 04: transient two-way manufactured solution

The transient state uses

$$
q(t)=1-\cos(\omega t),\qquad
A^*(s,t)=A_0+a q(t)\sin(ks),
$$

with `d`, `u`, and `lambda` defined by the same exact nonlinear wall map as in
Tutorial 03. The exact flow velocity is selected from continuity:

$$
U^*(s,t)=-\frac{a\dot q(t)}{k}
\frac{1-\cos(ks)}{A^*(s,t)}.
$$

Consequently `A_t + (AU)_s = 0` exactly and both endpoint velocities vanish.
At `t=0`, all perturbation and multiplier fields are zero while the area state
is the metadata value `A0`. The solid source is `rho_s u_tt - div sigma(u*)`
and the flow source is derived from the native momentum residual; neither is a
discrete residual cancellation.

Run the production-path input after configuring the project:

```bash
build-metric-flow-x-debug/metric_flow_x_elastodynamics_debug \
  build-metric-flow-x-debug/tutorials/metric_flow_x_elastodynamics/04_transient_mms.prm
```

The input sets a nonzero final time (`0.1`) and one time step. The executable
therefore performs an actual IDA step for the transient manufactured state;
the spatial stationary-exact case is kept separate in Tutorial 03.

The registered gate `metric_flow_x_elastodynamics_mms_verification` checks the
analytical formulas, independent exact-gradient finite differences, assembled
discrete virtual work, two-way residual/sign composition, the full coupled
Jacobian by finite differences, and a small actual transient IDA solve. It is a
verification baseline and does not claim asymptotic convergence. Tutorial 04
remains genuinely transient; the stationary spatial case is a separate driver
path. The driver also contains opt-in temporal and combined diagnostics. Run
them with:

```bash
cd build-metric-flow-x-debug/gtests
IMMERSX_RUN_MMS_STUDIES=1 mpirun -np 2 ./metric_flow_x_elastodynamics_mms_verification_debug \
  --gtest_filter='OneVesselMMSDriver.MPI_ActualFourLevelTemporalStudy:OneVesselMMSDriver.MPI_ActualFourLevelCombinedStudy'
```

Both studies print physical error tables and write CSV files below
`build-metric-flow-x-debug/test_output/metric-flow-x-elastodynamics-mms/`.
Their rates are diagnostic only. The current production full-system linear
solver blocks on the finest coupled levels, so complete asymptotic temporal or
combined convergence is an explicit follow-up rather than a result of this PR.
