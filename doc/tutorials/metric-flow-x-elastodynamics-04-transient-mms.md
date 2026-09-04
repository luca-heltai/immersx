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

The registered gate `metric_flow_x_elastodynamics_mms_convergence` currently
checks the four-level analytical convergence table together with the assembled
two-way residual composition. An IDA time solve remains a separate production
application check. The driver also contains four-level actual temporal and
combined studies. Run them with:

```bash
cd build-metric-flow-x-debug/gtests
IMMERSX_RUN_MMS_STUDIES=1 mpirun -np 2 ./metric_flow_x_elastodynamics_mms_convergence_debug \
  --gtest_filter='OneVesselMMSDriver.MPI_ActualFourLevelTemporalStudy:OneVesselMMSDriver.MPI_ActualFourLevelCombinedStudy'
```

Both studies print physical error tables and write CSV files below
`build-metric-flow-x-debug/test_output/metric-flow-x-elastodynamics-mms/`.
