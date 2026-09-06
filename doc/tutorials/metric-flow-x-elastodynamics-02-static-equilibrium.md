# Tutorial 02: static two-way vessel-wall equilibrium

The constant-amplitude benchmark uses

$$
d(s)=d_0,\qquad A^*=\pi(R_0+d_0)^2,\qquad U^*=0,
$$

and the radial solid displacement `u*=d0 phi(r) er`. The inner and outer
profiles are `r/R0` and

$$
\phi_{out}(r)=\frac{R_0}{R_0^2-R_1^2}
\left(r-\frac{R_1^2}{r}\right).
$$

For `sigma = 2 mu epsilon + lambda_s tr(epsilon) I`, the radial traction
jump is

$$
K=\frac{2(2\mu+\lambda_s)R_1^2}
        {R_0(R_1^2-R_0^2)}.
$$

The assembled residual is `F_solid + B lambda = 0`, hence the independent
sign oracle uses `lambda* = -K d0`. The interaction sends `-lambda` to the
native MetricFlowX external-pressure operation, preserving action/reaction.

Tutorial 02 sets `MMS case = static_equilibrium` in the
`MetricFlowX elastodynamics tutorial` subsection of its input.
It evaluates the time-independent two-way residual at the analytical
equilibrium; it does not substitute a transient solve for a steady problem.

```bash
build-metric-flow-x-debug/metric_flow_x_elastodynamics_debug \
  build-metric-flow-x-debug/tutorials/metric_flow_x_elastodynamics/02_static_equilibrium.prm
```

The configured input is
`build-metric-flow-x-debug/tutorials/metric_flow_x_elastodynamics/02_static_equilibrium.prm`.
The lower-level two-way residual and sign gate is:

```bash
mpirun -np 2 build-metric-flow-x-debug/gtests/gtests_debug \
  --gtest_filter='MetricFlowXVesselWallConstraint.*'
```
