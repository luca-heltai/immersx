# Tutorial 01: area-to-wall kinematics

This first step isolates the nonlinear geometric map used by the production
vessel-wall interaction:

$$
R_0=\sqrt{A_0/\pi},\qquad
\delta R(A)=\sqrt{A/\pi}-R_0,
\qquad P(A)=\delta R(A)n.
$$

`A0` is read from MetricFlowX vessel metadata. It is not replaced by `a_d` or
`r_d`. At `A=A0`, the displacement is exactly zero. The selected TensorProduct
cross-section modes are 3 and 7, with the same scalar coefficient, so their
reconstructed field is `delta R n`.

The derivative gate checks

$$
\frac{d\delta R}{dA}=\frac{1}{2\sqrt{\pi A}},
$$

against a centered finite difference. The pressure-work factor is evaluated
as `2*pi*R*dR/dA = 1`; this is the circumference normalization that gives the
multiplier pressure units.

Tutorial 01 runs the production executable with `MMS case = kinematics` in the
`MetricFlowX elastodynamics tutorial` subsection. It
initializes a constant perturbed area and zero wall state, then evaluates the
assembled two-way residual so that the `A -> delta R n` representation is
exercised.

```bash
build-metric-flow-x-debug/metric_flow_x_elastodynamics_debug \
  build-metric-flow-x-debug/tutorials/metric_flow_x_elastodynamics/01_wall_kinematics.prm
```

Run the independent mathematical and two-way residual gate with:

```bash
ctest --test-dir build-metric-flow-x-debug -V \
  -R '^metric_flow_x_elastodynamics_mms_verification$'
```

The configured input is
`build-metric-flow-x-debug/tutorials/metric_flow_x_elastodynamics/01_wall_kinematics.prm`.
Output is under `build-metric-flow-x-debug/test_output/tutorial-output/metric-flow-x-elastodynamics-01`.
