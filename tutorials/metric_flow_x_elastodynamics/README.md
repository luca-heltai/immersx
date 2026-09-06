# MetricFlowX vessel-wall inputs

The four inputs in this directory run the
`metric_flow_x_elastodynamics` application with `MMS case` set to
`kinematics`, `static_equilibrium`, `spatial`, or `transient`. Each case uses
the same 1D-in-3D MetricFlowX flow Problem, 3D elastodynamics Problem, and
`MetricFlowXVesselWallConstraint`.

The `.prm.in` files are configured into the build tree. Their analytical
helpers and residual/Jacobian checks are in
`gtests/metric_flow_x_elastodynamics_mms.h` and the corresponding tests.
