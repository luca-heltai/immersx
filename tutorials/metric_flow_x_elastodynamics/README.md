# MetricFlowX–elastodynamics examples

The two_way.prm.in input retains ordinary extended-solid coupling.
The mms_cube*.prm.in cases remove projected fictitious elasticity and viscosity
while retaining mass.

See [the coupling tutorial](../../doc/tutorials/metric-flow-x-elastodynamics-coupling.md)
and [the manufactured-solution derivation](../../doc/tutorials/metric-flow-x-elastodynamics-mms.md).

~~~bash
cmake -S . -B build-mms -DCMAKE_BUILD_TYPE=Release \
  -DMetricFlowX_DIR=/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build build-mms --target metric_flow_x_elastodynamics -j2
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases base refined finer
~~~

Use MetricFlowX with the native external-pressure state-Jacobian API and matching
headers and libraries. The reproducibility revision is main commit
`e98b2333ed41d0fcef027849e0efb460d9ada64d` (version 1.0.0); see the
manufactured-solution derivation for the required APIs. Older incompatible
packages give an error before mesh setup when the correction is enabled.

The base and refined cases use UMFPACK; finer uses factored serial iteration.
The iterative case compares solvers on the base mesh. The optional MPI input
also uses iteration; only the explicitly named MUMPS input requires MUMPS.
Outputs and JSON errors remain in separate build-tree directories. The finest
case has 65,536 solid cells. Its optional finer_direct benchmark exceeded
59 GiB during initial factorization and is outside the ordinary sequence.

The MMS uses distinct radial and axial strains, nonzero solid viscosity and
forcing, and source-free continuity. The radial case checks $h=0$; tight checks
sensitivity to adaptive time tolerances. No convergence rate is assumed.
