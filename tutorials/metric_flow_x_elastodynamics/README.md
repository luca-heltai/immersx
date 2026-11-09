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

Serial cases use UMFPACK. The optional MPI input uses the iterative solver;
only the explicitly named MUMPS input requires MUMPS. Outputs and JSON errors
remain in separate build-tree directories. The finer case has 65,536 solid
cells and may require substantially more memory and factorization time.

The MMS uses distinct radial and axial strains, nonzero solid viscosity and
forcing, and source-free continuity. The radial case checks $h=0$; tight checks
sensitivity to adaptive time tolerances. No convergence rate is assumed.
