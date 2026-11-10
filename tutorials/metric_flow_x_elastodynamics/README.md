# MetricFlowX–elastodynamics examples

The two_way.prm.in input retains ordinary extended-solid coupling.
The mms_cube*.prm.in cases remove projected fictitious elasticity and viscosity
while retaining mass.

See [the coupling tutorial](../../doc/tutorials/metric-flow-x-elastodynamics-coupling.md)
and [the manufactured-solution derivation](../../doc/tutorials/metric-flow-x-elastodynamics-mms.md).

~~~bash
cmake -S . -B build-mms -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_COUPLED_PROBLEMS=OFF \
  -DMetricFlowX_DIR=/path/to/metric-flow-x/lib/cmake/MetricFlowX
cmake --build build-mms --target metric_flow_x_elastodynamics -j2
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases base refined
~~~

Use MetricFlowX with the native external-pressure state-Jacobian API and matching
headers and libraries. The reproducibility revision is main commit
`e98b2333ed41d0fcef027849e0efb460d9ada64d` (version 1.0.0); see the
manufactured-solution derivation for the required APIs. Older incompatible
packages give an error before mesh setup when the correction is enabled.

Run a supplied iterative case with MPI through the same runner:

~~~bash
python3 scripts/mfx_modal_convergence.py --build build-mms --run \
  --cases finer --mpi-ranks 4
~~~

The default --mpi-ranks 1 retains serial execution. Multiple ranks require
finer, finer-original, iterative or mpi cases and a matching MPI launcher on PATH. Saved output
and execution metadata can be reprocessed with the same --output path and no
--run flag.

The base and refined cases use UMFPACK; finer uses factored iteration.
The measured finest case uses four MPI ranks. Ordinary flow meshes have 1/2/4
cells for 4/8/16 solid axial cells, with four axial quadrature repetitions.
The finer-original diagnostic retains the former eight-flow-cell pairing.
Common tolerances remain unchanged; all analytical targets and densities are
retained. The finest ordinary multiplier error is still 55%, so these
measurements do not establish pressure convergence or general stability.
The iterative case compares solvers on the base mesh. The optional MPI input
also uses iteration; only the explicitly named MUMPS input requires MUMPS.
Outputs and JSON errors remain in separate build-tree directories. The finest
case has 65,536 solid cells. An earlier SDK 0.1.0 diagnostic of the optional
finer_direct benchmark exceeded 59 GiB during initial factorization. That
benchmark is outside the ordinary sequence.

The MMS uses distinct radial and axial strains, nonzero solid viscosity and
forcing, and source-free continuity. The radial case checks $h=0$; tight checks
sensitivity to adaptive time tolerances. No convergence rate is assumed.

The local MMS family starts from a 1×1×1 solid grid with zero initial refinement.
Enable `/MetricFlowX coupling/Co-refinement/Enabled` to refine around the
physical wall-lift supports. The group exposes the radius-based refinement
factor, maximum level, global pre-refinement and local post-refinement cycles.
The `local`, `local-refined` and `local-finer` cases add 0/1/2 local post passes;
`local-tight` checks temporal tolerance sensitivity on the base local mesh.
See the MMS derivation for commands and the mesh and error diagnostics.
