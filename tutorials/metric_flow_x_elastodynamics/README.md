# MetricFlowX vessel-wall coupling

`two_way.prm.in` is the nonzero-inflow example for the
`metric_flow_x_elastodynamics` application. It configures the flow and solid
Problems, the tensor-product wall lift, the material data, time integration,
initial and boundary functions, and output directories.

The template is generated below the build tree. Run the application with that
generated file, as described in the [tutorial](../../doc/tutorials/metric-flow-x-elastodynamics.md).

The [mathematical formulation](../../doc/tutorials/metric-flow-x-elastodynamics-coupling.md)
is documented in the tutorial documentation.

The [one-vessel manufactured solution](../../doc/tutorials/metric-flow-x-elastodynamics-mms.md)
derives a continuous, piecewise solid displacement and nonzero pressure-jump
target. Its inlet supplies the changing vessel volume. Run
`mms_cube.prm.in`, `mms_cube_refined.prm.in`, and
`mms_cube_mpi.prm.in` through their generated build-tree `.prm`
files. These inputs use the corrected momentum and continuity rows, with the
native pressure feedback described in the tutorial. The application does not
compute their errors automatically.

The default cube inputs prescribe exact time-dependent displacement and velocity
on the outer boundary. `mms_cube_neumann.prm.in` and
`mms_cube_neumann_refined.prm.in` retain the exact-traction version on the base
and refined meshes. They use separate output directories so both boundary
choices can be compared.

`mms_cube_mumps.prm.in` selects native `dealii::SparseDirectMUMPS` for serial
or MPI runs. It requires deal.II configured with `DEAL_II_WITH_MUMPS`.

The cube MMS cases share `single_vessel_centered.vtk`. Their flow refinement
levels are `1` (base and MPI) and `2` (refined); separate pre-refined VTK inputs
are not needed. Domain retains `a0` while refining the native coarse mesh.
