# CCO growth in a sphere

The [tutorial](../../doc/tutorials/cco-sphere.md) describes the spherical
`Domain`, shared CCO controls, standalone generation, physical archive and
MetricFlowX transfer, and the draft coupled input.

CMake expands these `.prm.in` templates below the build tree. With inlet
radius 0.1 m and local mesh scaling, standalone generation reaches all 56
requested terminals and produces 111 physical branches. The checked run,
physical archive and ParaView meshes are under
`build/cco_sphere/radius-halving-20261103/radius-0.1/generation`.
The coupled transient and ParaView views remain pending.
