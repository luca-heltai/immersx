# Coupled 1D–3D Elasticity

This tutorial describes a pulsatile one-dimensional network coupled to a
three-dimensional tissue elasticity problem. The reduced vascular tree drives
the surrounding tissue through the tensor-product coupling representation.

This is the current application-level workflow on `master`: the executable
owns the orchestration of this particular 1D–3D example. In the architectural
direction described in {doc}`../core-architecture`, the network and tissue
physics can eventually contribute to the same residual through reusable
Representations and Interactions, with monolithic or partitioned execution
selected outside the physics layer. That common adapter is a roadmap item,
not an API claim about this tutorial.

## Quasi-Static Vascular-Tree Traveling Wave

File: `tutorials/elasticity/vascular_tree_quasistatic_wave_3d.prm`

Run from the repository root with:

```bash
./build/elasticity_debug tutorials/elasticity/vascular_tree_quasistatic_wave_3d.prm
```

This example couples 3D elasticity to `data/tests/mstree_100.vtk`. The
`path_distance` point field is exposed through `Input file fields` and used in
both the reduced thickness and the traveling load:

```{math}
h(s) = 0.01\exp\left(-\frac{10}{3}s\right),
\qquad
g(s,t) = 0.05\sin\left(2\pi\left(t-\frac{s}{0.69}\right)\right).
```

The tree starts at thickness `0.01`, reaches approximately `0.001` at
`path_distance = 0.69`, and the pulsatile load travels that distance in one
second. The quasi-static run lasts three seconds, producing three periods
without inertial dynamics. This example requires a SymEngine-enabled deal.II
build because the reduced coupling expressions depend on the imported VTK
field.

```{figure} assets/vascular_tree_pulsatile_displacement.gif
:name: fig-vascular-tree-pulsatile-displacement
:alt: Animation of the pulsatile vascular-tree tissue displacement contour

Evolution of the $10^{-5}$ contour of the tissue displacement caused by the
pulsatile deformations. The contour travels through the vascular tree as the
reduced load propagates along `path_distance`.
```
