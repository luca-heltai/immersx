# Vascular-tree elasticity

The `elasticity` application can run a quasi-static 3D elasticity problem with
a reduced vascular-tree load. The input is
`tutorials/elasticity/vascular_tree_quasistatic_wave_3d.prm`.

Run it from the repository root with:

```bash
./build/elasticity_debug \
  tutorials/elasticity/vascular_tree_quasistatic_wave_3d.prm
```

The input reads `data/tests/mstree_100.vtk`, exposes its `path_distance` point
field through `Input file fields`, and uses that field for the thickness and
traveling load:

```{math}
h(s) = 0.01\exp\left(-\frac{10}{3}s\right),
\qquad
g(s,t) = 0.05\sin\left(2\pi\left(t-\frac{s}{0.69}\right)\right).
```

The example requires deal.II with SymEngine support. It runs for three seconds
and produces three periods of the quasi-static traveling load. The
`coupled_elasticity` executable is a separate 3D/1D `lib1dsolver` application;
see the [application reference](../reference/applications) for its build
condition and command-line interface.
