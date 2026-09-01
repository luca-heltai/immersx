# Coupled Poisson

This tutorial composes two independent Poisson Problems with a scalar
Lagrange-multiplier Interaction. The complete steady workflow is:

```text
Problem + Problem + Interaction -> LinearAdapter -> solve -> accept -> output
```

The application adds the bulk and embedded Problems to a `LinearAdapter`, then
adds the continuity Interaction using their semantic solution Fields. After
the solve, each accepted Field is handed back to its owner before native output
is written. The multiplier is accepted and written by the Interaction itself.

Run the generated parameter file with:

```bash
build-debug/coupled_poisson_debug \
  build-debug/tutorials/coupled_poisson/coupled_poisson.prm
```

The output directory and all physical output names are configured in the
parameter file.
