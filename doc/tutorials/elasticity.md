# Static elasticity

This tutorial extends the Poisson example to a vector-valued finite-element
problem. You will configure material coefficients, displacement boundary data,
and a linear solve for a small 2D body.

The executable is `elastic_static`, implemented by `apps/app_elastic_static.cc`
and `ElasticStaticProblem`. The canonical input is
`tutorials/elastic_static/elastic_static.prm.in`:

```{literalinclude} ../../tutorials/elastic_static/elastic_static.prm.in
:language: ini
```

The square uses a vector `FE_Q` space. Three faces have prescribed displacement
and the remaining face has a Neumann load. The `Material properties` subsection
defines Lamé coefficients; the function expressions have one component per
spatial dimension.

Run the configured input from the repository root:

```bash
cmake --build build -j
./build/elastic_static_debug \
  build/tutorials/elastic_static/elastic_static.prm
```

Results are written below `build/test_output/tutorial-output/elastic-static`.
The application smoke test runs this exact input and checks the generated
output. The older `elasticity` executable remains available for immersed and
verification workflows; its strong/weak/Neumann and MMS cases are kept under
`tutorials/elasticity/` and are not part of this introductory path.

For boundary-condition choices, see [Choose boundary conditions](../how-to/boundary-conditions).
For the time-dependent extension, continue with [Elastodynamics](elastodynamics).
