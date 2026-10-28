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
output. The `elasticity` executable provides immersed and verification
workflows; its strong/weak/Neumann and MMS cases are kept under
`tutorials/elasticity/` and are separate from this introductory example.

For boundary-condition choices, see [Choose boundary conditions](../how-to/boundary-conditions).
For the time-dependent extension, continue with [Elastodynamics](elastodynamics).

## Simplex meshes

For `elastic_static` and `elastic_static_adapter`, set
`Elastic static/Domain/Triangulation type` to
`fullydistributed` and provide a triangle or tetrahedron mesh through
`Elastic static/Domain/Grid generator`. Set the runtime dimensions to match
the mesh. Keep one refinement cycle; initial refinement is performed before
distribution.

The default `Displacement space/Finite element = auto` selects `FE_SimplexP`
with `FE degree` and one component per spatial dimension. Explicit finite
elements must match the mesh reference cell. Cell assembly, Neumann loads,
Dirichlet interpolation, and output use a matching mapping and quadrature.
Both application lifecycles skip the convergence table on simplices;
`compute_error()` reports that this table requires hypercube meshes.
