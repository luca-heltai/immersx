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

## Robin tractions

A boundary spring with an applied traction imposes

$$
\sigma(u)n + \beta(x)u = g(x), \qquad \beta(x)\geq 0,
$$

where $n$ is the outward normal and
$\sigma(u)=2\mu\varepsilon(u)+\lambda\operatorname{div}(u)I$.
The coefficient is scalar; the load has one component per ambient spatial
coordinate. Robin boundary ids must be disjoint from Dirichlet and Neumann
ids. With a colored unit square, the following replaces the right-face
Neumann load by a Robin condition:

```ini
subsection Elastic static
  set Dirichlet boundary ids = 0, 2, 3
  set Neumann boundary ids =
  set Robin boundary ids = 1
  subsection Functions
    subsection Robin coefficient
      set Function expression = 2+x
    end
    subsection Robin boundary conditions
      set Function expression = 1; 0
    end
  end
end
```

Set `Domain/Grid generator arguments = 0: 1: true` for that face numbering.
Boundary-specific values use numbered subsections such as
`Functions/Robin coefficient 1` and `Functions/Robin boundary conditions 1`.
Unspecified entries inherit the corresponding global function. The vector
load also accepts the modulation frequency and phase parameters used by other
static elasticity loads. Both functions are evaluated at time zero in a
static solve.

The same parameters apply to `elastic_static_adapter`, the solid in
`coupled_poisson_elasticity`, and [lifted static elasticity](../how-to/lifted-static-elasticity)
under their respective parameter subsections. Lifted elasticity also requires
Robin ids to be disjoint from weak Dirichlet and normal-flux ids. Robin face
integration supports the same hypercube and simplex meshes as the volume
problem.
