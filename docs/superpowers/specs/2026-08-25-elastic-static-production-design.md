# ElasticStatic production slice design

**Date:** 2026-08-25

**Scope:** Work packages 0 and 1 of the post-#115 maturation roadmap.

## Current capability matrix

This matrix records the state observed at `33cf0f7` before this slice. “Adapter”
means the post-#69 semantic `LinearAdapter`/`IDAAdapter` composition path, not
an older solver-specific driver embedded in a physics class. “Partial” means
that the capability exists in a specialized or monolithic path but is not yet a
general standalone Problem feature.

| Capability | legacy `ElasticityProblem` | `PoissonSolver` | `ElasticStaticProblem` | `ElastodynamicsSolver` | `NavierStokesSolver` |
|---|---|---|---|---|---|
| ParameterAcceptor | yes | yes | no | yes | yes |
| Mesh source | generated, file, legacy special grids | generated or file fallback | hard-coded hypercube | generated or file | generated only |
| Distributed triangulation | yes | yes | yes | yes | yes |
| Fullydistributed triangulation | yes | yes | no | yes | yes |
| FE degree | yes | yes | constructor only | yes | velocity/pressure degrees |
| Material ids | yes | no | no | no | no |
| Parsed material data | yes, `MaterialProperties` | no | no | scalar constants | scalar constants |
| Parsed RHS | yes | yes | setter only | yes | yes |
| Material-specific RHS | yes | no | no | no | no |
| Strong Dirichlet BC | yes, configurable ids and parsed data | yes, configurable ids and parsed data | hard-coded id 0, zero data | configurable ids and parsed data | configurable ids and parsed data |
| Neumann BC | yes, parsed traction/flux paths | no | no | no | no |
| Parsed initial data | yes | no | no | yes | yes |
| Output | yes | yes | no | yes | yes |
| Exact solution/error | yes | yes | no | yes | optional parsed field/error table |
| Standalone LinearAdapter/IDA execution | no semantic adapter | LinearAdapter path via `poisson_residual` | LinearAdapter path | IDA path via semidiscrete contributor | IDA path via semidiscrete contributor |
| One-way coupling | legacy immersed load paths | representation/load interaction | usable as a target in the new slice | specialized drivers only | no |
| Multiplier coupling | legacy immersed multiplier paths | reduced/coupling implementations | no | specialized fiber interaction only | no |
| Bidirectional coupling | partial, legacy integrated paths | no | no | specialized fiber interaction only | no |

The matrix is a developer-facing baseline, not a promise to add every missing
cell. In particular, weak Dirichlet conditions and nonlinear constitutive laws
are outside this slice.

## Goals

`ElasticStaticProblem<dim, spacedim>` becomes a parameter-driven replacement
candidate for the useful static, strong-BC portion of legacy elasticity. The
problem will support full-dimensional 2D and 3D meshes, generated or file
input, uniform scale, initial global refinement, distributed and
fullydistributed triangulations, vector-valued parsed body force and boundary
data, parsed Neumann traction, and material-id-specific isotropic Lamé data.

The application-facing workflow is:

```cpp
ElasticStaticParameters<dim> parameters;
initialize_parameters(file);
ElasticStaticProblem<dim> problem(parameters);
problem.setup();
LinearAdapter adapter(...);
const auto fields = adapter.add(problem);
auto state = adapter.make_state();
adapter.solve(state);
problem.set_solution(adapter.field(state, fields.displacement));
problem.output_results();
```

The Problem does not own an adapter, global block numbers, interactions,
multiplier fields, or external coupling loads. Its native residual remains
`K u - f`; external interactions add to the adapter residual independently.

## Parameter and material design

Add `ElasticStaticParameters<dim, spacedim>` as a `ParameterAcceptor` with a
configurable root subsection. It composes three user-facing parsed functions:

- vector body force;
- vector Dirichlet data;
- vector Neumann traction.

The parameter object owns boundary-id sets, grid source/type/name/arguments,
scale, refinement, FE degree, and output naming. It also owns the default
`MaterialProperties` and material-id-to-tag configuration. Material override
acceptors are created during the same two-pass parameter initialization already
used by legacy elasticity. The material acceptor root is configurable so two
elastic Problems can coexist without subsection collisions.

The first implementation supports linear isotropic elasticity only. During
cell assembly, the material is selected from `cell->material_id()` and its
Lamé parameters are used at every quadrature point. Unknown material ids fall
back to the configured default material unless an explicit tag is supplied.

## Mesh, assembly, and output

The Problem stores a variant of deal.II distributed and fullydistributed
triangulations. Generated grids use `GridGenerator::generate_from_name_and_arguments`;
file input uses the existing deal.II grid readers. Scaling occurs before
global refinement. Fullydistributed meshes are built from a serial source mesh
and copied through deal.II’s triangulation description facilities.

Stiffness and load assembly happen after constraints are built. One assembly
pass preserves the affine correction from nonzero Dirichlet values while adding
body and Neumann loads. Parsed functions are evaluated at the current static
time (zero for this slice). A convenience `set_forcing()` may replace the
parsed body force for tests or programmatic callers, but normal applications
configure the RHS through parameters.

Output is a Problem-owned observation helper using `DataOut`, writing the
displacement and the material-id field to the configured directory/name. No
solver policy is introduced into the Problem.

## Testing and acceptance gates

Tests are added before implementation for parameter registration, parsed
nonzero Dirichlet data, parsed body force, Neumann traction, material-id
assembly, distributed execution, and fullydistributed execution. The focused
residual oracle is independent of the adapter:

```text
R = K u - f
```

The application is exercised from its generated build-tree parameter file and
from an unrelated working directory. The final validation includes a Debug
build, serial tests, two-rank MPI tests for both triangulation backends, the
relevant CTest application/regression test, and a Release build because the
parameter templates and explicit template instantiations change.

## Non-goals

This slice does not add weak Dirichlet conditions, nonlinear materials,
dynamic/static shared parameter abstractions, multiplier coupling,
bidirectional coupling, or legacy Elasticity removal. Those remain later
vertical slices and must reuse the proven static vocabulary rather than
introducing speculative common base classes.
