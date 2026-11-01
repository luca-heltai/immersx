# Solve Poisson with a lifted modal multiplier

`lifted_poisson` is the preferred compositional path for the tested 1D
representative to 2D surface coupling in 3D. It composes a scalar Poisson weak
term and a modal multiplier interaction without calling `ReducedPoisson` or
`ReducedCoupling`. The legacy
application remains available, including its zero-dimensional cases.

The new application supports a one-dimensional representative mesh in three
dimensions. A tensor-product lift places its modal basis on a two-dimensional
immersed surface. There are exactly two `Domain` objects and two configurable
`FiniteElementSpace` objects. Imported radius data belong to the representative
Domain; they do not require another application FE space.

## Run the application

For a `DebugRelease` build:

```bash
cmake --build build --target lifted_poisson_debug gtests_debug -j
build/lifted_poisson_debug "$PWD/build/gtests/parameters/lifted_poisson.prm"
mpirun -np 2 build/lifted_poisson_debug \
  "$PWD/build/gtests/parameters/lifted_poisson.prm"
```

The generated example uses an imported, varying radius, three modes, nonzero
forcing and Dirichlet data, and two adaptive bulk solves. Its input and output
paths are absolute build-tree paths. The same parameter file works when the
executable is launched from another directory. Do not run the serial and MPI
examples concurrently with the same output directory.

The groups below live under `subsection Lifted Poisson`:

- `Background` and `Representative` configure the two Domains.
- `Background space` selects the scalar bulk FE.
- `Modal space` selects a primitive FE with one component per selected mode,
  for example `FESystem<1,3>[FE_Q<1,3>(1)^3]`.
- `Lift` configures the reference section, selected modes, thickness, and
  representative/section quadrature. Its geometry and basis use the existing
  `TensorProductLiftSupport` and `ReferenceCrossSection` implementations.
- `Radius field` selects an imported scalar field. Enable `Read fields` under
  `Representative/VTK support`. Without a radius field, `Lift/Thickness` is a
  parsed spatial expression, evaluated at time zero.
- `Forcing` and `Boundary` contain parsed scalar functions. `Dirichlet boundary
  ids` selects the strongly prescribed boundaries. The solve is steady, at
  time zero.
- `Modal data` contains semicolon-separated coefficient expressions in selected
  mode order. Default-selected modes are resolved before parsing this list.
- `Modal data fields` exposes imported field aliases to those expressions using
  `InputFieldSelector` syntax. For example, `r=point:radius,q=cell:rhs` allows
  `Modal data = 0.1*r*r+q; 2*r+0.01*pi+t` for two selected modes. Enable
  representative `Read fields` first. Association qualifiers distinguish point
  and cell data; component selectors such as `v=velocity[0]` also work. Imported
  expressions require SymEngine, as in the native symbolic path. Coordinate-only
  expressions do not. Both paths accept `pi`, `E`, and time `t=0`.

For example, selecting indices `2,0` requires two modal FE components and two
expressions. Component zero and the first expression refer to mode **2**, not
mode zero. A component-count mismatch throws in Debug and Release.

## Equations and modal data

The residual is

```text
F_u      = A u + B^T lambda - f,
F_lambda = B u - g.
```

`A` is the ordinary scalar gradient weak term. `B` integrates bulk values
against representative FE shapes multiplied by the selected transverse basis
functions on the lifted geometry.

The prescribed data are modal coefficients, not a scalar function sampled on
the surface. For representative test shape `phi_i` in selected slot `k`,

```text
g_(i,k) = integral_representative phi_i(s) g_k(s) section_measure(radius(s)) ds.
```

The multiplier metric is component-diagonal representative mass with the same
section measure. It is not an unweighted line mass matrix. Radii must be finite
and positive. Lifted integration points must lie in the background mesh;
assembly fails rather than silently dropping points outside it.

Strong bulk Dirichlet values also contribute to the affine modal residual.
Changing those values through the same boundary-condition object updates that
contribution without rebuilding `B`. A changed mesh, radius, section, mode
selection, or constraint master structure requires rebuilding the lifted
observable and interaction.

Scalar and [vector static elasticity](lifted-static-elasticity) use the same
weak-term affine contribution storage. Scalar and vector terms automatically
read the current source offsets when evaluating the residual. No opt-in is
needed in either the weak term or `LiftedMultiplierInteraction`. Constraint
indices and master coefficients must remain fixed while reusing the matrix.
The combined implementation includes the field-preserving co-refinement from
[PR #259](https://github.com/luca-heltai/immersx/pull/259), static elasticity from
[PR #262](https://github.com/luca-heltai/immersx/pull/262), and the reindexed
mixed-field constraint-row correction from
[PR #241](https://github.com/luca-heltai/immersx/pull/241).

## Compose or extend the interaction

After creating the two spaces and applying bulk boundary conditions, the core
composition is:

```cpp
auto bulk_view = bulk_space.view();
auto modal_view = modal_space.view();
LiftedPoissonAdapter adapter(parameters.solver, bulk.communicator());
const auto u = adapter.add(algebraic_field(bulk_view.field("u"))).fields();
adapter.add(weak_term(gradient(u), gradient(test(u))));
// Known terms are residual contributions: the forcing term is -f.
adapter.add(known_term(negative_forcing, test(u)));
const auto geometry = modal_lift(modal_view.field("lambda"), parameters.lift);
const auto multiplier =
  adapter.add(LiftedMultiplierInteraction(u, geometry, modal_data)).fields();
auto state = adapter.make_state();
adapter.solve(state);
```

Here `negative_forcing` is a `KnownSource<double,3>`. `modal_data` may be a shared
`dealii::Function<3>` or a `ModalDataEvaluator` callback returning all coefficient
values for a representative `CellId` and point. The application creates that
callback with `modal_data_evaluator(representative, expressions, n_modes,
selector)`, reusing `InputFieldSelector` and `SymbolicFieldEvaluator`. It evaluates
retained ghosted input coefficients on the current representative cell before
applying nonlinear expressions, rather than interpolating expressions from
nodal values. No third application FE space is constructed. Supply the ordinary
bulk local preconditioner when using an iterative solver; `run_lifted_poisson`
registers AMG.

Keep the spaces, views, lift parameters, and imported Domain alive for the
observable and adapter. The views are borrowed objects, not owning snapshots.
For imported radius data, pass the third `modal_lift` argument:

```cpp
const auto input = representative.field("radius");
const auto radius = modal_radius(frozen(input.field(), input.coefficients()));
const auto geometry =
  modal_lift(modal_view.field("lambda"), parameters.lift, radius);
```

`modal_radius` retains a ghosted, read-only coefficient vector for local
representative-cell evaluation. The interaction hides particle redistribution,
search, coupling matrices, and metric assembly. Problems and application code
do not choose global execution blocks.

To add another physical contribution, add a standard weak/known term to the
bulk field or implement an Interaction accepting semantic Fields and a
`SemidiscreteBuilder`. Keep its physical residual separate from adapter solver
policy. `ModalLiftedObservable` exposes the scalar lifted quantity while
preserving every modal FE component; it is not an ambient vector observable.

## Solvers, refinement, and diagnostics

`Solver/Solver` accepts `automatic`, `iterative`, `direct`, or `mumps`. For
iterative runs, choose `Solver/Preconditioner = schur` or
`augmented_lagrangian`. `Tolerance` is the outer absolute residual tolerance;
`Augmented Lagrangian parameter` controls the augmentation. The generic AL
policy uses the diagonal inverse of the section-weighted metric, not the
legacy inverse-mass-squared preconditioner. Both solve the same physical
residual; iteration counts and performance are not claimed to match.

[Local co-refinement](local-co-refinement) runs before either application FE
space is attached and preserves imported radius data. Subsequent adaptive
cycles refine/coarsen the bulk mesh with Kelly indicators; the representative
mesh stays fixed. `global`, `fixed_fraction`, and `fixed_number` strategies are
available. Bulk adaptivity requires a distributed background Domain. Each cycle
redistributes bulk DoFs and rebuilds contributors, the adapter, and geometric
stencils. There is no transient history to transfer.

Each cycle writes:

- bulk FE output and all selected representative coefficient fields;
- a lifted particle VTU with reconstructed `lambda` and quadrature `weight`;
- a row of `diagnostics.csv` with cell/DoF counts, physical residual norm, and
  bulk/multiplier coefficient-vector norms.

Output, indicators, and error integration reconstruct hanging-node values.
Coefficient-vector norms are not FE L2 norms. Set `Compute errors = true` and
provide `Exact solution` for L2 and full H1 error norms; otherwise their CSV
entries are `nan`.

## Run the primitive Coral graph

Enable Coral plugins and graph tests as described in [the Coral guide](coral-backend).
CMake generates `build/coral/examples/lifted_poisson_primitives_3d.json` and its
parameter file from the checked-in templates. Run that generated graph with the
3D plugin, not the `.in` template:

```bash
coral -p "$PWD/build/coral/libcoral_backend_immersx_3_debug.so" run \
  "$PWD/build/coral/examples/lifted_poisson_primitives_3d.json"
ctest --test-dir build -R 'Coral.Graph.LiftedPoisson3D' --output-on-failure
```

Use the plugin suffix and extension for your build/platform. The graph exposes
each Domain, co-refinement, both FE spaces, scalar Fields, Dirichlet data,
Poisson and forcing terms, imported radius, modal lift/data, multiplier
interaction, AL execution adapter, state/residual checks, and native/lifted
output. There is no application node. CTest runs isolated one- and two-rank
copies with absolute build-tree input/output paths.

## Numerical checks and current limits

`LiftedPoisson.BOTH_*` compares coupling entries, modal loads, and every bulk
and multiplier coefficient against `ReducedPoisson`, matching DoFs by support
point and FE component. Cases cover a cylinder, mode order `{2,0}`, varying
imported radius, and a branched network. Both legacy Schur/AL and the new
Schur/AL paths are checked with nonzero forcing. An additional network case
compares an imported-radius-dependent modal load using a qualified alias,
`r*r`, `pi`, `E`, and `t=0`.

Legacy comparison uses homogeneous Dirichlet data: the legacy lifted coupling
does not retain the nonhomogeneous affine shift. Independent tests check a
nonzero constant boundary update and the manufactured solution `2+z-z*z` with
forcing `2`, nonzero boundary data, and hanging nodes. Application tests check
adaptive output, diagnostics, and default mode selection. These tests run in
serial and MPI. Existing production and independent quadrature tests remain.

This path does not implement the legacy zero-dimensional representative
backend, moving geometry, or representative adaptivity after initial
co-refinement. It does not establish large-network performance parity or make
the legacy application obsolete.
