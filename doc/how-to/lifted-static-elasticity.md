# Compose static elasticity with a modal surface lift

`lifted_elasticity` solves small-strain isotropic elasticity in 2D or 3D.
In 3D it can constrain displacement on a cylindrical surface swept along a
1D representative mesh, or apply prescribed modal pressure coefficients there.
It does not call `Elasticity` or `ElasticStatic` to assemble or solve the model.
The legacy applications remain available.

## Run the application

Build and run from an out-of-source build:

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --target lifted_elasticity_debug -j
./build-debug/lifted_elasticity_debug \
  "$PWD/build-debug/data/lifted_elasticity/multiplier_3d.prm"
```

For Release, build `lifted_elasticity` instead. The generated examples under
`data/lifted_elasticity/` also include uncoupled 2D and variable-radius pressure
cases. Use these **build-tree** files, not their `.in` source templates. Output
paths in the generated files are absolute, so the executable does not depend
on its working directory. MPI uses the same file:

```sh
mpirun -np 2 ./build-debug/lifted_elasticity_debug \
  "$PWD/build-debug/data/lifted_elasticity/multiplier_3d.prm"
```

The `Lifted elasticity` section selects `Coupling = none`, `multiplier`, or
`pressure`. Its `Domain` and `Immersed domain` subsections configure the two
Domains. `Displacement space/Finite element` selects a primitive vector FE
with one component per spatial dimension. Its `auto` default repeats `FE_Q`
using the inherited `FE degree`. `Multiplier space/Finite element`
selects a scalar FE repeated once per selected mode, or an explicit primitive
`FESystem` with that many components. Modal coefficients are independent scalar
Fields on **one** coefficient FE space, not separate meshes or execution blocks.
As in the native formulation, the background trace must resolve these
coefficient tests: a rank-deficient coupling leaves multipliers undetermined,
even when the displacement and algebraic residual are correct.

`Lift/Cross section/Selected modes` gives the ordered cross-section basis
indices. `Modal data expressions` must contain one expression per selected
index in that order. These are representative-domain coefficients, not Cartesian
components to be sampled on the swept surface. For example:

```text
subsection Lifted elasticity
  set Coupling = multiplier
  set Modal data expressions = 0.01, -0.02, 0.03
  subsection Lift
    set Thickness = 0.1
    subsection Cross section
      set Selected modes = 0, 1, 2
    end
  end
end
```

For an imported VTK radius, enable `Immersed domain/VTK support/Read fields`
and set `Radius field = radius`. The immersed Domain owns and transfers the
imported field; the application does not allocate another property FE space.
The lift evaluates this frozen field at representative quadrature points,
using locally relevant coefficient ghosts when a curve cell crosses an MPI
partition.

`Modal data fields` uses `InputFieldSelector` syntax to expose imported
scalar/component aliases to `Modal data expressions`. For example, select
`radius` and set `0.1*radius, -0.2*radius, 0.05*radius` for three modes. These
expressions require SymEngine, as in native `ReducedCoupling`. Coordinate-only
expressions, with `pi`, `E`, and static `t=0`, do not. Imported data are sampled
with deal.II `FEValues` on the retained source cell, without another point search.

## Equations and solver policy

The bulk term is

```text
integral [2 mu epsilon(u):epsilon(v) + lambda div(u) div(v)] - integral f.v.
```

Material-id parameters select `mu` and `lambda` cell by cell. Strong Dirichlet,
Neumann traction, normal displacement, and symmetric face-Nitsche conditions
use disjoint boundary-id sets. Nitsche uses the full vector datum, the adjacent
material, and `penalty / cell_diameter`. Normal-flux ids constrain `u.n = g.n`;
they are not a prescribed stress traction. Existing material-specific forcing
and boundary-specific function parameters are inherited from the static
elasticity parameter family.

For selected physical vector mode `phi_k`, the coupling is

```text
B_k(u, q_k) = integral_surface u . (q_k phi_k).
g_k(q_k)    = integral_representative coefficient_k q_k section_measure(radius).
```

`lift(value(component), geometry).mode(k)` preserves the selected mode's
amplitude and frame rotation. The older default lift behavior is unchanged.
Multiplier coupling imposes `B u = g` and adds the transpose reaction to the
bulk row. All modes form one unified `ConstraintSum` and one semantic multiplier
field. Its metric is the component-diagonal representative mass weighted by
the physical section measure, matching the native tensor-product formulation.

Pressure coupling first solves `M p = g` with that metric, then solves
`A u = f + B^T p`. It does not silently ignore the pressure load.

`Linear adapter` owns solver selection and execution storage. The default is
iterative: automatic Schur for multiplier coupling and block preconditioning
for pressure or uncoupled systems, with a local AMG elasticity policy. Select
`Preconditioner = augmented_lagrangian` to use the same section-weighted modal
metric in AL preconditioning. `Solver = direct` and the existing MUMPS selection
remain available; direct solution is not required for multiple vector modes.

## Compose or extend the model

The application composition lives in
`include/immersx/physics/lifted_elasticity.h`. Ordinary C++ usage is:

```cpp
LiftedElasticityParameters<3> parameters;
initialize_parameters(parameter_file);
LiftedElasticity<3> problem(parameters);
problem.run();
```

A custom application can use the same pieces directly:

1. Construct the background and representative `Domain`s; run
   `LocalCoRefinement` before constructing their two `FiniteElementSpace`s.
2. Select the displacement vector Field and scalar component Fields of the
   modal `FESystem`. Keep all coupling component names the same when collecting
   them into one semantic multiplier.
3. Add the bulk algebraic Field and `static_elasticity_term` to `LinearAdapter`.
   Add `static_elasticity_nitsche` when needed. `IsotropicElasticStress` exposes
   the constitutive evaluation; `isotropic_elasticity_term` builds a constant
   material weak term from FEValuesViews strain and divergence.
4. Pair `value(displacement)` with `test(lift(value(component), geometry).mode(k))`
   using `weak_term`, and collect the terms in `ConstraintSum`.
5. Use `lifted_modal_load` to assemble each representative coefficient load
   and metric. Sum the component contributions, then add
   `lifted_modal_constraint(terms, metric, rhs)` directly to the adapter. This
   small Interaction owns the geometric metric and reuses the unified
   `Constraint`; it does not introduce a second constraint hierarchy.
6. Register native component Fields with `OutputHandler` and call
   `adapter.write_output(handler, state, time)`. Register all FE components in
   one handler. The adapter binds semantic state internally.

For other physical terms, provide a callable contributor that receives the
existing semidiscrete builder, or add ordinary `WeakTerm`s. Keep material and
geometry in the Problem/Interaction and solver combinations in the adapter.
`Constraint::add(builder, &metric_operator)` accepts a supplied physical metric;
without it, existing constraints keep their default FE mass metric.

## Refinement, output, and diagnostics

Initial co-refinement supports `space`, `embedded`, and `both`, with lower and
upper diameter bounds. It preserves imported fields and supports a
fullydistributed representative Domain. Static cycles rebuild the two FE spaces
and their lifted search/assembly state after refining the distributed bulk
mesh. Choose `Refinement strategy = global`, `fixed_fraction`, or
`fixed_number`; the latter two use Kelly indicators. A fullydistributed bulk
Domain supports one cycle only. No time history or accepted transient state is
maintained.

Each cycle writes displacement and all modal coefficient components through
`OutputHandler`. The inherited convergence table reports the configured exact
solution errors or solution norms. `residual_norm()` reports the algebraic
static residual from the latest solve. The application rejects nonfinite
residuals. Each CLI cycle prints its residual, and
`initial_co_refinement_report()` exposes the initial geometry diagnostics,
which also appear on stdout. Component accessors expose native FE coefficient
vectors for physical diagnostics without exposing execution block numbers.

### Lifted-wall particles

With `Coupling = multiplier` or `pressure`, each cycle also writes
`<Output name>_particles_<cycle>.vtu` and the record
`<Output name>_particles.pvd`. There is one particle per lifted quadrature
point: each representative quadrature point times each cross-section
quadrature point. Points outside the background mesh are dropped.
`Particle output/Properties` is a comma-separated list of what each particle
carries:

| Property | Particle data |
|---|---|
| `weight` | Quadrature weight of the wall point |
| `mode_values` | Selected mode shapes `mode_<k>`, vectors for elasticity |
| `modal_coefficients` | Modal coefficients `<Coupling>_<k>` at the representative point |
| `lifted_value` | `<Coupling>`, the sum over modes of coefficient times mode shape |
| `modal_data` | Prescribed modal data `data_<k>` |
| `representative_point` | Curve point the wall point is lifted from |
| `radius` | Distance from the representative point |
| `indices` | `source_cell`, `representative_qpoint`, `section_qpoint` |
| `bulk_field` | Background displacement interpolated at the wall point |

`<k>` is the selected-mode slot. The default is `lifted_value, weight`; an
empty list disables the particle output.

## Coral graph

The configured `build-coral/coral/examples/lifted_elasticity_3d.json` is an
internal-component graph, not an application facade. Its sources are
`coral/examples/lifted_elasticity_3d.json.in` and `.prm.in`; run the generated
build-tree inputs, not these templates. Its nodes construct the two Domains, co-refine them,
construct the two FE spaces, apply boundary conditions, create the material
law and elasticity weak term, select three modal components, lift and test each
component, project **nonzero** modal data, sum the physical metric and weak
terms, add one interaction, solve, check the residual, and write both spaces.
Its accompanying `.prm` selects Schur execution.

With an installed Coral package, enable plugins and graph tests as described in
[Coral backend](coral-backend). For a Debug build, set the absolute build path
from the repository root, then run the generated graph even from an unrelated
directory:

```sh
build="$PWD/build-coral"
mkdir -p "$build/coral/lifted_elasticity_status"
(cd /tmp && coral -p "$build/coral/libcoral_backend_immersx_3_debug.so" \
  run "$build/coral/examples/lifted_elasticity_3d.json" \
  --graph "$build/coral/lifted_elasticity_3d.dot" \
  --touch-dir "$build/coral/lifted_elasticity_status")
```

The generated graph uses absolute parameter and output paths. Output goes to
`build-coral/coral/lifted_elasticity_output`, not the calling directory.
`--touch-dir` also keeps Coral's node-status markers in the build tree. CTest
configures separate inputs and output directories for each rank/build variant.
Run the registry and graph checks with:

```sh
ctest --test-dir build-coral -R 'ImmersX.Coral.*(Registry.3d|LiftedElasticity)' \
  --output-on-failure
```

The graph must run with the registry generated by the same plugin build.
Serial and two-rank tests check graph execution and both output files.

## Legacy capability boundaries

| Static capability | This application |
| --- | --- |
| Uncoupled 2D and 3D isotropic elasticity | Supported |
| Material ids, forcing, strong/weak Dirichlet, traction, normal displacement | Supported |
| 3D tensor-product curve-to-surface multiplier and pressure coupling | Supported |
| Independent vector modes, nonstandard indices, imported radius | Supported |
| Modal expressions using imported field aliases | Supported with SymEngine; coordinate-only expressions do not require it |
| Schur and section-weighted AL; direct alternative | Supported |
| Initial two-Domain co-refinement and bulk static refinement cycles | Supported as described above |
| Legacy 2D point inclusions / 0D representative spaces | Not implemented; use legacy `Elasticity` |
| Legacy non-tensor-product inclusion variants | Not implemented; use legacy `Elasticity` |
| Selectable higher-order curved mappings | Not implemented; the FE spaces use `MappingQ1` |
| Transient, frequency-domain, nonlinear or fiber elasticity | Outside this static application |
| Legacy stress/reaction reports, weighted errors, condition-number estimates and iteration-summary format | Not reproduced; standard error tables and adapter residual/solver diagnostics are available |
| Legacy exact/material/subdomain arrays and aggregate PVD naming | Not reproduced; output contains displacement and modal coefficients in per-cycle VTU/PVD files |
| Legacy particle dumps | Replaced by lifted-wall particles with selectable properties (see above) |

Native parity uses homogeneous strong boundary data with nonzero body/modal
loads and compares full displacement and multiplier fields, RHS vectors, and
operator actions. An independent constant-displacement manufactured test covers
nonzero strong data and hanging nodes in the lifted constraint. Oblique
two-segment curves exercise partition-crossing radius ghosts and imported-field
modal expressions. The legacy lifted assembly drops
the inhomogeneous source-constraint shift in that case; the new path retains it
rather than matching the incorrect legacy RHS. Lifted affine data are assembled
for the static constrained space; this application makes no claim about changing
boundary constraints during transient integration.
