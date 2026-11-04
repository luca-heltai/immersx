# Choose where a weak term is integrated

A weak term combines a finite-element expression, a test function, and a
geometric integration support. The support chooses the cells, faces, or
physical quadrature points used by the integral. It does not need to own a
finite-element space.

## Registration and assembly

Adding a weak term with `.add(builder)` records its assembly actions. It does
not assemble or retain the term's matrix. Request a matrix explicitly when it
will be reused, and keep that matrix in the caller. To assemble only selected
contributions, register them with a separate builder. For a serial deal.II
`SparseMatrix`, use the owning result returned by `matrix()`, or provide a
destination with caller-owned sparsity when using `materialize_into_matrix()`.
The standard iterative `LinearAdapter` solve materializes a matrix-based system
once and owns that matrix for the solve. Custom solve callbacks and systems
without complete matrix provenance keep their operator path.

Changing prescribed Dirichlet values with the same constraint structure does
not change a linear operator's matrix. Affine residual contributions on matching
cell and boundary supports are reintegrated only on cells with nonzero trial
constraint offsets. Local matrices are temporary; they are not retained for
later boundary updates. This also accounts for offsets propagated through
closed hanging-node constraints.

## Configure a field space

Register space parameters before reading the application parameter file. The
owning space distributes DoFs on a Domain's distributed triangulation; retain
both the space and its view while Fields or execution adapters use them:

```cpp
DomainParameters<1, 2> domain_parameters("/Line/Domain/");
FiniteElementSpaceParameters<1, 2> multiplier_parameters(
  "/Line/Multiplier space/", "FE_DGQ<1>(0)");
initialize_parameters(parameter_file);
Domain<1, 2> line(domain_parameters);
line.make_grid();
FiniteElementSpace<1, 2> multiplier_space(line.triangulation(),
                                         multiplier_parameters);
auto multiplier_view = multiplier_space.view();
auto lambda = multiplier_view.field("lambda");
```

The `Finite element` entry accepts names understood by
`dealii::FETools::get_fe_by_name()`. Coupled Poisson exposes this entry under
`Coupled Poisson/Multiplier space`, defaulting to `FE_DGQ<1>(0)`. Its generated
regression input `gtests/parameters/coupled_poisson_multiplier_q1.prm` selects
`FE_DGQ<1>(1)` with nonzero loads.

Static elasticity exposes `Elastic static/Displacement space/Finite element`.
The default `auto` entry retains `FE degree` and builds `spacedim` copies of
`FE_Q` on hypercubes or `FE_SimplexP` on simplices.
For example, `FESystem<2>[FE_Q<2>(2)^2]` selects quadratic displacement in 2D.
The element must be primitive with `spacedim` components; assembly quadrature
uses the selected element's actual degree. Problems still own their native
matrices, boundary constraints, material data, and accepted solution.

For a new Problem, own its Domain and FiniteElementSpace, update the space's
constraints for the physical boundary data, and expose the native DoFHandler,
mapping, constraints, and relevant indices. Contributors can use
`finite_element_space_view(problem)` without repeating those accessors. Keep
native residual assembly when it carries affine boundary corrections; changing
ownership does not require replacing it with a weak term.

## Simplex meshes

Each Domain and integration support must contain one reference-cell type:
triangles, tetrahedra, quadrilaterals, hexahedra, or line segments. Mixed cell
meshes are rejected. Simplex Domains use `Triangulation type = fullydistributed`.
Refine the serial mesh during preparation before finalizing its ownership.

Select a matching element, such as `FE_SimplexP<2>(1)` for a scalar triangle
space or `FE_SimplexDGP<2>(0)` for a cellwise multiplier. The owning
`FiniteElementSpace::view()` selects deal.II's linear mapping for that element.
A non-owning view keeps the explicit mapping supplied by its Problem.

Weak terms, known sources, boundary terms, and multiplier metrics select
Gauss quadrature from the reference cell. Independent integration meshes use
their own reference cell and mapping when generating physical support points.
Particle searches and immersed partitioning also select a matching background
mapping when none is supplied explicitly.

Lifting a triangular sheet into a volume uses simplex Gauss quadrature on the
sheet. Its `Representative quadrature/Type` must be `gauss`; the configured
point count and number of repetitions still apply. Line representatives retain
their configured one-dimensional quadrature.

## Cell and boundary supports

Use a domain support to select all cells or one material id. Use a boundary
support to select all boundary faces or one boundary id:

```cpp
auto dOmega = domain.support(1);
auto mass = value(u) * test(v) * dOmega;
auto laplace = weak_term(gradient(u), gradient(test(v)))
                 .on(domain.support());

auto wall = domain.boundary_support(0);
auto boundary_mass = value(u) * test(v) * wall;
auto normal_flux = gradient(u) * wall.normal() * test(v) * wall;
```

The multiplication form and `weak_term(...).on(support)` form select the same
support. The matching-mesh path uses deal.II `FEValues` or `FEFaceValues`.
Linear terms on matching cell or boundary supports, and terms that pair a bulk
field with a lifted test expression, retain the local contributions eliminated
by the source Field's affine constraints. Each residual evaluation reads the
current inhomogeneities and includes those contributions automatically. No
option on `weak_term` is needed. Changing the offsets does not rebuild the
matrix. Update time-dependent boundary data before evaluating the residual;
a Problem can register a `builder.context_update` callback for this purpose.
Constraint updates at a candidate time do not accept that physical state.

The mesh, mapping, constrained DoFs, and master coefficients must remain fixed
while reusing the matrix. A change to the constraint structure is rejected
collectively; rebuild the term after such a change.

## Classify cells against a background mesh

`set_overlap_material_ids()` assigns source-cell material ids from quadrature
points located in a background mesh. The search is collective and supports
independent source and background partitions. Source owners classify their
cells; deal.II exchanges the resulting ids with source ghost cells.

```cpp
#include <immersx/grid/overlap_classification.h>

// line and tissue are already initialized Domains; mappings match their meshes.
dealii::GridTools::Cache<3> background(tissue.triangulation(), tissue_mapping);
ImmersX::OverlapMaterialIds ids{10, 20, 30}; // outside, partial, inside
ImmersX::set_overlap_material_ids(line.triangulation(),
                                  line_mapping,
                                  dealii::QGauss<1>(3),
                                  background,
                                  ids);
auto inside_cells = line.support(ids.inside);
```

A cell is outside if no quadrature point is found, inside if all points are
found, and partial otherwise. These labels describe the supplied samples;
they do not certify an exact geometric intersection. The operation overwrites
existing source material ids. It leaves background material ids unchanged.
Source and background communicators must be identical or congruent.

To select every cell with at least one located sample through one support,
use the same id for `partial` and `inside`, for example `{10, 20, 20}`, then
select `line.support(20)`. The `outside` id must remain distinct. A material
support selects entire cells: this does not clip quadrature or permit exterior
points in existing nonmatching assemblers on partially covered cells.

The overload taking a point generator accepts physical wall samples or other
cell-associated points instead of mapped reference-cell quadrature:

```cpp
ImmersX::set_overlap_material_ids(
  line.triangulation(), background,
  [&](const auto &cell) {
    return physical_wall_points.at(cell->global_active_cell_index());
  }, ids);
```

The generator is called once per locally owned active source cell and must
return a nonempty vector of finite `dealii::Point<3>` values. Group existing
lifted-support points by their `source_entity_id`, which is the representative
cell's `global_active_cell_index()`. Classify the lifted wall when the coupling
uses that wall; centreline containment alone does not imply wall containment.

All ranks participate, including ranks without source cells. Serial meshes,
`parallel::distributed` and `parallel::fullydistributed` source meshes are
supported. Other parallel source types must support deal.II's
`GridTools::exchange_cell_data_to_ghosts()`; in particular,
`parallel::shared` requires artificial cells to be enabled. Search tolerance
is given in background reference-cell coordinates and defaults to `1e-10`.
Reclassify after changing a mesh, mapping or sampling geometry. Classification
does not add multiplier constraints or change the coupled equations.

## Couple partially overlapping domains

`make_continuity_constraint(bulk, embedded, lambda)` partitions the multiplier
quadrature against the full-dimensional participant. The multiplier mesh must
also carry an embedded participant when one is present. Between two bulk
fields it can have an independent mesh; the overlap then contains only samples
located in both bulk domains. The partition is fixed when the constraint
is constructed, before the solve. It preserves both meshes' physical material
ids.

Both participant pairings and their transposed reactions use the located
samples. The multiplier row is

```text
integral_overlap (bulk - embedded) * test(lambda)
  - integral_outside lambda * test(lambda) = 0.
```

The exterior integral includes missing quadrature points in partially covered
cells. Its mass matrix enters the physical multiplier block with a minus sign,
so the positive Schur complement gains the positive exterior mass. A completely
exterior multiplier space therefore has the homogeneous mass equation and
produces no coupling force. On a continuous basis spanning both regions, the
row is one combined weak equation; it does not impose independent pointwise
conditions on the two regions.

Custom Interactions can construct the same complementary supports explicitly:

```cpp
#include <immersx/core/overlap_integration_support.h>

auto supports = ImmersX::make_overlap_supports(
  line.support(), line_mapping, dealii::QGauss<1>(3), background);
auto fields = ImmersX::make_constraint(
  ImmersX::weak_term(ImmersX::value(bulk), ImmersX::test(lambda))
    .on(supports.overlap) -
  ImmersX::weak_term(ImmersX::value(embedded), ImmersX::test(lambda))
    .on(supports.overlap)).add(builder);
const auto registered_lambda = fields.multiplier;
ImmersX::weak_term(-1. * ImmersX::value(registered_lambda),
                  ImmersX::test(registered_lambda))
  .on(supports.outside).add(builder);
```

A material filter supplied through `line.support(id)` still selects the source
cells. The quadrature partition keeps that filter and does not replace material
ids. Use the full multiplier mesh as the source support when the exterior
mass must cover every multiplier DoF. Unlike the cell classification alone,
these supports select individual quadrature samples and retain their original
weights. They use the same fixed quadrature in both pairings and the exterior
mass. Rebuild the supports and the
associated operators after changing a mesh, mapping, partition, or sampling
geometry.

For a lifted interaction, pass its complete source-owned physical sample vector
between the representative quadrature and the background cache. Records contain
`point`, `weight`, `source_entity_id`, `representative_qpoint`, and
`section_qpoint`. Use one geometry cloud, shared by all modes. Section indices
start at zero and cover each representative quadrature point exactly once.
Lifted assembly selects physical samples with their original wall weights;
representative assembly sums those weights over the selected section. A cut
modal section requires the full sampled Gram matrix, including cross-mode
entries. Do not rescale the original orthogonal diagonal by a coverage fraction
or renormalize the remaining wall weights.

The search and ghost exchange are collective. Empty source ranks participate.
Use the default search tolerance for the existing particle assembly backends.
An explicit lifted cell support must use the representative field mesh.
The partition approximates overlap through the supplied samples; it does not
clip cells at the geometric intersection. Raw nonmatching weak terms retain
their existing search behavior when no overlap support is supplied.

Prescribed pressure loads retain their nominal, full-support coefficients and
restrict only the applied load to the covered samples. A pressure reconstructed
from multiplier traction uses the same covered wall samples and nominal
normalization. Tensor-product pressure output retains its centerline mean;
the interior pressure closure uses the full-support projected datum and the
existing cell-averaged material coefficients.

Legacy point and modal saddle systems use the same exterior mass block and the
full sampled Gram matrices. The `Assemble full AL system` option in Reduced
Poisson assembles a diagonal-Gram approximation for the preconditioner; the
augmented operator retains the full Gram matrix. The augmented row transform
also acts on the exterior multiplier block and the constraint right-hand side.

## Integrate bulk fields on an independent mesh

An immersed domain can supply integration geometry for trial and test fields
that both live on a background mesh:

```cpp
auto dGamma = immersed_domain.support();
auto trace_mass = value(u_background) * test(v_background) * dGamma;
trace_mass.add(builder);
```

The immersed mesh needs no `DoFHandler` or `Field`. Its quadrature points and
physical weights are located in the background mesh, and each field is
evaluated on its own background `DoFHandler`. Linear pairings accept independent
supports of any positive dimension up to the background dimension; nonlinear
pairings currently require codimension one. Both require fields on the same
background triangulation and mapping. The [particle penalty comparison](../performance/particle-penalty)
checks the extended linear path against independent deal.II assembly.
It uses `QGauss(p + 2)` on each support cell without splitting that
cell at background-cell boundaries. This approximates a piecewise FE trace,
not exact integration over mesh intersections. A Gauss rule exact for a global
polynomial need not integrate a trace whose derivatives jump at those
interfaces. Refine support cells until the integral or operator action
stabilizes; increasing only the background resolution does not control this
error.

`WeakTerm.IndependentSupportConvergesForPiecewiseFETrace` checks a Q1 trace
with three slope jumps against `MatrixCreator::create_mass_matrix` on a fitted
1D mesh. After eight support refinements, the energy error must be below
`1e-5` and one thousandth of the coarse error. Individual levels need not
improve monotonically. Constant-field and global-polynomial checks remain
separate measure and mapping checks. No intersection subdivision or Nitsche
assembler is provided.

The same physical-point assembly accepts composed observables, including
nonlinear expressions with frozen finite-element inputs:

```cpp
// response is a composed observable of active and frozen fields.
weak_term(response, test(v_background)).on(dGamma).add(builder);
```

The residual and Jacobian use the current active state and the frozen input
coefficients. Frozen FE inputs must share the source mesh and mapping.
Linear pairings on independent geometric supports retain essential source
offsets. Supported linear pairings refresh eliminated Dirichlet contributions
during residual evaluation while constraint entries stay fixed. Rebuild after
changing constraint structure or geometry.
The three-argument `fe_space(dof_handler, mapping, constraints)` view extracts
locally relevant DoFs. Pass an explicit set when one is already available.
Distributed nonlinear evaluation needs locally relevant ghost values in active
states and frozen coefficient vectors. `ImportedFiniteElementFields` exposes
owned coefficients; import them into a ghosted vector before calling
`frozen(field, coefficients)`. That call copies the vector. Recreate the
observable to change its frozen coefficients.

An ordinary `IntegrationSupport` is a non-owning mesh view. A new assembly
reads its current geometry; a materialized linear operator is a snapshot and
must be assembled again after a geometry change. A lifted support stores
generated points: call `rebuild()` before reassembling. Geometry must stay fixed
during each residual/Jacobian evaluation. The geometric shape derivative is
not included in the weak-term Jacobian.

## Use a lifted support

`Domain::lift()` turns a representative mesh and a `TensorProductLift` into
physical quadrature support. It creates no finite-element space on the lifted
geometry:

```cpp
auto dV = centerline.lift(lift);
auto lifted_mass = value(u_background) * test(v_background) * dV;
lifted_mass.add(builder);
```

For lifted and independent immersed supports, trial and test fields must use
the same full-dimensional background triangulation and mapping, and all
support points must lie in its searchable domain. Geometry is fixed during
residual and Jacobian evaluation. Call `dV.rebuild()` after changing the mesh,
lift descriptor, or thickness evaluator. See [Integrate over a lifted
geometry](lifted-integration) for ownership, normal, and benchmark details.

Known callback sources can use the same physical support:

```cpp
KnownSource<double, 3> forcing([](const KnownTermContext<3> &context) {
  return context.point[0] + context.time;
});
known_term(forcing, test(u_background)).on(dV).add(builder);
```

The callback is evaluated at the physical quadrature points, with the physical
weights supplied by the support. Its material id is the background cell's material id.
These terms have no state derivative. Physical supports can have any positive
dimension up to the background dimension, including independent volume meshes.

Mesh uniformity refers to physical cells. A fully distributed triangulation may
keep an artificial hypercube on a rank with no locally owned cells. Mesh and
space setup identify the physical reference cell collectively, so these ranks
use the same finite element family as the other ranks. Field views and geometry
accessors remain local operations. Independent geometry assembly and search
setup also use the physical reference cell.

Construct lifted quadrature supports and call their `rebuild()` method
collectively. Reading their already generated local quadrature points remains
a local operation.

Nonlinear weak terms import the candidate field coefficients before evaluating
local cells. Candidate vectors may contain only locally owned entries. This
setup is collective, including ranks without physical cells. Residual assembly
uses the target Field constraints, as the Jacobian assembly does, so essential
rows are omitted and hanging-node contributions are redistributed.
