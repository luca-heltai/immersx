# Choose where a weak term is integrated

A weak term combines a finite-element expression, a test function, and a
geometric integration support. The support chooses the cells, faces, or
physical quadrature points used by the integral. It does not need to own a
finite-element space.

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
