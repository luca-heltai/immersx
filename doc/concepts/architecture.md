# Architecture concepts

ImmersX separates physical discretizations, relations between discretizations,
and solver policy. An application adds its Problems and Interactions directly
to an execution adapter.

## Problems and fields

A Problem owns physical and discretization-specific data. Depending on the
Problem, this includes a mesh and DoFHandler, finite element and mapping,
material parameters, native matrices or operators, forcing and boundary data,
and the accepted physical state. A Problem does not know the global execution
block layout and does not need to inherit an ImmersX base class. A native
provider from another library can be adapted through its public accessors.

A `Field` gives a state or residual row a semantic identity. A field has a
name, an FE-space view, an extractor, and a `FieldId` in a `StateLayout`; it
does not own coefficient storage. A field identifier is not a global matrix
block, a native Problem block, or an execution-adapter block. The adapter maps
semantic fields to its own execution vectors.

`HistoryGroupId` identifies a timeline shared by one or more fields. Accepted
snapshots are stored by `StateHistory`; `StateHistoryRegistry` stores separate
histories for independent groups. `StateView` supplies non-owning values for
one evaluation, while `EvaluationContext` supplies the state and optional
state derivative to contributors.

## FE expressions and Observables

`FiniteElementSpaceView` is a non-owning view of a deal.II DoFHandler, mapping, affine
constraints, and locally relevant indices. Its `field()` methods create named
fields with a deal.II `FEValuesExtractor`.

`Observable<Field, Operation>` is a typed FE expression. The primitive
operations delegate to deal.II's `FEValuesViews`: `value`, `gradient`,
`divergence`, `symmetric_gradient`, and `curl` where the extractor supports
them. Pointwise arithmetic and symbolic kernels can compose observables. The
resulting value type and tensor operations remain deal.II types.

A frozen observable uses supplied coefficients instead of an active state
field. It has no active dependency, but it uses the same FE-space operation
machinery. Imported finite-element data therefore enters an application as an
ordinary FE space and a frozen observable.

An observable can also be lifted between a representative geometry and a bulk
finite-element space. Tensor-product quadrature, point search, retained
stencils, and redistribution are implementation details of that observable or
interaction.

## Weak terms and constraints

A `WeakTerm` pairs a trial observable with an explicit test expression. The
test expression is created with `test(field)` and is independent of the
current state. For fields `u` and `v`, examples include:

```cpp
weak_term(value(u), test(v));
weak_term(gradient(u), gradient(test(v)));
weak_term(divergence(u), test(v));
weak_term(symmetric_gradient(u), symmetric_gradient(test(v)));
```

The term contributes to the residual row represented by the test field. Its
linearization is generated from the active fields in the trial observable.
Registering a linear term retains its assembly actions, without constructing
its matrix. Callers explicitly materialize and retain matrices that they reuse.
Nonlinear terms evaluate the current state and provide the corresponding
derivative.

For a trial expression `E` and test expression `T`, the assembled matrix has
the form

```{math}
A_{ij} = \int_\Gamma
  \left\langle E(\phi_j), T(\phi_i)\right\rangle\,d\mu.
```

The same term API supports same-DoFHandler assembly, paired DoFHandlers, and
nonmatching lifted geometry. The selected traversal and storage are backend
choices.

A `Domain` can supply a non-owning cell integration support. `support()` selects
all active cells, while `support(material_id)` selects cells with that deal.II
material id. `boundary_support()` selects all boundary ids and
`boundary_support(id)` selects one id. Attach a support directly or with
`.on()`:

```cpp
auto mass = value(u) * test(v) * domain.support(1);
auto laplace = weak_term(gradient(u), gradient(test(v)))
                 .on(domain.support());
auto dGamma = domain.boundary_support(0);
auto boundary_mass = value(u) * test(v) * dGamma;
auto normal_flux = gradient(u) * dGamma.normal() * test(v) * dGamma;
```

The support selects cells or boundary faces and their physical measure. When
the support is native to a field, matching integrals retain deal.II's native
`FEValues` or `FEFaceValues` path. A codimension-one support may also provide
physical quadrature points for a background-to-background trace mass pairing:

```cpp
auto dGamma = immersed_domain.support();
auto trace_mass = value(u) * test(u) * dGamma;
```

Here both DoF spaces belong to the background mesh; the immersed support owns
only the integration geometry. This path supports value and gradient pairings
where the trial and test fields share a background triangulation and mapping.
Geometry-only supports such as `domain.lift(descriptor)` use the same syntax;
their locally owned physical points and weights define the integration rule.
Nonlinear composed observables with frozen finite element inputs can use these
physical supports as well; their residual and Jacobian evaluate at the same
points on the shared background triangulation.
It uses deal.II `GridTools::Cache` for
serial point location and `ParticleCoupling` to route support quadrature points
to distributed background cells. `FEValues` evaluates the background basis.
The current support rule uses `QGauss(p + 2)` per support cell and does not split
at background-cell boundaries, so support-mesh refinement is needed when a
support cell crosses such a boundary. deal.II does not provide a native
background-to-background trace-matrix oracle for this independent geometry.
`normal()` is available when the mesh and fields have the
same dimension; embedded
surfaces provide `conormal()`. A planar two-dimensional boundary also provides
`tangent()`.

Matching cell and boundary terms reintegrate affine source contributions on
cells with nonzero constraint offsets when the prescribed values change.
They do not retain local cell matrices. Nonmatching and lifted linear
pairings currently require homogeneous source constraints and reject nonzero
offsets, including changes made after assembly. Linear multiplier reaction rows
also require homogeneous multiplier constraints. Rebuild explicitly retained
matrices when their coefficients, constraint structure, meshes, or mappings
change.

A `Constraint` is an Interaction assembled from weak terms with one
independently defined multiplier field. The multiplier has its own FE space
and is an algebraic field. The constraint contributes its signed weak-term sum
to the multiplier row and contributes the transpose reactions to participant
rows. For example:

```cpp
auto continuity = make_constraint(
  weak_term(value(u), test(lambda)) -
  weak_term(value(v), test(lambda)));
```

Other Interactions can own geometry, search, transfer operators, and auxiliary
fields. Applications express each coupling through the terms and fields that
define its residual contribution.

## Residual and execution adapters

The solver-neutral semidiscrete contract is

```{math}
F(t,y,\dot y)=0.
```

Contributors expose the two linearizations separately:

```{math}
\frac{\partial F}{\partial y},\qquad
\frac{\partial F}{\partial \dot y}.
```

Candidate residual evaluations do not accept a physical state or update
Problem history. Acceptance and native output occur in the application-owned
accepted-state callback or handoff.

`LinearAdapter` executes affine systems. It registers contributors with
`add()`, creates execution vectors with `make_state()`, exposes field views
with `field()`, and solves the assembled linear system with its configured
deal.II solver and preconditioner.

`IDAAdapter` executes the residual as a differential-algebraic system. It
uses separate `TimeIntervalParameters` and `IDAParameters` objects for the
simulation interval/output schedule and IDA configuration,
builds the differential-component mask from field metadata, and forms the
solver Jacobian

```{math}
J = \frac{\partial F}{\partial y}
  + \alpha\frac{\partial F}{\partial \dot y}.
```

The adapter owns `alpha`, execution blocks, vectors, solver callbacks, output
callbacks, and optional consistent-initial-condition or restart callbacks.

`KINSOLAdapter` executes a steady nonlinear residual `F(y)=0` when deal.II is
built with SUNDIALS. It uses the same semantic contributor model, but rejects
contributors that register derivative terms. Its `KINSOLAdapterParameters`
select the Newton or line-search strategy and nonlinear stopping controls.

`ProblemHandle` connects an adapter to the fields returned by a contributor.
It is the application's typed way to access semantic field identities while
the adapter keeps execution storage private.

## Output

`Problem::output_results()` is legacy/native Problem-owned output of an
accepted state. It remains useful when a Problem owns the output lifecycle.
`OutputHandler` is solver-neutral output of semantic execution `Field`s. It
reads fields through `StateAccessor`, so it can be used by a linear, IDA, or
KINSOL execution without adding solver knowledge to the output code.

An `OutputHandler` reconstructs the native FE vector from the registered
fields. This matters for a mixed Stokes space: velocity and pressure are
separate semantic fields with separate execution vectors, but both refer to
one native mixed `DoFHandler`. The velocity and pressure `Field`s retain their
native-to-execution mappings, so output can write one dataset with a vector
`velocity` and scalar `pressure` without exposing execution block numbers to
the handler.
