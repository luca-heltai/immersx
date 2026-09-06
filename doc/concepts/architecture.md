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

`FESpaceView` is a non-owning view of a deal.II DoFHandler, mapping, affine
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
Linear terms can use cached deal.II operators. Nonlinear terms evaluate the
current state and provide the corresponding derivative.

For a trial expression `E` and test expression `T`, the assembled matrix has
the form

```{math}
A_{ij} = \int_\Gamma
  \left\langle E(\phi_j), T(\phi_i)\right\rangle\,d\mu.
```

The same term API supports same-DoFHandler assembly, paired DoFHandlers, and
nonmatching lifted geometry. The selected traversal and storage are backend
choices.

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

Other Interactions can own geometry, search, transfer operators, native
provider access, and auxiliary fields. The `MetricFlowXVesselWallConstraint`
is one such interaction: it connects a MetricFlowX area field to a nonlinear
wall displacement observable and sends the wall multiplier back as native
external pressure.

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
uses `TimeParameters` for the common time settings and IDA configuration,
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
