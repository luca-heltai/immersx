# Current implementation architecture

This page describes the implementation on the current `master` branch. It
complements {doc}`../../concepts/architecture`, which defines the ownership and
residual contracts. The two pages describe the same repository: the concepts
page includes the design rationale and roadmap, while this page identifies the
classes and execution paths that are currently available.

## Two current paths

Current ImmersX contains both a semantic/composable path and an established
reduced/tensor-product path. They coexist; the latter has not been rewritten in
terms of the former.

### Semantic and composable execution

The semantic core is implemented by `FieldId`, `FieldDescriptor`,
`StateLayout`, `StateView`, `StateAccessor`, `EvaluationContext`, and
`SemiDiscreteModel`. Problems and Interactions add residual terms and separate
`dF/dy` and `dF/dydot` operators. `Representation` and its typed derivatives
expose observables and lifts derived from semantic fields.

`LinearAdapter` executes affine steady systems. `IDAAdapter` executes the
canonical DAE residual `F(t,y,ydot)=0` through deal.II's SUNDIALS IDA wrapper.
Both adapters privately own execution block layouts, vectors, callbacks, and
solver policy. Applications add Problems and Interactions directly to an
adapter through `ProblemHandle`; `FieldId` is semantic identity, not a global
matrix block number. The public composition API is exercised by the distributed
IDA tests, including mixed fields and multiplier Interactions.

For affine systems, `LinearAdapter` can materialize the assembled block
operator or expose its deal.II `LinearOperator` view. Its default policy uses
iterative GMRES for single-field systems and FGMRES for coupled systems, with
Problem-registered local preconditioners. Block-diagonal, block-triangular,
saddle-point Schur, and matrix-based augmented-Lagrangian actions are available
for explicit selection. The augmented-Lagrangian path assembles the primal
superblock and uses the configured multiplier metric, including a positive
algebraic fallback for constrained rows whose lumped physical diagonal is zero.
The direct policy uses deal.II's serial direct solver and, for a multi-rank
Trilinos execution, the Amesos2 MUMPS backend after redistributing the
assembled matrix to a contiguous Epetra map. These are backend choices of the
execution adapter, not requirements imposed on Problems or Interactions.

```{mermaid}
flowchart LR
  P["Problem"] --> F["FieldId / FieldDescriptor"]
  F --> R["Representation"]
  R --> I["Interaction"]
  P --> M["SemiDiscreteModel"]
  I --> M
  M --> LA["LinearAdapter"]
  M --> IDA["IDAAdapter"]
```

### Reduced and tensor-product execution

The established reduced path is used by `PoissonProblem`, elasticity and
reduced-Poisson applications, and the coupling classes built around them.
`ReducedCoupling` owns relation-specific coupling assembly; `TensorProductSpace`
constructs the reduced representative geometry and reference cross-section;
`ParticleCoupling` and `ImmersedRepartitioner` provide distributed search and
ownership services. `Inclusions` remains the legacy input adapter for the
corresponding applications.

For imported reduced fields, VTK or legacy input is read into a `FieldCatalog`.
`InputFieldSelector` resolves selected fields, `ReducedFieldValues` extracts
their finite-element values, and `SymbolicFieldEvaluator` evaluates expressions
such as thickness and reduced loads. `ImportedFiniteElementFields` is the
reusable finite-element import path used by semantic representations and
problem modules. The shared `ReducedFieldUtils` transfer copies coefficients
between serial and distributed DoFHandlers by matching active `CellId`s; it
does not interpolate, search for points, or project onto a different mesh.

```{mermaid}
flowchart LR
  VTK["VTK / legacy input"] --> C["FieldCatalog"]
  C --> S["InputFieldSelector"]
  S --> RFV["ReducedFieldValues"]
  RFV --> E["SymbolicFieldEvaluator"]
  E --> TPS["TensorProductSpace"]
  TPS --> RC["ReducedCoupling"]
  RC --> P["Reduced Problems / coupling operators"]
  C --> IFS["ImportedFiniteElementFields"]
  IFS --> REP["FiniteElement Representation"]
```

The reduced and semantic paths can share deal.II meshes, fields, operators,
and representations where the current APIs connect them, but they retain
different setup and storage lifecycles. In particular, `TensorProductSpace`
still owns its reduced property DoFHandler and coefficients, while
`ImportedFiniteElementFields` owns its reusable imported-field storage.

## Current ownership boundaries

- A Problem owns its physical equations, discretization, native operators, and
  accepted physical state.
- A Field is semantic state/residual identity; execution storage and block
  numbering belong to an adapter.
- A Representation is a typed observable or lifting and does not own the
  physical relation between Problems.
- An Interaction owns relation-specific coupling, search/transfer state,
  auxiliary fields, and additive residual terms.
- An execution adapter owns solver and time-integration policy.

These boundaries apply to the semantic path and are also the direction for
incremental integration of the established reduced path. There is no separate
user-visible `CoupledSystem` object in the current composition API.

## API entry points

The generated API reference provides class-level details for the current
implementation:

- {doc}`../../api/class_immers_x_1_1_elasticity_problem`
- {doc}`../../api/class_immers_x_1_1_poisson_problem`
- {doc}`../../api/class_immers_x_1_1_elastic_static_problem`
- {doc}`../../api/class_immers_x_1_1_inclusions`
- {doc}`../../api/class_immers_x_1_1_tensor_product_space`
- {doc}`../../api/class_immers_x_1_1_imported_finite_element_fields`
- {doc}`../../api/class_immers_x_1_1_representation`
- {doc}`../../api/class_immers_x_1_1_linear_adapter`
- {doc}`../../api/class_immers_x_1_1_lagrange_multiplier_interaction`
- {doc}`../../api/class_immers_x_1_1_constraint_equation`

For runnable examples, see {doc}`../../developer/design/semidiscrete-contributors`,
{doc}`../../developer/design/time-residual-sundials`, and
{doc}`../../tutorials/index`. The remaining roadmap includes broader execution
adapters such as ARKode/IMEX, moving geometry, and partitioned or multirate
execution; those are not presented here as current APIs.

## Reading the architecture pages together

| Page | Purpose |
| --- | --- |
| {doc}`current-implementation` | Classes and data paths currently available on `master`. |
| {doc}`../../concepts/architecture` | Normative ownership, residual, representation, and execution design. |
| {doc}`../design/architecture-status` | Implemented, validated, and planned capability status. |
| {doc}`../../concepts/mathematical-background` | Mathematical motivation and reduced Lagrange-multiplier context. |
| {doc}`../../tutorials/index` | Runnable learning workflows. |
