# Architecture concepts

ImmersX separates physical problems from the relations that connect them and
from the solver that executes the resulting equations.

## Core vocabulary

**Problem** owns physical and discretization-specific information: mesh,
finite element, material data, native operators, forcing, boundary data, and
accepted physical state. A Problem does not know global coupled-system block
numbers and is not required to inherit an ImmersX base class. An external
library Problem, such as a MetricFlowX problem, can be adapted from the
outside through its public discretization accessors.

**Field** is a semantic identity for a state and residual row. It is not a
global matrix block, native Problem block, or execution-adapter block.

The finite-element vocabulary is deliberately layered:

```text
DoFHandler (+ Mapping/constraints view) -> FE space
FE space/subspace -> named Field
Field(s) -> Observable<T>
```

`fe_space(...)` is a non-owning view over an existing deal.II
`DoFHandler`, mapping, and constraints. A `Field` adds semantic naming and
optional extractor information; it owns no solution vector. `Observable<T>` is
a typed, composable physical quantity derived from one or more Fields. Its
public contract describes dependencies and differentiability, while point
search, quadrature orchestration, and evaluation caches remain implementation
details.

**State** is the collection of field values supplied for an evaluation.
`StateView` and `StateAccessor` provide current, frozen, historical, or
interpolated values without making a candidate state the accepted state.

**Representation** is a typed observable or lifting derived from one or more
Fields and geometric/discretization data. It does not own the physical
coupling relation.

**Interaction** owns terms that exist because two or more systems are related.
It may own transfer, search, geometry, auxiliary fields, and coupling residuals.

**Residual contributor** is a Problem or Interaction that adds terms to shared
residual rows. Contributions are additive.

**Execution adapter** owns solver and time-integration policy. It may own the
semantic composition state, execution block layout, vectors, masks, and
callbacks internally. Applications should add Problems and Interactions to an
adapter rather than introduce a separate user-visible coupled-system holder.

## The residual contract

The common semidiscrete form is

```{math}
F(t,y,\dot y)=0.
```

Problems and Interactions contribute to this residual using the state supplied
by the execution adapter. Linearization information keeps the two derivatives
separate:

```{math}
\frac{\partial F}{\partial y},\qquad
\frac{\partial F}{\partial \dot y}.
```

An adapter may combine them for its solver. For IDA, for example,

```{math}
J=\frac{\partial F}{\partial y}
  +\alpha\frac{\partial F}{\partial \dot y}.
```

The contributor does not know IDA's `alpha`. Evaluating a residual at a
candidate state also does not accept that state or update physical history.

## Semantic fields and execution storage

A Problem can own several Fields, such as displacement and velocity, while
retaining a native block matrix. An adapter maps semantic Fields to its own
execution storage. That mapping is private to the adapter, so changing from a
block vector to another deal.II vector layout does not change the physical API.

The [developer design status](../developer/design/architecture-status) records
which parts of the architecture are currently implemented, validated, or
planned. The [generated API](../api/library_root) gives the exact signatures of
merged interfaces.
