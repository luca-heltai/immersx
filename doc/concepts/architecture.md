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
Field -> typed FE expression
```

`fe_space(...)` is a non-owning view over an existing deal.II
`DoFHandler`, mapping, and constraints. A `Field` adds semantic naming and a
deal.II extractor; it owns no solution vector. `Observable<Field,Operation>` is
a lightweight FE expression whose operation forwards to the corresponding
`FEValuesViews` method. Its result type is consequently defined by deal.II,
while dependencies, scaling, and frozen state remain ImmersX metadata.

`frozen(field, values)` uses an unregistered FE `Field` only for its support
and finite-element metadata, and retains the supplied coefficients as a
dependency-free observable. A `weak_term` built from it contributes a fixed
residual and therefore does not register a source-field Jacobian.

## FE expressions and weak terms

For a Field (u_h\in V_h), an FE expression (E(u_h)) is one of the
first-order operations supplied by the deal.II view: (u_h),
\(\nabla u_h\), \(\nabla\!\cdot u_h\), \(\varepsilon(u_h)\), or
\(\operatorname{curl}u_h\), where supported by that view. The public helpers
`value`, `gradient`, `divergence`, `symmetric_gradient`, and `curl` select
these operations at compile time. Invalid extractor/operation combinations
are rejected by the compiler.

The trial side of a weak term is an Observable. The residual row is always an
explicit state-independent test expression made with `test(field)` and the
same deal.II FE operation helpers. For example:

```cpp
weak_term(value(u), test(v));                         // ∫ u · v
weak_term(gradient(p), gradient(test(q)));            // ∫ ∇p · ∇q
weak_term(divergence(u), test(q));                    // ∫ (∇·u) q
weak_term(value(p), divergence(test(v)));             // ∫ p (∇·v)
weak_term(symmetric_gradient(u),
          symmetric_gradient(test(v)));               // ∫ ε(u) : ε(v)
```

Observables may also be composed pointwise with native deal.II algebra. For
example, `gradient(u) * u` denotes `(grad u) u`, and its directional
derivative is `(grad du) u + (grad u) du`.

For a linear trial expression and test expression, the matrix entry is

```{math}
A_{ij}=\int_\Gamma
\left\langle E(\phi_j^u),F(\phi_i^v)\right\rangle\,d\mu.
```

The natural deal.II tensor product supplies scalar multiplication, vector
inner products, and tensor contractions. The user-facing mathematics is
independent of the geometric backend:

```text
same DoFHandler              -> one direct FE cell loop
different DoFHandlers/grid   -> as_dof_handler_iterator()
different geometries         -> prepared and cached particle coupling
```

The term is contributed through the ordinary semantic builder; its FE
assembly/search strategy remains an implementation detail and can be prepared
once for fixed geometry.

**State** is the collection of field values supplied for an evaluation.
`StateView` and `StateAccessor` provide current, frozen, historical, or
interpolated values without making a candidate state the accepted state.

**Observable** is a typed quantity derived from one or more Fields. A lifting
or particle search is an execution backend for an Observable, not a second
mathematical ownership layer.

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
