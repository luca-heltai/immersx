# ImmersX core architecture

This page is the architectural specification for ImmersX. It describes the
stable concepts that should support mixed-dimensional finite-element physics,
selective multiphysics coupling, and more than one execution strategy.

The specification deliberately separates three things that are easy to mix
up:

1. what is implemented by the current `master` branch;
2. interfaces validated by the distributed execution gate on this branch; and
3. normative design direction and longer-term roadmap.

The current production classes are the starting point for the design. The
architecture is not a requirement to rewrite `PoissonProblem`,
`ElasticityProblem`, `ReducedPoisson`, or the tensor-product coupling in one
step.

:::{admonition} Implementation status
:class: important

**Merged on `master`.** The repository contains problem-specific Poisson,
elasticity, and standalone Navier--Stokes discretizations, the semantic
`FieldId`/`FieldDescriptor`/`StateLayout` and `SemiDiscreteModel` core,
external-state evaluation, PackagedOperation residuals, separate
LinearOperator Jacobian blocks, DAE metadata, state history/interpolation,
immersed and reduced coupling classes, `Representation`/`TensorProductSpace` machinery,
particle/search support, multiplier specializations, and linear algebra
solvers.

**Validated distributed execution on this branch.** The semantic core is
connected to `IDA<LA::MPI::BlockVector>` through the public `IDAAdapter`, which
assigns one private execution block per Field. Real two-rank tests validate
direct block binding, differential and algebraic masks, standard deal.II
Jacobian operators, mixed-FE velocity Representation dependencies, and short
Elastodynamics, unsteady-Stokes, and five-field fiber IDA solves. The
validation covers the configured MPI/deal.II backend and is not a general
performance or production-robustness claim.

The real PDE adapters are composable contributors. An application adds
Problems directly to `IDAAdapter` or `LinearAdapter`, chooses prefixes and
`HistoryGroupId`s through the contributor scope, and can register multiple
instances of the same Problem class.
The full-order fiber test combines two Elastodynamics contributors and one
vector multiplier Interaction in the five-field layout
`matrix.displacement`, `matrix.velocity`, `fiber.displacement`,
`fiber.velocity`, and `fiber_coupling.lambda`.

### Capability baseline after PR #115

The first post-#115 maturation slice records the current feature baseline in
the [ElasticStatic production design](https://github.com/luca-heltai/immersx/blob/master/docs/superpowers/specs/2026-08-25-elastic-static-production-design.md).
That matrix is intentionally capability-oriented: it distinguishes the
legacy monolithic Elasticity path from the semantic adapter path and identifies
which parity gaps are concrete work rather than speculative roadmap items.

**Roadmap.** ARKode/IMEX for Navier--Stokes convection, broader execution
adapters, moving geometry, multirate/partitioned runs, term-level policies,
and a lightweight global composer may be integrated incrementally after
review.
:::

:::{admonition} Design pressure / non-goals
:class: note

ImmersX is not currently building a dynamic graph framework, an automatic
solver factory, a universal `Interaction` base class, heterogeneous
vector-backend type erasure, automatic `LinearOperator -> sparse matrix`
conversion, or a giant matrix-owning `CoupledSystem`. Those mechanisms can be
useful later, but they must not be smuggled into the core vocabulary before
the residual and field contracts are stable.
:::

## A. Scope and design status

The architectural boundary is:

```text
Problem                 owns equations and discretization for its state fields
Field                   names a semantic part of a global state and residual
State / StateAccessor   supplies current, frozen, historical, or interpolated values
Representation          exposes typed physical observables derived from fields
Interaction             owns relations and additive terms between participants
Residual contributor    adds residual and, when available, Jacobian actions
Execution adapter       applies a residual model under a solver/driver policy
```

The words *Problem*, *Field*, *Representation*, *Interaction*, *Residual*, and
*Execution adapter* have the meanings defined below. A class may implement
more than one role internally today, but new interfaces should not silently
merge their ownership responsibilities.

The term *background* may still describe a local geometric search mesh. It is
not an ownership concept: no problem becomes the owner of a coupled model just
because a particle search happens to use its triangulation.

## B. Core vocabulary

### Problem

A **Problem** owns the equations and discretization of the physics it solves.
It may own a finite-element space, constitutive data, boundary/initial data,
native operators, and one or more semantic state fields. A Problem can
contribute to several residual rows, and a residual row can receive
contributions from several Problems and Interactions.

A Problem does not need to be a PDE with exactly one unknown. In particular,
Stokes remains one `StokesProblem` owning velocity and pressure, even though
those fields have different roles in the differential-algebraic system. Do
not create `VelocityProblem` and `PressureProblem` objects merely to make the
field partition visible.

### Field

A **Field** identifies a semantically meaningful part of the global state and
the corresponding residual row or block. A field is not necessarily an
independent PDE. It is a stable name used to connect state storage, residual
contributions, representations, coupling selection, and differential/
algebraic metadata.

For a Stokes-like Problem, the semantic fields are typically `velocity` and
`pressure`:

$$
\begin{aligned}
F_u(u,p) &= M\dot u + N(u) + A u + B^T p - f,\\
F_p(u,p) &= B u - g.
\end{aligned}
$$

Both rows belong to the same Problem. The field partition is semantic; it does
not prescribe whether the implementation stores the state as a monolithic
vector, a block vector, or several backend-specific vectors.

### State, StateView, and StateAccessor

The **state** is the collection of field values supplied to an evaluation. A
`StateView`/`StateAccessor` is the conceptual read interface for that state.
The important property is that a Problem or Representation can evaluate an
externally supplied state rather than assuming that its own member named
`solution` is the only possible input.

Conceptual usage is:

```text
state = accessor.current_or_interpolated(field, time)
problem.add_residual(context.with_state(state), residual)
```

The concrete type can remain backend-specific. The architecture requires the
ownership and lifetime semantics, not one universal vector type.

### Representation

A **Representation** exposes a physical observable derived from one or more
Fields. It maps representative/reduced coefficients to values, gradients,
tractions, or other typed quantities on a represented physical support.

Keep these domains distinct:

- the **representative/reduced domain**, where coefficients or reduced physics
  live;
- the **represented physical support**, where an interaction evaluates the
  observable; and
- the ambient `spacedim`, in which that support is embedded.

A Representation can be reused by several Interactions. It is a view/evaluator
of state, not the owner of a coupling relation.

### Interaction

An **Interaction** owns the physical relation between participants and every
additive term that disappears when that relation is removed. Depending on the
relation, it may own geometric search, quadrature, transfer, auxiliary spaces,
coupling residuals, Jacobian actions, and geometry-dependent scaling.

The rule remains useful even when the relation is not a constraint:

> An Interaction owns the terms that exist because the participating physical
> systems are related.

### Residual contributor

A **Residual contributor** is a Problem or Interaction that adds terms to a
shared residual accumulator. Contributions are additive. A contributor may
also expose a Jacobian action and optional contributor-specific assembly
capabilities.

The contributor boundary prevents the execution layer from learning whether a
term came from elasticity, pressure, diffusion, a multiplier, a prescribed
source exchange, or a co-simulation port.

### Execution adapter

An **Execution adapter** consumes the same residual/Jacobian model under a
solver policy. The validated IDA adapter uses
`FieldVectorType = LA::MPI::Vector` and
`GlobalVectorType = LA::MPI::BlockVector`: its explicit Field-to-block map is
local to that adapter and does not constrain other storage choices. Future
examples include an ARKode IMEX adapter, an MRIStep-style multirate adapter,
and a partitioned or co-simulation driver. Problems and Interactions should
not know which adapter is calling them.

The solver used by an adapter owns no physics: it consumes residuals, Jacobian
actions, states, and optional preconditioning data supplied by the model.

## C. The semi-discrete model: $F(t,y,\dot y)=0$

The common denominator is a semi-discrete residual, not a collection of
matrices:

$$
F(t,y,\dot y) = 0.
$$

Here $y$ and $\dot y$ are field-partitioned states, while $t$ carries explicit
time dependence in coefficients, geometry, and prescribed data. A residual
evaluation receives the candidate state from the execution layer:

```text
residual.clear()
for contributor in contributors:
    contributor.add_residual(context, residual)
```

The names below are conceptual and are not a promise that they already exist
on `master`:

```cpp
// Conceptual API; exact vector and context types remain implementation-specific.
contributor.add_residual(EvaluationContext{t, y, ydot}, accumulator);
```

Problems contribute their own physical equations. Interactions add coupling,
exchange, contact, interface, or auxiliary-field equations. Several
contributors may add to the same Field row. A prescribed datum is data, not a
fake Problem.

Concrete Problems customize this contract through an ADL-discoverable
`contribute(builder, problem)` function. They do not own the caller's
`StateLayout` or `SemiDiscreteModel`; the execution adapter supplies the field
scope and model that receives additive terms. Thus two instances of one
Problem type can coexist without semantic-name collisions or coupling to
native block numbering. A concrete Interaction may add to Problem-owned rows
and register its own auxiliary algebraic row, as the vector
Lagrange-multiplier coupling does for `fiber_coupling.lambda`.

`LinearAdapter` is the affine steady execution path. `IDAAdapter` is the
transient DAE path. Both expose semantic `make_state()` and `field(state,id)`
access while keeping execution block mapping private.

This model supports Newton/KINSOL, IDA, ARKode, multirate stages, nonlinear
partitioned iterations, adjoint/sensitivity evaluations, and external state
tests without duplicating the physics layer for each driver.

## D. Multi-field Problems and native block operators

Fields are a semantic partition and do not impose storage layout. A Problem
may continue to assemble and maintain a native block matrix such as

$$
\begin{bmatrix}
A_{uu} & A_{up}\\
A_{pu} & A_{pp}
\end{bmatrix}.
$$

An adapter can gather semantic fields into the native block vector, apply the
native operator, and scatter the result back into residual fields. This is a
supported implementation path, not a violation of the Field abstraction.

For Stokes, the expected ownership is:

- `StokesProblem` owns both `velocity` and `pressure` fields and both native
  equation rows;
- a velocity Representation can expose only `velocity` to an external
  Interaction;
- pressure remains internal to the Stokes equations unless a separate
  Representation explicitly exposes it;
- an Interaction can add to an existing velocity row or introduce an auxiliary
  field without splitting the Problem.

The corresponding data flow is shown in the Stokes diagram in section N.

## E. Typed physical Representations and field selection

Representations should return strongly typed physical values whenever their
observable has a known shape: scalar, vector, tensor, or a future physical
quantity with a documented type. FE extractors or an equivalent mechanism must
allow a Representation to select only the relevant subfields from a mixed
finite-element system.

For example, a mixed Stokes FE system may expose a vector-valued velocity
Representation that excludes pressure. The selection is explicit in the
Representation contract and does not require a second FE system.

The “one Representation has exactly one source Field” rule is intentionally
not part of the architecture. A traction such as

$$
\sigma(u,p)n
$$

depends on more than one field. A Representation instead declares the Fields
on which its observable depends, ideally using semantic `FieldId` values rather
than opaque problem-specific tags.

Representations may preserve efficient scalar `IdentityRepresentation` and
tensor-product paths while adding vector/tensor value types. They do not
require a full embedded DoF system or explicit global $R$ and $R^T$ matrices.
At physical quadrature points, the useful contract is instead the ability to
provide, directly or through a reusable evaluator:

```text
physical point and measure:  x_q, w_q
selected DoF indices:        i_1, ..., i_k
typed basis/observable data: value, gradients, components, transforms
source Field dependencies:   {FieldId, ...}
```

## F. Interaction families and auxiliary Fields

The concrete interaction family is determined by the physical relation, not
by a single implementation pattern.

Possible families include:

- scalar or vector continuity;
- normal-flux or traction exchange;
- Robin, penalty, and Nitsche relations;
- Lagrange-multiplier constraints;
- nonlinear contact;
- source or conservative exchange;
- circuit, interface-state, and co-simulation ports.

`LagrangeMultiplierInteraction` is a concrete scalar continuity interaction,
not the definition of Interaction. Some Interactions introduce no multiplier
and no auxiliary Field.

An Interaction may introduce first-class auxiliary unknowns such as Lagrange
multipliers, contact variables, interface states, or circuit variables. Those
unknowns receive semantic Fields and residual rows. Do not create fake
Problems merely to host a multiplier or prescribed datum.

The ownership rule is unchanged for an Interaction that contributes to a row
already owned by a Problem: the Problem owns its physical row, while the
Interaction owns its additive relation term.

## G. Jacobian actions and optional assembly

The universal linearization capability is a Jacobian action or
`LinearOperator`, not an assembled sparse matrix. For an implicit or DAE
linearization, the solver supplies coefficients $a$ and $b$ and requests the
action

$$
Jv = \left(a\,\frac{\partial F}{\partial y}
       + b\,\frac{\partial F}{\partial \dot y}\right)v.
$$

Each Problem or Interaction contributes its part of $Jv$ to a shared output.
The action can use matrix-free quadrature, a native block operator, or an
assembled matrix wrapped as a `LinearOperator`.

Assembled sparse or block matrices remain a useful optional capability. A
contributor that can assemble a particular block may expose that capability
explicitly. There is no generic promise that any `LinearOperator` can be
converted back into a sparse matrix; assembly is contributor-specific.

Existing native block matrices are therefore preserved and wrapped, not
disassembled solely to satisfy the architecture.

## H. Differential/algebraic Fields and SUNDIALS mapping

Each `FieldDescriptor` stores a `differential_components` `IndexSet` with the
same global size as the field vector. A global component participates in
`ydot` exactly when it belongs to this set; components outside the set are
algebraic. The set may be empty, complete, or an arbitrary subset, so a single
semantic field can contain both differential and algebraic components.

Typical classifications are:

- Navier--Stokes/Stokes: the velocity mask is complete and the pressure mask is
  empty;
- multiplier coupling: the multiplier mask is empty;
- first-order elasticity: displacement and velocity masks are complete;
- mixed external state vectors: the mask contains only the components that
  participate in the differential equations.

An IDA-style adapter unions the field masks into its execution-level
differential mask while keeping the residual contributor API independent of
SUNDIALS. For the production block-vector path the source chain is
`FieldDescriptor::differential_components` -> private execution layout -> IDA
differential mask; callers do not duplicate the DAE classification in a second
metadata object. The adapter owns solver-specific vector conversion,
tolerances, callbacks, and masks; a Problem does not include IDA policy in its
physical equations.

For the validated Navier--Stokes path, the continuous operator is exposed as
separate velocity mass, pressure metric, and spatial blocks. The pressure
metric is used only by the native preconditioner. The semantic IDA residual
uses `rho*M_u*u_dot + C_uu*u + C_up*p - rho*f(t)` and `C_pu*u`, where the
`C_*` blocks are the assembled weak-form blocks; the pressure metric is not
part of `dF/dydot`.

## I. Term selection, IMEX, and multirate execution

Implicit/explicit and fast/slow are execution decisions, not intrinsic
properties of an entire Problem. A Problem may contain multiple additive
terms, for example:

- mass or inertia;
- convection;
- diffusion or elasticity;
- pressure gradient;
- incompressibility;
- interface traction;
- kinematic constraint;
- source exchange.

A future `TermId`/`TermSelection` (provisional names) can let an execution
adapter request a subset of terms or assign evaluation policies. This should
remain a small selection mechanism, not a new hierarchy or general workflow
framework. The driver decides which terms are implicit, explicit, fast, slow,
frozen, or interpolated for a particular step.

## J. State history, interpolation, and partitioned execution

State history is part of the architecture early, not a late convenience.
Partitioned and multirate drivers may request another subsystem or Field at an
intermediate time. A history-backed StateAccessor should provide current,
frozen, interpolated, or extrapolated values without changing a Problem,
Representation, or Interaction.

Independent time grids for different subsystems or fields should remain
possible. The physics layer is evaluated against the state requested by the
driver; it is not duplicated into monolithic and partitioned versions.

This is also the boundary for co-simulation: a co-simulation driver owns time
window negotiation, exchange schedule, rollback, and convergence policy, while
Problems and Interactions expose the same residual contributors used by a
monolithic run.

## K. Moving geometry, versions, and cache invalidation

The physical geometry of a Representation is not assumed to be immutable for
the entire simulation. Full moving-ALE or FSI implementation is outside this
revision, but the contract must leave room for it.

Conceptual hooks are:

```text
representation.update_geometry(context)
representation.geometry_version()
interaction.invalidate_search_or_coupling_cache(version)
interaction.rebuild_cache_if_needed()
```

An Interaction that performs point search or uses a `ParticleHandler` must be
able to invalidate and rebuild search/coupling caches when represented
geometry changes. Cached quadrature points are an implementation cache, not an
immutable property of the model.

## L. TensorProductSpace and reduced-to-physical lifting

`TensorProductRepresentation` is the reusable lifting mechanism. It maps
coefficients from a representative/reduced domain to a physical field on a
higher-dimensional support. `TensorProductSpace` remains a compatibility and
geometry-construction helper for existing reduced Problems; it is not the
Interaction-facing observable. Keep the dimension labels explicit:

```text
representative/reduced dimension -> represented physical dimension
                                      -> ambient spacedim
```

For the vessel-wall pattern,
`NetworkFlow<1,3> -> TensorProductRepresentation<1,2,3>` means that
one-dimensional network coefficients are evaluated as a two-dimensional wall
field embedded in three-dimensional space. It does not create a two-
dimensional wall Problem.

The lifting may be consumed by elasticity, traction, transport, or more than
one other Interaction. No full embedded DoF system and no explicit global
restriction/lifting matrices are required.

New code should construct the lifting from a representative view:

```{code-block} cpp
auto surface = make_tensor_product_representation(centerline, tensor_space);
```

The returned Representation is non-owning and reusable. The reduced Problem
continues to own its mesh, DoFs, and state, while the lifting owns only the
mapping and transfer view.

The geometric metrics used by reduced coupling also remain distinct:

- $M$ is the multiplier-space mass/Riesz metric or another metric of the
  reduced discretization;
- $W$ is the augmented-Lagrangian or preconditioning metric used by a solver
  strategy;
- $h_\gamma$ is a characteristic represented/reduced geometry scale;
- $H_\Omega$ is a bulk mesh or search-mesh scale.

A relation such as

$$
W^{-1} \approx h_\gamma^{-\alpha} M^{-1}
$$

may be appropriate for a particular stability/preconditioning regime, but it
does not make $M$ and $W$, or $h_\gamma$ and $H_\Omega$, interchangeable.
Geometry-dependent scaling is prepared by the relevant Representation or
Interaction; the Solver consumes the resulting operator or action.

## M. Linear constraint specialization

For linear constraints, a `ConstraintEquation` specialization remains useful:

$$
\sum_i C_i u_i = d.
$$

It is a compact linear-algebra representation of one relation and may be
handled by a specialized saddle-point or Schur/augmented-Lagrangian solver.
It is not the global residual model, not the definition of Interaction, and not
a requirement that all future relations introduce a multiplier.

The current scalar multiplier path can therefore remain production code while
the broader residual architecture is introduced around it incrementally.

## N. Architecture diagrams and data flow

The following diagrams are normative data-flow views. They use conceptual
names where the corresponding common API is still a roadmap item.

### Core ownership and data flow

```{mermaid}
flowchart LR
  subgraph physics["Physics layer"]
    P["Problem(s)"] --> F["semantic Fields"]
    F --> R["typed Representations"]
  end
  R --> I["Interactions"]
  P --> RC["residual contributors"]
  I --> RC
  RC --> RJ["Residual and Jacobian actions"]
  RJ --> E["Execution adapters"]
```

### Multi-field Stokes with selective coupling

```{mermaid}
flowchart LR
  SP["one StokesProblem"] --> U["velocity Field u"]
  SP --> P["pressure Field p"]
  SP --> EQ["Stokes equations F_u, F_p"]
  U --> VR["vector velocity Representation"]
  VR --> XI["external Interaction"]
  P -. "not selected by this Interaction" .-> INT["pressure stays internal"]
  XI --> RU["additive contribution to velocity row"]
  EQ --> RU
```

### Semi-discrete residual composition and Jacobian action

```{mermaid}
flowchart LR
  SV["external StateView y, ydot, t"] --> PC["Problem contributors"]
  SV --> IC["Interaction contributors"]
  PC --> F["additively composed F(t,y,ydot)"]
  IC --> F
  F --> J["compositional J*v"]
  J --> LS["solver linearization"]
```

### Monolithic and partitioned execution use the same physics

```{mermaid}
flowchart LR
  C["same residual contributors"] --> M["shared residual model"]
  M --> MONO["monolithic: IDA / ARKode / Newton-KINSOL"]
  M --> PART["partitioned or co-simulation driver"]
  HIST["history-backed StateAccessor"] --> PART
  PART --> HIST
```

### Tensor-product vessel wall and representation reuse

```{mermaid}
flowchart LR
  NF["NetworkFlow<1,3>\ncoefficients on representative centerline"] --> TP["TensorProductRepresentation<1,2,3>"]
  TP --> WALL["represented physical wall <2,3>"]
  WALL --> TISSUE["Interaction with 3D tissue"]
  TP --> OTHER["another reusable wall Interaction"]
```

### Moving geometry and cache invalidation

```{mermaid}
flowchart LR
  S["state or geometry update"] --> G["Representation.update_geometry"]
  G --> V["geometry_version increments"]
  V --> X["Interaction invalidates search/coupling cache"]
  X --> B["search and quadrature cache rebuild"]
  B --> E["next residual/Jacobian evaluation"]
```

### Optional native block and assembled path

```{mermaid}
flowchart LR
  F["semantic Fields"] <--> GA["gather / scatter adapter"]
  GA <--> NV["native Problem block vector"]
  NV --> NB["native block matrix or operator"]
  NB --> LO["LinearOperator action"]
  LO --> GC["global residual/Jacobian composer"]
  GC --> OUT["semantic residual or J*v"]
```

The native path is optional and local to a contributor. It demonstrates that
semantic Fields do not forbid block storage or assembled operators.

## O. What is implemented, prototyped, and planned

The status boundary is intentionally repeated here because it prevents a
conceptual API from being mistaken for a merged one.

| Status | Meaning in this specification |
| --- | --- |
| **Merged on `master`** | Current Poisson, elasticity, standalone Navier--Stokes, semantic Field/state/residual core, reduced-coupling, tensor-product, particle/search, multiplier, and solver classes. Existing tutorials describe these paths. |
| **Validated on this branch** | Direct one-block-per-Field `IDA<LA::MPI::BlockVector>` binding, real distributed Elastodynamics and unsteady-Stokes residual/Jacobian actions, DAE masks, mixed-FE velocity Representation dependencies, short two-rank IDA solves, and the five-field fiber semantic residual/Jacobian against backward Euler + Schur. |
| **Roadmap** | Navier--Stokes ARKode/IMEX convection, five-field IDA integration, broader execution adapters, independent time-grid policies, moving-geometry integration, co-simulation, and a lightweight global composer/registry. |

Existing `Poisson`, `Elasticity`, `ReducedPoisson`, and production coupling
classes can be adapted incrementally. The intended migration is an adapter
around existing assembly/native block operations followed by selective
extraction of residual contributors; it is not a big-bang rewrite.

## P. Explicit non-goals and deferred decisions

The following are deliberately deferred or excluded from the core contract:

- a dynamic runtime graph with automatic discovery and scheduling;
- automatic solver, preconditioner, or time-integrator selection;
- a universal interaction class that hides physically different relations;
- an invariant that every Representation has exactly one source Field;
- mandatory full embedded DoF systems or explicit global $R/R^T$ matrices;
- a generic conversion from arbitrary `LinearOperator` to sparse matrix;
- a requirement that all Interactions use Lagrange multipliers;
- a requirement that an entire Problem be assigned one IMEX or fast/slow role;
- a single heterogeneous, type-erased vector backend for every field;
- full moving-ALE/FSI implementation in the current documentation revision.

These decisions leave room for efficient deal.II-native implementations while
keeping the semantic residual architecture available to future monolithic,
partitioned, multirate, and co-simulation drivers.
