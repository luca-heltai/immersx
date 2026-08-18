# ImmersX Core Architecture

This page is the foundational architectural specification for ImmersX. It
defines the concepts that should remain stable as the repository grows from
two-problem immersed couplings to general mixed-dimensional systems. The
examples use Poisson, network-flow, and elasticity problems, but the ownership
rules apply to any discrete problem that can expose a physical representation
and participate in an interaction.

The central separation is:

```text
Problem          owns equations for algebraic unknowns
Representation   says what physical field those unknowns represent
Interaction      couples two or more representations
Solver           solves the algebraic system produced by the first three
```

This is a design specification, not a claim that every current class already
has this exact interface. Existing classes such as `PoissonProblem`,
`ElasticityProblem`, `ParticleCoupling`, `ReducedCoupling`, and
`TensorProductSpace` are the implementation points from which this design is
being evolved. The repository-wide class and data-flow inventory remains in
{doc}`architecture-diagram`, while the mathematical background is in
{doc}`background`.

## Architecture at a glance

The four concepts have different sources of truth and different ownership.
Problems own their diagonal physics. Representations provide reusable views of
problem states on physical domains. Interactions own every term that involves
more than one problem. Solvers receive the resulting algebraic operators and
do not reconstruct any of that meaning.

```{mermaid}
flowchart LR
  subgraph P["Problems / discrete problems"]
    P0["Problem 0\nstate, equations, FE space"]
    P1["Problem 1\nstate, equations, FE space"]
  end

  subgraph R["Representations / physical views"]
    R0["Representation 0\nreduced or direct field view"]
    R1["Representation 1\nembedded physical field view"]
  end

  subgraph I["Interactions / shared terms"]
    I0["Interaction\nsearch, multiplier, coupling terms"]
  end

  subgraph G["Global algebraic system"]
    G0["Problem diagonal blocks\n+ interaction contributions"]
  end

  S["Solver\noperator and preconditioner only"]

  P0 -->|"algebraic state"| R0
  P1 -->|"algebraic state"| R1
  R0 -->|"consumed by"| I0
  R1 -->|"consumed by"| I0
  P0 -->|"diagonal physics"| G0
  P1 -->|"diagonal physics"| G0
  I0 -->|"off-diagonal and constraint terms"| G0
  G0 -->|"algebraic operator"| S
```

The word *background* is deliberately not part of the long-term ownership
model. A search or assembly algorithm may use one mesh as its geometric
background, but that implementation choice does not make the corresponding
problem the owner or coordinator of the coupled model.

## Problems and discrete problems

A **Problem** is the owner of the equations for the algebraic state that it
actually solves. In finite-element terms, a problem owns at least:

- its unknown or state vector and the associated algebraic DoF space;
- its operator, residual, or time-discrete equations;
- its right-hand side and problem-specific boundary or initial data;
- its diagonal Jacobian or block operator;
- the finite-element/discrete space intrinsic to that PDE;
- the mesh and discretization data needed to assemble or apply its own terms.

A problem does **not** own knowledge of its couplings. It may expose read-only
access to its state, discretization, and algebraic operators so that another
object can build an interaction, but it must not decide which other problems
exist or assemble terms whose meaning depends on another problem.

The notation `Problem<dim, spacedim>` describes a PDE discretized on a domain
of dimension `dim` embedded in an ambient space of dimension `spacedim`. Typical
problem nodes include:

| Problem | Interpretation |
| --- | --- |
| `Poisson<2,2>` | scalar Poisson equation on a two-dimensional domain |
| `Laplace-Beltrami<1,2>` or `Poisson<1,2>` | scalar equation on a curve in the plane |
| `NetworkFlow<1,3>` | one-dimensional flow physics on a vessel network in three-dimensional space |
| `Elasticity<3,3>` | three-dimensional tissue or solid elasticity |

The dimensional label describes where the problem's coefficients live. It does
not by itself say whether those coefficients are later represented on another
physical domain. That second statement belongs to a Representation.

## Representations

A **Representation** maps a problem's algebraic state to a physical field on a
domain where an interaction can occur. It separates three domains that are
often conflated:

1. the **representative/reduced domain**, where the problem coefficients live;
2. the **represented physical domain**, where another problem should see the
   field;
3. the **ambient space**, in which that physical domain is embedded.

For example, a one-dimensional vessel-network problem can have coefficients on
the network centerline while exposing a two-dimensional field on a vessel wall
embedded in three-dimensional space. No two-dimensional vessel-wall PDE is
introduced merely because the one-dimensional coefficients are represented
there.

### The abstract mapping

Let $V_r$ be the algebraic/discrete space of the representative problem and
let $V_\Gamma$ be the conceptual discrete space of fields on the represented
physical domain $\Gamma$. The representation supplies the pair

$$
P = R^T : V_r \longrightarrow V_\Gamma,
\qquad
R : V_\Gamma^* \longrightarrow V_r^*.
$$

Here $P$ lifts representative coefficients to a field on $\Gamma$, while
$R$ restricts or projects physical functionals back to the representative
dual space. The distinction between primal and dual spaces matters: an
interaction commonly evaluates a represented trial or test field at physical
quadrature points, and the resulting functional must be accumulated into the
problem's algebraic DoFs.

```{mermaid}
flowchart LR
  Vr["Representative space V_r\nproblem coefficients"]
  Vg["Represented physical space V_Γ\nfield on Γ"]
  Amb["Ambient space R^spacedim\ngeometry and embedding"]

  Vr -->|"P = Rᵀ\nlift / represent"| Vg
  Vg -->|"R\nrestrict physical functionals"| Vr
  Amb -.->|"contains Γ"| Vg
```

If a full embedded problem with operator $A_\Gamma$ and right-hand side
$g_\Gamma$ is first written down, a reduced algebraic problem may have the
form

$$
R A_\Gamma R^T w = R g_\Gamma.
$$

This equation explains the role of a representation, but it is not an
architectural requirement that every reduced PDE be obtained by projecting a
full PDE. A one-dimensional vascular equation may be derived independently as
reduced physics and still expose a representation of its pressure, flow, or
traction on a vessel wall. The representation describes how its solved state is
seen by other problems; it does not prescribe how the reduced equations were
derived.

### Direct and identity representations

The special case in which the represented field is already the problem's own
finite-element field is a **direct** or **identity representation**:

$$
R = I, \qquad P = I.
$$

This is useful even when the two problems have different meshes or dimensions.
For example, a `Poisson<1,2>` problem can expose its ordinary one-dimensional
finite-element field directly on the curve that an interaction matches to a
`Poisson<2,2>` problem. The interaction still owns the nonmatching search,
quadrature, and coupling matrices.

### Tensor-product representations

Conceptually, `TensorProductSpace` is a Representation: it turns coefficients
on a representative/reduced domain into basis functions for a higher-
dimensional physical field. Its template dimensions are interpreted as

```cpp
TensorProductSpace<reduced_dim, dim, spacedim, n_components>
```

- `reduced_dim` is the dimension where the problem coefficients live;
- `dim` is the dimension of the represented physical domain;
- `spacedim` is the ambient geometric dimension;
- `n_components` is the number of scalar components in the represented field.

The component count may have a default in a concrete declaration, so shorthand
such as `TensorProductSpace<0, 1, 2>` is used when appropriate. The dimensions
are not a request to solve a new PDE on the represented domain.

Two representative cases are:

- `TensorProductSpace<0,1,2>`: coefficients associated with representative
  points are lifted to a one-dimensional field embedded in the plane;
- `TensorProductSpace<1,2,3>`: coefficients on a vessel centerline are lifted
  to a two-dimensional vessel-wall field embedded in three-dimensional space.

The second case is the important vascular pattern: `NetworkFlow<1,3>` remains
one-dimensional physics, while its wall representation is a reusable port for
elasticity, surface transport, or any other problem that can consume a field on
the same physical wall.

### Representation contract without global transfer matrices

The notation $R$ and $R^T$ is a useful design-level contract. An efficient
implementation does not need to construct explicit global matrices for either
operator. A representation can fold both actions into quadrature and assembly.

At a physical quadrature point $x_q \in \Gamma$, the low-level contract is
that the representation can provide, directly or through a reusable evaluator:

```text
physical point and measure:      x_q, w_q
algebraic DoF indices:           i_1, ..., i_k
represented basis values:        φ^Γ_i(x_q)
optional gradients/components:  ∇φ^Γ_i(x_q), component transforms, ...
```

An interaction uses those values to assemble, for example,

$$
C_{a i} \mathrel{+}= w_q\,\mu_a(x_q)\,\phi_i^\Gamma(x_q),
$$

where $\mu_a$ is a multiplier/test basis function. The represented basis
value already contains the action that a global $R$ or $R^T$ would have
provided. Particle locations, repartitioning, and search data can be used to
make this evaluation distributed and matrix-free.

This contract keeps the interaction independent of whether a representation is
an identity FE view, a tensor-product evaluator, or a future representation
based on another reduced model. It also makes clear why a tensor-product
representation can be reused: it is a view/evaluator of one problem state, not
the owner of any particular coupling.

## Interactions

An **Interaction** consumes Representations, not concrete PDE classes. It owns
the variational and algebraic terms that involve more than one problem. Its
responsibilities include, as applicable:

- geometric matching, search, and point evaluation;
- particles, repartitioning, and distributed ownership of coupling data;
- coupling, mortar, or transfer matrices and their matrix-free actions;
- the multiplier or auxiliary space required by the constraint;
- interaction residuals, Jacobian blocks, and right-hand sides;
- scaling data associated with the represented/reduced geometry.

An interaction may use one mesh as a *geometric search background* because a
search algorithm needs a triangulation in which to locate particles. That is a
local implementation role. Neither participating problem is privileged as the
background owner in the long-term architecture.

### Continuity constraints

Suppose representations $P_i u_i$ and $P_j u_j$ must agree on a coupling
space $Q$. The interaction expresses continuity as

$$
C_i u_i - C_j u_j = 0,
$$

where $C_i$ and $C_j$ evaluate the two represented fields against the
chosen multiplier/test basis. With problem operators $A_i$ and $A_j$, a
standard Lagrange-multiplier system is

$$
\begin{bmatrix}
A_i & 0 & C_i^T \\
0 & A_j & -C_j^T \\
C_i & -C_j & 0
\end{bmatrix}
\begin{bmatrix}
u_i \\
u_j \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
f_i \\
f_j \\
0
\end{bmatrix}.
$$

For a direct `Poisson<2,2>` to `Poisson<1,2>` coupling, if the multiplier
space and the one-dimensional finite-element space coincide, the
one-dimensional side is evaluated by its own identity representation and
$C_j$ is the one-dimensional mass matrix. The two-dimensional side still
requires nonmatching trace/search evaluation, so $C_i$ is assembled by the
interaction.

For a tensor-product representation, $C_i$ already includes the action of
$R$ through represented basis evaluation. The interaction therefore does not
need to create or solve a full embedded DoF system on $V_\Gamma$. It assembles
the reduced coupling directly between the problem DoFs and the multiplier
DoFs.

### Interaction ownership rule

An interaction owns every term that disappears when one of its participating
problems is removed. This includes both off-diagonal blocks and the multiplier
constraint block. A problem may provide the read-only data needed to evaluate
its side of the term, but it does not own the term itself.

## Solvers and coupled systems

Solvers own no physics. They consume algebraic operators, residuals, states,
and preconditioning data produced by Problems and Interactions. They should
not know whether a block came from elasticity, Poisson, network flow, an
identity representation, or a tensor-product representation.

The current `AugmentedLagrangianSolver` should remain mesh- and
dimension-agnostic. In particular, it should receive `invW` directly rather
than deriving it from a mesh, a dimension, or an assumed background grid. The
object that understands the represented geometry and multiplier discretization
is responsible for preparing that operator or operator action.

Two metrics must remain conceptually distinct:

- $M$ describes the multiplier discretization or its Riesz/mass metric on
  $Q$;
- $W$ describes the norm or preconditioning metric used by the augmented
  Lagrangian treatment.

For a characteristic represented/reduced scale $h_\gamma$, an interaction or
representation may use a relation such as

$$
W^{-1} \approx h_\gamma^{-\alpha} M^{-1}.
$$

The scale $h_\gamma$ is the characteristic resolution of the immersed,
represented, or reduced geometry. It is not automatically the mesh size
$H_\Omega$ of a bulk/search-background mesh. A stability regime may commonly
choose $H_\Omega \simeq h_\gamma$, but that relation does not make the two
quantities interchangeable, nor should it move geometry-dependent scaling into
the solver.

The intended boundary is therefore:

```text
Representation / Interaction
    knows geometry, quadrature, multiplier space, h_γ, and scaling
    prepares M, W, or an invW action

Solver
    receives invW and algebraic blocks
    applies/inverts/preconditions them according to solver policy
```

## N-problem graph structure

The general coupled model is a graph, or more precisely a hypergraph when an
interaction connects more than two problems:

- **nodes** are Problems and their algebraic states;
- **ports/views** are Representations exposed by those problems;
- **edges/hyperedges** are Interactions consuming one or more representations.

One problem may expose several representations. One representation may be
consumed by several interactions. Adding a third problem should therefore add
a new node, representation, and interaction edge; it should not require
editing either of the existing problem implementations.

```{mermaid}
flowchart LR
  P0["P0\nProblem"]
  P1["P1\nProblem"]
  P2["P2\nProblem"]

  R0a["P0 representation A"]
  R0b["P0 representation B"]
  R1a["P1 representation A"]
  R1b["P1 representation B"]
  R2a["P2 representation A"]
  R2b["P2 representation B"]

  I01["I01\ninteraction"]
  I02["I02\ninteraction"]
  I12["I12\ninteraction"]

  P0 --> R0a
  P0 --> R0b
  P1 --> R1a
  P1 --> R1b
  P2 --> R2a
  P2 --> R2b

  R0a --> I01
  R1a --> I01
  R0b --> I02
  R2a --> I02
  R1b --> I12
  R2b --> I12
```

The graph is a composition of independently derived physics. It is not a
class hierarchy in which a special `CoupledElasticityProblem` must become the
owner of every other PDE. A future `CoupledSystem` may aggregate the graph,
but it should remain a container for states, diagonal blocks, and interaction
contributions rather than a new source of physical equations.

### Two Poisson problems: the identity case

The first validation case is a two-dimensional Poisson problem coupled to a
one-dimensional Poisson problem in the plane. Both sides expose direct/identity
representations, and the interaction is responsible for matching them.

```{mermaid}
flowchart LR
  A["Poisson<2,2>\nunknown u_Ω"] --> RA["Identity representation\nfield on Γ"]
  B["Poisson<1,2>\nunknown w_γ"] --> RB["Identity representation\nfield on γ"]
  RA --> X["Lagrange-multiplier interaction\nC_Ω, C_γ, Q, search"]
  RB --> X
  X --> K["Global saddle-point system"]
```

The corresponding algebraic layout is

$$
\left[
\begin{array}{ccc}
A_\Omega & 0 & C_\Omega^T \\
0 & A_\gamma & -C_\gamma^T \\
C_\Omega & -C_\gamma & 0
\end{array}
\right]
\left[
\begin{array}{c}
u_\Omega \\
w_\gamma \\
\lambda
\end{array}
\right]
=
\left[
\begin{array}{c}
g_\Omega \\
g_\gamma \\
0
\end{array}
\right].
$$

The problem blocks $A_\Omega$ and $A_\gamma$ are assembled by the two
Poisson problems. The interaction assembles $C_\Omega$, $C_\gamma$, and
the multiplier metric. If the one-dimensional multiplier basis is the same as
the one-dimensional FE basis, $C_\gamma$ is its mass matrix. No problem
needs to know that the other one exists.

### Vessel wall, elasticity, and a third problem

The mixed-dimensional vascular example makes the distinction between physics
and representation explicit. A one-dimensional network-flow problem can
represent a wall field in two dimensions, embedded in three dimensions. A
three-dimensional elasticity problem can expose a boundary representation on
the same physical wall. The same network-wall representation can then be
reused by a third surface problem.

```{mermaid}
flowchart LR
  NF["NetworkFlow<1,3>\ncoefficients on centerline"]
  TP["TensorProductRepresentation<1,2,3>\nvessel-wall field on Γ"]
  EL["Elasticity<3,3>\n3D tissue displacement"]
  EB["BoundaryRepresentation<3,2,3>\ntrace on Γ"]
  SP["SurfaceProblem<2,3>\nwall/surface physics"]
  SI["Direct representation<2,3>\nfield on Γ"]
  IE["Wall–elasticity interaction\nsearch, particles, Q, coupling"]
  IS["Wall–surface interaction\nindependent interaction"]

  NF --> TP
  EL --> EB
  SP --> SI
  TP --> IE
  EB --> IE
  TP --> IS
  SI --> IS
```

The third problem is added by attaching `SurfaceProblem<2,3>` to the existing
wall representation through a new interaction. `NetworkFlow` and `Elasticity`
do not acquire a new coupling-specific method, and the network representation
does not become owned by either interaction. This is the intended extension
path for pressure-to-wall, wall-to-tissue, transport, and other multi-physics
relations.

## Ownership rules and design commandments

The following rules are normative for new architecture and should guide the
refactoring of existing code:

1. A Problem owns the equations for the coefficients/state it solves.
2. A Problem owns its intrinsic FE/discrete space, operator/residual, right
   hand side, and diagonal Jacobian/block.
3. A Problem does not own knowledge of its couplings or of the other problems
   in a coupled model.
4. A Representation says what physical field those coefficients represent on
   another domain.
5. A Representation distinguishes representative/reduced dimension,
   represented physical dimension, and ambient `spacedim`.
6. A Representation may be reused by multiple Interactions.
7. An Interaction consumes Representations, not concrete PDE classes.
8. An Interaction owns all terms that disappear when the other participating
   problems are removed: search, particles, multiplier spaces, coupling
   matrices, residuals, Jacobian blocks, and geometry-dependent scaling.
9. “Background” may describe a local geometric search mesh, never a permanent
   privilege in the coupled-system ownership model.
10. A Solver owns no physics. It consumes algebraic operators and
    preconditioning data.
11. The coupled/global system merely aggregates problem diagonal blocks/states
    and interaction contributions.
12. A global $R$ or $R^T$ matrix is a conceptual contract, not a required
    data structure. Implementations should evaluate represented basis functions
    and fold the action into quadrature and assembly whenever possible.

## Staged development path

The architecture should be validated with small, composable cases before a
general graph container is introduced. The intended sequence is:

1. Make `PoissonSolver` composable by exposing read-only discretization
   accessors and a configurable parameter subsection, without giving it
   coupling ownership.
2. Build an identity representation for `Poisson<1,2>`.
3. Extract a generic, representation-driven Lagrange-multiplier interaction
   assembler. Reuse `ParticleCoupling` and its search/repartitioning machinery
   where appropriate, but keep ownership of the coupling terms with the
   interaction.
4. Solve `Poisson<2,2>` coupled to `Poisson<1,2>` as the $R=I$ case.
5. Replace one side's identity representation with
   `TensorProductSpace<0,1,2>` without changing the interaction assembly.
6. Validate `TensorProductSpace<1,2,3>` as a vessel-wall representation
   coupled to `Elasticity<3,3>` through a boundary representation, and
   optionally attach a third `SurfaceProblem<2,3>` to the same wall view.
7. Only after these examples work, introduce a generic `CoupledSystem` or
   graph container that aggregates problem blocks and interaction
   contributions.

This sequence tests the abstraction at the points where it matters: direct
nonmatching coupling, reduced representation, reuse of a representation, and
more than two problems. It also prevents a first two-problem implementation
from accidentally making one PDE class the permanent owner of the architecture.
