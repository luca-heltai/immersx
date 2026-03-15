# Mathematical Background

This repository is about partial differential equations posed on a bulk domain and coupled with thin embedded structures. Typical examples are:

- elastic materials reinforced by slender fibers;
- vascularized tissues, where a three-dimensional elastic matrix interacts with one-dimensional vessels;
- scalar diffusion or Poisson problems with lower-dimensional inclusions.

The common theme is that the embedded structure is geometrically thin, so resolving it with a fully conforming mesh is often too expensive or unnecessary. The code therefore uses mixed-dimensional models together with weak coupling conditions enforced by Lagrange multipliers. The main references behind this viewpoint are {cite:p}`AlzettaHeltai-2020-a`, {cite:p}`HeltaiCaiazzo-2019-a`, {cite:p}`HeltaiZunino-2023-a`, and {cite:p}`BelponerCaiazzoHeltai-2025-a`.

## Geometric Setting

Let

$$
\Omega \subset \mathbb{R}^d
$$

be a bulk domain, typically with $d=2$ or $d=3$. Inside $\Omega$ there is a thin structure $\Gamma$:

- a curve inside a three-dimensional body;
- a set of centerlines representing vessels or fibers;
- a lower-dimensional manifold obtained by dimensional reduction of a thin inclusion.

In the fully resolved description, one would work on a thin physical inclusion $\Omega_f \subset \Omega$. In the reduced description, the inclusion is replaced by a lower-dimensional characteristic set $\gamma$, for example the centerline of a vessel network.

The mathematical problem is to transfer information between the bulk field on $\Omega$ and the reduced field on $\gamma$ or on the interface of the thin inclusion.

## Bulk PDE Models

### Scalar model

The scalar case is represented by Poisson or diffusion-type problems. A prototype equation is

$$
-\nabla \cdot (\kappa \nabla u) = f \qquad \text{in } \Omega,
$$

with suitable boundary conditions on $\partial\Omega$.

Its weak form is:

Find $u \in V$ such that

$$
a(u,v) = \ell(v)
\qquad \forall v \in V,
$$

where typically

$$
a(u,v) = \int_\Omega \kappa \nabla u \cdot \nabla v \, dx,
\qquad
\ell(v) = \int_\Omega f v \, dx.
$$

This is the mathematical background behind `PoissonProblem` and the reduced scalar coupling tools in `include/reduced_poisson.h`.

### Elasticity model

The vector-valued case is linear elasticity. Let

$$
u : \Omega \to \mathbb{R}^d
$$

be the displacement field and let

$$
\varepsilon(u) = \frac{1}{2}\bigl(\nabla u + \nabla u^T\bigr)
$$

be the linearized strain tensor. A prototype strong form is

$$
-\nabla \cdot \sigma(u) = b \qquad \text{in } \Omega,
$$

with constitutive relation

$$
\sigma(u) = \mathbb{C}\,\varepsilon(u).
$$

The weak form reads:

Find $u \in V^d$ such that

$$
\int_\Omega \sigma(u) : \varepsilon(v)\, dx = \int_\Omega b \cdot v \, dx
\qquad \forall v \in V^d.
$$

This is the bulk model behind `ElasticityProblem` and the coupled elasticity applications.

## Why Lagrange Multipliers Appear

Suppose now that the bulk field must satisfy a condition on an embedded object. A simple model is:

$$
u|_\Gamma = g
$$

or, more generally, a coupling relation between a bulk unknown $u$ and an inclusion unknown $w$:

$$
T_\Omega u = T_\Gamma w,
$$

where $T_\Omega$ and $T_\Gamma$ are trace, averaging, or projection operators.

Enforcing such a condition strongly is inconvenient when:

- the bulk mesh does not fit the inclusion;
- the inclusion is lower-dimensional;
- the coupling space is itself reduced.

We use a weak enforcement strategy. One introduces a Lagrange multiplier $\lambda$ and writes a saddle-point problem of the form:

$$
\begin{aligned}
a(u,v) + b(v,\lambda) &= \ell(v), \\
b(u,\mu) &= g(\mu),
\end{aligned}
\qquad
\forall (v,\mu).
$$

Here:

- $a(\cdot,\cdot)$ is the bulk bilinear form;
- $b(\cdot,\cdot)$ represents the coupling operator;
- $\lambda$ lives in a multiplier space $Q$ defined on the interface or on the reduced geometry.

In matrix form, this becomes the familiar block system

$$
\begin{bmatrix}
A & B^T \\
B & 0
\end{bmatrix}
\begin{bmatrix}
u \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
f \\
g
\end{bmatrix}.
$$

This structure is exactly what appears in the coupled parts of the codebase.

## Reduced Lagrange Multiplier Spaces

In a full formulation, the multiplier would live on the full interface of the thin inclusion. If the inclusion is a slender three-dimensional tube, that interface is two-dimensional. The reduced formulation replaces that full interface space with a lower-dimensional space constructed on a characteristic set, typically the centerline.

Schematically:

$$
Q_{\text{full}} \xrightarrow{\,R\,} Q_{\text{red}}.
$$

Here $R$ is the restriction operator introduced in the reduced Lagrange multiplier framework: it maps a multiplier defined on the full interface space to a multiplier represented on the reduced space. In the code and in the theory, the reduced space is not chosen arbitrarily. It is built so that the action of $R$ preserves the coupling modes that are relevant for the bulk problem, while still being much cheaper to discretize and solve.

The companion operator $R^T$ lifts reduced data back to the full interface space. At the continuous level, one should think of:

$$
R : Q_{\text{full}} \to Q_{\text{red}},
\qquad
R^T : Q_{\text{red}} \to Q_{\text{full}}.
$$

The reduced formulation is then obtained by replacing the full multiplier by an object of the form

$$
\lambda_{\text{full}} \approx R^T \lambda_{\text{red}}.
$$

Equivalently, the reduced operator seen by the bulk problem is obtained by composing the original coupling with $R^T$, while reduced data on the interface are obtained by applying $R$ to full-interface quantities.

The construction used in the repository follows this idea:

1. choose a lower-dimensional characteristic manifold $\gamma$;
2. choose a reference cross section $D$;
3. build basis functions on $D$;
4. form tensor-product basis functions on $\gamma \times D$;
5. use these to define the restriction operator $R$ and its lifting $R^T$ between full and reduced spaces.

At a formal level, one can think of a reduced multiplier as an expansion

$$
\lambda(s,y) \approx \sum_{i=1}^N \lambda_i(s)\,\phi_i(y),
$$

where:

- $s \in \gamma$ is the coordinate on the reduced manifold;
- $y \in D$ is the transverse coordinate on the reference cross section;
- $\phi_i$ are transverse modes;
- $\lambda_i(s)$ are unknown coefficients along the reduced geometry.

This is the reason the repository contains classes such as:

- `ReferenceCrossSection`, which defines the cross-sectional basis;
- `TensorProductSpace`, which combines reduced and transverse directions;
- `ParticleCoupling`, which manages the interaction between reduced geometry data and the background discretization.

## What Is Being Reduced

The reduction is not the same in every application. The code supports different interpretations.

### Fibers in elasticity

For reinforced materials, the thin structure represents fibers embedded in an elastic matrix. The reduced model keeps the centerline geometry and a small set of transverse modes instead of resolving the full fiber boundary.

Mathematically, one replaces a detailed fiber-matrix interface by a weak coupling condition against a reduced multiplier space. This preserves the effect of the fibers on the matrix while avoiding a fully conforming mesh around every fiber.

### Vessels in tissues

For vascularized tissues, the thin structure represents a vessel network. The bulk variable is the tissue displacement, while the lower-dimensional object carries information related to pressure, forcing, or local deformation induced by the fluid phase.

In this case, the reduced coupling is useful because:

- the vessels are thin relative to the tissue size;
- imaging or input data are often available only at coarse scales;
- one still wants the bulk tissue model to retain the influence of the vasculature.

### Scalar transport and Poisson problems

The same mixed-dimensional machinery also applies to scalar PDEs. In that case, the coupling may represent transfer, forcing, or interface constraints between a bulk scalar field and a reduced lower-dimensional object.

This is why the repository is not purely an elasticity code: the scalar case is mathematically simpler, but it is also the cleanest setting in which to study the stability and approximation properties of the reduced multiplier construction.

## Typical Weak Form With Reduction

A useful prototype for the repository is the following.

Let $u$ be a bulk field, let $\lambda_{\mathrm{full}} \in Q_{\mathrm{full}}$ be the multiplier of the unreduced problem, and let $\lambda_{\mathrm{red}} \in Q_{\mathrm{red}}$ be the reduced multiplier. The reduced ansatz is

$$
\lambda_{\mathrm{full}} \approx R^T \lambda_{\mathrm{red}}.
$$

Substituting this into the weak formulation gives the reduced problem:

$$
\begin{aligned}
a(u,v) + \langle B^T R^T \lambda_{\mathrm{red}}, v \rangle &= \ell(v), \\
\langle RBu, \mu \rangle - \langle M \lambda_{\mathrm{red}}, \mu \rangle &= r(\mu),
\end{aligned}
$$

for all test functions $v$ and $\mu$.

Here:

- $B$ is the coupling operator from the bulk unknowns to the full interface constraints;
- $R$ restricts full-interface quantities to the reduced space;
- $R^T$ lifts reduced multipliers to the full interface representation;
- $M$ is a reduced mass or scaling matrix;
- $r(\mu)$ is the reduced right-hand side assembled from lower-dimensional data.

From the implementation point of view, this means that the assembled coupling operator is never the unreduced $B$ alone. The matrix called `coupling_matrix` in the repository should always be interpreted as the reduced operator, namely $RB$, or equivalently through its transpose action $B^T R^T$ depending on which block of the saddle-point system is being assembled or applied.

This abstract form explains several implementation details in the code:

- assembly of a `coupling_matrix`;
- assembly of a reduced mass matrix for the multiplier basis;
- use of cross-sectional quadrature and basis functions;
- explicit right-hand side assembly on the reduced space.

## Two Concrete Examples

### Example A: Elastic body with an immersed one-dimensional inclusion

Let $\Omega \subset \mathbb{R}^3$ be an elastic body and let $\gamma$ be the centerline of a slender inclusion. One seeks a displacement field $u$ in the bulk and a reduced multiplier $\lambda$ on $\gamma$ such that:

$$
\int_\Omega \sigma(u):\varepsilon(v)\,dx
+
\langle B^T\lambda, v\rangle
=
\int_\Omega b\cdot v\,dx
$$

for all bulk test functions $v$, together with a reduced constraint equation expressing compatibility between the bulk displacement and the inclusion model.

This is the mathematical template behind the reduced elasticity solvers and the classes handling immersed inclusions, in particular `ElasticityProblem`, `Inclusions`, and `TensorProductSpace`.

### Example B: Bulk scalar field coupled to a reduced interface

Let $u$ solve a Poisson problem in $\Omega$, but suppose a lower-dimensional inclusion imposes an interface condition only in a reduced sense. Then the weak problem becomes:

$$
\int_\Omega \kappa \nabla u \cdot \nabla v\,dx
+
\langle B^T \lambda, v \rangle
=
\int_\Omega f v\,dx.
$$

The second equation determines $\lambda$ so that the reduced interface condition is satisfied in the multiplier space.

This is the clean scalar analogue of the elasticity machinery and is useful for understanding the repository before tackling the vector-valued case.

## Mathematical References

The papers in `doc/papers/` develop different parts of the mathematical picture described above:

- {cite:p}`AlzettaHeltai-2020-a` develops the immersed non-matching coupling viewpoint for fiber-reinforced elasticity.
- {cite:p}`HeltaiCaiazzo-2019-a` applies similar ideas to vascularized tissues and singular lower-dimensional forcing.
- {cite:p}`HeltaiZunino-2023-a` develops the abstract reduced Lagrange multiplier framework and its stability/error analysis.
- {cite:p}`BelponerCaiazzoHeltai-2025-a` extends the reduced framework to mixed-dimensional vascular tissues and elasticity with more realistic coupling conditions.

These papers are not separate from the repository: they describe, analyze, and motivate the same family of methods that the code implements in different variants.
