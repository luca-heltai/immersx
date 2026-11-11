# Pressure on the vessel walls

ImmersX solves vessel–tissue problems on an extended solid: the elastic
material fills the whole box, including the inside of every vessel, and the
wall displacement is imposed with a Lagrange multiplier. The multiplier is
therefore not the pressure of the tissue on the wall. It is the jump between
the normal stress of the tissue and that of the fictitious material inside the
vessel. This page collects what ImmersX does with that interior stress.

Two code paths turn a wall multiplier into a vessel pressure, and they remove
the interior stress in different ways:

- `VesselElasticityProblem` (the `coupled_elasticity` and `pseudocoupling1D`
  applications) solves the extended problem unchanged and adds the stress of
  the analytic interior solution to the multiplier when it computes the
  pressure. This is a postprocessing step, controlled by
  `Remove interior stress from pressure`.
- `metric_flow_x_elastodynamics` subtracts the discrete elastic and viscous
  response of two interior modes from the solid momentum equation, controlled
  by `Remove fictitious modal response`. The multiplier itself then supplies
  the pressure feedback to the flow.

The `elasticity` application uses `ElasticityProblem`, which computes no vessel
pressure. `Output pressure` and `Remove interior stress from pressure` are
declared in its parameter file but have no effect there.

## Signs and units

The wall normal $\mathbf{n}$ points out of the vessel into the tissue.
`VesselElasticityProblem` stores the mean normal stress
$p_v = \overline{\mathbf{n}\cdot\sigma\mathbf{n}}$ on the wall of vessel $v$
in `pressure_at_immersed_surface`, in the stress units of the material
parameters. It is positive in tension, so tissue that pushes on the vessel
gives $p_v < 0$. The physical external pressure is $-p_v$.

`coupled_elasticity` passes $p_e = -10\,p_v$ to the one-dimensional flow model
as its external pressure; the factor 10 converts Pa to dyn/cm². With
`Output pressure = true`, `VesselElasticityProblem` writes $p_v$, not $p_e$,
to `<Output name>_externalPressure.txt` in serial runs and to the dataset
`externalPressure_<cycle>` of `<Output name>_externalPressure.h5` in MPI runs.

In `metric_flow_x_elastodynamics` the multiplier $\lambda$ is a scalar radial
traction amplitude with pressure units. It enters the solid momentum residual
as $+B\lambda$, and MetricFlowX receives the additional external pressure
$p_{\mathrm{add}} = -\lambda$. See
[MetricFlowX elastodynamics coupling](../tutorials/metric-flow-x-elastodynamics-coupling.md).

## The multiplier is a jump of normal stress

Let $V$ be the inside of a vessel, $\Gamma = \partial V$ its wall, $\sigma^-$
the stress of the material in $V$, and $\sigma^+$ the stress of the tissue in
$\Omega \setminus V$. If $\operatorname{div}\sigma = -\mathbf{f}$ holds in
both parts, integration by parts on each part gives

$$
\int_\Omega \sigma : \varepsilon(\mathbf{v})\, dx
= \int_\Omega \mathbf{f}\cdot\mathbf{v}\, dx
- \int_\Gamma [\![\sigma]\!]\mathbf{n}\cdot\mathbf{v}\, d\Gamma,
\qquad
[\![\sigma]\!]\mathbf{n} = \sigma^+\mathbf{n} - \sigma^-\mathbf{n}.
$$

Comparing with $a(\mathbf{u},\mathbf{v}) + \langle B^T R^T \Lambda,
\mathbf{v}\rangle = \ell(\mathbf{v})$ identifies the multiplier load with the
jump,

$$
\langle B^T R^T \Lambda, \mathbf{v} \rangle
= \langle [\![\sigma]\!]\mathbf{n}, \mathbf{v} \rangle .
$$

This is Corollary 1 of {cite:p}`BelponerCaiazzoHeltai-2025-a` for the reduced
elasticity formulation. For the Poisson problem it is the classical result of
Glowinski, Pan and Périaux (Comput. Methods Appl. Mech. Engrg. 111, 1994) and
Girault and Glowinski (Japan J. Indust. Appl. Math. 12, 1995).

The pressure computed from the multiplier alone is therefore

$$
p_v^{\lambda} = \frac{1}{|\Gamma_v|}\int_{\Gamma_v}
\left(\sigma^+_{nn} - \sigma^-_{nn}\right) d\Gamma ,
$$

and the normal stress of the tissue is

$$
p_v = p_v^{\lambda} + \frac{1}{|\Gamma_v|}\int_{\Gamma_v} \sigma^-_{nn}\, d\Gamma .
$$

In pressure form, with $p_{\mathrm{ext}} = -\sigma^+_{nn}$ and
$p_{\mathrm{int}}^{\mathrm{fict}} = -\sigma^-_{nn}$, the multiplier is
$p_{\mathrm{int}}^{\mathrm{fict}} - p_{\mathrm{ext}}$, and the pressure sent to
the flow model without any correction is
$p_{\mathrm{ext}} - p_{\mathrm{int}}^{\mathrm{fict}}$.

The interior term is not small. For a cavity of radius $r$ expanded by
$\delta R$ in an unbounded medium with Lamé parameters $\lambda_L$ and $\mu_L$,
the tissue stress at the wall is $\sigma^+_{rr} = -2\mu_L\,\delta R/r$, while
the interior stress is $\sigma^-_{rr} = 2(\lambda_L+\mu_L)\,\delta R/r$ (see
below). The multiplier then gives $-2(\lambda_L + 2\mu_L)\,\delta R/r$, which
overestimates the tissue stress by the factor $(\lambda_L+2\mu_L)/\mu_L$: 3.5
for $\lambda_L = 1.5\mu_L$ ($\nu = 0.3$), and much more for a nearly
incompressible tissue.

## The analytic interior stress

`VesselElasticityProblem` adds the stress of the exact interior solution for
the imposed wall displacement, not the stress of the discrete solution inside
$V$:

- the vessels are thin, and the radius is often smaller than the mesh size, so
  the discrete stress inside $V$ is not resolved;
- as the mesh is refined, the discrete multiplier converges to the exact jump,
  whose interior part is the stress of the exact interior solution. Adding the
  exact interior stress gives the tissue stress in the limit.

The interior solution is used in the thin-vessel, plane-strain setting of the
reduced model: the cross section is a disc $D$ of radius $r$, the displacement
does not change along the vessel, $\varepsilon_{zz} = 0$, and there is no body
force inside the vessel. Let $\mathbf{x}$ be the in-plane position measured
from the center, so that $\mathbf{x} = r\mathbf{n}$ on the wall. Since
$\operatorname{div}\sigma^- = 0$ in $D$ and
$\sigma = \lambda_L(\operatorname{div}\mathbf{u})I + 2\mu_L\varepsilon$ with
$\varepsilon_{zz}=0$,

$$
\int_{\partial D} \sigma^-\mathbf{n}\cdot\mathbf{x}\, ds
= \int_{D} \sigma^- : \nabla\mathbf{x}\, dA
= \int_{D} (\sigma^-_{xx} + \sigma^-_{yy})\, dA
= 2(\lambda_L+\mu_L) \int_{D} \operatorname{div}\mathbf{u}\, dA
= 2(\lambda_L+\mu_L) \int_{\partial D} \mathbf{u}\cdot\mathbf{n}\, ds .
$$

With $\mathbf{x} = r\mathbf{n}$ on the wall, the mean normal stress of the
interior material on the wall is

$$
\bar\sigma^-_{nn} = \frac{K}{r}\, \bar u_n ,
\qquad
K = 2(\lambda_L+\mu_L),
\qquad
\bar u_n = \frac{1}{|\partial D|}\int_{\partial D} \mathbf{u}\cdot\mathbf{n}\, ds .
$$

This holds for any imposed wall displacement, not only for a uniform
inflation. Only the mean normal displacement $\bar u_n$ matters, so only the
radial modes of the data enter, just as only the radial modes of the
multiplier enter the normal force. For the uniform inflation
$\mathbf{u} = \delta R\,\mathbf{n}$, the interior solution is
$\mathbf{u} = (\delta R/r)\,\mathbf{x}$ and
$\sigma^-_{rr} = 2(\lambda_L+\mu_L)\,\delta R/r$.

In 2D (plane strain) the formula is exact. In 3D it is the leading-order
approximation for a vessel whose wall displacement varies slowly along the
vessel compared with its radius; the axial strain inside the vessel and the
ends of the vessels are neglected.

## The radial modes

In the frame of a vessel, with the vessel along $z$ and $\theta$ the angle
around it, the wall normal is
$\mathbf{n} = \cos\theta\,\mathbf{e}_x + \sin\theta\,\mathbf{e}_y$. A
multiplier mode $\varphi(\theta)\,\mathbf{e}_c$ has a nonzero mean normal
component on a complete circle only if $\varphi\,\mathbf{e}_c$ is not
orthogonal to $\mathbf{n}$. Among the Fourier modes $1, \cos\theta,
\sin\theta, \cos 2\theta, \sin 2\theta, \dots$ times $\mathbf{e}_x,
\mathbf{e}_y, \mathbf{e}_z$, only

$$
\cos\theta\,\mathbf{e}_x \quad\text{and}\quad \sin\theta\,\mathbf{e}_y
$$

qualify. With the coefficient index $c = m\,d + \text{component}$ (Fourier
mode $m$, $d$ vector components), these are the indices $d$ and $2d+1$: 3 and
7 in 3D, 2 and 5 in 2D. The formulas below use the 3D indices. These are also
the modes that impose a radial displacement (inflation or deflation) of the
vessel, so they carry both the normal force of the multiplier and the mean
normal displacement $\bar u_n$ of the interior correction.

ImmersX evaluates the pressure on the physical wall samples. Samples that lie
outside the bulk mesh contribute zero, and every mean is normalized by the
complete nominal wall or section. The closed formulas in terms of the
coefficients 3 and 7 hold on a complete, resolved wall. On a partly covered
wall the other modes no longer cancel, and the sampled sums are used.

## Point coupling

### One multiplier block per inclusion

Inclusion $k$ has center $\mathbf{c}_k$, direction $\mathbf{d}_k$ (its length
is $\ell_k = |\mathbf{d}_k|$), radius $r_k$, and rotation $R_k$ from the
reference frame to the vessel frame. It is sampled at the $n_q$ points

$$
\mathbf{x}_{k,q} = \mathbf{c}_k + r_k R_k (\cos\theta_q, \sin\theta_q, 0),
\qquad \theta_q = \frac{2\pi q}{n_q},
$$

with weights $w_q = |S_k|/n_q$, where $|S_k| = 2\pi r_k \ell_k$ is the area of
its wall ($2\pi r_k$ in 2D). The basis functions are the unnormalized Fourier
modes $\varphi_0 = 1$, $\varphi_{2j-1} = \cos(j\theta)$,
$\varphi_{2j} = \sin(j\theta)$. Multiplier entry $(k, j)$ uses the coefficient
$c_j$ of the selection, with mode $m_j = c_j / d$ and direction
$\mathbf{e}_{c_j \% d}$. The constraint rows are divided by $|S_k|$:

$$
(B\mathbf{u})_{k,j} = \frac{1}{|S_k|} \sum_q \mathbf{u}(\mathbf{x}_{k,q})
\cdot R_k \mathbf{e}_{c_j \% d}\, \varphi_{m_j}(\theta_q)\, \frac{|S_k|}{n_q}
= \frac{1}{n_q} \sum_q \mathbf{u}(\mathbf{x}_{k,q}) \cdot R_k
\mathbf{e}_{c_j \% d}\, \varphi_{m_j}(\theta_q).
$$

$(B\mathbf{u})_{k,j}$ is a displacement, so $\lambda_{k,j}$ is a force. The
term $B^T\lambda$ applies at each point the force $w_q\,\mathbf{t}_{k,q}$, with
the traction

$$
\mathbf{t}_{k,q} = \frac{1}{|S_k|} \sum_j \lambda_{k,j}\,
\varphi_{m_j}(\theta_q)\, R_k \mathbf{e}_{c_j \% d}.
$$

Let $\chi_q = 1$ if the sample lies in the bulk mesh and $0$ otherwise, and
$\mathbf{n}_{k,q} = R_k(\cos\theta_q, \sin\theta_q, 0)$. The pressure of
inclusion $k$ and of vessel $v$ are

$$
p_k = \frac{1}{|S_k|}\sum_q w_q\,\chi_q\,\sigma_{k,q},
\qquad
p_v = \frac{\sum_{k\in v} |S_k|\, p_k}{\sum_{k\in v}|S_k|},
\qquad
\sigma_{k,q} = \mathbf{t}_{k,q}\cdot\mathbf{n}_{k,q} + \sigma^-_{nn,k}.
$$

The interior term $\sigma^-_{nn,k}$ is zero unless
`Remove interior stress from pressure` is true; it is defined in the
interior-stress subsection below. $p_k$ is stored in
`pressure_at_immersed_surface_by_inclusion` and $p_v$ in
`pressure_at_immersed_surface`.

On a complete wall ($\chi_q = 1$ for every $q$) with $n_q$ larger than the
highest selected frequency plus one,

$$
\frac{1}{n_q}\sum_q \cos^2\theta_q = \frac{1}{n_q}\sum_q \sin^2\theta_q =
\frac12 ,
$$

and every other product of a basis function with $\cos\theta_q$ or
$\sin\theta_q$ averages to zero. ImmersX then uses the exact moments, and the
multiplier part of the pressure is

$$
p_k^{\lambda} = \frac{\lambda_{k,3} + \lambda_{k,7}}{2\,|S_k|},
$$

where $\lambda_{k,3}$ and $\lambda_{k,7}$ are the entries of the coefficients
3 and 7 (zero if they are not selected). For a vessel of constant radius $r$
and length $L_v = \sum_{k\in v} \ell_k$,

$$
p_v^{\lambda} = \frac{1}{2\pi r L_v} \sum_{k \in v} \frac{\lambda_{k,3} +
\lambda_{k,7}}{2}.
$$

If the vessel is split into $m$ times as many inclusions, each of them has
$1/m$ of the wall area and receives $1/m$ of the force, so $\lambda_k$ and
$|S_k|$ both scale with $1/m$ and $p_k$ and $p_v$ do not change.

### Segments

With `Cluster inclusions with segments` the inclusions whose centers lie in
the same bulk cell form a segment $s$, and the multiplier has one block per
segment. The constraint of a segment is the sum of the constraints of its
inclusions,

$$
(B\mathbf{u})_{s,j} = \sum_{k \in s} (B\mathbf{u})_{k,j},
$$

so

$$
\langle B^T \lambda, \mathbf{v} \rangle
= \sum_s \sum_j \lambda_{s,j} \sum_{k\in s} (B\mathbf{v})_{k,j}
= \sum_k \sum_j \lambda_{s(k),j} (B\mathbf{v})_{k,j} .
$$

The segment multiplier acts on every inclusion of the segment exactly as if
that inclusion had $\lambda_{s(k)}$ as its own multiplier. The formulas above
hold with $\lambda_k$ replaced by $\lambda_{s(k)}$; on a complete wall

$$
p_k^{\lambda} = \frac{\lambda_{s(k),3} + \lambda_{s(k),7}}{2\,|S_k|}.
$$

A segment may contain inclusions of different vessels; each inclusion
contributes to its own vessel. The mean normal stress on the wall of a
segment with $n_s$ inclusions is

$$
p_s^{\lambda} = \frac{n_s}{\sum_{k\in s}|S_k|}\,
\frac{\lambda_{s,3} + \lambda_{s,7}}{2},
$$

that is, the segment force divided by the mean wall area of its inclusions.
If the inclusions of a segment are made $m$ times more numerous, the
constraint of the segment is the same wall integral sampled with $m$ times
more points, so $(B\mathbf{u})_s$ and the data both scale with $m$,
$\lambda_s$ scales with $1/m$, and $p_k$, $p_s$ and $p_v$ do not change.

### Interior stress

The mean normal displacement is read from the multiplier right-hand side, as
it enters the constraint, so it is the same for a data file, reference data,
or a boundary function. The right-hand side is assembled on the complete
nominal wall, independently of $\chi_q$. Its entries for the coefficients 3
and 7 of block $b$ are

$$
g_{b,3} = \frac{1}{n_q}\sum_q u_x(\theta_q)\cos\theta_q,
\qquad
g_{b,7} = \frac{1}{n_q}\sum_q u_y(\theta_q)\sin\theta_q,
$$

with $u_x$ and $u_y$ the components of the imposed wall displacement in the
vessel frame. Their sum is the sampled mean of $\mathbf{u}\cdot\mathbf{n}$, so
$\bar u_{n,k} = g_{b,3} + g_{b,7}$. For data with coefficients $a_3$ and $a_7$
this is $(a_3 + a_7)/2$. With segments, the right-hand side of a segment is
the sum over its $n_s$ inclusions, and each inclusion receives the mean
$\bar u_{n,k} = (g_{s,3} + g_{s,7})/n_s$.

$\bar K_k$ is the arithmetic mean of $2(\lambda_L+\mu_L)$ over the samples of
inclusion $k$ that lie in the bulk mesh, each evaluated in the material of the
bulk cell that contains it; it is zero if no sample is found. Then

$$
\sigma^-_{nn,k} = \frac{\bar K_k}{r_k}\,\bar u_{n,k},
$$

and on a complete wall

$$
p_k = \frac{\lambda_{b(k),3} + \lambda_{b(k),7}}{2\,|S_k|}
+ \frac{\bar K_k}{r_k}\,\bar u_{n,k}.
$$

## Tensor-product coupling

The multiplier is a finite element field on the reduced grid. Component $c$
of the field is the coefficient $\Lambda_c(s)$ of the selected cross-section
mode $\phi_c$, and the coupling term is

$$
\langle B^T R^T \Lambda, \mathbf{v} \rangle = \int_{\gamma} \int_{D(s)}
\mathbf{v} \cdot \mathbf{t}\, dD\, ds,
\qquad
\mathbf{t}(s, \xi) = \sum_c \Lambda_c(s)\, \phi_c(\xi),
$$

where $D(s)$ is the cross section at $s$: for elasticity, the circle of radius
$r(s)$, i.e., the wall itself. The multiplier is a force per unit area. The
modes are orthogonal on the reference cross section and scaled to maximum
one, as the Fourier modes of point coupling, so the two radial modes are

$$
\phi_3 = \cos\theta\,\mathbf{e}_x, \qquad
\phi_7 = \sin\theta\,\mathbf{e}_y ,
$$

and $\mathbf{n} = \phi_3 + \phi_7$ on the wall.

Let $s_j$ be the quadrature points of the reduced grid, with weights $v_j$,
and let $\xi_{j\ell}$ be the wall samples of the cross section at $s_j$, with
nominal weights $w_{j\ell}$ and $\chi_{j\ell} = 1$ if the sample lies in the
bulk mesh. The normal $\mathbf{n}_{j\ell}$ is the direction from the
centerline point to the physical sample. All selected modes enter the
traction before its normal component is taken:

$$
p(s_j) = \frac{\sum_\ell w_{j\ell}\,\chi_{j\ell}\,\sigma_{j\ell}}
{\sum_\ell w_{j\ell}},
\qquad
\sigma_{j\ell} = \sum_c \Lambda_c(s_j)\,\phi_c(\xi_{j\ell})\cdot
\mathbf{n}_{j\ell} + \sigma^-_{nn}(s_j).
$$

On a complete section only the radial modes survive, and

$$
p^{\lambda}(s) = \sum_c k_c\, \Lambda_c(s),
\qquad
k_c = \frac{1}{|D_{\text{ref}}|}\int_{D_{\text{ref}}} \phi_c \cdot
\mathbf{n}\, dD,
$$

with $k_3 = k_7 = 1/2$ and $k_c = 0$ otherwise:

$$
p^{\lambda}(s) = \frac{\Lambda_3(s) + \Lambda_7(s)}{2}.
$$

The reference circle is a polygon, so the sampled $k_3$ and $k_7$ are close
to, but not exactly, $1/2$: 0.475 on the 8-sided polygon of
`Refinement level = 1`. These are the values consistent with the quadrature
that assembles the coupling.

The pressure of vessel $v$ is the mean over its centerline $\gamma_v$,

$$
p_v = \frac{\sum_{j\in\gamma_v} v_j\, p(s_j)}{\sum_{j\in\gamma_v} v_j}.
$$

When the radius is constant along the vessel this is the mean normal stress on
its wall, because $|\Gamma_v| = 2\pi r |\gamma_v|$. With a variable radius the
wall-area mean would weight $p(s)$ by $|D(s)|$; the implementation weights by
arc length.

`pressure_at_immersed_surface_by_inclusion` holds one value per support point
of the reduced grid, in the order of the degrees of freedom of the first
component of the reduced `DoFHandler`. The values are the coefficients of the
$L^2$ projection of $p(s)$ onto the scalar finite element space of one
multiplier component, with the centerline mass matrix
$M_{in} = \sum_j v_j \psi_i(s_j)\psi_n(s_j)$. On a complete section the
multiplier part $p^{\lambda}$ is a combination of multiplier components and
lies in that space, so the projection returns its nodal values
$\sum_c k_c\,\lambda_{n,c}$.

### Interior stress

The coefficients $a_c(s)$ of the imposed wall displacement solve $M a = g$,
where $g$ is the multiplier right-hand side and
$M = \int \psi_m\psi_n |D(s)|\, m_c\, ds$ is the coupling mass matrix, the
Gram matrix of the multiplier space. Both are assembled on the complete
nominal sections, so $a$ does not depend on $\chi$. The mean normal
displacement at $s_j$ is the sampled mean over the nominal section of the
radial modes,

$$
\bar u_n(s_j) = \frac{\sum_\ell w_{j\ell} \sum_{c\in\{3,7\}} a_c(s_j)\,
\phi_c(\xi_{j\ell})\cdot\mathbf{n}_{j\ell}}{\sum_\ell w_{j\ell}},
$$

which is $(a_3 + a_7)/2$ on a resolved circle. Then

$$
\sigma^-_{nn}(s_j) = \frac{\bar K}{r(s_j)}\,\bar u_n(s_j),
$$

with $r(s)$ the thickness, evaluated as for the coupling (constant,
expression, or input field), and $\bar K$ the arithmetic mean of
$2(\lambda_L+\mu_L)$ over the samples of the reduced cell that lie in the bulk
mesh (zero if there are none). On a complete section

$$
p(s) = \frac{\Lambda_3 + \Lambda_7}{2}
+ \frac{\bar K}{r(s)}\,\frac{a_3 + a_7}{2}.
$$

## Comparing point and tensor-product coupling

Both couplings use modes with maximum one ($\cos\theta$), so a coefficient
$X$ for index 3 imposes the displacement $X\cos\theta$ in both. The
multiplier differs: in point coupling it is a force per inclusion, in
tensor-product coupling a force per unit area. The formulas above remove this
difference, so for the same imposed wall displacement the two couplings give
the same pressure up to discretization error, with and without the interior
stress.

The operators also differ in what they integrate. Point coupling averages
over the wall $S$ of each inclusion: the coupling matrix is
$\frac{1}{|S|}\oint_S \mathbf{v}\cdot\phi_j$, the right-hand side
$a_j\langle\phi_j^2\rangle$, and the inclusion matrix $\oint_S \phi_j^2$.
Tensor-product coupling integrates over the wall: the coupling matrix is
$\int_\gamma \int_{D(s)} \mathbf{v}\cdot\psi\phi_c\, dD\, ds$, the
right-hand side $\int_\gamma \psi\, a_c\, |D(s)|\, m_c\, ds$, and the mass
matrix $\int_\gamma \psi_m\psi_n\, |D(s)|\, m_c\, ds$, with
$m_c = \frac{1}{|D|}\int_D \phi_c\cdot\phi_c$ ($1/2$ for $\cos\theta$).

## Prescribed vessel pressure

With `Prescribed vessel pressure = true` there is no constraint. ImmersX
computes $\lambda = M^{-1}g$ from the data and applies the load $B^T\lambda$
to the extended solid. The prescribed value is therefore a jump of normal
stress across the wall, and part of it is carried by the material inside the
vessel. No interior correction is applied to the data in this mode: the
interior stress depends on the wall displacement, which is not known before
the solve.

## MetricFlowX: removing the interior modes from the solve

`metric_flow_x_elastodynamics` uses the same extended solid, so without a
correction its multiplier is also the jump
$\lambda = p_{\mathrm{int}}^{\mathrm{fict}} - p_{\mathrm{ext}}$, and the flow
receives $-\lambda = p_{\mathrm{ext}} - p_{\mathrm{int}}^{\mathrm{fict}}$.
With

```text
subsection MetricFlowX coupling
  set Remove fictitious modal response = true
end
```

the solid momentum equation loses the elastic and viscous response of two
spatially constant affine interior modes, $u_r = g\,r$ and $u_x = h\,x$. Their
elastic normal stresses are

$$
q_r = 2(\lambda_L+\mu_L)\,g + \lambda_L h,
\qquad
q_z = 2\lambda_L g + (\lambda_L+2\mu_L)\,h,
$$

plus the corresponding viscous terms. For a static, plane-strain interior
($h = 0$) with $g = \bar u_n / r$, $q_r$ is the analytic interior stress
$2(\lambda_L+\mu_L)\,\bar u_n / r$ used by `VesselElasticityProblem`. The two
corrections remove the same leading-order interior stress, but in different
ways:

| | `VesselElasticityProblem` | `metric_flow_x_elastodynamics` |
| --- | --- | --- |
| Parameter (default) | `Remove interior stress from pressure` (`true`) | `Remove fictitious modal response` (`false`) |
| Where | Added to the computed pressure; the solve is unchanged | Subtracted from the solid momentum equation; the multiplier changes |
| Interior amplitude | Analytic, from the imposed wall data | Discrete projection of the solid strain at the Gauss points inside the cylinder |
| Modes | Mean radial displacement, per inclusion or reduced quadrature point | Radial and axial affine modes, constant over the whole cylinder |
| Physics | Static, elastic, plane strain | Elastic and viscous; interior mass and inertia are kept |
| Geometry | Any network, point or tensor-product coupling | One straight, uniform-radius, fully covered cylinder with homogeneous interior material |

With the MetricFlowX correction the interior stress that remains in the
multiplier comes from the retained inertia and from interior strains outside
the two modes. In the manufactured solution the interior strain lies in the
two modes, and $\lambda^\star = B - K$ is the exterior radial stress at the
wall. The pressure sent to the flow is still $-\lambda$, and no further
correction is applied to it. The derivation and its limits are in
[One vessel in a cube](../tutorials/metric-flow-x-elastodynamics-mms.md).

## Validation

`ElasticityValidation.BOTH_PaperAxisSymmetricPressure` solves the annulus of
Section 5.1 of {cite:p}`BelponerCaiazzoHeltai-2025-a` (inclusion radius
$r_i = 0.2$, outer radius $R = 1$, inflation $\bar u = 0.1$, displacement
zero on the outer boundary) with $\lambda_L = 3$, $\mu_L = 1$. Outside the
inclusion $u_r = A\rho + B/\rho$ with $A = -r_i\bar u/(R^2-r_i^2)$ and
$B = r_i\bar u R^2/(R^2-r_i^2)$, and inside $\mathbf{u} = \bar u\,\mathbf{x}/r_i$.
At the wall

$$
\sigma^+_{rr} = 2(\lambda_L+\mu_L)A - \frac{2\mu_L B}{r_i^2} = -1.2083,
\qquad
\sigma^-_{rr} = \frac{2(\lambda_L+\mu_L)\,\bar u}{r_i} = 4,
\qquad
\sigma^+_{rr} - \sigma^-_{rr} = -5.2083 .
$$

With $Q_2$ elements, four global refinements and one further refinement of
the cells that contain the circle, ImmersX gives $-5.2670$ without the
interior stress and $-1.2670$ with it. The difference of the two is the
interior stress, 4, to round-off. The remaining difference from the exact
values, 0.059 in both cases, is the discretization error of the multiplier
(1.1% of the jump). The test accepts 2% of the jump.

`gtests/coupled_elasticity_03.cc` checks the formulas of this page on a
straight cylinder in a cube: the multiplier part for point coupling with and
without segments and for tensor-product coupling; the interior stress alone,
with the multiplier set to zero, for the same three cases; zero extension of
wall samples outside the bulk mesh; the arc-length vessel mean with a variable
radius; and the nominal projection and cell-averaged material of the
tensor-product interior stress.

## Limitations

- A body force inside the vessels is not included in the interior solution.
- In 3D the interior stress is the plane-strain value; axial strain and the
  vessel ends are neglected.
- The interior material is averaged per inclusion or reduced cell, so a
  vessel that crosses a material interface uses a mean value there.
- The tensor-product vessel mean is weighted by arc length, which is the
  wall-area mean only for a constant radius.
- Prescribed vessel pressure data are applied as a stress jump, without
  interior correction.
- The MetricFlowX modal correction is limited to a single straight cylinder;
  see its tutorial for the full list of restrictions.
