# Convergence of the vessel multiplier

This page continues [A vessel in a cylinder](cylinder-vessel-vs-hole.md). It
measures how fast the multiplier of an immersed vessel converges, first in a
static cube, then in time with Newmark steps, and compares the result with the
MetricFlowX study of [One vessel in a cube](metric-flow-x-elastodynamics-mms.md)
(issue [#379](https://github.com/luca-heltai/immersx/issues/379)). One
ingredient is added at a time:

1. the static cylinder of the previous tutorial: the multiplier is the
   traction jump;
2. a static cube with the manufactured exterior field of the MetricFlowX
   study, refined near the wall: the rate of the jump in $h/a$;
3. an axial strain along the vessel: the jump does not change, the tissue
   stress does;
4. Newmark steps with inertia and damping: the time error of the multiplier;
5. the dynamic manufactured solution under spatial refinement.

## Setting

The cube is $[-\ell,\ell]^3$ with a vessel of radius $a$ on the $x$ axis,
$\lambda = \mu = 1$, tensor-product coupling with the modes 3 and 7 (the mean
radial displacement), a cross section with 32 sides and eight reduced cells.
The manufactured field is

$$
u = \bigl(h\,x,\; g\,W_y,\; g\,W_z\bigr),\qquad
W = \frac{a^2}{\max(r^2, a^2)}\,(0, y, z),\qquad r^2 = y^2 + z^2,
$$

a uniform dilation $g$ inside the vessel, the field $g a^2/r$ outside, and the
same axial strain $h$ on both sides. All faces of the cube get the exact field,
and the wall data are the mean radial displacement $g a$.

The multiplier is the jump of the traction across the wall,
$\lambda = (\sigma^{\rm out} - \sigma^{\rm in})\,n$ (see the previous
tutorial). Here

$$
\sigma^{\rm out}_{rr} = \lambda h - 2\mu g,\qquad
\sigma^{\rm in}_{rr} = 2(\lambda+\mu) g + \lambda h,\qquad
J = \sigma^{\rm out}_{rr} - \sigma^{\rm in}_{rr} = -(4\mu + 2\lambda)\, g .
$$

The jump does not depend on $h$.

The mesh starts with $8^3$ cells. Each refinement pass refines the cells within
one cell diameter of the wall $r = a$. $h$ below is the size of the smallest
cell, at the wall. The tables give the mean of the multiplier coefficients
(both modes) at the reduced support points with $|x| \le \ell/2$; the mean
over all support points, including the two ends on the faces of the cube,
differs by less than the error and converges at the same rate.

The tensor coupling's own `Space post-refinement cycles` cannot be used for
this study: they refine only the cells around the vessel axis, not around the
wall ([#465](https://github.com/luca-heltai/immersx/issues/465)).

## Static cube

$\ell = 1$, $a = 0.2$, $g = 0.5$, $h = 0$: exact jump $J = -3$.

| passes | cells | dofs | $h/a$ | multiplier | error |
|---:|---:|---:|---:|---:|---:|
| 1 | 2304 | 8775 | 0.625 | −3.922 | 0.922 |
| 2 | 8128 | 29115 | 0.313 | −3.249 | 0.249 |
| 3 | 26944 | 93303 | 0.156 | −3.148 | 0.148 |
| 4 | 114752 | 388599 | 0.078 | −3.080 | 0.080 |
| 5 | 458816 | 1544367 | 0.039 | −3.030 | 0.030 |

Without refinement ($h/a = 1.25$) the solve does not converge. Below
$h/a \approx 0.3$ the error falls roughly by two per pass: first order in
$h/a$. A cross section with 64 or 128 sides, four times more reduced cells,
or four times more axial quadrature points change the multiplier at pass 3 by
less than 0.005.

The same study with the proportions of the MetricFlowX cube,
$\ell = 0.120685$, $a = 0.00987$ ($a/\ell = 0.082$) and $g = 1$ (exact jump
$-6$):

| passes | cells | dofs | $h/a$ | multiplier | relative error |
|---:|---:|---:|---:|---:|---:|
| 2 | 5440 | 20211 | 0.76 | −7.455 | 24 % |
| 3 | 15296 | 54927 | 0.38 | −6.929 | 15 % |
| 4 | 51136 | 174939 | 0.19 | −6.298 | 5.0 % |
| 5 | 183744 | 620295 | 0.096 | −6.168 | 2.8 % |

The relative error depends on $h/a$ much more than on $a/\ell$.

## Axial strain

With $h = 0.25$ the multiplier is the same as with $h = 0$ at every pass, to
all printed digits, and so are the displacement errors: the axial field is the
same on both sides of the wall and $Q_1$ represents it exactly. The tissue
stress on the wall changes, from $-1$ to $-0.75$.

`VesselElasticityProblem` reports as pressure the multiplier plus the
plane-strain interior stress $2(\lambda+\mu)u_n/a$. That is $J + 2 = -1$ in
the limit for both values of $h$; `coupled_elasticity` reports $-2.1106$ for
both on a uniform $16^3$ mesh. The interior stress $\lambda h$ of the axial
strain is missing, and so is the viscous interior stress in damped dynamic
runs ([#464](https://github.com/luca-heltai/immersx/issues/464)). The
multiplier itself is not affected.

## Newmark steps

The dynamic manufactured solution of the MetricFlowX study, with ImmersX's
coupling instead of MetricFlowX's: the same cube and vessel, $\rho = 1$,
Kelvin-Voigt viscosity $\eta = 0.1$ and Rayleigh $\beta_R = 0.2$, $g = t^3$,
$h = 2t^4$, body force $\rho\,\ddot u$, exact displacement on the faces, wall
data $g(t)\,a$, final time $T = 0.1$. The interior is a fictitious solid
with the same viscosity, so the multiplier is the jump of the elastic and
viscous traction:

$$
J(t) = -(4\mu + 2\lambda)\, g - \bigl[2\eta + \beta_R(4\mu + 2\lambda)\bigr]\,\dot g
= -6t^3 - 4.2t^2,\qquad J(0.1) = -0.048 .
$$

The Newmark steps impose the wall displacement at every step
(`coupled_elasticity` with a density, PR
[#449](https://github.com/luca-heltai/immersx/pull/449)). The multiplier of a
displacement constraint in a Newmark step balances the acceleration of the
increment, so a constraint residual $\delta$ shows up in the multiplier as
about $M\delta/(\beta\,dt^2)$. In the first version of #449 the solver could
stop with the constraint unmet, and the multiplier oscillated with an
amplitude growing like $1/dt^2$. PR
[#463](https://github.com/luca-heltai/immersx/pull/463) fixes the stopping
test. With it, at pass 2 ($h/a = 0.76$), $\lambda_h/J$ at $T = 0.1$ is

| $dt$ | 0.01 | 0.0025 | 0.00125 | 0.000625 |
|---|---:|---:|---:|---:|
| $\lambda_h/J$ | 1.2471 | 1.2495 | 1.2474 | 1.2468 |

An oscillation from the equilibrium start decays over the first steps; for
$t \ge 0.05$, $\lambda_h/J$ stays within 1.246–1.253 for $dt \le 0.0025$.
The radial displacement sampled on the circle $r = a$ at $x = 0$ stays at
0.997 of its target at pass 2 and at 0.981 at pass 5, at every step. The
decrease under refinement is not explained yet. The time error is negligible: the static solve on the same mesh
gives 1.2425.

Under spatial refinement:

| passes | $h/a$ | $dt$ | $\lambda_h/J$ at $T = 0.1$ | static |
|---:|---:|---:|---:|---:|
| 2 | 0.76 | 0.0025 | 1.2495 | 1.2425 |
| 3 | 0.38 | 0.0025 | 1.1576 | 1.1548 |
| 4 | 0.19 | 0.0025 | 1.0513 | 1.0497 |
| 5 | 0.096 | 0.0025 | 1.0298 | 1.0280 |
| 5 | 0.096 | 0.00125 | 1.0276 | 1.0280 |

The dynamic multiplier has the error of the static one at every pass, to
within 0.7 %. The support points at the ends of the vessel, near the faces
of the cube, behave differently: there the dynamic error is larger than the
static one (10 % against 3 % at pass 5) and does not change when $dt$ is
halved. Including them, the mean ratio at pass 5 is 1.061 for
$dt = 0.0025$ and 1.059 for $dt = 0.00125$, against 1.030 static. This is
an effect of the end faces, where the vessel meets the boundary data, not of
the time integration.

At pass 5 a step solves a system with 620295 displacement dofs. The inner
solve with the augmented block needs an AMG preconditioner; without it its
iterations double with each refinement (870 per solve at pass 5, fixed in
#463).

## What this means for #379

MetricFlowX removes the two affine interior modes, elastic and viscous, from
the solid and reports the exterior radial stress
$\lambda^\star = J + q_r = -0.0152$ at $T = 0.1$, where
$q_r = 0.0328$ is the removed interior stress. The jump is about
$3.2$ times larger than $\lambda^\star$. If $q_r$ is removed exactly, an
error $e$ of the jump becomes an error $3.2\,e$ of $\lambda^\star$.

- The static jump error at the meshes of #379 ($h/a$ between 0.4 and 0.8) is
  15 to 28 %, which gives 50 to 90 % on $\lambda^\star$. #379 measures 26 to
  72 %, and 55 % on the uniform $16\times64\times64$ mesh ($h/a = 0.38$).
- To reach 5 % on $\lambda^\star$ the jump must be accurate to about 1.5 %,
  which needs $h/a \approx 0.05$ near the wall with first-order convergence.
  The fitted grid of #379 gives 5.9 %; its $h/a$ at the wall is not
  reported there.
- The IDA tolerances of #379 barely change the multiplier, and here the time
  step does not either once the constraint is solved accurately: away from
  the cube faces the dynamic multiplier has the static error. #463
  concerns the iterative solve of the ImmersX Newmark step; MetricFlowX
  integrates with IDA and other linear solvers, and its tolerance study shows
  no comparable effect.

So the multiplier error of #379 is consistent with the first-order spatial
convergence of the jump at $h/a$ near 0.5, amplified by $|J/\lambda^\star|$.
No error source other than the spatial resolution of the wall is needed to
explain it. A convergence study of #379 needs refinement that reaches
$h/a \approx 0.05$ at the wall, not only around the axis.

## Reproducing

`gtests/vessel_multiplier_convergence_01.cc` builds the static cube with the
band refinement above and checks the first two passes and the independence
of the axial strain (`VesselMultiplierValidation`, nightly). The tables used
the same setup with more passes, Release builds and four or eight MPI ranks.
The dynamic runs call `ElasticityProblem` with
`newmark_wall_constraint = true`, as `coupled_elasticity` does, and need
#449 and #463.
