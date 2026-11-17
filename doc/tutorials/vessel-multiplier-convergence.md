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
3. the displacement near the wall, and the refinement of the centerline;
4. an axial strain along the vessel: the jump does not change, the tissue
   stress does;
5. Newmark steps with inertia and damping: the time error of the multiplier;
6. the dynamic manufactured solution under spatial refinement;
7. meshes fitted to the vessel wall.

The result: the multiplier converges at first order in the size $h$ of the
cells at the wall, relative to the vessel radius $a$, and at second order on
meshes fitted to the wall. Time integration adds
no visible error. The multiplier error reported in #379 is the size this
spatial error predicts at the meshes used there.

## Setting

The cube is $[-\ell,\ell]^3$ with a vessel of radius $a$ on the $x$ axis,
$\lambda = \mu = 1$, tensor-product coupling with the modes 3 and 7 (the mean
radial displacement), a cross section with 32 sides and a centerline with
eight cells unless stated otherwise. The manufactured field is

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
cell, at the wall. Unless stated otherwise the tables give the mean of the
multiplier coefficients (both modes) at the centerline nodes with
$|x| \le \ell/2$, called interior nodes below. The nodes on the faces of the
cube behave differently; see the section on centerline refinement below.

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
$h/a$. A cross section with 64 or 128 sides, four times more quadrature
repetitions, or four times more axial quadrature points change the multiplier
at pass 3 by less than 0.005.

The same study with the proportions of the MetricFlowX cube,
$\ell = 0.120685$, $a = 0.00987$ ($a/\ell = 0.082$) and $g = 1$ (exact jump
$-6$):

| passes | cells | dofs | $h/a$ | interior | error | all nodes | $L^2$ error of $u$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5440 | 20211 | 0.76 | −7.455 | 24 % | −7.687 | 3.5e−5 |
| 3 | 15296 | 54927 | 0.38 | −6.929 | 15 % | −7.040 | 2.3e−5 |
| 4 | 51136 | 174939 | 0.19 | −6.298 | 5.0 % | −6.325 | 6.4e−6 |
| 5 | 183744 | 620295 | 0.096 | −6.168 | 2.8 % | −6.178 | 3.2e−6 |

The relative error depends on $h/a$ much more than on $a/\ell$:

```{figure} ../_static/vessel-multiplier-convergence/convergence.svg
:alt: Relative error of the interior multiplier against h/a on log-log axes, for the static cube with a/l = 0.2, the static cube with a/l = 0.082, and the Newmark runs with a/l = 0.082. All three fall with slope about one.

Relative error of the interior multiplier against $h/a$. The Newmark curve
is the value at $T = 0.1$ with $dt = 0.0025$.
```

The displacement error falls more slowly than $h$ in $H^1$ (0.0090,
0.0072, 0.0045, 0.0036): the refinement is only near the wall, and the
coarse cells away from it set the $H^1$ error.

## Displacement at the wall

The radial displacement $u_r$ at $x = 0$, averaged over 64 angles, divided by
the exact value:

| $h/a$ | $r = a/2$ | $0.9a$ | $0.97a$ | $a$ | $1.03a$ | $1.1a$ | $2a$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.76 | 1.248 | 1.107 | 1.030 | 0.997 | 1.025 | 1.076 | 1.221 |
| 0.38 | 1.148 | 1.090 | 1.026 | 0.997 | 1.027 | 1.087 | 1.136 |
| 0.19 | 1.045 | 1.046 | 1.026 | 0.995 | 1.015 | 1.035 | 1.036 |
| 0.096 | 1.025 | 1.021 | 1.005 | 0.980 | 1.000 | 1.014 | 1.015 |

```{figure} ../_static/vessel-multiplier-convergence/radial-displacement.svg
:alt: Ratio of the computed to the exact radial displacement against r/a for four meshes. Away from r = a the ratio falls toward one under refinement; at r = a it dips below one.

$u_r/u_r^{\rm exact}$ at $x = 0$ for the four meshes.
```

Away from the wall the error falls with $h$. At $r = a$ the ratio decreases
under refinement, from 0.997 to 0.980. That is expected: the exact $u_r$ has
a kink at the wall, slope $+g$ inside and $-g$ outside, and a $Q_1$ field
cannot follow the corner unless a mesh plane passes through it. It cuts the
peak by up to $g h/2$, by about $g h/4$ on average over the angles: 2.4 % at
$h/a = 0.096$. On the coarse meshes the whole field is 10 to 25 % too large
and hides the cut. The constraint is met (the mean over the cross-section
polygon of $u \cdot n$ equals $g a$); the values just inside and outside the
wall are a little above the exact ones to compensate.

## Centerline refinement

The centerline is the 1D grid of the multiplier. With eight cells its nodes
are $\ell/4 = 0.030$ apart, much more than the cells at the wall. Refining it
at fixed 3D mesh, static, interior mean:

| $h/a$ | 8 cells | 16 cells | 32 cells | 64 cells |
|---:|---:|---:|---:|---:|
| 0.76 | −7.455 | −7.436 | fails | fails |
| 0.38 | −6.929 | −6.919 | −6.917 | fails |
| 0.19 | −6.298 | −6.295 | −6.295 | −6.295 |
| 0.096 | −6.168 | −6.178 | −6.178 | −6.178 |

The interior multiplier does not depend on the centerline. When a centerline
cell is about as small as the 3D cells at the wall (32 cells at
$h/a = 0.76$, 64 cells at $h/a = 0.76$ and $0.38$) the solve does not
converge: the multiplier space is then too rich for the displacement trace
(an inf-sup failure). A centerline cell of at least about twice the 3D cell
works.

The nodes on the two faces of the cube, where the vessel meets the boundary
data, are different. Value of $\lambda_h/J$ at pass 5 ($h/a = 0.096$):

| centerline cells | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| face node | 1.037 | 1.060 | 1.107 | 1.244 |
| next node | 1.026 | 1.024 | 1.017 | 0.996 |
| centre ($x = 0$) | 1.028 | 1.030 | 1.030 | 1.030 |

```{figure} ../_static/vessel-multiplier-convergence/centerline.svg
:alt: Multiplier over exact jump along the vessel at h/a = 0.096 for 8, 16, 32 and 64 centerline cells. The curves agree at about 1.03 in the interior and spike at the two faces of the cube, more for finer centerlines.

$\lambda_h/J$ along the vessel at $h/a = 0.096$, static.
```

The face node gets worse as the centerline is refined, and the neighbouring
nodes start to alternate. With the ratio of centerline cell to 3D cell fixed
at eight, the face node goes from 1.262 to 1.114 to 1.107 at passes 3, 4,
5: it does not converge. Its test function has half its support, and its
row of the constraint couples mostly to displacement degrees of freedom
fixed by the boundary data. Averages over the whole vessel therefore depend
on the centerline at the ends; the interior values do not.

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
amplitude growing like $1/dt^2$: at $h/a = 0.76$, the mean of
$\lambda_h/J$ over all nodes at $T$ was 1.21 for $dt = 0.0025$ and $-10.4$
for $dt = 0.00125$, alternating in sign from step to step. PR
[#463](https://github.com/luca-heltai/immersx/pull/463) fixes the stopping
test. With it, at $h/a = 0.76$, $\lambda_h/J$ at $T = 0.1$ is

| $dt$ | 0.01 | 0.0025 | 0.00125 | 0.000625 |
|---|---:|---:|---:|---:|
| $\lambda_h/J$ | 1.2471 | 1.2495 | 1.2474 | 1.2468 |

```{figure} ../_static/vessel-multiplier-convergence/time-step.svg
:alt: Multiplier over exact jump against time at h/a = 0.76 for four time steps. After an initial oscillation all curves settle at about 1.247, just above the static value 1.2426.

$\lambda_h/J$ against $t$ at $h/a = 0.76$, interior nodes. The dotted line is
the static solve on the same mesh.
```

An oscillation from the equilibrium start decays over the first steps; for
$t \ge 0.05$, $\lambda_h/J$ stays within 1.246–1.253 for $dt \le 0.0025$.
The time error is negligible: the static solve on the same mesh gives 1.2425.

Under spatial refinement:

| passes | $h/a$ | $dt$ | $\lambda_h/J$ at $T = 0.1$ | static |
|---:|---:|---:|---:|---:|
| 2 | 0.76 | 0.0025 | 1.2495 | 1.2425 |
| 3 | 0.38 | 0.0025 | 1.1576 | 1.1548 |
| 4 | 0.19 | 0.0025 | 1.0513 | 1.0497 |
| 5 | 0.096 | 0.0025 | 1.0298 | 1.0280 |
| 5 | 0.096 | 0.00125 | 1.0276 | 1.0280 |

The dynamic multiplier has the error of the static one at every pass, to
within 0.7 %. What the dynamic terms change, at pass 3 with $dt = 0.0025$
($\lambda_h/J$ of mode 3 at the centre node and at a face node, at
$T = 0.1$ for the Newmark runs):

| run | centre node | face node |
|---|---:|---:|
| static | 1.154 | 1.262 |
| Newmark | 1.152 | 1.322 |
| Newmark, $\rho = 10^{-6}$ | 1.150 | 1.321 |
| Newmark, $h = 0$ | 1.158 | 1.288 |

Inertia changes nothing. In the interior the multiplier has the static value.
At the face nodes the dynamic error is larger, about half of the excess
comes from the axial strain, and it does not change when $dt$ is halved
(10.1 % and 9.7 % at pass 5 for $dt = 0.0025$ and $0.00125$, mean over the
nodes with $|x| > \ell/2$). The mean over all
nodes, including the faces, is therefore larger than the static one: 1.061
at pass 5 against 1.030.

## Fitted meshes

The first order comes from the kink of the exact field at the wall, which an
unfitted $Q_1$ mesh cuts. On a mesh whose faces lie on the wall, $Q_1$
represents the kink. The `fitted_vessel_mesh` application (see
[One vessel in a cube](metric-flow-x-elastodynamics-mms.md)) writes such a
mesh of the same cube; the static problem is solved on it unchanged, with
`Domain type = file`, no band refinement, and the same exact field and
material inside and outside:

| generator refinement | cells | dofs | wall sides | multiplier | error |
|---:|---:|---:|---:|---:|---:|
| 0 | 2016 | 7527 | 8 | −6.873 | 14.6 % |
| 1 | 16128 | 54075 | 16 | −6.228 | 3.8 % |
| 2 | 129024 | 409395 | 32 | −6.048 | 0.79 % |

The error falls by four per refinement, second order, against two per pass
on the unfitted meshes. At the same cost the fitted mesh is four times more
accurate: 3.8 % against 15 % at 54 000 dofs, and 0.79 % at 409 000 dofs
against 2.8 % at 620 000.

```{figure} ../_static/vessel-multiplier-convergence/fitted.svg
:alt: Relative error of the interior multiplier against the number of displacement dofs on log-log axes, for unfitted meshes refined near the wall and for fitted meshes. The fitted error is lower and falls faster.

Relative error of the interior multiplier against the number of
displacement dofs, static.
```

The radial displacement at $r = a$ is within 0.5 % on the fitted meshes
at refinements 1 and 2, with no dip, and within 0.1 % at $r = a/2$ and $2a$
at refinement 2.

## Solver

Two fixes in [#463](https://github.com/luca-heltai/immersx/pull/463) were
needed for these runs:

- FGMRES stops on the norm of the whole residual. In a Newmark step the
  momentum row is scaled by $1/(\beta\,dt^2)$, and the constraint row holds
  only the small residual of the predictor, so the momentum row alone decided
  when FGMRES stopped. The constraint row is now weighted by the same factor,
  and the preconditioner undoes the weight.
- The inner CG solve with the augmented block $K + \gamma B^T W^{-1} B$ had
  no preconditioner. Its iterations doubled with each refinement: about 70,
  185, 400, 870 per solve at passes 2 to 5. The augmentation has the rank of
  the multiplier space, so the AMG of $K$ preconditions it: 21 iterations at
  pass 3 and 47 at pass 5. A pass-5 run with 80 steps takes 40 minutes on
  eight ranks instead of about two hours, and the multiplier does not change.

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
  which needs $h/a \approx 0.05$ near the wall with first-order convergence
  on unfitted meshes.
- On fitted meshes the jump converges at second order. The fitted grids of
  #379 are the ones in the table above, at the same refinement. The
  predicted bias of $\lambda^\star$, 3.2 times the jump error, is 47 % at
  refinement 0 and 12 % at refinement 1. #379 measures a mean bias of 44.7 %
  and 5.0 %, the latter after co-refinement added half again as many
  cells. The unfitted grid of #379 ($h/a \approx 0.38$ at the wall) gives a
  predicted 50 % against a measured 63 %.
- The IDA tolerances of #379 barely change the multiplier, and here the time
  step does not either once the constraint is solved accurately: away from
  the cube faces the dynamic multiplier has the static error.
- #379 reports the $L^2$ error of the multiplier over the whole centerline,
  including the ends on the cube faces, and the mean multiplier. Their
  difference is small there (55 % and 53 % on the finest unfitted grid), so
  the face nodes do not dominate yet. On finer centerlines they add an error
  that does not converge; interior values avoid it.
- The trace identity that #379 found with two solid axial cells per flow
  cell is the inf-sup failure of the centerline refinement section: a
  multiplier grid as fine as the 3D mesh.

So the multiplier error of #379 is the spatial error of the jump at the
wall, amplified by $|J/\lambda^\star|$: first order in $h/a$ on unfitted
meshes, second order on fitted ones. No error source other than the
resolution of the wall is needed to explain the measured values on either
kind of grid. A convergence study of #379 needs fitted meshes, or unfitted
refinement that reaches $h/a \approx 0.05$ at the wall, not only around the
axis, and a centerline cell of at least twice the wall cell.

## Reproducing

The input files are in `tutorials/vessel_multiplier/`: the static case,

```{literalinclude} ../../tutorials/vessel_multiplier/static_cube.prm.in
:language: text
```

and the dynamic one, which adds the density and damping, the time
parameters, and time-dependent data:

```{literalinclude} ../../tutorials/vessel_multiplier/dynamic_cube.prm.in
:language: text
:lines: 13-45
```

Its wall data are `Reduced right hand side = 0.0098700932*t^3;
0.0098700932*t^3`. `vessel_x.vtk.in` is the centerline with eight cells. The study used two
small drivers built against the library. They read these files with
`initialize_parameters`, refine the band around the wall, solve
(`solve_static`, or `initialize_newmark_at_equilibrium` and `solve_newmark`
with `newmark_wall_constraint = true`, as `coupled_elasticity` does), and
write the multiplier at the centerline nodes. Their sources are attached to
[#379](https://github.com/luca-heltai/immersx/issues/379). The dynamic driver
needs #449 and #463. All runs used Release builds and eight MPI ranks.

The fitted runs use the static input with `Domain type = file`,
`Grid generator` set to the mesh written by
`fitted_vessel_mesh --refinement N`, `Initial refinement = 0`, and the radius
of that mesh, $a = 0.009870093245078749$.

`gtests/vessel_multiplier_convergence_01.cc` builds the static cube with the
band refinement above and checks the first two passes and the independence
of the axial strain (`VesselMultiplierValidation`, nightly).
