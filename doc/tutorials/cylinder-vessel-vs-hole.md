# A vessel in a cylinder: immersed vs resolved

This tutorial compares two ways of modelling a straight vessel inside a solid
cylinder:

- **resolved**: the vessel is a cylindrical hole in the mesh, and the wall
  displacement is a Dirichlet condition on the hole;
- **immersed**: the mesh fills the whole cylinder, and a tensor-product
  coupling imposes the same wall displacement through a Lagrange multiplier.

Both cases have the same exact solution. The tutorial compares the normal
stress that the tissue exerts on the vessel wall: computed on the hole surface
in the resolved case, and as the vessel pressure in the immersed case. It also
checks what the multiplier itself represents.

## Problem and exact solution

The solid is a cylinder of radius $r_e = 2$ and length 4, with Lamé
coefficients $\lambda = \mu = 1$. The vessel has radius $r_i = 0.2$ and lies
on the cylinder axis. Its wall moves radially by $d_i = 0.1$, and the outer
surface is fixed. With zero axial displacement on the end faces the problem is
plane strain, and the exact displacement is the radial Lamé field

$$
u_r(r) = A\left(\frac{r_e^2}{r} - r\right), \qquad
A = \frac{r_i d_i}{r_e^2 - r_i^2}, \qquad r_i \le r \le r_e.
$$

Its radial stress at the wall, on the tissue side, is

$$
\sigma_{rr}^{\rm out}(r_i) = -2A\left(\mu\frac{r_e^2}{r_i^2} + \lambda + \mu\right)
= -1.030.
$$

In the immersed case the mesh also covers the vessel, and the material inside
is fictitious. The coupling only constrains the wall, so the interior follows
the uniform dilation $u_r = d_i r / r_i$. Its radial stress is

$$
\sigma_{rr}^{\rm in} = 2(\lambda + \mu)\frac{d_i}{r_i} = 2.
$$

## What the multiplier and the pressure are

In the immersed problem the multiplier $\lambda_h$ enters the momentum
equation as an extra load on the wall $\Gamma$:

$$
a(u, v) + \int_\Gamma \lambda_h \cdot v \, ds = (f, v).
$$

Integrate $a(u, v)$ by parts separately inside and outside the vessel, with
$n$ pointing out of the vessel. The wall terms are
$\int_\Gamma (\sigma^{\rm in} - \sigma^{\rm out})\, n \cdot v \, ds$, so

$$
\lambda_h = (\sigma^{\rm out} - \sigma^{\rm in})\, n ,
$$

the **jump** of the traction across the wall, not the tissue stress. For this
problem the jump is $\sigma_{rr}^{\rm out} - \sigma_{rr}^{\rm in} = -1.030 - 2 =
-3.030$.

`VesselElasticityProblem` reports as vessel pressure the mean normal
component of $\lambda_h$ over the wall. With
`Remove interior stress from pressure = true` (the default), it adds back the
interior stress $2(\lambda + \mu) u_n / r_i$ computed from the imposed wall
displacement $u_n$. The pressure is then the tissue-side normal stress
$\sigma_{rr}^{\rm out} = -1.030$. This is the quantity to compare with the
resolved hole, where the mean normal stress $n \cdot \sigma n$ on the hole
surface is $\sigma_{rr}^{\rm out}$ directly. The normal of the hole points out
of the solid, $n = -e_r$, and $n \cdot \sigma n = \sigma_{rr}$ whichever way
$n$ points.

[Pressure on the vessel walls](../concepts/vessel-pressure.md) explains the
pressure for both couplings in detail.

## Input files

The resolved case runs with the `elasticity` application. `cylinder_shell`
builds the hole around the $z$ axis, with boundary ids 0 (hole), 1 (outer
surface), 2 and 3 (end faces):

```{literalinclude} ../../tutorials/cylinder_vessel/hole.prm.in
:language: text
```

The immersed case runs with `coupled_elasticity`. `subdivided_cylinder` builds
the cylinder around the $x$ axis, with boundary ids 0 (hull), 1 and 2 (end
faces). The reduced grid `data/tests/cylinder_vessel_x.vtk.in` is the
vessel axis with four cells and the cell field `vessel_id`, which the pressure
computation needs:

```{literalinclude} ../../tutorials/cylinder_vessel/vessel.prm.in
:language: text
```

Both inputs impose the exact field on the end faces, so the axial
displacement there is zero and the problem stays plane strain. A
no-normal-flux condition on the end faces would describe the same problem.
It currently gives NaN in the solver on locally refined meshes and in
parallel ([issue #458](https://github.com/luca-heltai/immersx/issues/458)).

## Running

Build the two applications and run the configured inputs:

```bash
cmake --build build-debug -j 4 --target elasticity_debug coupled_elasticity_debug
build-debug/elasticity_debug build-debug/tutorials/cylinder_vessel/hole.prm
build-debug/coupled_elasticity_debug build-debug/tutorials/cylinder_vessel/vessel.prm
```

Both applications compute the wall quantity on the final mesh only. To get one
value per refinement level, run each input once per level:

```bash
for k in 1 2 3 4; do
  for case in hole vessel; do
    sed -e "s/^\(  set Initial refinement *\)= .*/\1= $k/" \
        -e "s/^\(  set Output name *\)= $case/\1= ${case}_k$k/" \
        build-debug/tutorials/cylinder_vessel/$case.prm > ${case}_k$k.prm
  done
  build-debug/elasticity_debug hole_k$k.prm
  build-debug/coupled_elasticity_debug vessel_k$k.prm
done
```

The results go to `build-debug/test_output/tutorial-output/cylinder-vessel/`:

- `hole_k<k>_forces.txt`: one row with, for each boundary id, the mean
  traction vector, the integral of $u \cdot n$ and the mean normal stress.
  The column `sigmaNN_0` is the hole wall. On a closed curved surface the mean
  traction vector averages to about zero, so use `sigmaNN_0`, not the
  traction components. As a check, `sigmaNN_2` and `sigmaNN_3` on the end
  faces are the plane-strain axial stress $\sigma_{zz} = -2A\lambda = -0.0101$
  on every mesh.
- `vessel_k<k>_externalPressure.txt` in serial runs, or `.h5` in parallel
  runs: the vessel pressure.

To see the bare multiplier, run the vessel input again with
`Remove interior stress from pressure = false`.

## Results

Q1 elements, uniform refinement, Debug build. Levels 1 to 3 ran serially,
level 4 on 4 ranks.

Resolved hole, mean normal stress on the hole wall (exact $-1.030$), and
displacement errors against the exact field:

| level | cells | dofs | $L^2$ error | $H^1$ error | `sigmaNN_0` | error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96 | 504 | 6.68e-2 | 0.372 | +0.215 | 1.245 |
| 2 | 768 | 3120 | 2.96e-2 | 0.244 | +0.031 | 1.061 |
| 3 | 6144 | 21600 | 1.04e-2 | 0.147 | −0.213 | 0.817 |
| 4 | 49152 | 159936 | 3.10e-3 | 0.081 | −0.474 | 0.556 |

Immersed vessel. The pressure is the default output (exact $-1.030$); the
multiplier is the output with the interior stress not removed (exact jump
$-3.030$):

| level | dofs (displacement + multiplier) | multiplier | pressure | pressure − multiplier | pressure error |
|---:|---:|---:|---:|---:|---:|
| 1 | 675 + 10 | −27.96 | −25.97 | 1.99 | 24.94 |
| 2 | 4539 + 10 | −6.926 | −4.933 | 1.993 | 3.90 |
| 3 | 33363 + 10 | −3.828 | −1.834 | 1.994 | 0.80 |
| 4 | 256035 + 10 | −3.480 | −1.486 | 1.994 | 0.46 |

What the numbers show:

- The multiplier converges to the jump $-3.030$ and the pressure to the
  tissue stress $-1.030$. The two differ at every level by the interior
  stress, 1.994 instead of 2 because the polygonal cross section (32 sides at
  `Refinement level = 3`) gives a mean wall displacement slightly below 0.1.
- Both wall quantities converge to $\sigma_{rr}^{\rm out}$, but at different
  speeds. The hole stress is evaluated from the displacement gradient at the
  wall. With Q1 that is first order, and here it is still pre-asymptotic: the
  stress varies like $1/r^2$, and `cylinder_shell` has no radial grading, so
  even at level 4 the cells at the wall are about half the vessel radius
  across. The multiplier is a reaction of the discrete equations, a
  consistent flux, and its error falls by a factor of 5 to 6 per level up
  to level 3.
- At level 4 the pressure error drops only from 0.80 to 0.46. A likely cause,
  not checked here: the pressure is the mean over the whole vessel, including
  the two ends, where the vessel meets the end faces with their Dirichlet
  data.
- The resolved displacement errors approach the expected rates: from level 3
  to level 4 the $L^2$ error falls by 3.4 (4 for second order) and the $H^1$
  error by 1.8 (2 for first order).

Finer meshes would need a Release build. Graded radial cells, or local
refinement near the wall, would help the resolved wall stress much more than
uniform refinement.

## Next step: the dynamic case

[One vessel in a cube: a manufactured
solution](metric-flow-x-elastodynamics-mms.md) runs the same comparison in
time, with inertia and viscosity, for the MetricFlowX coupling. There the
multiplier is also a traction jump, and removing the fictitious interior
response turns it into the exterior radial stress.
