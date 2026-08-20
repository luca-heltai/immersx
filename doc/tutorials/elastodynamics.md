# Elastodynamics

This tutorial explains the standalone `ElastodynamicsSolver` application. It
is the first-order-in-time sibling of the existing `ElasticityProblem`: it
solves uncoupled, full-dimensional linear elastodynamics and keeps the
displacement and velocity as separate state fields on the same vector-valued
finite-element space.

The tutorial inputs are manufactured-solution convergence tests for strong
Dirichlet data, the adapted Neumann MMS field, and Kelvin--Voigt damping:

- `tutorials/elastodynamics/strong_dirichlet.prm`
- `tutorials/elastodynamics/neumann.prm`
- `tutorials/elastodynamics/kelvin_voigt.prm`

The `ElastodynamicsSolver` deliberately has no immersed coupling, particles,
Lagrange multipliers, weak boundary penalty, or SUNDIALS dependency. The
existing `ElasticityProblem` and its tutorial remain the reference for those
features.

## What Problem Is Solved?

The fundamental state is

```{math}
y = [d,v],
```

where $d$ is displacement and $v$ is velocity. The first-order system is

```{math}
\dot d-v=0,
```

```{math}
\rho\dot v-\nabla\cdot\sigma(d)-\nabla\cdot\sigma_v(v)=f.
```

The elastic stress is the linear isotropic law

```{math}
\sigma(d)=2\mu\,\varepsilon(d)+\lambda\,\operatorname{div}(d)I,
\qquad
\varepsilon(d)=\tfrac12(\nabla d+\nabla d^T).
```

The optional Kelvin--Voigt stress is

```{math}
\sigma_v(v)=2\eta_s\,\varepsilon(v)+\eta_b\,\operatorname{div}(v)I.
```

After spatial discretization, the solver keeps the operators separate:

```{math}
M\dot d-Mv=0,
\qquad
M\dot v+Kd+Dv=f.
```

Here the same consistent mass matrix $M$ is used for the weak kinematic
equation and for the physical velocity equation. The separate $M$, $K$, and
$D$ operators are exposed for a future semantic residual and SUNDIALS
adapter.

## Where The Implementation Lives

Main files:

- `apps/app_elastodynamics.cc`
- `include/elastodynamics.h`
- `source/elastodynamics.cc`
- `gtests/elastodynamics_01.cc`

The public classes are `ElastodynamicsParameters` and
`ElastodynamicsSolver`. The name is intentional: the repository already has
an `ElasticityProblem`, and this is a separate uncoupled first-order
application.

Run a parameter file with:

```bash
./build/elastodynamics[_debug] tutorials/elastodynamics/<input_file.prm>
```

The application reads the root-level `dimension` and `space dimension`
parameters. This tutorial uses full-dimensional 2D inputs; the application
also supports the corresponding full-dimensional 3D instantiation.

## Spatial State And Time Integration

One vector-valued `FESystem` based on `FE_Q` and one `DoFHandler` represent the
physical vector space $V_h$. Displacement and velocity are separate algebraic
vectors on $V_h$:

```text
V_h: one vector FE space

displacement: vector on V_h
velocity:     vector on V_h
```

The standalone driver uses backward Euler. For a step of size $\Delta t$,

```{math}
\frac{M}{\Delta t}d^{n+1}-Mv^{n+1}
  =\frac{M}{\Delta t}d^n,
```

```{math}
Kd^{n+1}+\left(\frac{M}{\Delta t}+D\right)v^{n+1}
  =f^{n+1}+\frac{M}{\Delta t}v^n.
```

The two fields are solved together with a distributed GMRES solve and a
Jacobi preconditioner. This matrix is only the standalone time-step driver;
the continuous semi-discrete operators remain available separately.

## Common Structure Of The Test Files

Each parameter file contains:

- root-level dimension selection;
- the `Elastodynamics` subsection with mesh, material, time, and solver data;
- vector-valued parsed functions for force, boundary data, and initial state;
- an `Exact solution` function for the convergence table;
- an `Error` subsection with the requested norms and rate calculation.

All three cases use five global mesh levels, one backward-Euler step, and
strong homogeneous or manufactured Dirichlet constraints. The error tables
measure the displacement error in the vector field.

## Test 1: Strong Dirichlet MMS

File: `tutorials/elastodynamics/strong_dirichlet.prm`

This is the baseline first-order elastodynamics convergence test. It uses the
static manufactured displacement field inherited from the bulk elasticity
example, zero velocity, and the corresponding body force. The one backward
Euler step exercises the coupled kinematic/dynamic block system while the
exact solution is time-independent.

```{literalinclude} ../../tutorials/elastodynamics/strong_dirichlet.prm
:language: bash
```

## Test 2: Adapted Neumann MMS Field

File: `tutorials/elastodynamics/neumann.prm`

This input retains the smooth manufactured displacement and body-force field
from the elasticity Neumann example. The standalone solver currently supports
essential Dirichlet constraints only, so all boundary ids are constrained
strongly here. The case is therefore an operator and convergence fixture for
`ElastodynamicsSolver`, not a test of a natural Neumann boundary operator.

```{literalinclude} ../../tutorials/elastodynamics/neumann.prm
:language: bash
```

## Test 3: Kelvin--Voigt Damping

File: `tutorials/elastodynamics/kelvin_voigt.prm`

This case activates the viscous operator with
`Damping shear = 0.1` and `Damping bulk = 0`. The manufactured state is

```{math}
\phi(x,y)=\begin{bmatrix}\sin(\pi x)\sin(\pi y)\\0\end{bmatrix},
\qquad
d=t\phi,
\qquad
v=\phi.
```

The body force is assembled consistently with $Kd+Dv$. Since the exact
displacement is linear in time, backward Euler introduces no temporal
truncation error for this fixture; the table primarily measures the spatial
discretization and exercises the Kelvin--Voigt matrix $D$.

```{literalinclude} ../../tutorials/elastodynamics/kelvin_voigt.prm
:language: bash
```

## Quick Comparison Table

| Test file | Boundary treatment | Material model | Time setup | Verification target |
| --- | --- | --- | --- | --- |
| `strong_dirichlet.prm` | Strong Dirichlet on 0,1,2,3 | Linear elastic | One backward-Euler step | Baseline first-order block and spatial convergence |
| `neumann.prm` | Strong Dirichlet adaptation on 0,1,2,3 | Linear elastic | One backward-Euler step | Smooth vector MMS and operator convergence |
| `kelvin_voigt.prm` | Homogeneous strong Dirichlet | Linear elastic + Kelvin--Voigt | One backward-Euler step | Damping operator and spatial convergence |

## Convergence Tables From The Current Runs

The following tables were produced by the Debug and Release applications. The
same values were obtained with two MPI ranks up to the displayed precision.

### strong_dirichlet.prm

```text
cells dofs      d_L2_norm      d_Linfty_norm      d_H1_norm
    4   18 1.94298208e-01    - 3.26541871e-01    - 1.15519249e+00    -
   16   50 5.51240854e-02 2.47 1.24582522e-01 1.89 5.52213788e-01 1.44
   64  162 1.42272441e-02 2.30 3.45490910e-02 2.18 2.71190077e-01 1.21
  256  578 3.58529948e-03 2.17 8.86317343e-03 2.14 1.34913787e-01 1.10
 1024 2178 8.98114929e-04 2.09 2.23012664e-03 2.08 6.73695281e-02 1.05
```

### neumann.prm

```text
cells dofs      d_L2_norm      d_Linfty_norm      d_H1_norm
    4   18 2.65238315e-01    - 3.26541871e-01    - 1.57606518e+00    -
   16   50 7.54244328e-02 2.46 1.24676555e-01 1.88 7.55304694e-01 1.44
   64  162 1.94780175e-02 2.30 3.45589742e-02 2.18 3.71168137e-01 1.21
  256  578 4.90908558e-03 2.17 8.86438880e-03 2.14 1.84681743e-01 1.10
 1024 2178 1.22974417e-03 2.09 2.23027938e-03 2.08 9.22250524e-02 1.05
```

### kelvin_voigt.prm

```text
cells dofs      d_L2_norm      d_Linfty_norm      d_H1_norm
    4   18 1.77858560e-03    - 3.01501830e-03    - 1.06686801e-02    -
   16   50 5.07742516e-04 2.45 1.16787723e-03 1.86 5.14549110e-03 1.43
   64  162 1.31274413e-04 2.30 3.24966823e-04 2.18 2.53323233e-03 1.21
  256  578 3.30962284e-05 2.17 8.34357706e-05 2.14 1.26105407e-03 1.10
 1024 2178 8.29151213e-06 2.09 2.09981918e-05 2.08 6.29810849e-04 1.05
```

The expected rates are approximately two in the L2 and maximum norms and one
in the H1 norm for the degree-one vector finite element.

## Future Use

The application is intentionally independent of the semantic residual core.
Its separate displacement and velocity fields and its public $M$, $K$, and
$D$ operators are intended to be adapted later to `SemiDiscreteModel` and
SUNDIALS. Fluid-solid coupling, moving geometry, nonlinear materials, and
interface interactions belong to future layers and are not part of this
tutorial.
