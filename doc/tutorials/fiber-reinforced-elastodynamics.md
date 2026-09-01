# Full-order fiber-reinforced elastodynamics

This tutorial implements a first coupled application with two independent,
nonmatching, full-dimensional Problems:

```text
matrix: ElastodynamicsSolver<d,d>
fiber:  ElastodynamicsSolver<d,d>
```

The canonical input is
`tutorials/fiber_reinforced_elastodynamics/parameters.prm.in`:

```{literalinclude} ../../tutorials/fiber_reinforced_elastodynamics/parameters.prm.in
:language: ini
```

A Debug run is:

```bash
./build/fiber_reinforced_elastodynamics_debug \
  build/tutorials/fiber_reinforced_elastodynamics/parameters.prm
```

## Geometry and material meaning

The 2D matrix occupies $[-1,1]^2$. The fiber is a thin rectangle strictly
inside it and is meshed independently, so the grids do not match. Both spaces
are full-dimensional vector finite-element spaces. The word *embedded* refers
to the geometric inclusion $\Omega_f\subset\Omega$, not to a reduced
`<1,3>` finite-element space.

The matrix coefficients fill the entire background domain. The fiber Problem
is an additive/excess contribution supported in the fiber region. Its density
and Lamé coefficients therefore describe the excess material contribution;
they do not silently mean “matrix material plus a second complete solid” in the
same region. The tutorial uses positive excess density so the free-fiber
effective block remains positive definite for the existing Schur solver.

## Coupling and time stepping

The matrix and fiber velocity representations are typed
`VectorFiniteElementRepresentation`s. `VectorLagrangeMultiplierInteraction`
uses the fiber vector FE space as the multiplier space and assembles

```{math}
C_{ij}=\int_{\Omega_f}
  \boldsymbol\phi_i^m\cdot\boldsymbol\psi_j^\lambda\,dx,
\qquad
Q_{jk}=\int_{\Omega_f}
  \boldsymbol\psi_j^\lambda\cdot\boldsymbol\phi_k^f\,dx.
```

Here $Q$ is an interaction pairing matrix, not the fiber’s physical mass
matrix. Fiber quadrature points are located in the distributed matrix mesh by
the existing `ParticleCoupling` search. Vector basis values are evaluated by
component, so an x basis function cannot couple to a y basis function.

The application driver, rather than either Problem’s standalone time loop,
owns the five-field IDA solve. Eliminating displacement gives

```{math}
A_m v_m^{n+1}+C\lambda^{n+1}=r_m,
```

```{math}
A_f v_f^{n+1}-Q^T\lambda^{n+1}=r_f,
```

```{math}
C^T v_m^{n+1}-Qv_f^{n+1}=0,
```

with $A=M/\Delta t+D+\Delta t K$ and
$r=f^{n+1}+Mv^n/\Delta t-Kd^n$. The existing
`LagrangeMultiplierSchurSolver` solves this block system. After the solve,
$d^{n+1}=d^n+\Delta t\,v^{n+1}$, and the driver records both velocity and
displacement compatibility diagnostics.

The initial displacement must already satisfy
$C^Td_m^0-Qd_f^0=0$; the tutorial uses the compatible zero state. The current
application assumes fixed reference geometry. Geometry versions remain part of
the representation/interaction seam so a future moving FSI relation can
invalidate assembled transfer data explicitly. Nonzero moving Dirichlet data
and adaptive refinement during the coupled run are not implemented here.

Output ownership follows the semantic composition boundary. The two Problems
accept their solved displacement and velocity states and write their native
mesh output. The vector Interaction accepts the multiplier state and writes
`velocity_multiplier` on the second (fiber) representation mesh; it is not
written through either Problem or through the execution adapter. In the IDA
path, the handoff occurs for the initial state, configured accepted output
states, and final accepted state, never from residual or Jacobian trial
evaluations.

## Semantic five-field execution

The application composes the two Problems, Representations, and Interaction
directly through the public `IDAAdapter` with one private semantic execution
layout containing

```text
matrix.displacement       differential
matrix.velocity            differential
fiber.displacement         differential
fiber.velocity             differential
fiber_coupling.lambda      algebraic
```

The two Elastodynamics contributors use caller-selected prefixes and distinct
`HistoryGroupId`s. The vector Interaction registers the algebraic multiplier
field and adds `C lambda`, `-Q^T lambda`, and `C^T v_matrix - Q v_fiber` to
the appropriate rows. Each Problem and the Interaction retain their own
accepted state and native output. The application’s accepted-state callback
hands displacement and velocity back to both Problems and the multiplier back
to the Interaction before output is written. Focused tests compare all five
residual rows and all five Jacobian rows with the native `M`, `K`, `D`, `C`,
and `Q` actions.

## Scope and future path

This is a full-order fiber application. It does not implement the future
reduced `<1,3>` TensorProduct fiber mechanics, moving geometry, FSI, adaptive
coupled refinement, or partitioned execution.

The MPI path is the same distributed point-search path used by the existing
coupling infrastructure. A meaningful two-rank transient test exercises the
nonmatching vector interaction and coupled algebra.
