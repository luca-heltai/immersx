# Elastostatic–MetricFlowX coupling: onboarding

## Goal and sequence

We want to couple vascular flow, computed by MetricFlowX, with the elastostatic
deformation of the surrounding tissue, computed by ImmersX.

MetricFlowX determines vessel cross-sectional areas. Their change determines a
radial wall displacement, imposed on the solid through a weak constraint. The
associated Lagrange multiplier measures the solid reaction and supplies the
pressure feedback to the flow problem.

Work through these stages:

1. Run the two applications independently.
2. Connect them through files.
3. Iterate the exchange to convergence.
4. Transfer the same procedure into memory.
5. Assemble the coupled residual and solve it with one `IDAAdapter`.

Each stage should produce a small reproducible example that remains a reference
for the next stage. Start with one straight vessel inside a simple elastic
domain. Add branches and larger geometries after the single-vessel case works.

This is a development plan. The existing applications and C++ interfaces provide
parts of the workflow; a complete file-based restart and exchange driver still
needs to be implemented and checked.

## Quantities and conventions

Let $s$ denote the centerline coordinate and $t$ physical time.

| Quantity | Meaning | SI units |
| --- | --- | --- |
| $A(s,t)$ | Cross-sectional area computed by MetricFlowX | $\mathrm{m}^2$ |
| $A_0(s)$ | Area in the solid's reference configuration | $\mathrm{m}^2$ |
| $R_0(s)$ | Reference radius | $\mathrm{m}$ |
| $\delta R(s,t)$ | Prescribed radial wall increment | $\mathrm{m}$ |
| $u(x,t)$ | Solid displacement | $\mathrm{m}$ |
| $\lambda(s,t)$ | Wall-constraint multiplier | Pressure after verifying modal normalization |

For circular sections,

$$
R_0=\sqrt{\frac{A_0}{\pi}},\qquad
\delta R(A)=\sqrt{\frac{A}{\pi}}-\sqrt{\frac{A_0}{\pi}}.
$$

Use this full relation. The approximation
$\delta R\simeq(A-A_0)/(2\pi R_0)$ is useful as a small-perturbation check.
Keep $A_0$ fixed during coupling; do not replace it with the previous
iteration's area.

Distinguish the geometric reference from the tube-law parameters. The
MetricFlowX implementation inspected for this guide uses the local diastolic
area `a_d` in its pressure law, while the ImmersX coupled example derives its
geometric radius from `a0`. Make those references consistent in the chosen case;
do not assume the two names are interchangeable.

## Stage 1: run each application independently

### MetricFlowX

Use the [MetricFlowX repository](https://github.com/luca-heltai/metric-flow-x)
and the [ImmersX repository](https://github.com/luca-heltai/immersx). Record the
commit used for each application so the exchange can be reproduced.

Build from the MetricFlowX repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

The documented invocation form is:

```bash
./build/metric_flow_x build/parameters/aortic.prm
```

This illustrates how to pass a CMake-generated parameter file. Prepare a
dedicated single-vessel input for coupling instead of starting with the full
aortic network. First check equilibrium, then apply a nonzero inlet perturbation.

Set explicit output directories. Save area and velocity along the whole vessel,
the physical time of each result, vessel and cell identifiers, and the data
needed to reconstruct the complete flow state for restart.

Midpoint probe CSV files are useful for plots, but do not describe the spatial
field needed for coupling. Parallel visualization output contains multiple
pieces: read the complete dataset when converting it.

### Elasticity

For the initial exchange, use the `elasticity` application with its
`TensorProduct` path in static mode. From the ImmersX repository root:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --target elasticity_debug -j2
./build-debug/elasticity_debug \
  "$PWD/build-debug/data/tests/elasticity_tensor_product_coupling_3d.prm"
```

The input is generated from
`data/tests/elasticity_tensor_product_coupling_3d.prm.in`. It checks the
tensor-product path; adapt its load to the required radial displacement.
Use generated build-tree inputs, with explicit output paths.

Recognize these settings:

```text
subsection Immersed Problem
  set Coupling type = TensorProduct
  set Prescribed vessel pressure = false
end
```

We prescribe displacement and compute its reaction. Prescribing pressure solves
the opposite problem from the one needed here.

Before connecting MetricFlowX:

1. Prescribe a constant, nonzero radial increment.
2. Check that the weak wall constraint is satisfied.
3. Check the reaction's sign and magnitude.
4. Ensure the outer solid boundary conditions remove rigid-body motions.

The lift's `Thickness` parameter sets the geometric scale of the reconstructed
cross section. Do not automatically interpret it as vessel-wall thickness.

## Stage 2: complete one exchange through files

```{mermaid}
flowchart TD
  F["MetricFlowX: area A and complete flow state"] --> P["Python: area to radial increment"]
  P --> E["elasticity: TensorProduct displacement constraint"]
  E --> L["Export and interpret the Lagrange multiplier"]
  L --> R["MetricFlowX: saved flow state and pressure feedback"]
```

### Export areas

Export $A(s,t)$ on the centerline, preserving geometry and the association
between values, vessels, and cells. Record units and time in the exchange data.

DG fields can have different traces at the same physical junction or interface.
Coordinates alone may merge distinct values. Preserve the incident cell identity
as well as the vessel identifier.

### Convert area to radial increment

The Python script must:

1. Read $A$ and the reference $A_0$.
2. Reject nonfinite values and nonpositive areas.
3. Compute $\delta R$.
4. Write an input readable by the tensor-product path.

A first format to implement and check is a VTK centerline with a scalar
`delta_r` field. Match the reader configuration to the chosen point/cell field
association and the software version in use. If projection is needed, document
the source and target FE spaces and check its error.

The relevant legacy parameters are `Reduced grid name`, `Input file fields`,
`Reduced right hand side`, and `Selected indices`. Imported-field expressions
require a compatible build, including SymEngine where used by this path.

The vascular example selects modes `3, 7` for radial reconstruction. Verify
their meaning and normalization before mapping `delta_r` to the prescribed
modal coefficients. The geometric target is

$$
P(A)=\delta R(A)n,
$$

where $n$ is the radial unit direction on the reconstructed wall. A scalar
centerline value is not directly a three-dimensional Cartesian vector.

Minimum conversion checks:

- $A=A_0$ gives $\delta R=0$.
- $A=4A_0$ gives $\delta R=R_0$.
- Invalid input produces an explicit error.
- Cell/value associations survive the conversion.

### Solve the solid and export the multiplier

For a prescribed area, solve schematically

$$
Ku+B\lambda=f,\qquad B^Tu=g(A).
$$

Here $g(A)$ is a dual-space vector assembled from radial displacement, test
functions, and the lift's geometric weights. It is not simply the list of nodal
$\delta R$ values.

Save displacement, multiplier, equilibrium residual, and constraint residual.
Inspect the legacy output before using it as an interface: do not assume
`lambda.txt` contains the complete tensor-product coefficient field. Add a
dedicated export if needed, including FE space, geometry, selected modes, and
normalization.

The tensor-product multiplier can have several modal coefficients. Identify the
radial pressure component and its normalization before sending it to the flow.
The current monolithic example documents

$$
\lambda=p_{\mathrm{ext}}-p_{\mathrm{int}}^{\mathrm{fitt}},
$$

with a $+B\lambda$ solid reaction and negative flow coupling. Check this
convention for the legacy path with a simple radial test; renaming raw modal
coefficients to pressure does not establish the conversion.

### Restart MetricFlowX

To advance from $t_n$ to $t_{n+1}$, restart from the accepted state at
$t_n$ and apply the assigned pressure feedback.

Save the complete state: area, velocity, and any additional terminal-condition,
trace, or constraint states. An area visualization file alone does not establish
a complete restart.

The inspected MetricFlowX C++ API provides `set_external_pressure_provider()`.
A complete CLI workflow reading both a checkpoint and a pressure field was not
identified. The reader, restart mechanism, and configuration are explicit tasks
for this stage. Verify restart by comparing a split run with a continuous run.

## Choose one pressure-feedback operator

The available pressure interfaces have different effects.

MetricFlowX's native external-pressure provider changes physical pressure and
also contributes to numerical fluxes, junctions, and some boundary conditions.
Its pressure convention is
$p_{\mathrm{internal}}=p_{\mathrm{tube}}+p_{\mathrm{external}}$.

The current ImmersX coupled application supplies native external pressure

$$
p_{\mathrm{external}}=-\lambda.
$$

This includes native volume, face, boundary, and junction pressure terms.
Use the same native provider convention in the file-based experiment.

Before comparing stages, fix one definition of the feedback operator
$C_p\lambda$, including sign, units, physical factors, and integration
weights. File-based and in-memory transfers must apply that same operator.

## Stage 3: iterate the two-way exchange

A single exchange uses lagged data. Iterate to obtain consistency at the new
time. Given the accepted flow state $z_n$:

1. Choose an initial pressure iterate $\lambda^{(0)}$.
2. Solve flow on $[t_n,t_{n+1}]$ using $\lambda^{(k)}$.
3. Convert $A^{(k+1)}$ to $\delta R^{(k+1)}$.
4. Solve elasticity to obtain $\widehat\lambda^{(k+1)}$.
5. Update the pressure, using relaxation if needed:

   $$
   \lambda^{(k+1)}=(1-\omega)\lambda^{(k)}
   +\omega\widehat\lambda^{(k+1)},\qquad 0<\omega\leq1.
   $$

6. Check area and multiplier changes, equilibrium, and the wall constraint.

Every attempt on the same time interval starts from $z_n$. Restarting from
the previous attempt's final state would advance physical time again.
Accept the new state only after convergence. If iteration fails, retain
$z_n$ and reconsider the time step or iteration strategy. Use separate
output directories for time steps and coupling iterations.

## Stage 4: transfer the procedure into memory

Remove files while keeping equations and iteration unchanged. Replace the
Python conversion with evaluation of $A\mapsto\delta R(A)$, and evaluate
the multiplier directly in the relevant FE space. Keep diagnostic exports so
the file-based and in-memory paths can be compared.

Keep these responsibilities:

- **Flow Problem:** physics, mesh, native state, and native residual.
- **Elastostatic Problem:** material, discretization, and equilibrium.
- **Interaction:** wall constraint, multiplier, and transfers.
- **Execution adapter:** global state, solution policy, and time integration.

Problems must not know global execution block numbers. Residual evaluations at
candidate states must not update accepted histories. Preserve distributed
ownership and ghost synchronization; a one-dimensional centerline can still be
distributed across MPI ranks.

This stage passes when it reproduces the converged file-based iteration within
the stated transfer and solver tolerances.

## Stage 5: one monolithic IDAAdapter

Let $z$ denote the complete MetricFlowX state and set
$y=(u,z,\lambda)$. The coupled semidiscrete problem is

$$
F(t,y,\dot y)=
\begin{bmatrix}
Ku-f(t)+B\lambda\\
F_{\mathrm{flow}}(t,z,\dot z)+C_p\lambda\\
B^Tu-g(A(z))
\end{bmatrix}=0.
$$

For the elastostatic model, $u$ and $\lambda$ are algebraic. MetricFlowX
retains its native differential/algebraic partition. Do not introduce solid
velocity or inertia: time describes the evolving flow and the sequence of solid
equilibria.

For fixed lift geometry, the derivatives have the structure

$$
F_y=
\begin{bmatrix}
K&0&B\\
0&F_{\mathrm{flow},z}&C_p\\
B^T&-g_A A_z&0
\end{bmatrix},\qquad
F_{\dot y}=
\begin{bmatrix}
0&0&0\\
0&F_{\mathrm{flow},\dot z}&0\\
0&0&0
\end{bmatrix}.
$$

The radial map has derivative

$$
\frac{d\,\delta R}{dA}=\frac{1}{2\sqrt{\pi A}}.
$$

If geometry, weights, or operators depend on the candidate state, include their
derivatives too. The block $C_p$ must appear in differentiation with respect
to $\lambda$, even if the native pressure provider treats pressure as fixed
when assembling its own Jacobians.

`IDAAdapter` constructs

$$
J=F_y+\alpha F_{\dot y}.
$$

Contributors expose the two derivatives separately. IDA's coefficient
$\alpha$ belongs to the adapter. Application code adds the elastostatic
Problem, MetricFlowX Problem, and wall Interaction directly to one adapter.

### Consistent initialization

Before integration:

1. Assign an initial flow state with positive area.
2. Compute $g(A(t_0))$.
3. Solve constrained equilibrium for $u(t_0)$ and $\lambda(t_0)$.
4. Determine an initial derivative compatible with the flow equations.
5. Check the complete residual.

Zero initialization is valid only when it satisfies the system. Verify
consistent initialization and DAE-index behavior on the smallest case. During
nonlinear iteration, evaluate area-dependent expressions only on admissible
positive-area states.

## Existing code to read

- [MetricFlowX–elastodynamics tutorial](metric-flow-x-elastodynamics) and
  [coupling formulation](metric-flow-x-elastodynamics-coupling): existing
  monolithic composition, area-to-radius map, constraint, and volume feedback.
  Adapt these to an algebraic solid; the existing example includes inertia.
- [Compositional lifted static elasticity](../how-to/lifted-static-elasticity):
  static tensor-product constraints, modal data, physical metrics, and output.
- [Vascular-tree elasticity](../how-to/vascular-tree-elasticity): legacy
  tensor-product parameter groups and imported-field expressions. Adapt runtime
  paths to the build-tree conventions used in this guide.
- [elastic_static.h](https://github.com/luca-heltai/immersx/blob/master/include/immersx/physics/elastic_static.h): `ElasticStaticProblem` and its
  algebraic displacement contribution.
- [metric_flow_x.h](https://github.com/luca-heltai/immersx/blob/master/include/immersx/physics/metric_flow_x.h): native flow-state contribution and
  semantic area/velocity fields.
- [app_metric_flow_x_elastodynamics.cc](https://github.com/luca-heltai/immersx/blob/master/apps/app_metric_flow_x_elastodynamics.cc): the current coupled executable.
- [sundials_ida_adapter.h](https://github.com/luca-heltai/immersx/blob/master/include/immersx/core/sundials_ida_adapter.h): time-integration policy.

Use the compositional static path when assembling the final coupled application;
retain the independent legacy run as a comparison where the formulations agree.
MetricFlowX integration requires a compatible installed package. For IDA, deal.II
must provide SUNDIALS support. Check that configuration reports
`MetricFlowX support enabled`, and use matching Debug/Release variants as
described in the existing coupled tutorial.

## Work split and completion checks

| Stage | Flow collaborator | Solid collaborator | Shared check |
| --- | --- | --- | --- |
| Independent runs | Minimal case, output, complete state | Radial static case, multiplier export | Geometry and units |
| File exchange | Feedback import and restart | Area-to-radius conversion and solid input | One reproducible complete exchange |
| Iteration | Replay the same time interval | Equilibrium for each candidate area | Convergence and common acceptance |
| In-memory exchange | Native residual access | Interaction and transfers | Agreement with file exchange |
| Monolithic solve | Flow and feedback derivatives | Equilibrium, constraint, and derivatives | Complete residual and Jacobian |

Before moving to branched networks, require:

- A radial case with a verified reaction sign and normalization.
- A restart comparison against continuous flow integration.
- A nonzero coupled case with checked equilibrium and constraint residuals.
- Directional Jacobian checks, including both cross-coupling directions.
- Agreement between converged partitioned and monolithic solutions.
- Time-step, mesh, and quadrature checks.
- Serial and two-rank MPI comparisons.

The first shared deliverable is one vessel with an operational README,
parameters, conversion script, exchange-format description, and numerical
checks. Keep that case through every stage up to the monolithic `IDAAdapter`.
