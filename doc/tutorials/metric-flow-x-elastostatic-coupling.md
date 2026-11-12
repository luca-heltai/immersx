# MetricFlowX and elastostatic coupling

The `metric_flow_x_elastostatic` application couples transient native
MetricFlowX flow to a solid in instantaneous elastic equilibrium. It uses
one IDA adapter. By default the flow drives the solid without pressure
feedback. The existing `metric_flow_x_elastodynamics` application
continues to include solid inertia.

## Equations and state

The solid displacement $u$ and radial traction multiplier $\lambda$ are
algebraic fields. The native flow state retains its own differential mask:
area and axial velocity, and any capacitive pressure states, are differential
only where MetricFlowX marks them so. Native trace and junction equations
remain algebraic.

With pressure feedback enabled, the coupled equations are

$$
\begin{aligned}
F_{\mathrm{flow}}(t,A,U,z,\dot A,\dot U,\dot z;-\lambda)&=0,\\
Ku+B\lambda-f(t)&=0,\\
B^T u-P_h(A)&=0.
\end{aligned}
$$

There is no independent solid velocity, solid mass, damping, or solid
rate term. The traction $+B\lambda$ enters the displacement equilibrium
row. When feedback is enabled, the native external-pressure interface receives
$-\lambda$, including
its pressure, flux, boundary, and junction contributions.

The reference area $A_0$ comes from the VTK vessel field. The radius map is

$$
R_0=\sqrt{A_0/\pi},\qquad \delta R(A)=\sqrt{A/\pi}-R_0.
$$

The wall constraint uses the same tensor-product radial lift and overlap
supports as the dynamic coupling. Its discrete pairing $P_h(A)$ integrates
this radius change against the multiplier basis. The multiplier basis must
match the native area basis. Outside the solid support, the multiplier is
set to zero through its own weak equation. The
[dynamic formulation](metric-flow-x-elastodynamics-coupling.md) describes the
lift quadrature and pressure convention in detail.

## Run a single vessel without pressure feedback

Build with MetricFlowX and a deal.II installation with SUNDIALS and VTK:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/dealii -DMetricFlowX_DIR=/path/to/MetricFlowX
cmake --build build-debug --target metric_flow_x_elastostatic_debug -j
build-debug/metric_flow_x_elastostatic_debug \
  build-debug/tutorials/metric_flow_x_elastostatic/single_vessel.prm
```

Use the generated build-tree parameter file. CMake expands its data and
output paths. The example has a nonzero, smoothly increasing inlet flow
and a time-dependent radial body load that is already nonzero initially. Its solid material and boundary data
live under `Elastic static`; there are no solid velocity or initial-velocity
parameters. The lift and multiplier subsections retain the names used by
the dynamic application.

The default setting is:

```text
subsection MetricFlowX coupling
  set Pressure feedback = false
end
```

The native transient flow then evolves independently of the solid and the
multiplier. The solid still follows the flow radius through
$Ku+B\lambda-f(t)=0$ and $B^Tu-P_h(A)=0$. Disabling feedback removes only
the external-pressure residual and its Jacobians from the flow equations;
it does not remove the wall constraint or the traction reaction. Set
`Pressure feedback = true` to restore the two-way equations above.

The first version accepts VTK vessel input. Generated networks, archived
networks, adaptive co-refinement, and fictitious-interior modal corrections
are not exposed by this application. It keeps the uncorrected extended
solid convention. Nonzero solid density, viscosity, and Rayleigh damping
coefficients are incompatible with this model and are rejected.

## Linear solves

The IDA application uses the iterative coupled solve provided by the adapter.
The elastic block uses AMG; an unchanged assembled stiffness matrix reuses its
AMG preconditioner. Matrix values are checked before reuse, so time-dependent
boundary springs invalidate it when they change the stiffness. This does
not cache a factorization of the complete saddle system.

With a MUMPS-enabled deal.II build, the flow block uses a MUMPS
factorization for each Newton matrix snapshot. Repeated actions during that
linear solve reuse the factor. Native mixed-state ownership can be
noncontiguous; the block inverse redistributes its backsolve result to the
native field layout. The coupling option `Flow block solver` selects
`mumps` or `ilu`. The global IDA `Linear solver/Type` remains `iterative`
for this block-preconditioned path.

## Initialization and regularity

The system is locally index one when the constrained elastic system and the
native algebraic flow trace Jacobian are regular. This requires suitable
solid boundary conditions to remove rigid motions on each connected
component of the solid mesh and a multiplier space
and quadrature with the required rank. A finer multiplier space can lose
rank even when the elastic stiffness itself is nonsingular.

The parameter example selects IDA `use_y_diff` initial-condition correction.
The supplied zero solid guess is inconsistent with the initial body load.
IDA holds the differential initial values while correcting algebraic
values and differential rates in the coupled residual. It does not make
arbitrary prescribed initial flow data compatible. In particular, boundary
and junction data, positive areas, and wall compatibility must admit a
solution. Initialization failures must be diagnosed from that coupled
system.

The algebraic wall constraint applies at accepted times. Saving at internal
IDA step endpoints avoids showing an interpolated state that violates the
nonlinear radius constraint. The application does not claim improved
pressure accuracy or validated convergence merely from removing inertia.

## Checkpoint and restart

Checkpoint/restart requires the native hooks from
[MetricFlowX PR #94](https://github.com/luca-heltai/metric-flow-x/pull/94),
reference revision
[`7d0937f`](https://github.com/luca-heltai/metric-flow-x/commit/7d0937f7ad92e0003dca8ba5defdf1bf8c2e4bc5).
A package without these hooks can run the coupled application but rejects
checkpoint requests explicitly.

For deal.II builds affected by the fully distributed serialization offset
defect, MPI checkpoints also require the correction in
[deal.II PR #20374](https://github.com/dealii/dealii/pull/20374). Regenerate
MPI checkpoints written by the affected serializer.

Use a fully distributed solid mesh (`Elastic static/Domain/Triangulation
type = fullydistributed`, the application default) and keep
`MetricFlowX coupling/Stop at output times = true`. The checkpoint records
an accepted state at the final time. For example, with two ranks:

```bash
mpirun -np 2 build-debug/metric_flow_x_elastostatic_debug \
  build-debug/tutorials/metric_flow_x_elastostatic/single_vessel.prm \
  --final-time 0.01 --checkpoint /scratch/run/checkpoint-001

mpirun -np 2 build-debug/metric_flow_x_elastostatic_debug \
  --restart /scratch/run/checkpoint-001 --final-time 0.02
```

Restart does not read the original parameter file, vessel VTK, or solid
mesh input. The archive contains resolved application parameters, parsed
functions, native flow physics and inflow data, both meshes, the native
flow state and derivative, solid displacement, multiplier and multiplier
derivative, and the complete reference-area FE field. Mesh and FE fields
use deal.II's native triangulation and `SolutionTransfer` serialization;
Boost archives hold the application metadata.

All ranks participate. Restart requires the same MPI rank count,
algebra backend, deal.II version, and compatible executables and finite
elements. The fixed linear FE mapping uses the stored vertices. Checkpoints
restore exact `FlatManifold` and `CylindricalManifold` objects from their
constructor parameters. Other manifold types and custom subclasses are
rejected when writing a checkpoint. Further mesh refinement is unavailable
after restart. A checkpoint is
published by renaming a complete staging directory; existing destinations
are rejected. Preserve the entire directory when copying it.

IDA starts a new integration history from the stored time and state pair.
Its internal BDF history is not saved. The restored algebraic state must
already satisfy the coupled equilibrium and wall equations; restart does
not replace a valid checkpoint with a new initial-condition solve.
Continuation writes to a new sibling output directory beside the
checkpoint, preserving earlier output. Add `--checkpoint NEW_DIRECTORY`
to write a later checkpoint.
