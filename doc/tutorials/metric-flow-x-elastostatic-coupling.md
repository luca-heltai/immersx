# MetricFlowX and elastostatic coupling

The `metric_flow_x_elastostatic` application couples transient native
MetricFlowX flow to a solid in instantaneous elastic equilibrium. It uses
one IDA adapter. By default the flow drives the solid without pressure
feedback. The existing `metric_flow_x_elastodynamics` application
continues to include solid inertia. The runnable example uses
`tutorials/metric_flow_x_elastostatic/single_vessel.prm.in`, with the same
single-vessel geometry and inflow as the [dynamic tutorial](metric-flow-x-elastodynamics.md).

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

MetricFlowX and ImmersX must use matching build variants and the same linear
algebra backend. When deal.II provides Trilinos, build MetricFlowX with
`-DMETRIC_FLOW_X_BACKEND=Trilinos`. DebugRelease requires a dual-variant package
exporting `MetricFlowX::metric_flow_x_Debug` and
`MetricFlowX::metric_flow_x_Release`; a Debug-only package requires a separate
Debug build. Use deal.II with SUNDIALS and VTK:

```bash
cmake -S . -B build-metric-debug -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/dealii -DMetricFlowX_DIR=/path/to/MetricFlowX
cmake --build build-metric-debug --target metric_flow_x_elastostatic_debug -j2
build-metric-debug/metric_flow_x_elastostatic_debug \
  build-metric-debug/tutorials/metric_flow_x_elastostatic/single_vessel.prm
```

The configure output must report `MetricFlowX support enabled`. For the
serial and two-rank MPI checks, build the test targets and run:

```bash
cmake --build build-metric-debug -j2
ctest --test-dir build-metric-debug -R MetricFlowXElastostatic --output-on-failure
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
To reuse one flow run for several solids, save it with MetricFlowX checkpoints
and [replay it](metric-flow-x-elastostatic-replay.md) with static solves.

The application accepts VTK vessel input or an existing native physical network
archive. The [IRCAD11 small tutorial](metric-flow-x-elastostatic-ircad11-small.md)
loads three already-grown networks with their saved microvascular connections;
it does not rerun growth. Co-refinement currently applies to VTK input. Network
generation and fictitious-interior modal corrections are not exposed here.
The application keeps the uncorrected extended-solid convention. Nonzero solid
density, viscosity, and Rayleigh damping coefficients are rejected.

## Time integration and output

As in the dynamic application, time bounds, output cadence, solver controls,
and tolerance ramps use MetricFlowX's native IDA section. The example sets:

```text
subsection MetricFlowSystem<1, 3>
  subsection IDA parameters
    set Initial time = 0
    set Final time = 0.02
    set Time interval between each output = 0.01
    subsection Running parameters
      set Initial step size = 0.0001
      set Maximum order of BDF = 2
    end
    subsection Error control
      set Absolute error tolerance = 1e-6
      set Relative error tolerance = 1e-5
    end
  end
end
```

`Initial step size` controls the adaptive integrator's first step.
`Time interval between each output` controls the shared flow, solid, and
multiplier callbacks. MetricFlowX's separate `Output time interval` is an
optional flow-file throttle; zero writes at every shared callback.
`Stop at output times = true` keeps these callbacks at accepted IDA step
endpoints, preserving the nonlinear wall constraint.

The generated example writes below
`build-metric-debug/test_output/metric-flow-x-elastostatic`. Open `flow.pvd`,
`solid.pvd`, and `pressure_jump.pvd` for the native flow, solid displacement,
and scalar wall multiplier. The multiplier output retains the dynamic
application's name `pressure_jump`. The flow and solid output directories
are set independently in `MetricFlowSystem<1, 3>` and `Elastic static`.

## Co-refine the solid and vessel wall lift

Co-refinement is disabled by default. Enable it before the initial solve with:

```text
subsection MetricFlowX coupling
  subsection Co-refinement
    set Enabled = true
    set Refinement strategy = both
    set Lower factor = 0.5
    set Upper factor = 2
    set Max refinement level = 10
    set Max section refinement level = 10
  end
end
```

The application compares local solid-cell sizes with the axial and physical
cross-section scales of the lifted wall quadrature. It refines the solid,
flow mesh, or section as needed before creating the application FE spaces.
The section cap is independent of the solid/flow cap. These size bounds check
local discretization compatibility; they do not establish an inf-sup condition.
Imported `a0` follows flow refinement, and the wall radius remains
$R_0 = \sqrt{a_0/\pi}$.

Checkpoints retain both refined native meshes, the transferred reference-area
field, and the final section refinement level. Restart loads that data and
skips co-refinement.

## Wall particles

Open `wall_particles.pvd` alongside the solid and flow output. Each particle is
one lifted quadrature point on the reference vessel wall. The default output
contains the quadrature weight and constraint reaction `wall_traction = lambda * n`, where `n` is
the normalized radial direction used by the coupling.

Select additional quantities with:

```text
subsection MetricFlowX coupling
  subsection Particle output
    set Properties = weight, mode_values, modal_coefficients, \
                     lifted_value, modal_data, representative_point, \
                     radius, indices, bulk_field
  end
end
```

`wall_traction_radial` is the scalar multiplier in Pa; `mode_radial` is its
unit radial direction. `data_radial` is the current target radial increment
`sqrt(A/pi) - sqrt(a0/pi)` in metres. `displacement` samples the accepted solid
solution at the wall. The representative point, radius, indices and weight
refer to the reference geometry. An empty `Properties` list disables particle
files. The PVD times are the accepted output times, including the initial state.

The solid mesh is configured under `Elastic static/Domain`. Native MetricFlowX
settings configure the 1D network; there is no separate `MetricFlowX domain`
subsection.

## Linear solves

The IDA application uses the iterative coupled solve provided by the adapter.
The solid stiffness matrix is assembled once after mesh setup, including any
initial co-refinement, or once when loading a restart. Candidate times update
only the loads and inhomogeneous Dirichlet lifting. The application reuses the
solid operator, IDA tangent and AMG hierarchy without scanning matrix entries.
Material and Robin spring coefficients remain fixed in this application;
body loads, tractions and prescribed displacements may vary in time.
The output reports `Static stiffness assemblies: 1` and the number of AMG
setups so this behavior can be checked on an actual run.

With a MUMPS-enabled deal.II build, the flow block uses a MUMPS
factorization for each Newton matrix snapshot. Repeated actions during that
linear solve reuse the factor. Native mixed-state ownership can be
noncontiguous; the block inverse redistributes its backsolve result to the
native field layout. The coupling option `Flow block solver` selects
`mumps` or `ilu`. The global IDA `Linear solver/Type` remains `iterative`
for this block-preconditioned path.

The coupling option `Wall pressure absolute tolerance` sets the multiplier's
absolute tolerance in Pa through the adapter's field-tolerance interface.
Zero retains the common IDA absolute tolerance. Set it independently when
tightening the common tolerance for the native flow coefficients.

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

Use the same executable and dependency versions to save and load a checkpoint.

Checkpoint/restart requires the native hooks from
[MetricFlowX PR #94](https://github.com/luca-heltai/metric-flow-x/pull/94).
The coupled checkpoint checks reported here used the frozen revision
[`7d0937f`](https://github.com/luca-heltai/metric-flow-x/commit/7d0937f7ad92e0003dca8ba5defdf1bf8c2e4bc5)
and a deal.II build with the serialization correction described below.
A package without the native hooks can run the application but rejects
checkpoint requests explicitly.

PR #94 has since merged at
[`49a88a5`](https://github.com/luca-heltai/metric-flow-x/commit/49a88a57cc3636008f28381ab1c8267b66b21a37).
Its native flow serializer uses a local compatibility triangulation and does
not require a patched deal.II installation for that path. API inspection found
unchanged mesh setup and extra-field hooks; the coupled ImmersX application has
not been compiled or run against this newer SDK in the checks reported here.
Upstream native checkpoint tests are separate evidence.

The solid checkpoint still calls deal.II's fully distributed triangulation
serializer directly. On deal.II builds affected by its serialization offset
defect, that solid path still needs the correction in
[deal.II PR #20374](https://github.com/dealii/dealii/pull/20374) or a version
containing it. MetricFlowX's local flow compatibility class does not replace
this solid serializer. Regenerate MPI checkpoints written by the defective
serializer, and retain the same executable and dependency versions when
comparing the coupled restart results.

Use a fully distributed solid mesh (`Elastic static/Domain/Triangulation
type = fullydistributed`, the application default) and keep
`MetricFlowX coupling/Stop at output times = true`. The checkpoint records
an accepted state at the final time. For example, with two ranks:

```bash
mpirun -np 2 build-metric-debug/metric_flow_x_elastostatic_debug \
  build-metric-debug/tutorials/metric_flow_x_elastostatic/single_vessel.prm \
  --final-time 0.01 --checkpoint /scratch/run/checkpoint-001

mpirun -np 2 build-metric-debug/metric_flow_x_elastostatic_debug \
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
