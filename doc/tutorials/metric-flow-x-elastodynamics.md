# MetricFlowX and elastodynamics

The `metric_flow_x_elastodynamics` application composes a 1D-in-3D
MetricFlowX flow Problem and a 3D elastodynamics Problem in one IDA adapter.
The runnable example uses one parameter file:

```text
tutorials/metric_flow_x_elastodynamics/two_way.prm.in
```

The [one-vessel manufactured solution](metric-flow-x-elastodynamics-mms)
provides a piecewise manufactured target in a cube, with forcing for all five
residual rows, volume supplied by the inlet, and measured field errors.

MetricFlowX must provide the same build configuration as ImmersX.
DebugRelease requires a dual-variant package exporting
`MetricFlowX::metric_flow_x_Debug` and
`MetricFlowX::metric_flow_x_Release`. ImmersX links each library and its
tests to the matching concrete target, keeping the deal.II variants separate.
Older packages exporting only `MetricFlowX::metric_flow_x` remain supported
in compatible single-variant builds; incompatible packages are disabled.
For a Debug-only installation, configure a separate Debug build:

```bash
cmake -S . -B build-metric-debug -DCMAKE_BUILD_TYPE=Debug \
  -DMetricFlowX_DIR="$HOME/dualistic/installations/lib/cmake/MetricFlowX"
cmake --build build-metric-debug -j2
ctest --test-dir build-metric-debug -R MetricFlowX --output-on-failure
```

The configure output must report `MetricFlowX support enabled`. CTest checks
the standalone residual and Jacobians against the native matrices and runs the
nonzero-inflow coupled example in serial and on two MPI ranks, using degree
two to check automatic multiplier selection. Coupled cases write to separate
build-tree output directories.

The coupled application can assign vessel cells to the owners of the solid
cells containing their centers:

```text
subsection MetricFlowX coupling
  set Embedded partitioning policy = immersed
end
```

The default value, `default`, retains ordinary Domain partitioning. With
`immersed`, the solid mesh is built first and supplies the ownership policy
when the flow Domain is finalized. The multiplier and imported vessel data
are then initialized on that final mesh. This policy improves proximity to
the solid; its load balance follows the solid partition and vessel locations.
It can leave ranks without vessels and does not guarantee connected vessel
subdomains. Vessel cell centers must lie inside the solid mesh.

With
[MetricFlowX PR #82](https://github.com/luca-heltai/metric-flow-x/pull/82),
both policies keep each RCR capacitor pressure on the rank owning its
terminal cell. MetricFlowX represents that pressure as component 4 of its
hp finite element: `FE_DGQ(0)` on capacitive terminal cells and `FE_Nothing`
elsewhere. Area, velocity, traces and capacitor pressures share the native
`DoFHandler` numbering; the adapter uses that numbering directly and retains
MetricFlowX's relevant indices for distant terminal links. There is no
separate FE-to-state conversion. Observable evaluation and weak-term
linearization select each cell's active element and local DoF count.
Imported reference areas and multiplier fields keep their own FE spaces.

Use a MetricFlowX version containing PR #82's hp representation. The earlier
rank-local layout with manually appended capacitor indices is no longer
supported by this adapter.

MetricFlowX and ImmersX must also use the same linear algebra backend. When
deal.II provides Trilinos, configure MetricFlowX with
`-DMETRIC_FLOW_X_BACKEND=Trilinos` for this application.

The inflow starts with $Q(0)=Q'(0)=0$, matching the initially resting solid
and zero initial acceleration. Displacement compatibility also imposes
conditions on the initial velocity and acceleration; a nonzero initial
inflow derivative is incompatible with these zero initial data.

After configuring and building with MetricFlowX and deal.II SUNDIALS enabled,
run the generated build-tree input:

```bash
build/metric_flow_x_elastodynamics_debug \
  build/tutorials/metric_flow_x_elastodynamics/two_way.prm
```

Both the standalone `metric_flow_x` application and this coupled application
use MetricFlowX's native IDA section for time bounds, output cadence, solver
controls, and tolerance ramps:

```text
subsection MetricFlowSystem<1, 3>
  subsection IDA parameters
    set Initial time = 0
    set Final time = 0.02
    set Time interval between each output = 0.01
    subsection Running parameters
      set Initial step size = 0.001
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
multiplier output callbacks. MetricFlowX's separate `Output time interval`
entry is an optional flow-file throttle; its default zero writes at every
shared callback.

Migrate older ImmersX inputs by moving `/IDA/` or
`/MetricFlowSystem<1, 3>/IDA/` into this section. Move the entries from
`Time interval` into the same section and rename `Output time interval` to
`Time interval between each output`. Remove `Fixed step`; this IDA-only
application no longer declares it. Move the old `MetricFlowX coupled solver`
selection to `IDA parameters/Linear solver/Type` (`iterative`, `umfpack`, or
`mumps`). The solid shares the same time interval
and IDA configuration. The IDA controls also include initial-condition
correction, nonlinear and linear solver options, and the startup tolerance ramp.

The wall relation is imposed weakly. With (A) the flow area, (A_0) its
reference area, (u) the solid displacement, (n) the radial direction and
(lambda) the pressure-jump multiplier, the lifted kinematic row represents

```text
integral_Gamma (u - (sqrt(A/pi) - sqrt(A0/pi))*n) dot (mu*n) dGamma = 0.
```

The application supplies native external pressure

```text
p_external = -lambda.
```

Native assembly includes volume, face, boundary, and junction contributions.
The solid receives the positive multiplier traction in momentum. All coupling
terms appear together in the application.

The application imports `a0` from the VTK mesh already configured for
MetricFlowX. It represents `a0` as a frozen `FE_DGQ(0)` Observable, so adjacent
cells can keep different reference areas. The same Observable defines both
the radial increment and the lift thickness. The multiplier has an independent
scalar `FiniteElementSpace<1,3>` on the centerline triangulation. Its default
`auto` selection follows the parsed MetricFlowX area degree. The application
checks that the local scalar FE basis matches. The IDA adapter advances both Problems,
and output includes flow, solid, and multiplier fields.

The solid can also be read from a mesh file, including a tetrahedral one:
set `Grid generator` in `subsection Elastodynamics/Domain` to the file name
and `Triangulation type = fullydistributed`, and use `Grid scale` when the
mesh and the VTK network use different length units. At startup the
application prints the solid boundary ids and the number of vessel cells with
a vertex outside the solid mesh. This only checks centreline vertices: the
lifted wall points lie one radius away from the centreline, so a count of zero
does not guarantee that the lifted wall is inside the solid. Lifted points
outside the solid stop the assembly with "Some lifted integration points lie
outside the background mesh".
