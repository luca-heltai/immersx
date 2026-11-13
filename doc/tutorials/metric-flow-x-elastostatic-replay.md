# Replay saved MetricFlowX flow on an elastostatic solid

The `metric_flow_x_elastostatic_replay` application drives a static solid with
blood flow that MetricFlowX computed beforehand. The flow runs once, on its
own, and saves its area at each output time. The replay then solves one static
elasticity problem with a wall multiplier at each saved time. The flow does
not see the solid.

This gives the same solid and multiplier as the
[elastostatic coupling](metric-flow-x-elastostatic-coupling.md) with
`Pressure feedback = false`, at the saved times. The difference is the
execution: the flow is not integrated again, and the solid solves are
independent linear solves through one `LinearAdapter` instead of algebraic
rows inside IDA.

## Equations

At each saved time $t_n$ the application reads the native flow area $A_n$ and
solves

$$
\begin{aligned}
Ku+B\lambda&=f(t_n),\\
B^T u&=P_h(A_n),
\end{aligned}
$$

with the radius map

$$
R_d=\sqrt{a_d/\pi},\qquad \delta R(A)=\sqrt{A/\pi}-R_d .
$$

$P_h(A)$ integrates $\delta R(A)$ against the multiplier basis on the lifted
wall overlap, and the multiplier is zero outside the solid. The reference area
$a_d(x)$ is the tapered diastolic area of MetricFlowX
(`BloodFlowSystem::reference_area()`), relative to which the flow evaluates its
wall laws. It is interpolated into the native area space, and its radius $R_d$
also places the lifted wall. When the vessel input gives `a0` equal to
$a_d$, as in the single-vessel example, this is the reference of the coupled
application.

$K$, $B$ and the multiplier metric are assembled once. Each step updates the
solid loads, the Dirichlet data and $P_h(A_n)$. With the `direct` or `mumps`
solver the saddle-point matrix is factored once and reused for every step.

## Run the single-vessel example

The example uses the geometry, inflow and solid of the single-vessel
elastostatic tutorial. Build ImmersX against a MetricFlowX package with
coupling field output and `reference_area()`
([MetricFlowX PR #97](https://github.com/luca-heltai/metric-flow-x/pull/97) and
[PR #100](https://github.com/luca-heltai/metric-flow-x/pull/100) or later):

```bash
cmake -S . -B build-metric-debug -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/dealii -DMetricFlowX_DIR=/path/to/MetricFlowX
cmake --build build-metric-debug \
  --target metric_flow_x_elastostatic_replay_debug -j2
```

First run the MetricFlowX `blood_flow` application with the flow input:

```bash
blood_flow \
  build-metric-debug/tutorials/metric_flow_x_elastostatic/single_vessel_flow.prm
```

The input sets a coupling output directory:

```text
subsection MetricFlowSystem<1, 3>
  set Coupling output directory = .../metric-flow-x-elastostatic-replay/coupling
  set Coupling output fields = area
end
```

`blood_flow` writes one native checkpoint `layout` at the first output time,
with the mesh, physics and DoF layout, and then only the cell area in a
`fields-<time>` directory at every output time. MetricFlowX does not overwrite
existing directories, so remove `coupling` before running the flow again.

Then replay the history:

```bash
build-metric-debug/metric_flow_x_elastostatic_replay_debug \
  build-metric-debug/tutorials/metric_flow_x_elastostatic/single_vessel_replay.prm
```

The replay input points to the history:

```text
subsection MetricFlowX replay
  set Flow history = .../metric-flow-x-elastostatic-replay/coupling
  subsection Linear adapter
    set Solver = direct
  end
end
```

The replay loads `layout` once and then reads only the area of every committed
`fields-<time>` directory, in time order. A directory written by
`blood_flow --checkpoint-directory` works too: the first `checkpoint-<time>` is
loaded once and the replay reads only the state of the others. The solid, wall
lift and multiplier sections have the same names and meaning as in the coupled
application, so the solid part of an existing coupled input can be reused. The
native flow sections are not read: mesh and physics come from the layout.

The output directory of `Elastic static` receives `solid.pvd`,
`pressure_jump.pvd` and `wall_particles.pvd`, with one record per saved time. The application prints the adapter residual of each step and the number
of stiffness assemblies.

The wall particles are the lifted quadrature points, written as in the
[coupled application](metric-flow-x-elastostatic-coupling.md).
Select their properties under the replay section:

```text
subsection MetricFlowX replay
  subsection Particle output
    set Properties = weight, lifted_value, modal_data, bulk_field
  end
end
```

`wall_traction` is the multiplier, `data_radial` the imposed radius change
$\sqrt{A_n/\pi}-\sqrt{a_d/\pi}$ of the step, and `displacement` the solid
solution at the wall. An empty list disables the particle files.

## Restrictions

- Run the replay with the same number of MPI ranks as `blood_flow`.
  MetricFlowX rejects data written with a different rank count or partition.
- All saved times must come from one run with a fixed flow mesh. MetricFlowX
  checks each one against the loaded layout.
- Loading the layout parses its archived parameter tree into the global one.
  The replay therefore clears the `ParameterAcceptor` registry before that
  load, and all parameters must be parsed before `setup()`.
- The flow mesh comes from the layout and cannot be refined, so the
  co-refinement of the coupled application is not available. Archived
  vascular networks are not supported either.

## Use the replay from C++

`MetricFlowXElastostaticReplay` in
`immersx/physics/metric_flow_x_elastostatic_replay.h` exposes the steps
separately:

```cpp
ParameterAcceptor::clear();
MetricFlowXElastostaticReplayParameters parameters;
initialize_parameters("replay.prm");

MetricFlowXElastostaticReplay replay(parameters);
replay.setup();
for (unsigned int step = 0; step < replay.n_steps(); ++step)
  {
    replay.solve_step(step);
    replay.output_step(step);
  }
```

`run()` does the same. A C++ flow solver that owns its time loop can write
the same history with `BloodFlowSystem::save_checkpoint()` for `layout` and
`save_coupling_fields()` for each `fields-<time>`. The wall rows come from
`metric_flow_x_prescribed_wall_constraint()`, which takes the flow area as a
`frozen()` expression sharing its coefficient vector. Overwriting that vector
changes only the constraint data of the next solve.

The `MetricFlowXElastostatic.BOTH_ReplayMatchesOneWayCoupledSolid` test runs
the coupled IDA application without pressure feedback, saves the flow at each
output time both as checkpoints and as coupling fields, replays both histories
and compares displacement and multiplier at every step.
