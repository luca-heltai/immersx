# Replay saved MetricFlowX flow on an elastostatic solid

The `metric_flow_x_elastostatic_replay` application drives a static solid with
blood flow that MetricFlowX computed beforehand. The flow runs once, on its
own, and saves native checkpoints. The replay then solves one static
elasticity problem with a wall multiplier at each saved time. The flow does
not see the solid.

This gives the same solid and multiplier as the
[elastostatic coupling](metric-flow-x-elastostatic-coupling.md) with
`Pressure feedback = false`, at the saved times. The difference is the
execution: the flow is not integrated again, and the solid solves are
independent linear solves through one `LinearAdapter` instead of algebraic
rows inside IDA.

## Equations

At each checkpoint time $t_n$ the application reads the native flow area
$A_n$ and solves

$$
\begin{aligned}
Ku+B\lambda&=f(t_n),\\
B^T u&=P_h(A_n),
\end{aligned}
$$

with the radius map of the coupled application,

$$
R_0=\sqrt{a_0/\pi},\qquad \delta R(A)=\sqrt{A/\pi}-R_0 .
$$

$P_h(A)$ integrates $\delta R(A)$ against the multiplier basis on the lifted
wall overlap, and the multiplier is zero outside the solid. The reference area
$a_0$ is the area of the first checkpoint in the history, so the solid is
displaced by the change of radius since that time. When the flow starts from
its reference area, this is the same $a_0$ as in the coupled application.
MetricFlowX does not yet expose its reference area;
[MetricFlowX issue #98](https://github.com/luca-heltai/metric-flow-x/issues/98)
tracks it.

$K$, $B$ and the multiplier metric are assembled once. Each step updates the
solid loads, the Dirichlet data and $P_h(A_n)$. With the `direct` or `mumps`
solver the saddle-point matrix is factored once and reused for every step.

## Run the single-vessel example

The example uses the geometry, inflow and solid of the single-vessel
elastostatic tutorial. Build ImmersX against a MetricFlowX package with native
checkpoints ([MetricFlowX PR #94](https://github.com/luca-heltai/metric-flow-x/pull/94)
or later):

```bash
cmake -S . -B build-metric-debug -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/dealii -DMetricFlowX_DIR=/path/to/MetricFlowX
cmake --build build-metric-debug \
  --target metric_flow_x_elastostatic_replay_debug -j2
```

First run the MetricFlowX `blood_flow` application with the flow input and a
checkpoint directory. It writes one checkpoint per output time:

```bash
blood_flow \
  build-metric-debug/tutorials/metric_flow_x_elastostatic/single_vessel_flow.prm \
  --checkpoint-directory \
  build-metric-debug/test_output/metric-flow-x-elastostatic-replay/history
```

Then replay the history:

```bash
build-metric-debug/metric_flow_x_elastostatic_replay_debug \
  build-metric-debug/tutorials/metric_flow_x_elastostatic/single_vessel_replay.prm
```

The replay input points to the history:

```text
subsection MetricFlowX replay
  set Checkpoint history = .../metric-flow-x-elastostatic-replay/history
  subsection Linear adapter
    set Solver = direct
  end
end
```

Every committed `checkpoint-<time>` directory in the history is replayed in
time order. The solid, wall lift and multiplier sections have the same names
and meaning as in the coupled application, so the solid part of an existing
coupled input can be reused. The native flow sections are not read: mesh,
physics and state come from the checkpoints.

The output directory of `Elastic static` receives `solid.pvd`,
`pressure_jump.pvd` and `wall_particles.pvd`, with one record per checkpoint
time. The application prints the adapter residual of each step and the number
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
$\sqrt{A_n/\pi}-\sqrt{a_0/\pi}$ of the step, and `displacement` the solid
solution at the wall. An empty list disables the particle files.

## Restrictions

- Run the replay with the same number of MPI ranks as `blood_flow`.
  MetricFlowX rejects checkpoints written with a different rank count.
- All checkpoints must come from one history with a fixed flow mesh. The
  replay checks the mesh and DoF numbering of each checkpoint against the
  first one.
- MetricFlowX loads each checkpoint into a new `BloodFlowSystem` and
  parses the global parameter tree. The replay therefore clears the
  `ParameterAcceptor` registry before each load. All parameters must be parsed
  before `setup()`. Each step reads a full restart checkpoint and repeats the
  native flow setup, although the replay uses only the cell area.
  [MetricFlowX issue #96](https://github.com/luca-heltai/metric-flow-x/issues/96)
  asks for a lighter state load and
  [issue #99](https://github.com/luca-heltai/metric-flow-x/issues/99) for
  writing and reading only selected fields.
- The flow mesh comes from the checkpoints and cannot be refined, so the
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

`run()` does the same. The wall rows come from
`metric_flow_x_prescribed_wall_constraint()`, which takes the flow area as a
`frozen()` expression sharing its coefficient vector. Overwriting that vector
changes only the constraint data of the next solve.

The `MetricFlowXElastostatic.BOTH_ReplayMatchesOneWayCoupledSolid` test runs
the coupled IDA application without pressure feedback, saves the flow at each
output time, replays it and compares displacement and multiplier at every
step.
