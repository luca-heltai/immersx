PR: #441

- Fresh MetricFlowX elastostatic runs can explicitly select `use_y_dot` to solve
  the native steady initial residual with all initial rates prescribed to zero.
  The default `use_y_diff` correction and checkpoint restart behavior are unchanged.
- The explicit `--steady-only` mode uses native KINSOL Newton with fresh
  Jacobians, frozen scaling from the first materialized Jacobian, strict area
  constraints, and a final residual gate. Accepted output and checkpoints are
  written at the initial time without advancing IDA.
- Dynamic MetricFlowX elastodynamics can explicitly initialize a steady
  equilibrium with native KINSOL line search in the same four-field adapter,
  retaining second-order condensation and the subsequent time-tolerance ramp.
  Full-residual scaling is computed from the existing global blocks, and the
  original linear and nonlinear post-gates precede adoption of the initial state.
