PR: #441

- Fresh MetricFlowX elastostatic runs can explicitly select `use_y_dot` to solve
  the native steady initial residual with all initial rates prescribed to zero.
  The default `use_y_diff` correction and checkpoint restart behavior are unchanged.
