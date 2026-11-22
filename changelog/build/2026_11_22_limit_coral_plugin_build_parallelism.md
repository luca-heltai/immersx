PR: #489

- Before: Coral Debug and Release jobs could each compile two template-heavy
  plugin targets at once. An overlapping Debug build lost `cc1plus` to memory
  pressure. After: they compile one plugin target at a time per configuration,
  reducing peak compiler memory while keeping the same targets and tests.
