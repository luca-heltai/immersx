PR: pending

- Before: coupled steady initialization made one KINSOL attempt from the initial
  state. After: an opt-in residual continuation rolls back and reduces eta after
  retryable nonlinear failures, with bounded attempts and a strict eta-one gate
  against the original coupled residual. The existing direct path remains the
  default.
