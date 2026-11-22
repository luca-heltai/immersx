PR: #488

- Before: steady residual continuation always used secant extrapolation after
  the first accepted stage. After: Use secant predictor chooses secant
  extrapolation (the default) or the latest accepted state for each stage,
  retaining the existing rollback and line-search behavior.
