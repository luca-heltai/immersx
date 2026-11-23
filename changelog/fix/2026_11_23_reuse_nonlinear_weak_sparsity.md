PR: #501

- Before: every nonlinear Jacobian evaluation rebuilt and distributed the
  global sparsity pattern, and serial matrices copied it again. After: each
  fixed weak-term state operator caches its structural pattern and reuses it
  across states, while assembling fresh state-dependent matrix values on every
  evaluation. Rebuild the term/model after changing the DoF layout, constraint
  graph, or selected support cells.
