PR: #N

- Before: coupled steady initialization did not report its original initial residual by semantic field. After: it reports raw L2 and maximum norms and row-scaled infinity norms for displacement, velocity, the full native flow state, and the physical wall multiplier, reusing the residual already evaluated by the solver.
