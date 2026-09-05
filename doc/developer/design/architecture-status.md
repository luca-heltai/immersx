# Architecture status

The current architecture is intentionally small:

```text
deal.II FE space -> Field -> Observable / typed FE expression
                 -> weak_term -> Constraint / Interaction
                 -> ExecutionAdapter
```

## Current path

- Problems and external/native providers own physical data, meshes, FE spaces,
  native operators, and accepted physical state.
- Fields provide semantic state and residual-row identity. Execution block
  numbering belongs to the adapter.
- Observables are typed quantities derived from Fields. Value, gradient,
  divergence, symmetric-gradient, and curl operations use deal.II's
  `FEValuesViews`.
- Weak terms and Constraints provide residual and transpose-consistent
  linearization contributions.
- Interactions own relation-specific geometry, search, transfer, and auxiliary
  multiplier Fields.
- Linear and IDA adapters own solver and time-integration policy.

Imported finite-element data is represented by ordinary FE-space views and
frozen coefficient vectors. Tensor-product lifting and particle search are
backends for those quantities, not public mathematical object hierarchies.
MetricFlowX remains external and is adapted through its public native FE-space
accessors. Its vessel-wall path uses an Area Field, a nonlinear radial
Observable, and `MetricFlowXVesselWallConstraint`.

## Protected boundary

`ElasticityProblem` and `ReducedPoisson` are protected production paths. Their
established reduced/tensor-product implementation may retain the minimal
deal.II coupling and inclusion closure required for unchanged behavior. New
applications and current documentation must not extend that closure.

## Validation status

The maintained correctness gates are the FE weak-term and Observable tests,
independent lifting/stencil adjointness tests, Constraint transpose tests,
application vertical tests, and the protected production smoke tests. Tests
whose only purpose was to preserve deleted transitional APIs have been removed.
