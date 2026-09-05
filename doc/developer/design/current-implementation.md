# Current implementation

The current implementation follows one composable semantic path for new
applications:

```text
Problem/native provider -> Field -> Observable -> weak_term
                         -> Constraint/Interaction -> ExecutionAdapter
```

`FieldId`, `FieldDescriptor`, `StateLayout`, `StateView`, `EvaluationContext`,
and `SemiDiscreteModel` provide semantic state and residual storage.
`FESpaceView` is a non-owning deal.II FE-space view, while `Field` selects a
named FE quantity. `Observable<Field, Operation>` delegates value and
first-order operations to deal.II `FEValuesViews`. Frozen coefficients use the
same operation machinery with no active dependencies.

`weak_term` assembles the duality between trial and test expressions. It can
use a same-space cell loop, paired DoFHandler traversal, or a prepared
nonmatching backend. Particle ownership, retained stencils, and tensor-product
lifting remain internal implementation choices.

`LinearAdapter` executes affine systems and `IDAAdapter` executes
`F(t,y,ydot)=0`. Both keep execution block numbering, vectors, solver policy,
and callbacks private. Contributors expose separate `dF/dy` and `dF/dydot`
actions; IDA-specific combinations are created only by the adapter.

## Reduced and protected legacy paths

`PoissonProblem`, `ElasticityProblem`, and `ReducedPoisson` retain established
deal.II reduced/tensor-product production paths. Their implementation is a
protected compatibility boundary for this cleanup. The associated coupling
and inclusion machinery is not a recommended extension API and is not used by
the current semantic application path.

Imported FE data is owned by `ImportedFiniteElementFields` and exposed through
ordinary Fields plus frozen coefficient vectors. MetricFlowX is adapted through
its public native DoFHandler and component accessors; it remains an external
Problem and is not made to inherit an ImmersX interface.
