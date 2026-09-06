# Constraints and interactions

A `Constraint` expresses a relation between fields through weak terms. It
owns the multiplier field and contributes the relation to the residual rows of
the multiplier and its participants. An Interaction is the broader category
for relation-specific terms that may also own geometry, search, transfer
operators, or native-provider state.

The multiplier is an algebraic field on an independently selected finite-
element space. It does not borrow a participant's DoFHandler. All weak terms
in one constraint use the same unregistered multiplier field:

```cpp
auto lambda = multiplier_space.field("lambda");
auto first = weak_term(value(u), test(lambda));
auto second = weak_term(value(v), test(lambda));
auto constraint = make_constraint(first - second, rhs);
auto fields = adapter.add(constraint, "continuity");
```

The multiplier row receives the signed weak-term sum minus `rhs`. Participant
rows receive the transpose reactions. The same construction handles scalar
and vector fields; the FE extractor determines the value type.

The weak-term backend chooses direct cell traversal, paired DoFHandler
traversal, or nonmatching point search from the spaces and geometry. This
choice does not change the constraint API. Distributed point ownership and
search use deal.II's distributed particle facilities where the coupling
requires them.

`MetricFlowXVesselWallConstraint` is a domain-specific Interaction. It relates
the MetricFlowX area field and the wall displacement observable, contributes
the solid-side coupling terms, and maps the multiplier to the native
MetricFlowX external-pressure operation. The application adds the native
MetricFlowX provider and the solid Problem to the same `IDAAdapter`.
