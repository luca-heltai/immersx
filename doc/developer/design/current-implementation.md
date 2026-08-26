# Current implementation architecture

This page is an inventory of the implementation currently visible on
`master`. It complements the normative design in
{doc}`../../concepts/architecture`: the class diagram below describes existing classes
and dependencies, while the core page describes the Field/residual/execution
direction that is being introduced incrementally.

The diagram emphasizes public headers, strong composition, and the active
VTK-driven tensor-product path. It is not a promise that every edge is a
future ownership rule.

## Current class and data-flow inventory

```{mermaid}
flowchart LR
  subgraph problems["Current Problem implementations"]
    PP["PoissonProblem"]
    RP["ReducedPoisson"]
    EP["ElasticityProblem"]
    CEP["CoupledElasticityProblem"]
  end

  subgraph representations["Current representations and reduced geometry"]
    INC["Inclusions"]
    RC["ReducedCoupling"]
    TPS["TensorProductSpace"]
    RCS["ReferenceCrossSection"]
    PC["ParticleCoupling"]
    IR["ImmersedRepartitioner"]
  end

  subgraph inputs["Current imported-field path"]
    VTK["VTK reduced grid"]
    IFS["InputFieldSelector"]
    RFV["ReducedFieldValues"]
    SEE["SymbolicFieldEvaluator"]
  end

  subgraph algebra["Current interaction and linear algebra"]
    CO["CouplingOperator"]
    LMI["LagrangeMultiplierInteraction"]
    CE["ConstraintEquation"]
    AL["Schur / augmented-Lagrangian solvers"]
  end

  PP --> INC
  PP --> CO
  RP --> RC
  RP --> CO
  EP --> INC
  EP --> RC
  CEP --> EP

  RC --> TPS
  RC --> PC
  RC --> IR
  TPS --> RCS
  TPS --> RFV
  VTK --> IFS
  IFS --> RFV
  RFV --> SEE
  RFV --> TPS
  SEE --> TPS

  INC --> CO
  PC --> CO
  RC --> CO
  CO --> LMI
  CE --> AL
  LMI --> AL
  CO --> AL
```

On the tensor-product path, the reduced VTK grid supplies geometry and field
metadata. `InputFieldSelector` resolves selected point/cell fields,
`ReducedFieldValues` evaluates them at reduced quadrature points, and
`SymbolicFieldEvaluator` evaluates expressions such as thickness or reduced
loads. `ReferenceCrossSection` and `TensorProductSpace` lift the reduced
description to the represented physical support. `ParticleCoupling` and
`ImmersedRepartitioner` provide search and distributed ownership services.

The scalar and elasticity coupling paths may use different subsets of these
classes. A local search mesh is an implementation detail, not a permanent
“background Problem” role.

## API entry points

The generated API reference provides the current class-level details:

- {doc}`../../api/class_immers_x_1_1_elasticity_problem`
- {doc}`../../api/class_immers_x_1_1_poisson_problem`
- {doc}`../../api/class_immers_x_1_1_inclusions`
- {doc}`../../api/class_immers_x_1_1_coupling_operator`
- {doc}`../../api/class_immers_x_1_1_particle_coupling`
- {doc}`../../api/class_immers_x_1_1_tensor_product_space`
- {doc}`../../api/class_immers_x_1_1_reference_cross_section`
- {doc}`../../api/class_immers_x_1_1_input_field_selector`
- {doc}`../../api/class_immers_x_1_1_reduced_field_values`
- {doc}`../../api/class_immers_x_1_1_symbolic_field_evaluator`
- {doc}`../../api/class_immers_x_1_1_lagrange_multiplier_interaction`
- {doc}`../../api/class_immers_x_1_1_constraint_equation`

For the future semantic architecture, start with
{doc}`../../concepts/architecture`, especially the sections on Fields, residual
contributors, typed Representations, and execution adapters. Prototype names
such as `FieldId`, `StateView`, `EvaluationContext`, or a SUNDIALS adapter
must not be read as merged API when browsing this current implementation page.

## Reading the two architecture pages together

| Page | Purpose |
| --- | --- |
| {doc}`current-implementation` | Existing classes and current data paths on `master`. |
| {doc}`../../concepts/architecture` | Normative ownership, residual, representation, and execution design. |
| {doc}`../../concepts/mathematical-background` | Mathematical motivation and reduced Lagrange-multiplier context. |
| {doc}`../../tutorials/index` | Runnable learning workflows. |
