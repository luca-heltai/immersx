# Architecture Diagram

This page collects a repository-wide class interaction diagram and links it to
the generated API reference.

The diagram below is intentionally organized by subsystem so that it stays
readable on a large code base. It focuses on:

- inheritance,
- strong composition,
- direct structural dependencies visible in public headers,
- the VTK-driven tensor-product coupling data flow.

The page is rendered with the Sphinx Mermaid extension so the diagram appears
directly in the generated site while keeping the source close to the prose.

## Mermaid source

```{mermaid}
classDiagram
direction LR

class PoissonProblem
class ReducedPoisson
class ElasticityProblem
class CoupledElasticityProblem
class MaterialProperties
class ModulatedParsedFunction
class Inclusions
class CouplingOperator
class BlockPreconditionerAugmentedLagrangian
class UtilitiesAL_BlockPreconditionerAugmentedLagrangian

class ReferenceCrossSection
class TensorProductSpace
class ParticleCoupling
class ReducedCoupling
class ImmersedRepartitioner
class "VTK reduced grid" as VTKReducedGrid
class "VTK field catalog" as VTKFieldCatalog
class InputFieldSelector
class ReducedFieldValues
class SymbolicFieldEvaluator
class "Triangulation signals" as TriangulationSignals
class SolutionTransfer

CoupledElasticityProblem --|> ElasticityProblem
ReducedCoupling --|> TensorProductSpace : tensor-product space
ReducedCoupling --|> ParticleCoupling : immersed particles

PoissonProblem *-- Inclusions : immersed data
PoissonProblem *-- CouplingOperator : matrix-free coupling

ReducedPoisson *-- ReducedCoupling : reduced interface
ReducedPoisson *-- CouplingOperator : matrix-free coupling

ElasticityProblem *-- Inclusions : immersed data
ElasticityProblem *-- MaterialProperties : materials
ElasticityProblem *-- ModulatedParsedFunction : rhs/bc
ElasticityProblem *-- ReducedCoupling : TensorProduct mode
ElasticityProblem ..> BlockPreconditionerAugmentedLagrangian : solves AL system
ElasticityProblem ..> UtilitiesAL_BlockPreconditionerAugmentedLagrangian : legacy/helper AL

TensorProductSpace *-- ReferenceCrossSection : modal section
TensorProductSpace *-- VTKReducedGrid : reads reduced mesh
TensorProductSpace *-- VTKFieldCatalog : discovers PointData/CellData
TensorProductSpace *-- SymbolicFieldEvaluator : thickness and rhs expressions
TensorProductSpace ..> InputFieldSelector : resolves selected fields
TensorProductSpace ..> ReducedFieldValues : evaluates fields at quadrature points
TensorProductSpace ..> TriangulationSignals : refinement callbacks
TriangulationSignals ..> SolutionTransfer : preserve properties
SolutionTransfer ..> TensorProductSpace : rebuilds refined properties

VTKReducedGrid ..> VTKFieldCatalog : field metadata
InputFieldSelector ..> VTKFieldCatalog : aliases and wildcard expansion
ReducedFieldValues ..> SymbolicFieldEvaluator : field values plus x,y,z,t

ReducedCoupling *-- ImmersedRepartitioner : repartitioning
ReducedCoupling ..> CouplingOperator : assembles reduced coupling

CouplingOperator o-- Inclusions : evaluates coupling on particles
```

The tensor-product path starts from a reduced VTK mesh and its point or cell
fields. `InputFieldSelector` resolves those fields into symbolic bindings;
`ReducedFieldValues` interpolates them at reduced quadrature points, where
`SymbolicFieldEvaluator` evaluates time-dependent thickness and reduced-load
expressions. `ReferenceCrossSection` lifts the reduced manifold into the
three-dimensional coupling geometry, while `ParticleCoupling` connects that
geometry to the bulk elasticity mesh.

During local refinement, `Triangulation` pre- and post-refinement signals drive
`SolutionTransfer`, so imported reduced-grid properties are transferred to the
new mesh before the tensor-product coupling is assembled again. This is the
data path used by the coupled 1D–3D vascular-network examples.

## API entry points

The following pages in the generated API reference are the most useful anchors
for navigating the classes shown above:

- {doc}`api/class_elasticity_problem`
- {doc}`api/class_poisson_problem`
- {doc}`api/class_inclusions`
- {doc}`api/class_coupling_operator`
- {doc}`api/class_particle_coupling`
- {doc}`api/class_tensor_product_space`
- {doc}`api/class_reference_cross_section`
- {doc}`api/class_input_field_selector`
- {doc}`api/class_reduced_field_values`
- {doc}`api/class_symbolic_field_evaluator`
- {doc}`api/file_include_vtk_utils.h`
- {doc}`api/file_include_reduced_coupling.h`
- {doc}`api/class_block_preconditioner_augmented_lagrangian`
- {doc}`api/class_utilities_a_l_1_1_block_preconditioner_augmented_lagrangian`

## Notes

- The diagram is derived from declarations in `include/`.
- It emphasizes architectural structure and the tensor-product data flow over
  every possible helper method.
- The VTK, symbolic-expression, and `SolutionTransfer` path is active when
  deal.II is built with VTK support.
- The `tests/tests.h` utility structs are intentionally omitted to keep the
  repository view focused on production code.
