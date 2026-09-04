# GoogleTest suites are assigned one primary category.  The quick label is
# applied to the unit and integration category tests below; MPI is orthogonal
# and is selected by the process-count-specific test command.
set(IMMERSX_UNIT_TEST_SUITES
    ContributorCore
    DimensionParameters
    FESpace
    InputFieldSelector
    MaterialParameters
    BoundaryConditionParameters
    RhsParameters
    ModulatedParsedFunction
    ReferenceCrossSection
    ReferenceFrame
    ReferenceInclusion2
    Inclusion2
    Inclusion3
    InclusionsBasis2
    CCO
    Constraint
    SymbolicExpressionKernel
    SymbolicFieldEvaluator
    StateHistory
    VTKUtils
    FieldCatalog
    WeakTerm
    VectorRepresentation
    RetainedSampling
    Representation
    MixedField
    TensorProductLift
    TensorProductLiftParity
    TensorProductSpace0D
    ReducedPoisson0D)

set(IMMERSX_INTEGRATION_TEST_SUITES
    ContributorPhysics
    TractionParameters
    CouplingOperator
    DistributedIDA
    DistributedLiftedQuadrature
    ElasticStaticDimensions
    ElasticStaticExecution
    ElasticStaticProblem
    ElasticityTest
    ElasticityCouplingParameters
    ElasticityCouplingConstruction
    ElasticityCouplingIntegration
    ElasticityCouplingParticleOutput
    ElasticityTensorProductCoupling
    TriangulationBackends/*
    Elastodynamics
    ElastodynamicsExecution
    ElastodynamicsDimension
    ExpressionFieldSources
    ExpressionRepresentation
    FiberReinforcedElastodynamics
    ImportedFiniteElementFields
    LargeNetworks
    LegacyInclusions
    LinearAdapter
    TimeParameters
    DeferredPressureExpression
    PublicComposition
    ReducedCoupling
    TensorProductSpace
    TensorProductCoupling
    ReducedCoupling0D
    VectorLagrangeMultiplierInteraction
    RepresentationContract
    RepresentationLoadInteraction
    WeakTermNonmatching)

set(IMMERSX_VALIDATION_TEST_SUITES
    OneVesselMMS
    OneVesselMMSDriver
    MetricFlowXVesselWallRepresentation
    MetricFlowXVesselWallInteraction
    NavierStokes)

set(IMMERSX_APPLICATION_TEST_SUITES
    AppExecutables
    ApplicationExecution
    Poisson
    PoissonExecution
    PrescribedPoisson
    ReducedPoisson)

set(_immersx_category_suites
    ${IMMERSX_UNIT_TEST_SUITES}
    ${IMMERSX_INTEGRATION_TEST_SUITES}
    ${IMMERSX_VALIDATION_TEST_SUITES}
    ${IMMERSX_APPLICATION_TEST_SUITES})
list(LENGTH _immersx_category_suites _immersx_category_suite_count)
list(REMOVE_DUPLICATES _immersx_category_suites)
list(LENGTH _immersx_category_suites _immersx_unique_category_suite_count)
if(NOT _immersx_category_suite_count EQUAL _immersx_unique_category_suite_count)
  message(FATAL_ERROR "ImmersX test category lists overlap")
endif()

set(IMMERSX_TEST_PRIMARY_CATEGORIES unit integration validation application)

# Keep this inventory next to the category assignment so adding a GoogleTest
# suite requires an explicit classification.  The configure-time comparison
# is intentionally small and fails closed when a suite is omitted or listed
# twice.
set(_immersx_expected_test_suites
    AppExecutables
    ContributorCore
    ContributorPhysics
    TractionParameters
    CouplingOperator
    DimensionParameters
    DistributedIDA
    DistributedLiftedQuadrature
    ElasticStaticDimensions
    ElasticStaticExecution
    ElasticStaticProblem
    ElasticityTest
    ElasticityCouplingParameters
    ElasticityCouplingConstruction
    ElasticityCouplingIntegration
    ElasticityCouplingParticleOutput
    ElasticityTensorProductCoupling
    TriangulationBackends/*
    WeakTermNonmatching
    Elastodynamics
    ElastodynamicsExecution
    ElastodynamicsDimension
    ExpressionFieldSources
    ExpressionRepresentation
    FESpace
    TimeParameters
    FiberReinforcedElastodynamics
    ImportedFiniteElementFields
    InclusionsBasis2
    CCO
    Constraint
    InputFieldSelector
    LargeNetworks
    LegacyInclusions
    LinearAdapter
    MaterialParameters
    BoundaryConditionParameters
    RhsParameters
    OneVesselMMS
    OneVesselMMSDriver
    MetricFlowXVesselWallRepresentation
    MetricFlowXVesselWallInteraction
    ModulatedParsedFunction
    NavierStokes
    Poisson
    ApplicationExecution
    PoissonExecution
    PrescribedPoisson
    DeferredPressureExpression
    PublicComposition
    ReducedCoupling
    TensorProductSpace
    ReferenceCrossSection
    ReferenceFrame
    TensorProductLift
    ReferenceInclusion2
    Inclusion3
    Inclusion2
    RepresentationContract
    Representation
    MixedField
    RepresentationLoadInteraction
    RetainedSampling
    StateHistory
    SymbolicExpressionKernel
    SymbolicFieldEvaluator
    TensorProductCoupling
    TensorProductLiftParity
    TensorProductSpace0D
    ReducedCoupling0D
    ReducedPoisson
    ReducedPoisson0D
    VectorLagrangeMultiplierInteraction
    VectorRepresentation
    FieldCatalog
    VTKUtils
    WeakTerm)
list(SORT _immersx_expected_test_suites)
set(_immersx_actual_test_suites ${_immersx_category_suites})
list(SORT _immersx_actual_test_suites)
if(NOT "${_immersx_actual_test_suites}" STREQUAL
       "${_immersx_expected_test_suites}")
  message(FATAL_ERROR
          "Every aggregate GoogleTest suite must have exactly one category")
endif()
