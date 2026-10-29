# GoogleTest suites are assigned one primary category.  The quick label is
# applied to the unit and integration category tests below; MPI is orthogonal
# and is selected by the process-count-specific test command.
set(IMMERSX_UNIT_TEST_SUITES
    VascularNetworkData
    TestCategoryTyped/*
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
    SymbolicExpressionKernel
    SymbolicFieldEvaluator
    StateHistory
    VTKUtils
    FieldCatalog
    MixedField
    TensorProductLift
    TensorProductSpace0D
    ReducedPoisson0D)

set(IMMERSX_INTEGRATION_TEST_SUITES
    VascularGeneration
    VascularCCO
    BoundaryConditions
    CoupledPoisson
    CoupledPoissonElasticity
    KnownTerm
    LargeNetworks
    ParticleCoupling
    ApplicationRoadmap
    ContributorPhysics
    FoundationalApplications
    TractionParameters
    DistributedIDA
    SundialsCompatibility
    DistributedLiftedQuadrature
    ElasticStaticDimensions
    ElasticStaticExecution
    ElasticStaticProblem
    ElasticityCouplingParameters
    ElasticityCouplingConstruction
    ElasticityCouplingIntegration
    ElasticityCouplingParticleOutput
    Elastodynamics
    ElasticityFreq
    FiberReinforcedElasticityFreq
    ImportedFiniteElementFields
    Domain
    LocalCoRefinement
    LiftedElasticity
    LegacyInclusions
    LiftedIntegrationSupport
    LinearAdapter
    PoissonPrimitives
    LiftedPoisson
    KINSOLAdapter
    TimeIntervalParameters
    ReducedCoupling
    TensorProductSpace
    TensorProductCoupling
    ReducedCoupling0D
    WeakTermNonmatching
    Constraint
    WeakTerm
    OutputHandler)

set(IMMERSX_VALIDATION_TEST_SUITES
    ElasticityCoupling
    ElasticStaticProblemValidation
    ElasticityValidation
    ElasticityExplicitValidation
    LaplacianValidation
    ReducedPoissonValidation
    ElasticityTensorProductCouplingValidation
    TriangulationBackendsValidation/*
    ElastodynamicsValidation
    ElastodynamicsExecutionValidation
    ElastodynamicsDimensionValidation
    ElastodynamicsAdapterMMS
    FiberReinforcedElastodynamicsValidation
    LargeNetworksValidation
    ElasticityCouplingIntegrationValidation
    NavierStokes)

set(IMMERSX_APPLICATION_TEST_SUITES
    AppExecutables
    ApplicationExecution
    Poisson
    PoissonExecution
    PrescribedPoisson
    LiftedPoissonApplication
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

# The compiled GoogleTest registry is checked by check_test_categories.cmake.
# Do not maintain a second manual inventory here.
