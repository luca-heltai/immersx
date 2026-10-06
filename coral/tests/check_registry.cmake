if(NOT DEFINED CORAL_EXECUTABLE OR NOT DEFINED PLUGIN OR
   NOT DEFINED REGISTRY OR NOT DEFINED SPACEDIM)
  message(FATAL_ERROR "Coral registry test requires executable, plugin, registry, and spacedim.")
endif()

execute_process(
  COMMAND "${CORAL_EXECUTABLE}" -p "${PLUGIN}" register
          --registry-path "${REGISTRY}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "Coral registry generation failed with ${_result}.\n${_stdout}\n${_stderr}")
endif()

if(NOT EXISTS "${REGISTRY}")
  message(FATAL_ERROR "Coral did not produce registry '${REGISTRY}'.")
endif()
file(READ "${REGISTRY}" _registry)

if(VTK_AVAILABLE)
  set(_vtk_variant_count 0)
  string(JSON _registry_size LENGTH "${_registry}")
  math(EXPR _registry_last_index "${_registry_size} - 1")
  foreach(_registry_index RANGE 0 ${_registry_last_index})
    string(JSON _registry_key MEMBER "${_registry}" ${_registry_index})
    string(JSON _operation ERROR_VARIABLE _operation_error
      GET "${_registry}" "${_registry_key}" operation)
    if(NOT _operation_error AND _operation STREQUAL "Write VTK")
      math(EXPR _vtk_variant_count "${_vtk_variant_count} + 1")
      string(JSON _vtk_variant GET
        "${_registry}" "${_registry_key}" variant_name)
      if(_vtk_variant MATCHES "^Default id fields\\. ")
        set(_vtk_expected_arity 2)
      elseif(_vtk_variant MATCHES "^Material id field\\. ")
        set(_vtk_expected_arity 3)
      elseif(_vtk_variant MATCHES "^Material and boundary id fields\\. ")
        set(_vtk_expected_arity 4)
      elseif(_vtk_variant MATCHES "^All id fields\\. ")
        set(_vtk_expected_arity 5)
      else()
        message(FATAL_ERROR
          "Registry ${SPACEDIM}d contains an unknown VTK writer variant "
          "'${_vtk_variant}'.")
      endif()

      string(JSON _vtk_argument_count LENGTH
        "${_registry}" "${_registry_key}" arguments)
      if(NOT _vtk_argument_count EQUAL _vtk_expected_arity)
        message(FATAL_ERROR
          "VTK writer variant '${_vtk_variant}' has ${_vtk_argument_count} "
          "arguments; expected ${_vtk_expected_arity}.")
      endif()

      string(JSON _vtk_output_count LENGTH
        "${_registry}" "${_registry_key}" outputs)
      if(NOT _vtk_output_count EQUAL 0)
        message(FATAL_ERROR
          "VTK writer variant '${_vtk_variant}' must not expose outputs.")
      endif()

      math(EXPR _vtk_last_argument "${_vtk_argument_count} - 1")
      foreach(_vtk_argument RANGE 0 ${_vtk_last_argument})
        string(JSON _vtk_connection_type GET
          "${_registry}" "${_registry_key}" arguments ${_vtk_argument}
          connection_type)
        if(NOT _vtk_connection_type STREQUAL "input")
          message(FATAL_ERROR
            "VTK writer variant '${_vtk_variant}' must use input arguments.")
        endif()
      endforeach()
    endif()
  endforeach()

  math(EXPR _expected_vtk_variant_count "${SPACEDIM} * 4")
  if(NOT _vtk_variant_count EQUAL _expected_vtk_variant_count)
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d contains ${_vtk_variant_count} VTK writer "
      "variants; expected ${_expected_vtk_variant_count}.")
  endif()

  foreach(_dim RANGE 1 ${SPACEDIM})
    if(_dim EQUAL SPACEDIM)
      set(_vtk_dimension "${_dim}D")
    else()
      set(_vtk_dimension "${_dim}D in ${SPACEDIM}D")
    endif()

    string(FIND "${_registry}"
      "dealii::Triangulation<${_dim}, ${SPACEDIM}>" _vtk_tria_found)
    if(_vtk_tria_found EQUAL -1)
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d is missing the VTK triangulation type for "
        "${_dim}D.")
    endif()

    foreach(_vtk_variant
        "Default id fields"
        "Material id field"
        "Material and boundary id fields"
        "All id fields")
      string(FIND "${_registry}"
        "\"operation\": \"Write VTK\"" _vtk_operation_found)
      string(FIND "${_registry}"
        "\"variant_name\": \"${_vtk_variant}. ${_vtk_dimension}\""
        _vtk_variant_found)
      if(_vtk_operation_found EQUAL -1 OR _vtk_variant_found EQUAL -1)
        message(FATAL_ERROR
          "Registry ${SPACEDIM}d is missing the ${_vtk_variant} VTK writer "
          "variant for ${_vtk_dimension}.")
      endif()
    endforeach()
  endforeach()
endif()

foreach(_dim RANGE 1 ${SPACEDIM})
  foreach(_family Poisson ElasticStatic Elastodynamics)
    string(FIND "${_registry}"
      "ImmersX::${_family}<${_dim},${SPACEDIM}>" _found)
    if(_found EQUAL -1)
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not contain ${_family}<${_dim},${SPACEDIM}>.")
    endif()

    string(FIND "${_registry}"
      "ImmersX::${_family}Parameters<${_dim},${SPACEDIM}>" _parameters_found)
    if(_parameters_found EQUAL -1)
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d is missing ${_family} parameter type.")
    endif()

  endforeach()

  set(_space_type "ImmersX::FiniteElementSpaceView<${_dim},${SPACEDIM}>")
  string(JSON _space_node_type ERROR_VARIABLE _space_error
    GET "${_registry}" "${_space_type}" node_type)
  if(_space_error OR NOT _space_node_type STREQUAL "output_only")
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d does not register ${_space_type} as output-only.")
  endif()

  foreach(_field_kind Scalar Vector)
    set(_field_type "ImmersX::Field<${_dim},${SPACEDIM},${_field_kind}>")
    string(JSON _field_node_type ERROR_VARIABLE _field_error
      GET "${_registry}" "${_field_type}" node_type)
    if(_field_error OR NOT _field_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_field_type} as output-only.")
    endif()

    foreach(_observable_kind value gradient)
      set(_observable_type
        "ImmersX::Observable<${_dim},${SPACEDIM},${_field_kind},${_observable_kind}>")
      string(JSON _observable_node_type ERROR_VARIABLE _observable_error
        GET "${_registry}" "${_observable_type}" node_type)
      if(_observable_error OR NOT _observable_node_type STREQUAL "output_only")
        message(FATAL_ERROR
          "Registry ${SPACEDIM}d does not register ${_observable_type} as output-only.")
      endif()
    endforeach()

    set(_test_type
      "ImmersX::TestExpression<${_dim},${SPACEDIM},${_field_kind}>")
    string(JSON _test_node_type ERROR_VARIABLE _test_error
      GET "${_registry}" "${_test_type}" node_type)
    if(_test_error OR NOT _test_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_test_type} as output-only.")
    endif()

    foreach(_test_observable_kind gradient)
      set(_test_observable_type
        "ImmersX::TestExpression<${_dim},${SPACEDIM},${_field_kind},${_test_observable_kind}>")
      string(JSON _test_observable_node_type ERROR_VARIABLE _test_observable_error
        GET "${_registry}" "${_test_observable_type}" node_type)
      if(_test_observable_error OR
         NOT _test_observable_node_type STREQUAL "output_only")
        message(FATAL_ERROR
          "Registry ${SPACEDIM}d does not register ${_test_observable_type} as output-only.")
      endif()
    endforeach()

    set(_boundary_type "ImmersX::BoundaryConditions<${_dim},${SPACEDIM}>")
    string(JSON _boundary_node_type ERROR_VARIABLE _boundary_error
      GET "${_registry}" "${_boundary_type}" node_type)
    if(_boundary_error OR NOT _boundary_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_boundary_type} as output-only.")
    endif()
  endforeach()

set(_function_type "ImmersX::Function<${SPACEDIM}>")
string(JSON _function_node_type ERROR_VARIABLE _function_error
  GET "${_registry}" "${_function_type}" node_type)
if(_function_error OR
   NOT _function_node_type STREQUAL "output_only")
  message(FATAL_ERROR
    "Registry ${SPACEDIM}d does not register ${_function_type} as output-only.")
endif()

string(FIND "${_registry}" "\"type\": \"ImmersX::ModulatedParsedFunction<${SPACEDIM}>\""
  _modulated_function_found)
if(_modulated_function_found EQUAL -1)
  message(FATAL_ERROR
    "Registry ${SPACEDIM}d does not expose the parameterized parsed function type.")
endif()

foreach(_parsed_variant Expression Parameterized)
  string(FIND "${_registry}"
    "\"operation\": \"Parsed function\"" _parsed_operation_found)
  string(FIND "${_registry}"
    "\"variant_name\": \"${_parsed_variant}. ${SPACEDIM}D\""
    _parsed_variant_found)
  string(FIND "${_registry}"
    "\"output_type\": \"ImmersX::Function<${SPACEDIM}>\""
    _parsed_output_found)
  if(_parsed_operation_found EQUAL -1 OR _parsed_variant_found EQUAL -1 OR
     _parsed_output_found EQUAL -1)
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d is missing the ${_parsed_variant} parsed-function variant.")
  endif()
endforeach()

  foreach(_observable_kind divergence "symmetric gradient")
    set(_observable_type
      "ImmersX::Observable<${_dim},${SPACEDIM},Vector,${_observable_kind}>")
    string(JSON _observable_node_type ERROR_VARIABLE _observable_error
      GET "${_registry}" "${_observable_type}" node_type)
    if(_observable_error OR NOT _observable_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_observable_type} as output-only.")
    endif()

    set(_test_observable_type
      "ImmersX::TestExpression<${_dim},${SPACEDIM},Vector,${_observable_kind}>")
    string(JSON _test_observable_node_type ERROR_VARIABLE _test_observable_error
      GET "${_registry}" "${_test_observable_type}" node_type)
    if(_test_observable_error OR
       NOT _test_observable_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_test_observable_type} as output-only.")
    endif()
  endforeach()

  if(SPACEDIM GREATER 1)
    set(_observable_type
      "ImmersX::Observable<${_dim},${SPACEDIM},Vector,curl>")
    string(JSON _observable_node_type ERROR_VARIABLE _observable_error
      GET "${_registry}" "${_observable_type}" node_type)
    if(_observable_error OR NOT _observable_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_observable_type} as output-only.")
    endif()
    set(_test_observable_type
      "ImmersX::TestExpression<${_dim},${SPACEDIM},Vector,curl>")
    string(JSON _test_observable_node_type ERROR_VARIABLE _test_observable_error
      GET "${_registry}" "${_test_observable_type}" node_type)
    if(_test_observable_error OR
       NOT _test_observable_node_type STREQUAL "output_only")
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d does not register ${_test_observable_type} as output-only.")
    endif()
  endif()
endforeach()

set(_parameter_acceptor_type "dealii::ParameterAcceptor")
string(JSON _parameter_acceptor_node_type ERROR_VARIABLE _parameter_acceptor_error
  GET "${_registry}" "${_parameter_acceptor_type}" node_type)
if(_parameter_acceptor_error OR
   NOT _parameter_acceptor_node_type STREQUAL "abstract")
  message(FATAL_ERROR
    "Registry ${SPACEDIM}d does not expose the abstract ParameterAcceptor type.")
endif()

string(JSON _parameter_acceptor_hash
  GET "${_registry}" "${_parameter_acceptor_type}" type)
string(JSON _parameter_acceptor_derived_count ERROR_VARIABLE _derived_error
  LENGTH "${_registry}" "${_parameter_acceptor_type}" derived)
if(_derived_error OR _parameter_acceptor_derived_count LESS 1)
  message(FATAL_ERROR
    "Registry ${SPACEDIM}d does not expose ParameterAcceptor derived types.")
endif()

foreach(_arity RANGE 1 8)
  if(_arity EQUAL 1)
    set(_variant "1 parameter object")
  else()
    set(_variant "${_arity} parameter objects")
  endif()
  string(FIND "${_registry}" "\"variant_name\": \"${_variant}\"" _variant_found)
  if(_variant_found EQUAL -1)
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d is missing the ${_variant} initialization operation.")
  endif()
endforeach()

foreach(_dim RANGE 1 ${SPACEDIM})
  foreach(_family Poisson ElasticStatic Elastodynamics)
    set(_parameter_type
      "ImmersX::${_family}Parameters<${_dim},${SPACEDIM}>")
    string(JSON _parameter_base ERROR_VARIABLE _parameter_base_error
      GET "${_registry}" "${_parameter_type}" base)
    if(_parameter_base_error OR
       NOT _parameter_base STREQUAL "${_parameter_acceptor_hash}")
      message(FATAL_ERROR
        "${_parameter_type} is not registered as a ParameterAcceptor derived type.")
    endif()
  endforeach()
endforeach()

if(SPACEDIM EQUAL 2)
  foreach(_primitive_type
      "ImmersX::DomainParameters<2,2>"
      "ImmersX::OwnedDomain<2,2>"
      "ImmersX::OwnedFiniteElementSpace<2,2>"
      "ImmersX::StaticScalarField<2,2>"
      "ImmersX::TestExpression<2,2,Scalar>"
      "ImmersX::TestExpression<2,2,Scalar,gradient>"
      "ImmersX::KnownTerm<2,2,ScalarParsedFunctionRhs>"
      "ImmersX::WeakTerm<2,2,ScalarStaticFieldRhs,ScalarTestExpression>"
      "ImmersX::WeakTerm<2,2,ScalarGradient,ScalarTestExpressionGradient>"
      "ImmersX::KnownSource<2,2,Vector>"
      "ImmersX::KnownTerm<2,2,Vector>"
      "ImmersX::WeakTerm<2,2,VectorDivergence,VectorTestExpressionDivergence>"
      "ImmersX::WeakTerm<2,2,VectorSymmetricGradient,VectorTestExpressionSymmetricGradient>")
    string(FIND "${_registry}" "${_primitive_type}" _primitive_found)
    if(_primitive_found EQUAL -1)
      message(FATAL_ERROR
        "Registry 2d is missing generic primitive type ${_primitive_type}.")
    endif()
  endforeach()

  foreach(_primitive_operation
      "Create domain"
      "Generate domain"
      "Create finite element space"
      "Boundary conditions"
      "Constant function"
      "Dirichlet boundary condition"
      "Apply boundary conditions"
      "Test"
      "Static scalar field"
      "Register algebraic field"
      "Weak term"
      "Add weak term to linear execution"
      "Known source"
      "Known term"
      "Boundary known term"
      "Add known term to linear execution"
      "Vector Known source"
      "Vector Known term"
      "Vector Boundary known term"
      "Vector Add known term to linear execution")
    string(FIND "${_registry}" "\"operation\": \"${_primitive_operation}\""
      _primitive_operation_found)
    if(_primitive_operation_found EQUAL -1)
      message(FATAL_ERROR
        "Registry 2d is missing generic primitive operation '${_primitive_operation}'.")
      endif()
  endforeach()

  string(FIND "${_registry}" "Set constant Dirichlet boundary condition"
    _legacy_boundary_condition_found)
  if(NOT _legacy_boundary_condition_found EQUAL -1)
    message(FATAL_ERROR
      "Registry still contains the specialized constant Dirichlet operation.")
  endif()

  foreach(_test_operation Test Value Gradient)
    string(FIND "${_registry}"
      "${_test_operation}::std::function<ImmersX::TestExpression"
      _test_signature_found)
    if(_test_signature_found EQUAL -1)
      message(FATAL_ERROR
        "Registry 2d is missing the ${_test_operation} TestExpression signature.")
    endif()
  endforeach()

  foreach(_parameter_type
      "ImmersX::FiniteElementSpaceParameters<1,2>"
      "ImmersX::LinearAdapterParameters"
      "ImmersX::OutputHandlerParameters")
    string(JSON _parameter_base ERROR_VARIABLE _parameter_base_error
      GET "${_registry}" "${_parameter_type}" base)
    if(_parameter_base_error OR
       NOT _parameter_base STREQUAL "${_parameter_acceptor_hash}")
      message(FATAL_ERROR
        "${_parameter_type} is not registered as a ParameterAcceptor derived type.")
    endif()
  endforeach()

  string(FIND "${_registry}" "\"operation\": \"Test gradient\""
    _specialized_test_gradient_found)
  if(NOT _specialized_test_gradient_found EQUAL -1)
    message(FATAL_ERROR
      "Registry still contains the specialized Test gradient operation.")
  endif()
endif()

set(_initialize_arity4_found FALSE)
string(JSON _registry_size LENGTH "${_registry}")
math(EXPR _registry_last_index "${_registry_size} - 1")
foreach(_registry_index RANGE 0 ${_registry_last_index})
  string(JSON _registry_key MEMBER "${_registry}" ${_registry_index})
  string(JSON _operation ERROR_VARIABLE _operation_error
    GET "${_registry}" "${_registry_key}" operation)
  if(NOT _operation_error AND _operation STREQUAL "Initialize parameters")
    string(JSON _variant ERROR_VARIABLE _variant_error
      GET "${_registry}" "${_registry_key}" variant_name)
    if(NOT _variant_error AND _variant STREQUAL "4 parameter objects")
      set(_initialize_arity4_found TRUE)
      string(JSON _argument_count LENGTH
        "${_registry}" "${_registry_key}" arguments)
      if(NOT _argument_count EQUAL 5)
        message(FATAL_ERROR
          "The 4-parameter Initialize parameters node has the wrong arity.")
      endif()
      foreach(_parameter_argument RANGE 0 3)
        string(JSON _connection_type GET
          "${_registry}" "${_registry_key}" arguments ${_parameter_argument}
          connection_type)
        if(NOT _connection_type STREQUAL "pass_through")
          message(FATAL_ERROR
            "Initialize parameters must pass through ParameterAcceptor arguments.")
        endif()
        string(JSON _argument_type GET
          "${_registry}" "${_registry_key}" arguments ${_parameter_argument} type)
        if(NOT _argument_type STREQUAL "${_parameter_acceptor_type}")
          message(FATAL_ERROR
            "Initialize parameters must accept ParameterAcceptor references.")
        endif()
        string(JSON _output_index GET
          "${_registry}" "${_registry_key}" outputs ${_parameter_argument})
        if(NOT _output_index EQUAL _parameter_argument)
          message(FATAL_ERROR
            "Initialize parameters must expose each ParameterAcceptor pass-through output.")
        endif()
      endforeach()
      string(JSON _file_connection_type GET
        "${_registry}" "${_registry_key}" arguments 4 connection_type)
      if(NOT _file_connection_type STREQUAL "input")
        message(FATAL_ERROR
          "Initialize parameters must consume the parameter file as input.")
      endif()
    endif()
  endif()
endforeach()
if(NOT _initialize_arity4_found)
  message(FATAL_ERROR
    "Registry ${SPACEDIM}d is missing the generic four-parameter initialization operation.")
endif()

string(FIND "${_registry}" "Multiple parameter objects" _multiple_parameter_objects)
if(NOT _multiple_parameter_objects EQUAL -1)
  message(FATAL_ERROR "Registry still contains Multiple parameter objects.")
endif()
string(FIND "${_registry}" "Load parameters" _legacy_parameter_loaders)
if(NOT _legacy_parameter_loaders EQUAL -1)
  message(FATAL_ERROR "Registry still contains legacy Load parameters operations.")
endif()

if(SPACEDIM GREATER 1)
  set(_fiber_type
    "ImmersX::FiberReinforcedElastodynamicsParameters<${SPACEDIM}>")
  string(FIND "${_registry}" "${_fiber_type}" _fiber_parameters_found)
  if(_fiber_parameters_found EQUAL -1)
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d is missing ${_fiber_type}.")
  endif()
endif()

foreach(_wrong_spacedim RANGE 1 3)
  if(NOT _wrong_spacedim EQUAL SPACEDIM)
    foreach(_family Poisson ElasticStatic Elastodynamics)
      string(FIND "${_registry}"
        "ImmersX::${_family}<1,${_wrong_spacedim}>" _found)
      if(NOT _found EQUAL -1)
        message(FATAL_ERROR
          "Registry ${SPACEDIM}d contains a ${_family} node for "
          "spacedim=${_wrong_spacedim}.")
      endif()
    endforeach()
  endif()
endforeach()

foreach(_type "std::string" "bool" "int" "unsigned int" "double")
  string(FIND "${_registry}" "\"${_type}\"" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "Registry is missing elementary type '${_type}'.")
  endif()
endforeach()

string(FIND "${_registry}" "Finite element space" _found)
if(_found EQUAL -1)
  message(FATAL_ERROR "Registry is missing the finite-element-space operation.")
endif()

foreach(_field_operation "Scalar field" "Vector field")
  string(FIND "${_registry}" "${_field_operation}" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR
      "Registry is missing the '${_field_operation}' operation family.")
  endif()
endforeach()

foreach(_observable_operation
    "Value"
    "Gradient"
    "Divergence"
    "Symmetric gradient")
  string(FIND "${_registry}" "${_observable_operation}" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR
      "Registry is missing the '${_observable_operation}' operation family.")
  endif()
endforeach()

foreach(_expression_operation "Scale term" "Nonlinear product")
  string(FIND "${_registry}" "${_expression_operation}" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR
      "Registry is missing the '${_expression_operation}' operation family.")
  endif()
endforeach()

if(SPACEDIM GREATER 1)
  string(FIND "${_registry}" "\"operation\": \"Curl\"" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "Registry is missing the 'Curl' operation family.")
  endif()
else()
  string(FIND "${_registry}" "\"operation\": \"Curl\"" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR "1D registry must not expose the unsupported Curl operation.")
  endif()
endif()

foreach(_test_operation "Test" "Value" "Gradient")
  string(FIND "${_registry}" "\"operation\": \"${_test_operation}\""
    _test_operation_found)
  if(_test_operation_found EQUAL -1)
    message(FATAL_ERROR
      "Registry is missing the '${_test_operation}' test-expression operation family.")
  endif()
endforeach()

string(FIND "${_registry}" "ImmersX::PoissonParameters<1,${SPACEDIM}>" _found)
if(_found EQUAL -1)
  message(FATAL_ERROR "Registry is missing stable Poisson parameter type name.")
endif()

if(SPACEDIM EQUAL 2)
  string(FIND "${_registry}" "ImmersX::CoupledPoisson<2>" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR "2D registry still contains the coupled Poisson façade.")
  endif()
  foreach(_operation
      "Create finite element space"
      "Registered scalar field"
      "Create linear execution"
      "Add problem to linear execution"
      "Problem solution field"
      "Continuity constraint"
      "Add constraint to linear execution"
      "Constraint multiplier field"
      "Create output handler"
      "Add scalar field"
      "Add vector field"
      "Write output"
      "Create linear state"
      "Solve linear state"
      "Evaluate linear residual"
      "Linear state norm"
      "Assert finite below")
    string(FIND "${_registry}" "${_operation}" _operation_found)
    if(_operation_found EQUAL -1)
      message(FATAL_ERROR
        "2D registry is missing the generic operation '${_operation}'.")
    endif()
  endforeach()
elseif(SPACEDIM EQUAL 1 OR SPACEDIM EQUAL 3)
  string(FIND "${_registry}" "ImmersX::CoupledPoisson<2>" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR
      "${SPACEDIM}D registry contains the 2D coupled Poisson façade.")
  endif()
endif()

if(SPACEDIM EQUAL 3)
  string(FIND "${_registry}" "ImmersX::CoupledPoissonElasticity<3>" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR
      "3D registry is missing the coupled Poisson-elasticity façade.")
  endif()
  string(FIND "${_registry}" "ImmersX::ReducedPoissonWorkflow<3>" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "3D registry is missing the ReducedPoisson workflow.")
  endif()
else()
  string(FIND "${_registry}" "ImmersX::CoupledPoissonElasticity<3>" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR
      "${SPACEDIM}D registry contains the 3D coupled Poisson-elasticity "
      "façade.")
  endif()
  string(FIND "${_registry}" "ImmersX::ReducedPoissonWorkflow<3>" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR
      "${SPACEDIM}D registry contains the 3D ReducedPoisson workflow.")
  endif()
endif()

if(SPACEDIM EQUAL 2)
  string(FIND "${_registry}" "ImmersX::FiberReinforcedElastodynamics<2>" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "2D registry is missing fiber elastodynamics.")
  endif()
  foreach(_type
      "ImmersX::FiberReinforcedElastodynamicsGraph<2>"
      "ImmersX::FiberMatrixProblem<2>"
      "ImmersX::FiberEmbeddedProblem<2>"
      "ImmersX::FiberVelocityContinuity<2>"
      "ImmersX::FiberExecutionAdapter<2>")
    string(FIND "${_registry}" "${_type}" _found)
    if(_found EQUAL -1)
      message(FATAL_ERROR "2D registry is missing compositional fiber type '${_type}'.")
    endif()
  endforeach()
elseif(SPACEDIM EQUAL 3)
  string(FIND "${_registry}" "ImmersX::FiberReinforcedElastodynamics<3>" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "3D registry is missing fiber elastodynamics.")
  endif()
  foreach(_type
      "ImmersX::FiberReinforcedElastodynamicsGraph<3>"
      "ImmersX::FiberMatrixProblem<3>"
      "ImmersX::FiberEmbeddedProblem<3>"
      "ImmersX::FiberVelocityContinuity<3>"
      "ImmersX::FiberExecutionAdapter<3>")
    string(FIND "${_registry}" "${_type}" _found)
    if(_found EQUAL -1)
      message(FATAL_ERROR "3D registry is missing compositional fiber type '${_type}'.")
    endif()
  endforeach()
else()
  foreach(_fiber_dim 2 3)
    string(FIND "${_registry}"
      "ImmersX::FiberReinforcedElastodynamics<${_fiber_dim}>" _found)
    if(NOT _found EQUAL -1)
      message(FATAL_ERROR
        "${SPACEDIM}D registry contains ${_fiber_dim}D fiber elastodynamics.")
    endif()
  endforeach()
endif()
