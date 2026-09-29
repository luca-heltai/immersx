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

    string(FIND "${_registry}"
      "Load parameters::std::function<void (ImmersX::${_family}Parameters<${_dim}, ${SPACEDIM}> &"
      _loader_found)
    if(_loader_found EQUAL -1)
      message(FATAL_ERROR
        "Registry ${SPACEDIM}d is missing the generic ${_family} parameter loader.")
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
  endif()
endforeach()

if(SPACEDIM GREATER 1)
  set(_fiber_type
    "ImmersX::FiberReinforcedElastodynamicsParameters<${SPACEDIM}>")
  string(FIND "${_registry}" "${_fiber_type}" _fiber_parameters_found)
  if(_fiber_parameters_found EQUAL -1)
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d is missing ${_fiber_type}.")
  endif()

  string(FIND "${_registry}"
    "Load parameters::std::function<void (ImmersX::FiberReinforcedElastodynamicsParameters<${SPACEDIM}> &"
    _fiber_loader_found)
  if(_fiber_loader_found EQUAL -1)
    message(FATAL_ERROR
      "Registry ${SPACEDIM}d is missing the generic fiber parameter loader.")
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
    "Field value"
    "Field gradient"
    "Field divergence"
    "Field symmetric gradient")
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
  string(FIND "${_registry}" "Field curl" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "Registry is missing the 'Field curl' operation family.")
  endif()
else()
  string(FIND "${_registry}" "Field curl" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR "1D registry must not expose the unsupported Field curl operation.")
  endif()
endif()

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
      "Create linear state"
      "Solve linear state"
      "Evaluate linear residual"
      "Linear state norm"
      "Assert finite below"
      "Write scalar field")
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
