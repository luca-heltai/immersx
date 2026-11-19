# Component-level static elasticity must remain a composable graph API.
if(SPACEDIM EQUAL 3)
  foreach(_type
      "ImmersX::ModalVectorLift"
      "ImmersX::ModalVectorLiftTest"
      "ImmersX::ModalVectorWeakTerm"
      "ImmersX::ModalVectorWeakSum"
      "ImmersX::ModalProjection"
      "ImmersX::LiftedModalInteraction"
      "ImmersX::IsotropicElasticStress<3>"
      "ImmersX::IsotropicElasticityTerm<3>")
    string(JSON _node_type ERROR_VARIABLE _error
      GET "${_registry}" "${_type}" node_type)
    if(_error OR NOT _node_type STREQUAL "output_only")
      message(FATAL_ERROR "Missing output-only elasticity component '${_type}'.")
    endif()
  endforeach()
  string(JSON _registry_size LENGTH "${_registry}")
  math(EXPR _registry_last_index "${_registry_size} - 1")
  foreach(_spec
      "Isotropic elastic law|2"
      "Elasticity energy term|2"
      "Add elasticity energy|2"
      "Add elasticity Robin boundary|3"
      "Modal coefficient field|3"
      "Vector tensor-product lift|3"
      "Lifted vector test|1"
      "Lifted vector weak term|2"
      "Modal weak sum|1"
      "Append modal weak term|2"
      "Project modal data|3"
      "Sum modal projections|2"
      "Modal multiplier interaction|2"
      "Add modal interaction|2"
      "Registered modal component|2")
    string(REPLACE "|" ";" _parts "${_spec}")
    list(GET _parts 0 _operation)
    list(GET _parts 1 _expected_arity)
    set(_registry_key "")
    foreach(_registry_index RANGE 0 ${_registry_last_index})
      string(JSON _candidate_key MEMBER "${_registry}" ${_registry_index})
      if(_candidate_key STREQUAL _operation)
        set(_registry_key "${_candidate_key}")
        break()
      endif()
      string(JSON _candidate_operation ERROR_VARIABLE _candidate_error
        GET "${_registry}" "${_candidate_key}" operation)
      if(_candidate_error STREQUAL "NOTFOUND" AND
         _candidate_operation STREQUAL _operation)
        set(_registry_key "${_candidate_key}")
        break()
      endif()
    endforeach()
    if(_registry_key STREQUAL "")
      message(FATAL_ERROR "Missing elasticity operation '${_operation}'.")
    endif()
    string(JSON _arity ERROR_VARIABLE _arity_error
      LENGTH "${_registry}" "${_registry_key}" arguments)
    if(NOT _arity_error STREQUAL "NOTFOUND" OR
       NOT _arity EQUAL _expected_arity)
      message(FATAL_ERROR "Elasticity operation '${_operation}' has the wrong arity.")
    endif()
  endforeach()
endif()
