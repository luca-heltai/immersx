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
    string(JSON _arity ERROR_VARIABLE _error
      LENGTH "${_registry}" "${_operation}" arguments)
    if(_error OR NOT _arity EQUAL _expected_arity)
      message(FATAL_ERROR "Elasticity operation '${_operation}' has the wrong arity.")
    endif()
  endforeach()
endif()
