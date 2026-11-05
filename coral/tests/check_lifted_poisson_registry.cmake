# The combined plugin must expose both scalar and vector modal components.
if(SPACEDIM EQUAL 3)
  foreach(_type
      "ImmersX::ModalLiftedObservable<1,2,3>"
      "ImmersX::LiftedMultiplierInteraction<3>"
      "ImmersX::ModalDataEvaluator<3>")
    string(JSON _node_type ERROR_VARIABLE _error
      GET "${_registry}" "${_type}" node_type)
    if(_error OR NOT _node_type STREQUAL "output_only")
      message(FATAL_ERROR "Missing output-only Poisson component '${_type}'.")
    endif()
  endforeach()
  foreach(_spec
      "Modal lift::ImmersX::ModalLiftFactory<1,2,3>()|2"
      "Modal lift::ImmersX::ImportedModalLiftFactory<1,2,3>()|3"
      "Modal data::ImmersX::ModalDataFactory<3>()|2"
      "Modal data::ImmersX::ImportedModalDataFactory<3>()|4"
      "Lifted multiplier interaction::ImmersX::LiftedMultiplierFactory<3>()|3"
      "Lifted multiplier interaction::ImmersX::ImportedLiftedMultiplierFactory<3>()|3"
      "Add lifted multiplier::ImmersX::AddLiftedMultiplier<3>()|3"
      "Modal output coefficient field::ImmersX::ModalCoefficientField<1,3>()|2"
      "Write modal lift::ImmersX::WriteModalLift<3>()|6")
    string(REPLACE "|" ";" _parts "${_spec}")
    list(GET _parts 0 _operation)
    list(GET _parts 1 _expected_arity)
    string(JSON _arity ERROR_VARIABLE _error
      LENGTH "${_registry}" "${_operation}" arguments)
    if(_error OR NOT _arity EQUAL _expected_arity)
      message(FATAL_ERROR "Poisson operation '${_operation}' has the wrong arity.")
    endif()
  endforeach()
  string(JSON _registry_length LENGTH "${_registry}")
  math(EXPR _registry_last "${_registry_length} - 1")
  set(_robin_found FALSE)
  foreach(_index RANGE 0 ${_registry_last})
    string(JSON _operation MEMBER "${_registry}" ${_index})
    if(_operation MATCHES "^Add scalar Robin boundary(::|$)")
      string(JSON _arity LENGTH "${_registry}" "${_operation}" arguments)
      if(NOT _arity EQUAL 3)
        message(FATAL_ERROR "Scalar Robin operation '${_operation}' has the wrong arity.")
      endif()
      set(_robin_found TRUE)
    endif()
  endforeach()
  if(NOT _robin_found)
    message(FATAL_ERROR "Missing scalar Robin boundary operation.")
  endif()
endif()
