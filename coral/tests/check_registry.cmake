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
  endforeach()
endforeach()

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

string(FIND "${_registry}" "ImmersX::PoissonParameters<1,${SPACEDIM}>" _found)
if(_found EQUAL -1)
  message(FATAL_ERROR "Registry is missing stable Poisson parameter type name.")
endif()

if(SPACEDIM EQUAL 2)
  string(FIND "${_registry}" "ImmersX::CoupledPoisson<2>" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "2D registry is missing the coupled Poisson façade.")
  endif()
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
else()
  string(FIND "${_registry}" "ImmersX::CoupledPoissonElasticity<3>" _found)
  if(NOT _found EQUAL -1)
    message(FATAL_ERROR
      "${SPACEDIM}D registry contains the 3D coupled Poisson-elasticity "
      "façade.")
  endif()
endif()
