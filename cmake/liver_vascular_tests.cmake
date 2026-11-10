# A synthetic surface and two 1D trees exercise the liver workflow without
# downloading patient data or requiring Python VTK/VMTK in CI.
if(NOT IMMERSX_PYTHON_EXECUTABLE OR NOT DEAL_II_WITH_VTK OR
   NOT DEAL_II_GMSH_WITH_API)
  return()
endif()
find_program(_liver_mpiexec NAMES mpirun mpiexec)
set(_liver_ranks 1)
if(_liver_mpiexec)
  list(APPEND _liver_ranks 2)
endif()
foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
  if(NOT _target MATCHES "^liver_vascular(_debug)?$")
    continue()
  endif()
  foreach(_ranks IN LISTS _liver_ranks)
    set(_name "ImmersX.Application.${_target}.${_ranks}Rank")
    add_test(NAME "${_name}"
      COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/test_liver_vascular.py"
        --executable "$<TARGET_FILE:${_target}>"
        --parameters "${CMAKE_BINARY_DIR}/data/vascularization/liver_smoke/smoke.prm"
        --output "${TEST_OUTPUT_DIR}/liver-smoke/${_target}/${_ranks}"
        --ranks "${_ranks}" --mpiexec "${_liver_mpiexec}" --keep-output)
    add_test(NAME "${_name}.Portal"
      COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/test_liver_vascular.py"
        --executable "$<TARGET_FILE:${_target}>"
        --parameters "${CMAKE_BINARY_DIR}/data/vascularization/liver_smoke/portal.prm"
        --output "${TEST_OUTPUT_DIR}/liver-smoke/${_target}/${_ranks}/portal"
        --portal --ranks "${_ranks}" --mpiexec "${_liver_mpiexec}" --keep-output)
    add_test(NAME "${_name}.Robin"
      COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/test_liver_vascular.py"
        --executable "$<TARGET_FILE:${_target}>"
        --parameters "${CMAKE_BINARY_DIR}/data/vascularization/liver_smoke/robin.prm"
        --output "${TEST_OUTPUT_DIR}/liver-smoke/${_target}/${_ranks}/robin"
        --robin --ranks "${_ranks}" --mpiexec "${_liver_mpiexec}" --keep-output)
    add_test(NAME "${_name}.InvalidOutput"
      COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/test_liver_invalid_output.py"
        --executable "$<TARGET_FILE:${_target}>"
        --parameters "${CMAKE_BINARY_DIR}/data/vascularization/liver_smoke/portal.prm"
        --output "${TEST_OUTPUT_DIR}/liver-invalid-output/${_target}/${_ranks}"
        --ranks "${_ranks}" --mpiexec "${_liver_mpiexec}")
    set_tests_properties("${_name}.InvalidOutput" PROPERTIES LABELS "application"
      TIMEOUT 115 PROCESSORS "${_ranks}"
      ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
    set_tests_properties("${_name}.Robin" PROPERTIES LABELS "application"
      TIMEOUT 115 PROCESSORS "${_ranks}"
      ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
    set_tests_properties("${_name}" PROPERTIES LABELS "application"
      TIMEOUT 115 PROCESSORS "${_ranks}"
      ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
    set_tests_properties("${_name}.Portal" PROPERTIES LABELS "application" TIMEOUT 115 PROCESSORS "${_ranks}"
      ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
    if(_ranks GREATER 1)
      set_property(TEST "${_name}" "${_name}.Portal" "${_name}.Robin" "${_name}.InvalidOutput" APPEND PROPERTY LABELS "mpi")
    endif()
  endforeach()
endforeach()
