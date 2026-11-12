# The exported-field quadrature checks require only the Python standard library.
if(ENABLE_GOOGLE_TESTING OR ENABLE_DEAL_II_APP_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME ImmersX.MetricFlowX.MMSQuadrature
      COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_SOURCE_DIR}/scripts/test_mfx_modal_quadrature.py")
    set_tests_properties(ImmersX.MetricFlowX.MMSQuadrature PROPERTIES
      LABELS "unit;quick" TIMEOUT 60)
    foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
      if(NOT _target MATCHES "^fitted_vessel_mesh(_debug)?$")
        continue()
      endif()
      set(_test "ImmersX.Application.${_target}.Mesh")
      add_test(NAME "${_test}"
        COMMAND "${Python3_EXECUTABLE}" -B
          "${CMAKE_SOURCE_DIR}/scripts/test_fitted_vessel_mesh.py"
          --executable "$<TARGET_FILE:${_target}>"
          --output "${TEST_OUTPUT_DIR}/fitted-vessel-mesh/${_target}")
      set_tests_properties("${_test}" PROPERTIES
        LABELS "integration;quick" TIMEOUT 90
        ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
    endforeach()
  endif()
endif()
