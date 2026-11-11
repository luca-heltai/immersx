# The exported-field quadrature checks require only the Python standard library.
if(ENABLE_GOOGLE_TESTING OR ENABLE_DEAL_II_APP_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME ImmersX.MetricFlowX.MMSQuadrature
      COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_SOURCE_DIR}/scripts/test_mfx_modal_quadrature.py")
    set_tests_properties(ImmersX.MetricFlowX.MMSQuadrature PROPERTIES
      LABELS "unit;quick" TIMEOUT 60)
  endif()
endif()
