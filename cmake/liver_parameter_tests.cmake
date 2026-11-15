if(ENABLE_GOOGLE_TESTING OR ENABLE_DEAL_II_APP_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME ImmersX.Liver.Runner
      COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_SOURCE_DIR}/tests/liver_icar/test_run_case.py")
    set_tests_properties(ImmersX.Liver.Runner PROPERTIES LABELS "unit;quick"
      TIMEOUT 60)
  endif()
endif()

# Parse the actual generated tutorial inputs without downloading liver assets.
if(IMMERSX_WITH_METRIC_FLOW_X AND DEAL_II_WITH_SUNDIALS)
  foreach(_build_type IN LISTS IMMERSX_REQUESTED_BUILD_TYPES)
    foreach(_profile tiny small medium large)
      set(_parameter "${CMAKE_CURRENT_BINARY_DIR}/liver-${_profile}-parse.prm")
      file(WRITE "${_parameter}"
        "include ${TEST_DATA_DIR}/liver_icar/${_profile}.prm\nsubsection MetricFlowX coupling\n  set Parse only = true\nend\n")
      set(_name "ImmersX${${_build_type}_ctest_suffix}.integration.LiverParameters.${_profile}")
      add_test(NAME "${_name}"
        COMMAND $<TARGET_FILE:metric_flow_x_elastodynamics${${_build_type}_postfix}>
          "${_parameter}")
      set_tests_properties("${_name}" PROPERTIES LABELS "integration;quick"
        TIMEOUT 60 RUN_SERIAL TRUE)
    endforeach()
  endforeach()
endif()
