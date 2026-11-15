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
    foreach(_case liver_icar/tiny liver_icar/small liver_icar/medium liver_icar/large
                  liver_icar_case11/tiny liver_icar_case11/small liver_icar_case11/medium)
      get_filename_component(_family "${_case}" DIRECTORY)
      get_filename_component(_profile "${_case}" NAME)
      set(_suite LiverParameters)
      if(_family STREQUAL "liver_icar_case11")
        set(_suite LiverCase11Parameters)
      endif()
      set(_parameter "${CMAKE_CURRENT_BINARY_DIR}/${_family}-${_profile}-parse.prm")
      file(WRITE "${_parameter}"
        "include ${TEST_DATA_DIR}/${_family}/${_profile}.prm\nsubsection MetricFlowX coupling\n  set Parse only = true\nend\n")
      set(_name "ImmersX${${_build_type}_ctest_suffix}.integration.${_suite}.${_profile}")
      add_test(NAME "${_name}"
        COMMAND $<TARGET_FILE:metric_flow_x_elastodynamics${${_build_type}_postfix}>
          "${_parameter}")
      set_tests_properties("${_name}" PROPERTIES LABELS "integration;quick"
        TIMEOUT 60 RUN_SERIAL TRUE)
    endforeach()
  endforeach()
endif()
