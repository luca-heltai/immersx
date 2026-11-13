# Exercise the installed application workflow with generated build-tree inputs.
if(IMMERSX_WITH_METRIC_FLOW_X AND DEAL_II_WITH_SUNDIALS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  foreach(_build_type IN LISTS IMMERSX_REQUESTED_BUILD_TYPES)
    foreach(_mode Serial MPI)
      if(_mode STREQUAL "MPI" AND NOT _immersx_mpiexec)
        continue()
      endif()
      set(_name "ImmersX${${_build_type}_ctest_suffix}.application.MetricFlowXElastostatic.${_mode}")
      set(_saved_output "${TEST_OUTPUT_DIR}")
      set(TEST_OUTPUT_DIR "${_saved_output}/${_build_type}/elastostatic-${_mode}")
      set(_parameter "${CMAKE_CURRENT_BINARY_DIR}/parameters/metric_elastostatic_${_build_type}_${_mode}.prm")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_elastostatic.prm.in"
                     "${_parameter}" @ONLY)
      set(_mpi_args)
      set(_labels "application")
      set(_processors 1)
      if(_mode STREQUAL "MPI")
        set(_mpi_args "-DMPIEXEC=${_immersx_mpiexec}" -DMPI_RANKS=2)
        set(_labels "application;mpi")
        set(_processors 2)
      endif()
      add_test(NAME "${_name}" COMMAND ${CMAKE_COMMAND}
        "-DTEST_EXECUTABLE=$<TARGET_FILE:metric_flow_x_elastostatic${${_build_type}_postfix}>"
        "-DPARAMETER_FILE=${_parameter}"
        "-DOUTPUT_DIRECTORY=${TEST_OUTPUT_DIR}/metric-flow-x-elastostatic"
        "-DPARTICLE_PROPERTIES=weight,mode_values,modal_coefficients,lifted_value,modal_data,representative_point,radius,indices,bulk_field"
        "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DPARTICLE_CHECKER=${CMAKE_SOURCE_DIR}/scripts/lifted_particle_output_checks.py"
        "-DEXPECTED_TIMES=0;0.001;0.002"
        -DAPPLICATION_TIMEOUT=300 ${_mpi_args}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/run_metric_flow_x_coupled.cmake")
      set_tests_properties("${_name}" PROPERTIES LABELS "${_labels}"
        TIMEOUT 330 PROCESSORS ${_processors} RESOURCE_LOCK ImmersX.TestOutput)
      set(_restart_args)
      if(_mode STREQUAL "MPI")
        set(_restart_args --mpiexec "${_immersx_mpiexec}" --ranks 2)
      endif()
      set(_restart_name "ImmersX${${_build_type}_ctest_suffix}.application.MetricFlowXElastostatic.Restart.${_mode}")
      add_test(NAME "${_restart_name}" COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/run_metric_flow_x_elastostatic_restart.py"
        --executable "$<TARGET_FILE:metric_flow_x_elastostatic${${_build_type}_postfix}>"
        --parameter "${_parameter}" --output-directory "${TEST_OUTPUT_DIR}/restart"
        --feedback both ${_restart_args})
      set_tests_properties("${_restart_name}" PROPERTIES LABELS "${_labels}"
        TIMEOUT 1200 PROCESSORS ${_processors} RESOURCE_LOCK ImmersX.TestOutput
        SKIP_RETURN_CODE 77)
      set(_refined_restart_name "ImmersX${${_build_type}_ctest_suffix}.application.MetricFlowXElastostatic.Restart.CoRefinement.${_mode}")
      add_test(NAME "${_refined_restart_name}" COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/run_metric_flow_x_elastostatic_restart.py"
        --executable "$<TARGET_FILE:metric_flow_x_elastostatic${${_build_type}_postfix}>"
        --parameter "${_parameter}" --output-directory "${TEST_OUTPUT_DIR}/refined-restart"
        --co-refinement --feedback one-way ${_restart_args})
      set_tests_properties("${_refined_restart_name}" PROPERTIES LABELS "${_labels}"
        TIMEOUT 1200 PROCESSORS ${_processors} RESOURCE_LOCK ImmersX.TestOutput
        SKIP_RETURN_CODE 77)
      set(TEST_OUTPUT_DIR "${_saved_output}")
    endforeach()
  endforeach()
endif()
