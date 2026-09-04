## ---------------------------------------------------------------------
##
## Copyright (C) 2000 2020 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------
set(_target gtests)

file(GLOB _test_files CONFIGURE_DEPENDS *cc)
list(FILTER _test_files EXCLUDE REGEX "metric_flow_x_01\\.cc$")
list(FILTER _test_files EXCLUDE REGEX "mms_convergence_main\\.cc$")

include(GoogleTest)
include(${CMAKE_CURRENT_SOURCE_DIR}/test_categories.cmake)

# Compile selected installed-facing headers independently. These translation
# units intentionally contain no project umbrella include or namespace using
# directive: a failure here is a public-header dependency or namespace error,
# not a runtime test failure.
set(_public_headers
    immersx/core/constraint_equation.h
    immersx/core/constraint_contributor.h
    immersx/core/contributor.h
    immersx/core/expression_representation.h
    immersx/core/fe_space.h
    immersx/core/field.h
    immersx/core/input_field_selector.h
    immersx/core/lifting.h
    immersx/core/load_interaction.h
    immersx/core/reduced_field_catalog.h
    immersx/core/reduced_field_utils.h
    immersx/core/reduced_field_values.h
    immersx/core/representation.h
    immersx/core/semidiscrete_pde_models.h
    immersx/core/state.h
    immersx/core/state_history.h
    immersx/core/sundials_ida_adapter.h
    immersx/core/time_parameters.h
    immersx/core/linear_adapter.h
    immersx/core/matrix_operator.h
    immersx/core/observable.h
    immersx/core/weak_term.h
    immersx/core/problem_handle.h
    immersx/core/symbolic_field_evaluator.h
    immersx/core/time_residual.h
    immersx/coupling/coupling_operator.h
    immersx/coupling/tensor_product_lift.h
    immersx/physics/elastic_static.h
    immersx/algebra/lagrange_multiplier_interaction.h
    immersx/algebra/lagrange_multiplier_schur_solver.h
    immersx/algebra/linear_algebra.h
    immersx/algebra/vector_lagrange_multiplier_interaction.h
    immersx/coupling/particle_coupling.h
    immersx/io/imported_finite_element_fields.h
    immersx/physics/elastodynamics.h
    immersx/physics/elastodynamics_semidiscrete.h
    immersx/physics/fiber_reinforced_elastodynamics.h
    immersx/physics/navier_stokes.h
    immersx/physics/navier_stokes_semidiscrete.h
    immersx/physics/poisson_residual.h)
if(IMMERSX_WITH_METRIC_FLOW_X)
  list(APPEND _public_headers
       immersx/physics/metric_flow_x.h
       immersx/physics/metric_flow_x_vessel_wall_representation.h
       immersx/algebra/vessel_wall_interaction.h)
endif()

set(_public_header_targets)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _public_build_type)
foreach(_public_header ${_public_headers})
  string(REPLACE "/" "_" _public_header_id "${_public_header}")
  string(REPLACE "." "_" _public_header_id "${_public_header_id}")
  set(PUBLIC_HEADER "${_public_header}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/public_header_compile.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/public_header_${_public_header_id}.cpp"
    @ONLY)

  set(_public_header_target "public_header_${_public_header_id}")
  add_executable(
    ${_public_header_target}
    "${CMAKE_CURRENT_BINARY_DIR}/public_header_${_public_header_id}.cpp")
  target_include_directories(${_public_header_target}
                             PRIVATE ${CMAKE_SOURCE_DIR}/include
                                     ${CMAKE_BINARY_DIR}/include)
  if(IMMERSX_WITH_METRIC_FLOW_X AND
     (_public_header STREQUAL "immersx/physics/metric_flow_x.h" OR
      _public_header STREQUAL
        "immersx/physics/metric_flow_x_vessel_wall_representation.h" OR
      _public_header STREQUAL "immersx/algebra/vessel_wall_interaction.h"))
    target_link_libraries(${_public_header_target}
                          MetricFlowX::metric_flow_x)
  endif()
  DEAL_II_SETUP_TARGET(${_public_header_target} ${_public_build_type})
  list(APPEND _public_header_targets ${_public_header_target})
endforeach()

add_custom_target(immersx_public_header_self_containment ALL
                  DEPENDS ${_public_header_targets})
add_test(NAME ImmersX.PublicHeaderSelfContainment
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                 --target immersx_public_header_self_containment)
set_tests_properties(ImmersX.PublicHeaderSelfContainment
                     PROPERTIES LABELS "unit;quick")

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")

    MESSAGE("-- Found ${_build_type} version of deal.II.")
    
    STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
    SET(_exe "${_target}${${_build_type}_postfix}")
    
    MESSAGE("-- Setting up ${_exe}")
    add_executable(${_exe} ${_test_files})
    target_include_directories(${_exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                               ${CMAKE_SOURCE_DIR}/apps)

    TARGET_LINK_LIBRARIES(${_exe} ${TEST_LIBRARIES_${_BUILD_TYPE}} ${GTEST_MAIN_LIBRARY} ${GTEST_LIBRARIES})   
    # Test paths are configured explicitly so the gtest binary can be launched
    # from any working directory.
    TARGET_COMPILE_DEFINITIONS(${_exe} PRIVATE
      SOURCE_DIR="${IMMERSX_SOURCE_DIR}"
      IMMERSX_SOURCE_DIR="${IMMERSX_SOURCE_DIR}"
      IMMERSX_BINARY_DIR="${IMMERSX_BINARY_DIR}"
      TEST_DATA_DIR="${TEST_DATA_DIR}"
      TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
    IF(ENABLE_COUPLED_PROBLEMS)
      TARGET_COMPILE_DEFINITIONS(${_exe} PRIVATE ENABLE_COUPLED_PROBLEMS)
    ENDIF()
    DEAL_II_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
    target_include_directories(${_exe} PRIVATE ${GTEST_INCLUDE_DIRS})
    LIST(APPEND IMMERSX_TEST_TARGETS ${_exe})

    string(JOIN ":" _unit_filter ${IMMERSX_UNIT_TEST_SUITES})
    string(JOIN ":" _integration_filter ${IMMERSX_INTEGRATION_TEST_SUITES})
    string(JOIN ":" _validation_filter ${IMMERSX_VALIDATION_TEST_SUITES})
    string(JOIN ":" _application_filter ${IMMERSX_APPLICATION_TEST_SUITES})
    set(_unit_mpi_suites)
    foreach(_suite ${IMMERSX_UNIT_TEST_SUITES})
      list(APPEND _unit_mpi_suites "${_suite}.MPI_*")
    endforeach()
    set(_integration_mpi_suites)
    foreach(_suite ${IMMERSX_INTEGRATION_TEST_SUITES})
      list(APPEND _integration_mpi_suites "${_suite}.MPI_*")
    endforeach()
    set(_validation_mpi_suites)
    foreach(_suite ${IMMERSX_VALIDATION_TEST_SUITES})
      list(APPEND _validation_mpi_suites "${_suite}.MPI_*")
    endforeach()
    set(_application_mpi_suites)
    foreach(_suite ${IMMERSX_APPLICATION_TEST_SUITES})
      list(APPEND _application_mpi_suites "${_suite}.MPI_*")
    endforeach()
    string(JOIN ":" _unit_mpi_filter ${_unit_mpi_suites})
    string(JOIN ":" _integration_mpi_filter ${_integration_mpi_suites})
    string(JOIN ":" _validation_mpi_filter ${_validation_mpi_suites})
    string(JOIN ":" _application_mpi_filter ${_application_mpi_suites})
    find_program(_immersx_mpiexec NAMES mpirun mpiexec)

    # One aggregate executable keeps the translation units compiled once.
    # CTest provides the user-facing category and process-count matrix.
    foreach(_category unit integration validation application)
      set(_category_quick FALSE)
      if(_category STREQUAL "unit")
        set(_category_filter "${_unit_filter}")
        set(_category_mpi_filter "${_unit_mpi_filter}")
        set(_category_quick TRUE)
      elseif(_category STREQUAL "integration")
        set(_category_filter "${_integration_filter}")
        set(_category_mpi_filter "${_integration_mpi_filter}")
        set(_category_quick TRUE)
      elseif(_category STREQUAL "validation")
        set(_category_filter "${_validation_filter}")
        set(_category_mpi_filter "${_validation_mpi_filter}")
      else()
        set(_category_filter "${_application_filter}")
        set(_category_mpi_filter "${_application_mpi_filter}")
      endif()

      set(_serial_test "ImmersX.${_category}.Serial")
      add_test(NAME ${_serial_test}
               COMMAND $<TARGET_FILE:${_exe}>
                       --gtest_filter=${_category_filter}-*.MPI_*)
      set(_labels "${_category}")
      if(_category_quick)
        list(APPEND _labels quick)
      endif()
      set_tests_properties(${_serial_test} PROPERTIES LABELS "${_labels}")

      if(_immersx_mpiexec)
        set(_mpi_test "ImmersX.${_category}.MPI")
        add_test(NAME ${_mpi_test}
                 COMMAND ${_immersx_mpiexec} -np 2
                         $<TARGET_FILE:${_exe}>
                         --gtest_filter=${_category_mpi_filter})
        set_tests_properties(${_mpi_test} PROPERTIES LABELS "${_labels};mpi")
      endif()
    endforeach()
    IF(IMMERSX_INSTALL_TESTS)
      INSTALL(TARGETS ${_exe}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/immersx/tests)
    ENDIF()
    
    # gtest_discover_tests(${_exe})
  ENDIF()
ENDFOREACH()

if(IMMERSX_WITH_METRIC_FLOW_X)
  foreach(_build_type ${_d2_build_types})
    if(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
       DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      string(TOUPPER "${_build_type}" _BUILD_TYPE)
      set(_metric_flow_x_test_target
          "metric_flow_x_tests${${_build_type}_postfix}")
      add_executable(
        ${_metric_flow_x_test_target}
        metric_flow_x_01.cc
        gtest_main.cc)
      target_include_directories(
        ${_metric_flow_x_test_target}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR}/include)
      target_compile_definitions(
        ${_metric_flow_x_test_target}
        PRIVATE SOURCE_DIR="${IMMERSX_SOURCE_DIR}"
                IMMERSX_SOURCE_DIR="${IMMERSX_SOURCE_DIR}"
                IMMERSX_BINARY_DIR="${IMMERSX_BINARY_DIR}"
                TEST_DATA_DIR="${TEST_DATA_DIR}"
                TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
      target_link_libraries(
        ${_metric_flow_x_test_target}
        ${TEST_LIBRARIES_${_BUILD_TYPE}}
        ${GTEST_MAIN_LIBRARY}
        ${GTEST_LIBRARIES})
      target_include_directories(${_metric_flow_x_test_target}
                                 PRIVATE ${GTEST_INCLUDE_DIRS})
      DEAL_II_SETUP_TARGET(${_metric_flow_x_test_target} ${_BUILD_TYPE})
      add_test(NAME ImmersX.integration.MetricFlowX.Serial
               COMMAND $<TARGET_FILE:${_metric_flow_x_test_target}>
                       --gtest_filter=MetricFlowX.*-*.MPI_*)
      set_tests_properties(ImmersX.integration.MetricFlowX.Serial
                           PROPERTIES LABELS "integration;quick")
      find_program(_immersx_mpiexec NAMES mpirun mpiexec)
      if(_immersx_mpiexec)
        add_test(NAME ImmersX.integration.MetricFlowX.MPI
                 COMMAND ${_immersx_mpiexec} -np 2
                         $<TARGET_FILE:${_metric_flow_x_test_target}>
                         --gtest_filter=MetricFlowX.MPI_*)
        set_tests_properties(ImmersX.integration.MetricFlowX.MPI
                             PROPERTIES LABELS "integration;quick;mpi")
      endif()
      list(APPEND IMMERSX_TEST_TARGETS ${_metric_flow_x_test_target})
      if(IMMERSX_INSTALL_TESTS)
        install(TARGETS ${_metric_flow_x_test_target}
                RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/immersx/tests)
      endif()
    endif()
  endforeach()

  # Keep the one-vessel MMS algebra and two-way residual composition as a
  # small, independently runnable vertical gate. The aggregate gtests
  # executable filters MPI tests by process count; this target lets CTest
  # invoke the gate directly.
  foreach(_build_type ${_d2_build_types})
    if(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
       DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      string(TOUPPER "${_build_type}" _BUILD_TYPE)
      set(_mms_gate_target
          "metric_flow_x_elastodynamics_mms_verification${${_build_type}_postfix}")
      add_executable(${_mms_gate_target}
                     metric_flow_x_vessel_wall_interaction_01.cc
                     metric_flow_x_elastodynamics_mms_01.cc
                     metric_flow_x_elastodynamics_mms_driver.cc
                     mms_convergence_main.cc)
      target_include_directories(
        ${_mms_gate_target}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_BINARY_DIR}/include)
      target_compile_definitions(
        ${_mms_gate_target}
        PRIVATE SOURCE_DIR="${IMMERSX_SOURCE_DIR}"
                IMMERSX_SOURCE_DIR="${IMMERSX_SOURCE_DIR}"
                IMMERSX_BINARY_DIR="${IMMERSX_BINARY_DIR}"
                TEST_DATA_DIR="${TEST_DATA_DIR}"
                TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")
      target_link_libraries(
        ${_mms_gate_target}
        ${TEST_LIBRARIES_${_BUILD_TYPE}}
        ${GTEST_LIBRARIES})
      target_include_directories(${_mms_gate_target}
                                 PRIVATE ${GTEST_INCLUDE_DIRS})
      DEAL_II_SETUP_TARGET(${_mms_gate_target} ${_BUILD_TYPE})
      find_program(_immersx_mpiexec NAMES mpirun mpiexec)
      if(_immersx_mpiexec)
        add_test(
          NAME metric_flow_x_elastodynamics_mms_verification
          COMMAND ${_immersx_mpiexec} -np 2
                  $<TARGET_FILE:${_mms_gate_target}>
                  --gtest_filter=OneVesselMMS.*:OneVesselMMSDriver.MPI_ActualTwoWayTransientSolve:MetricFlowXVesselWallInteraction.MPI_TwoWayResidualAndPressureSign:MetricFlowXVesselWallInteraction.MPI_TwoWayCompositionResidualSmoke:MetricFlowXVesselWallInteraction.MPI_DiscreteVirtualWorkNormalization:MetricFlowXVesselWallInteraction.MPI_FullCoupledJacobianFiniteDifference)
        set_tests_properties(metric_flow_x_elastodynamics_mms_verification
                             PROPERTIES LABELS "validation;mpi")
      endif()
    endif()
  endforeach()
endif()
