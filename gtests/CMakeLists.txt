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

include(GoogleTest)
include(${CMAKE_CURRENT_SOURCE_DIR}/test_categories.cmake)

# Compile selected installed-facing headers independently. These translation
# units intentionally contain no project umbrella include or namespace using
# directive: a failure here is a public-header dependency or namespace error,
# not a runtime test failure.
set(_public_headers
    immersx/vascularization/metric_flow_x.h
    immersx/vascularization/application.h
    immersx/vascularization/io.h
    immersx/vascularization/distribution.h
    immersx/vascularization/reference_flow.h
    immersx/vascularization/mesh_scale.h
    immersx/vascularization/metrics.h
    immersx/vascularization/discretization.h
    immersx/vascularization/generation.h
    immersx/vascularization/growth_work.h
    immersx/vascularization/network.h
    immersx/compatibility/reference_cell.h
    immersx/core/boundary_conditions.h
    immersx/core/constraint.h
    immersx/core/contributor.h
    immersx/core/domain.h
    immersx/grid/local_co_refinement.h
    immersx/grid/overlap_classification.h
    immersx/grid/local_co_refinement_parameters.h
    immersx/core/fe_space.h
    immersx/core/field.h
    immersx/core/field_contributor.h
    immersx/core/overlap_integration_support.h
    immersx/core/integration_support.h
    immersx/core/lifted_integration_support.h
    immersx/core/input_field_selector.h
    immersx/core/known_term.h
    immersx/core/reduced_field_catalog.h
    immersx/core/reduced_field_utils.h
    immersx/core/reduced_field_values.h
    immersx/core/semidiscrete_pde_models.h
    immersx/core/second_order_preconditioner.h
    immersx/core/state.h
    immersx/core/state_history.h
    immersx/core/sundials_ida_adapter.h
    immersx/compatibility/dealii_9_9/ida.h
    immersx/compatibility/dealii_9_9/n_vector.h
    immersx/compatibility/dealii_9_9/n_vector.templates.h
    immersx/compatibility/dealii_9_9/utilities.h
    immersx/core/sundials_kinsol_adapter.h
    immersx/core/time_parameters.h
    immersx/core/ida_tolerance_ramp.h
    immersx/core/linear_adapter.h
    immersx/core/matrix_operator.h
    immersx/core/observable.h
    immersx/core/observable_lift.h
    immersx/core/robin_boundary.h
    immersx/core/robin_term.h
    immersx/core/weak_term.h
    immersx/core/problem_handle.h
    immersx/core/symbolic_field_evaluator.h
    immersx/core/time_residual.h
    immersx/coupling/tensor_product_lift.h
    immersx/physics/elastic_static.h
    immersx/physics/coupled_elasticity.h
    immersx/algebra/lagrange_multiplier_schur_solver.h
    immersx/algebra/linear_algebra.h
    immersx/algebra/local_preconditioner.h
    immersx/coupling/particle_coupling.h
    immersx/io/imported_finite_element_fields.h
    immersx/io/lifted_particle_output.h
    immersx/io/output_handler.h
    immersx/physics/elasticity_freq.h
    immersx/physics/elasticity_freq_contributor.h
    immersx/physics/elastodynamics.h
    immersx/physics/elastodynamics_semidiscrete.h
    immersx/physics/fiber_reinforced_elasticity_freq.h
    immersx/physics/fiber_reinforced_elastodynamics.h
    immersx/physics/navier_stokes.h
    immersx/physics/navier_stokes_semidiscrete.h
    immersx/physics/poisson_residual.h)
if(IMMERSX_WITH_METRIC_FLOW_X)
  list(APPEND _public_headers immersx/physics/metric_flow_x.h
                             immersx/physics/metric_flow_x_pressure.h)
endif()

set(_public_header_targets)
set(_public_header_variant_targets)
foreach(_build_type ${_d2_build_types})
  if(_build_type IN_LIST IMMERSX_REQUESTED_BUILD_TYPES)
    string(TOUPPER "${_build_type}" _BUILD_TYPE)
    string(TOLOWER "${_build_type}" _build_type_lower)
    set(_public_header_variant_targets_for_build)
    foreach(_public_header ${_public_headers})
      string(REPLACE "/" "_" _public_header_id "${_public_header}")
      string(REPLACE "." "_" _public_header_id "${_public_header_id}")
      set(PUBLIC_HEADER "${_public_header}")
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/public_header_compile.cpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/public_header_${_public_header_id}.cpp"
        @ONLY)

      set(_public_header_target
          "public_header_${_public_header_id}${${_build_type}_postfix}")
      add_library(
        ${_public_header_target} OBJECT
        "${CMAKE_CURRENT_BINARY_DIR}/public_header_${_public_header_id}.cpp")
      target_include_directories(${_public_header_target}
                                 PRIVATE ${CMAKE_SOURCE_DIR}/include
                                         ${CMAKE_BINARY_DIR}/include)
      target_compile_options(${_public_header_target} PRIVATE
                             $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
                             $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
                             $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>)
      if(IMMERSX_WITH_METRIC_FLOW_X AND
         _public_header MATCHES "^immersx/(physics|vascularization)/metric_flow_x")
        target_link_libraries(${_public_header_target}
                              ${IMMERSX_METRIC_FLOW_X_TARGET_${_build_type}})
      endif()
      DEAL_II_SETUP_TARGET(${_public_header_target} ${_BUILD_TYPE})
      list(APPEND _public_header_targets ${_public_header_target})
      list(APPEND _public_header_variant_targets_for_build
           ${_public_header_target})
    endforeach()
    add_custom_target(
      "immersx_public_header_self_containment_${_build_type_lower}"
      DEPENDS ${_public_header_variant_targets_for_build})
    list(APPEND _public_header_variant_targets
         "immersx_public_header_self_containment_${_build_type_lower}")

    set(_public_header_test
        "ImmersX${${_build_type}_ctest_suffix}.PublicHeaderSelfContainment")
    add_test(NAME ${_public_header_test}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                     --target
                     "immersx_public_header_self_containment_${_build_type_lower}")
    set_tests_properties(${_public_header_test}
                         PROPERTIES LABELS "unit;quick" TIMEOUT 300 RUN_SERIAL TRUE)
  endif()
endforeach()

add_custom_target(immersx_public_header_self_containment ALL
                  DEPENDS ${_public_header_variant_targets})

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build requested variants that are provided by deal.II.
  IF(_build_type IN_LIST IMMERSX_REQUESTED_BUILD_TYPES)

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
      TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}"
      IMMERSX_WEAK_TERM_TESTING)
    TARGET_COMPILE_OPTIONS(${_exe} PRIVATE
      $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
      $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
      $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>)
    IF(ENABLE_COUPLED_PROBLEMS)
      TARGET_COMPILE_DEFINITIONS(${_exe} PRIVATE ENABLE_COUPLED_PROBLEMS)
    ENDIF()
    # When the optional library is absent, still compile its clients so their
    # public interfaces remain checked. Enabled application targets already
    # provide this compilation in the normal build.
    if(ENABLE_COUPLED_PROBLEMS)
      add_dependencies(${_exe} coupled_elasticity${_p} pseudocoupling1D${_p})
    elseif(OpenMP_CXX_FOUND AND
       NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
      set(_optional_1d_clients "immersx_optional_1d_clients${_p}")
      add_library(${_optional_1d_clients} OBJECT
        "${CMAKE_SOURCE_DIR}/apps/app_coupled_elasticity.cc"
        "${CMAKE_SOURCE_DIR}/apps/app_pseudocoupling1D.cc")
      target_compile_definitions(${_optional_1d_clients} PRIVATE ENABLE_COUPLED_PROBLEMS)
      target_compile_options(${_optional_1d_clients} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>)
      target_link_libraries(${_optional_1d_clients} ${TEST_LIBRARIES_${_BUILD_TYPE}})
      DEAL_II_SETUP_TARGET(${_optional_1d_clients} ${_BUILD_TYPE})
      add_dependencies(${_exe} ${_optional_1d_clients})
    endif()
    DEAL_II_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
    target_include_directories(${_exe} PRIVATE ${GTEST_INCLUDE_DIRS})
    LIST(APPEND IMMERSX_TEST_TARGETS ${_exe})

    add_test(
      NAME "ImmersX${${_build_type}_ctest_suffix}.TestCategoryCoverage"
      COMMAND ${CMAKE_COMMAND}
              "-DTEST_EXECUTABLE=$<TARGET_FILE:${_exe}>"
              "-DCATEGORY_FILE=${CMAKE_CURRENT_SOURCE_DIR}/test_categories.cmake"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/check_test_categories.cmake")
    set_tests_properties(
      "ImmersX${${_build_type}_ctest_suffix}.TestCategoryCoverage"
      PROPERTIES LABELS "unit;quick" TIMEOUT 60)

    add_test(
      NAME "ImmersX${${_build_type}_ctest_suffix}.TestCategoryGuard"
      COMMAND ${CMAKE_COMMAND}
              "-DTEST_EXECUTABLE=$<TARGET_FILE:${_exe}>"
              "-DCATEGORY_FILE=${CMAKE_CURRENT_SOURCE_DIR}/test_categories.cmake"
              "-DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/category-guard-${_build_type}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/test_category_coverage_guard.cmake")
    set_tests_properties(
      "ImmersX${${_build_type}_ctest_suffix}.TestCategoryGuard"
      PROPERTIES LABELS "unit;quick" TIMEOUT 60)

    # Discover the compiled registry after linking. Integration suites get
    # independent CTest entries without compiling their translation units twice.
    set(_groups_file "${CMAKE_CURRENT_BINARY_DIR}/${_exe}_test_groups.cmake")
    set(_groups_include "${CMAKE_CURRENT_BINARY_DIR}/${_exe}_test_include.cmake")
    set(_groups_script "${CMAKE_CURRENT_SOURCE_DIR}/discover_test_groups.cmake")
    find_program(_immersx_mpiexec NAMES mpirun mpiexec)
    if(NOT _immersx_mpiexec)
      set(_immersx_mpiexec "")
    endif()
    add_custom_command(TARGET ${_exe} POST_BUILD
      BYPRODUCTS "${_groups_file}"
      COMMAND ${CMAKE_COMMAND}
              "-DTEST_EXECUTABLE=$<TARGET_FILE:${_exe}>"
              "-DCATEGORY_FILE=${CMAKE_CURRENT_SOURCE_DIR}/test_categories.cmake"
              "-DTEST_NAME_PREFIX=ImmersX${${_build_type}_ctest_suffix}"
              "-DTEST_RESULTS_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-results"
              "-DMPIEXEC=${_immersx_mpiexec}"
              "-DOUTPUT_FILE=${_groups_file}"
              -P "${_groups_script}"
      VERBATIM)
    set_property(TARGET ${_exe} APPEND PROPERTY LINK_DEPENDS
      "${_groups_script}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/check_test_categories.cmake")
    file(GENERATE OUTPUT "${_groups_include}" CONTENT
"if(EXISTS [==[${_groups_file}]==])
  include([==[${_groups_file}]==])
else()
  add_test(ImmersX${${_build_type}_ctest_suffix}.GoogleTests.NOT_BUILT NOT_BUILT)
endif()
")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_groups_include}")
    IF(IMMERSX_INSTALL_TESTS)
      INSTALL(TARGETS ${_exe}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/immersx/tests)
    ENDIF()
    
    # gtest_discover_tests(${_exe})
  ENDIF()
ENDFOREACH()

if(IMMERSX_WITH_METRIC_FLOW_X)
  foreach(_build_type ${_d2_build_types})
    if(_build_type IN_LIST IMMERSX_REQUESTED_BUILD_TYPES)
      string(TOUPPER "${_build_type}" _BUILD_TYPE)
      set(_metric_flow_x_test_target
          "metric_flow_x_tests${${_build_type}_postfix}")
      add_executable(
        ${_metric_flow_x_test_target}
        metric_flow_x_01.cc
        vascularization_metric_flow_x_01.cc
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
      target_compile_options(
        ${_metric_flow_x_test_target}
        PRIVATE $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
                $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
                $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>)
      target_link_libraries(
        ${_metric_flow_x_test_target}
        ${TEST_LIBRARIES_${_BUILD_TYPE}}
        ${GTEST_MAIN_LIBRARY}
        ${GTEST_LIBRARIES})
      target_include_directories(${_metric_flow_x_test_target}
                                 PRIVATE ${GTEST_INCLUDE_DIRS})
      DEAL_II_SETUP_TARGET(${_metric_flow_x_test_target} ${_BUILD_TYPE})
      set(_vascular_transfer_filter
        "MetricFlowX.BOTH_PreparesVascularInputWithNativeBeds:MetricFlowX.BOTH_PreparesThreeCompartmentInputWithIndependentPortalBoundary:MetricFlowX.BOTH_PreparesExplicitSealedInputTerminals")
      set(_metric_flow_x_serial_test
          "ImmersX${${_build_type}_ctest_suffix}.integration.MetricFlowX.Serial")
      add_test(NAME ${_metric_flow_x_serial_test}
               COMMAND $<TARGET_FILE:${_metric_flow_x_test_target}>)
      set_tests_properties(${_metric_flow_x_serial_test}
                           PROPERTIES LABELS "integration;quick" TIMEOUT 300
                                      RESOURCE_LOCK ImmersX.TestOutput
                                      ENVIRONMENT "GTEST_FILTER=MetricFlowX.*-*.MPI_*:${_vascular_transfer_filter}")
      find_program(_immersx_mpiexec NAMES mpirun mpiexec)
      if(_immersx_mpiexec)
        set(_metric_flow_x_mpi_test
            "ImmersX${${_build_type}_ctest_suffix}.integration.MetricFlowX.MPI")
        add_test(NAME ${_metric_flow_x_mpi_test}
                 COMMAND ${_immersx_mpiexec} -np 2
                         $<TARGET_FILE:${_metric_flow_x_test_target}>)
        set_tests_properties(${_metric_flow_x_mpi_test}
                             PROPERTIES LABELS "integration;quick;mpi"
                                        TIMEOUT 300 PROCESSORS 2
                                        RESOURCE_LOCK ImmersX.TestOutput
                                        ENVIRONMENT "GTEST_FILTER=MetricFlowX.MPI_*:MetricFlowX.BOTH_*-${_vascular_transfer_filter}")
        set(_metric_flow_x_rcr_mpi4_test
            "ImmersX${${_build_type}_ctest_suffix}.integration.MetricFlowX.RCR.MPI4")
        add_test(NAME ${_metric_flow_x_rcr_mpi4_test}
                 COMMAND ${_immersx_mpiexec} -np 4
                         $<TARGET_FILE:${_metric_flow_x_test_target}>)
        set_tests_properties(${_metric_flow_x_rcr_mpi4_test}
                             PROPERTIES LABELS "integration;quick;mpi"
                                        TIMEOUT 300 PROCESSORS 4
                                        RESOURCE_LOCK ImmersX.TestOutput
                                        ENVIRONMENT "GTEST_FILTER=MetricFlowX.BOTH_BackgroundOwnershipKeepsRCRAndFieldValuesLocal:MetricFlowX.BOTH_PressureOperatorOverwritesImportedDestination")
      endif()
      foreach(_ranks 1 2 4)
        if(_ranks GREATER 1 AND NOT _immersx_mpiexec)
          continue()
        endif()
        set(_vascular_name "ImmersX${${_build_type}_ctest_suffix}.integration.MetricFlowX.Vascular.${_ranks}Rank")
        if(_ranks EQUAL 1)
          add_test(NAME ${_vascular_name} COMMAND $<TARGET_FILE:${_metric_flow_x_test_target}>)
        else()
          add_test(NAME ${_vascular_name}
            COMMAND ${_immersx_mpiexec} -np ${_ranks} $<TARGET_FILE:${_metric_flow_x_test_target}>)
        endif()
        set_tests_properties(${_vascular_name} PROPERTIES LABELS "integration;quick"
          TIMEOUT 115 PROCESSORS ${_ranks} RESOURCE_LOCK ImmersX.TestOutput
          ENVIRONMENT "GTEST_FILTER=${_vascular_transfer_filter};${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}"
          SKIP_REGULAR_EXPRESSION "Installed MetricFlowX has no in-memory input API")
        if(_ranks GREATER 1)
          set_property(TEST ${_vascular_name} APPEND PROPERTY LABELS mpi)
        endif()
      endforeach()
      foreach(_case Coupled Generated MMS MMSNeumann RCRDefault RCRImmersed RCRSimplexDefault RCRSimplexImmersed)
        set(_coupled_modes Serial MPI)
        if(DEAL_II_WITH_UMFPACK)
          list(APPEND _coupled_modes SerialUMFPACK)
        endif()
        if(DEAL_II_WITH_MUMPS)
          list(APPEND _coupled_modes SerialMUMPS MPIMUMPS)
          if(_case MATCHES "^RCR" OR _case STREQUAL "Generated")
            list(APPEND _coupled_modes MPI4MUMPS)
          endif()
        endif()
        foreach(_mode IN LISTS _coupled_modes)
          if(_mode MATCHES "^MPI" AND NOT _immersx_mpiexec)
            continue()
          endif()
          set(_coupled_name
              "ImmersX${${_build_type}_ctest_suffix}.application.MetricFlowX${_case}.${_mode}")
          set(_saved_output_dir "${TEST_OUTPUT_DIR}")
          set(TEST_OUTPUT_DIR "${_saved_output_dir}/${_build_type}/metric-${_case}-${_mode}")
          if(_case STREQUAL "Coupled")
            set(_coupled_output_directory "${TEST_OUTPUT_DIR}/metric-flow-x-elastodynamics")
            set(_coupled_output_times "0;0.01;0.02")
          elseif(_case STREQUAL "Generated")
            set(_coupled_output_directory "${TEST_OUTPUT_DIR}/metric-flow-x-generated")
            set(_coupled_output_times "0;0.001;0.002")
          elseif(_case STREQUAL "MMS")
            set(_coupled_output_directory "${TEST_OUTPUT_DIR}/metric-flow-x-mms-cube-resolved")
            set(_coupled_output_times "0;0.01;0.02")
          elseif(_case STREQUAL "MMSNeumann")
            set(_coupled_output_directory "${TEST_OUTPUT_DIR}/metric-flow-x-mms-cube-neumann")
            set(_coupled_output_times "0;0.01;0.02")
          elseif(_case MATCHES "^RCR")
            set(_coupled_output_directory "${TEST_OUTPUT_DIR}/metric-flow-x-rcr")
            set(_coupled_output_times "0;0.001;0.002")
          endif()
          set(_coupled_input "${CMAKE_CURRENT_BINARY_DIR}/parameters/metric_${_case}_${_build_type}_${_mode}.prm")
          if(_case STREQUAL "Generated")
            set(_coupled_template "${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_generated_coupled.prm.in")
          elseif(_case STREQUAL "MMS")
            set(_coupled_template "${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_mms_coupled.prm.in")
          elseif(_case STREQUAL "MMSNeumann")
            set(_coupled_template "${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_mms_neumann_coupled.prm.in")
          elseif(_case STREQUAL "Coupled")
            set(_coupled_template "${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_elastodynamics.prm.in")
          elseif(_case STREQUAL "RCRSimplexDefault" OR _case STREQUAL "RCRSimplexImmersed")
            set(_coupled_template "${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_rcr_simplex_coupled.prm.in")
          elseif(_case MATCHES "^RCR")
            set(_coupled_template "${CMAKE_CURRENT_SOURCE_DIR}/parameters/metric_flow_x_rcr_coupled.prm.in")
            if(_case STREQUAL "RCRImmersed")
              set(PARTITIONING_POLICY "immersed")
            elseif(_case STREQUAL "RCRDefault")
              set(PARTITIONING_POLICY "default")
            endif()
          endif()
          configure_file(
            "${_coupled_template}"
            "${_coupled_input}" @ONLY)
          if(_case STREQUAL "RCRSimplexImmersed")
            file(APPEND "${_coupled_input}"
              "\nsubsection MetricFlowX coupling\n  set Embedded partitioning policy = immersed\nend\n")
          endif()
          if(_mode STREQUAL "SerialUMFPACK")
            file(APPEND "${_coupled_input}"
              "\nsubsection MetricFlowSystem<1, 3>\n  subsection IDA parameters\n    subsection Linear solver\n      set Type = umfpack\n    end\n  end\nend\n")
          elseif(_mode STREQUAL "SerialMUMPS" OR _mode STREQUAL "MPIMUMPS" OR _mode STREQUAL "MPI4MUMPS")
            file(APPEND "${_coupled_input}"
              "\nsubsection MetricFlowSystem<1, 3>\n  subsection IDA parameters\n    subsection Linear solver\n      set Type = mumps\n    end\n  end\nend\n")
          endif()
          set(TEST_OUTPUT_DIR "${_saved_output_dir}")
          set(_coupled_mpiexec "")
          set(_coupled_labels "application")
          set(_coupled_ranks 2)
          if(_mode STREQUAL "MPI4MUMPS")
            set(_coupled_ranks 4)
          endif()
          if(_mode MATCHES "^MPI")
            set(_coupled_mpiexec "${_immersx_mpiexec}")
            set(_coupled_labels "application;mpi")
          endif()
          set(_coupled_timeout 115)
          set(_coupled_application_timeout 110)
          add_test(NAME "${_coupled_name}"
            COMMAND "${CMAKE_COMMAND}"
              "-DTEST_EXECUTABLE=$<TARGET_FILE:metric_flow_x_elastodynamics${${_build_type}_postfix}>"
              "-DPARAMETER_FILE=${_coupled_input}"
              "-DOUTPUT_DIRECTORY=${_coupled_output_directory}"
              "-DEXPECTED_TIMES=${_coupled_output_times}"
              "-DMPIEXEC=${_coupled_mpiexec}"
              "-DMPI_RANKS=${_coupled_ranks}"
              "-DREQUIRE_GENERATED_RESPONSE=$<STREQUAL:${_case},Generated>"
              "-DAPPLICATION_TIMEOUT=${_coupled_application_timeout}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/run_metric_flow_x_coupled.cmake")
          set_tests_properties("${_coupled_name}" PROPERTIES
                               TIMEOUT ${_coupled_timeout} LABELS "${_coupled_labels}"
                               ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
          if(_case STREQUAL "Generated")
            set_tests_properties("${_coupled_name}" PROPERTIES
              SKIP_REGULAR_EXPRESSION "Generated flow requires MetricFlowX with in-memory NetworkPhysics")
          endif()
          if(_mode MATCHES "^MPI")
            set_property(TEST "${_coupled_name}" PROPERTY PROCESSORS ${_coupled_ranks})
          endif()
        endforeach()
      endforeach()
      list(APPEND IMMERSX_TEST_TARGETS ${_metric_flow_x_test_target})
      if(IMMERSX_INSTALL_TESTS)
        install(TARGETS ${_metric_flow_x_test_target}
                RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/immersx/tests)
      endif()
    endif()
  endforeach()

endif()

include("${CMAKE_SOURCE_DIR}/cmake/metric_flow_x_robin_tests.cmake")
