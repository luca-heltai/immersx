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

include(GoogleTest)

# Compile selected installed-facing headers independently. These translation
# units intentionally contain no project umbrella include or namespace using
# directive: a failure here is a public-header dependency or namespace error,
# not a runtime test failure.
set(_public_headers
    immersx/core/constraint_equation.h
    immersx/core/constraint_contributor.h
    immersx/core/contributor.h
    immersx/core/field.h
    immersx/core/input_field_selector.h
    immersx/core/load_interaction.h
    immersx/core/reduced_field_catalog.h
    immersx/core/reduced_field_utils.h
    immersx/core/reduced_field_values.h
    immersx/core/representation.h
    immersx/core/semidiscrete_pde_models.h
    immersx/core/state.h
    immersx/core/state_history.h
    immersx/core/sundials_ida_adapter.h
    immersx/core/linear_adapter.h
    immersx/core/symbolic_field_evaluator.h
    immersx/core/time_residual.h
    immersx/coupling/coupling_operator.h
    immersx/physics/elastic_static.h
    immersx/algebra/lagrange_multiplier_interaction.h
    immersx/algebra/lagrange_multiplier_schur_solver.h
    immersx/algebra/linear_algebra.h
    immersx/algebra/vector_lagrange_multiplier_interaction.h
    immersx/coupling/particle_coupling.h
    immersx/physics/elastodynamics.h
    immersx/physics/elastodynamics_semidiscrete.h
    immersx/physics/fiber_reinforced_elastodynamics.h
    immersx/physics/navier_stokes.h
    immersx/physics/navier_stokes_semidiscrete.h
    immersx/physics/poisson_residual.h)

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
                             PRIVATE ${CMAKE_SOURCE_DIR}/include)
  DEAL_II_SETUP_TARGET(${_public_header_target} ${_public_build_type})
  list(APPEND _public_header_targets ${_public_header_target})
endforeach()

add_custom_target(immersx_public_header_self_containment ALL
                  DEPENDS ${_public_header_targets})
add_test(NAME ImmersX.PublicHeaderSelfContainment
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                 --target immersx_public_header_self_containment)

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
    target_include_directories(${_exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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
    IF(IMMERSX_INSTALL_TESTS)
      INSTALL(TARGETS ${_exe}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/immersx/tests)
    ENDIF()
    
    # gtest_discover_tests(${_exe})
  ENDIF()
ENDFOREACH()
