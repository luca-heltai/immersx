foreach(_build_type IN LISTS IMMERSX_REQUESTED_BUILD_TYPES)
  string(TOUPPER "${_build_type}" _BUILD_TYPE)
  set(_penalty_target "particle_penalty_benchmark${${_build_type}_postfix}")
  add_executable(${_penalty_target} particle_penalty.cc)
  target_include_directories(${_penalty_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(${_penalty_target} PRIVATE
    DEAL_II_PACKAGE_VERSION="${DEAL_II_PACKAGE_VERSION}")
  target_link_libraries(${_penalty_target} ${TEST_LIBRARIES_${_BUILD_TYPE}})
  DEAL_II_SETUP_TARGET(${_penalty_target} ${_BUILD_TYPE})
  add_test(NAME ImmersX.ParticlePenalty.${_build_type}
    COMMAND ${_penalty_target} --refinements=1 --repetitions=1)
  set_tests_properties(ImmersX.ParticlePenalty.${_build_type} PROPERTIES
    LABELS "integration;quick" TIMEOUT 180)
  find_program(_penalty_mpiexec NAMES mpirun mpiexec)
  if(_penalty_mpiexec)
    add_test(NAME ImmersX.ParticlePenalty.${_build_type}.MPI
      COMMAND ${_penalty_mpiexec} -np 2 $<TARGET_FILE:${_penalty_target}>
        --refinements=1 --repetitions=1)
    set_tests_properties(ImmersX.ParticlePenalty.${_build_type}.MPI PROPERTIES
      LABELS "integration;quick;mpi" PROCESSORS 2 TIMEOUT 180)
  endif()

  set(_target "weak_term_assembly_benchmark${${_build_type}_postfix}")
  add_executable(${_target} weak_term_assembly.cc)
  target_include_directories(${_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(
    ${_target} PRIVATE DEAL_II_PACKAGE_VERSION="${DEAL_II_PACKAGE_VERSION}")
  target_link_libraries(${_target} ${TEST_LIBRARIES_${_BUILD_TYPE}})
  DEAL_II_SETUP_TARGET(${_target} ${_BUILD_TYPE})
endforeach()

foreach(_build_type IN LISTS IMMERSX_REQUESTED_BUILD_TYPES)
  string(TOUPPER "${_build_type}" _BUILD_TYPE)
  set(_target "lifted_integration_support_benchmark${${_build_type}_postfix}")
  add_executable(${_target} lifted_integration_support.cc)
  target_compile_definitions(
    ${_target} PRIVATE DEAL_II_PACKAGE_VERSION="${DEAL_II_PACKAGE_VERSION}")
  target_include_directories(${_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(${_target} ${TEST_LIBRARIES_${_BUILD_TYPE}})
  DEAL_II_SETUP_TARGET(${_target} ${_BUILD_TYPE})
endforeach()

# Standard PMPI interposition is confined to these optional Unix benchmarks.
if(UNIX AND DEAL_II_WITH_MPI)
  find_program(_vascular_profile_mpiexec NAMES mpirun mpiexec)
  foreach(_build_type IN LISTS IMMERSX_REQUESTED_BUILD_TYPES)
    string(TOUPPER "${_build_type}" _BUILD_TYPE)
    set(_target "vascular_network_profile${${_build_type}_postfix}")
    add_executable(${_target} ../apps/app_vascular_network.cc vascular_mpi_profile.cc)
    target_include_directories(${_target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${_target} ${TEST_LIBRARIES_${_BUILD_TYPE}})
    set_target_properties(${_target} PROPERTIES ENABLE_EXPORTS ON)
    DEAL_II_SETUP_TARGET(${_target} ${_BUILD_TYPE})
    set(_profile_test "vascular_mpi_profile_test${${_build_type}_postfix}")
    add_executable(${_profile_test} vascular_mpi_profile_test.cc vascular_mpi_profile.cc)
    set_target_properties(${_profile_test} PROPERTIES ENABLE_EXPORTS ON)
    DEAL_II_SETUP_TARGET(${_profile_test} ${_BUILD_TYPE})
    if(_vascular_profile_mpiexec AND IMMERSX_PYTHON_EXECUTABLE)
      add_test(NAME "ImmersX.VascularProfile.${_build_type}.MPI2"
        COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
          "${CMAKE_SOURCE_DIR}/scripts/test_vascular_mpi_profile.py"
          --executable "$<TARGET_FILE:${_profile_test}>"
          --mpiexec "${_vascular_profile_mpiexec}"
          --output "${TEST_OUTPUT_DIR}/vascular_profile/${_build_type}")
      set_tests_properties("ImmersX.VascularProfile.${_build_type}.MPI2" PROPERTIES
        LABELS "integration;quick;mpi" PROCESSORS 2 TIMEOUT 180)
    endif()
  endforeach()
endif()
