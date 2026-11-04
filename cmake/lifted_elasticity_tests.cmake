# Exercise the parameter-driven application, including output and static cycles.
find_program(_lifted_elasticity_mpiexec NAMES mpirun mpiexec)
set(_lifted_elasticity_ranks 1)
if(_lifted_elasticity_mpiexec)
  list(APPEND _lifted_elasticity_ranks 2)
endif()
foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
  if(NOT _target MATCHES "^lifted_elasticity(_debug)?$")
    continue()
  endif()
  foreach(_case uncoupled_2d multiplier_3d pressure_radius_3d)
    foreach(_ranks IN LISTS _lifted_elasticity_ranks)
      set(_name "ImmersX.Application.${_target}.${_case}.${_ranks}Rank")
      set(_working_directory
        "${CMAKE_BINARY_DIR}/application_tests/${_target}/${_case}_${_ranks}")
      file(MAKE_DIRECTORY "${_working_directory}")
      add_test(NAME "${_name}"
        COMMAND "${CMAKE_COMMAND}"
          "-DEXECUTABLE=$<TARGET_FILE:${_target}>"
          "-DPARAMETERS=${CMAKE_BINARY_DIR}/data/lifted_elasticity/${_case}.prm"
          "-DOUTPUT_DIRECTORY=${TEST_OUTPUT_DIR}/lifted_elasticity/${_case}"
          "-DCASE=${_case}" "-DRANKS=${_ranks}"
          "-DMPIEXEC=${_lifted_elasticity_mpiexec}"
          -P "${CMAKE_CURRENT_LIST_DIR}/run_lifted_elasticity_test.cmake")
      set_tests_properties("${_name}" PROPERTIES
        LABELS "application"
        TIMEOUT 115
        WORKING_DIRECTORY "${_working_directory}"
        RESOURCE_LOCK ImmersX.TestOutput)
      if(_ranks GREATER 1)
        set_property(TEST "${_name}" APPEND PROPERTY LABELS "mpi")
        set_property(TEST "${_name}" PROPERTY PROCESSORS "${_ranks}")
      endif()
    endforeach()
  endforeach()
endforeach()
