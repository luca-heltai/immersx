# Exercise parameter-file dispatch and both displacement formulations.
find_program(_harmonic_fiber_mpiexec NAMES mpirun mpiexec)
set(_harmonic_fiber_ranks 1)
if(_harmonic_fiber_mpiexec)
  list(APPEND _harmonic_fiber_ranks 2)
endif()
foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
  if(NOT _target MATCHES "^fiber_reinforced_elasticity_freq(_debug)?$")
    continue()
  endif()
  foreach(_pair 1-1 1-2 1-3 2-2 2-3 3-3)
    string(REPLACE "-" ";" _dimensions "${_pair}")
    list(GET _dimensions 0 _dim)
    list(GET _dimensions 1 _spacedim)
    foreach(_formulation independent shared)
      foreach(_ranks IN LISTS _harmonic_fiber_ranks)
        set(_case "${_dim}DIn${_spacedim}D.${_formulation}.${_ranks}Rank")
        set(_name "ImmersX.Application.${_target}.${_case}")
        set(_working_directory "${CMAKE_BINARY_DIR}/application_tests/${_target}/${_case}")
        file(MAKE_DIRECTORY "${_working_directory}")
        add_test(NAME "${_name}"
          COMMAND "${CMAKE_COMMAND}"
            "-DEXECUTABLE=$<TARGET_FILE:${_target}>"
            "-DPARAMETERS=${CMAKE_BINARY_DIR}/tutorials/fiber_reinforced_elasticity_freq/parameters.prm"
            "-DWORKING_DIRECTORY=${_working_directory}"
            "-DDIM=${_dim}" "-DSPACEDIM=${_spacedim}" "-DFORMULATION=${_formulation}"
            "-DRANKS=${_ranks}" "-DMPIEXEC=${_harmonic_fiber_mpiexec}"
            -P "${CMAKE_CURRENT_LIST_DIR}/run_harmonic_fiber_test.cmake")
        set_tests_properties("${_name}" PROPERTIES
          LABELS application TIMEOUT 300 WORKING_DIRECTORY "${_working_directory}")
        if(_ranks GREATER 1)
          set_property(TEST "${_name}" APPEND PROPERTY LABELS mpi)
          set_property(TEST "${_name}" PROPERTY PROCESSORS "${_ranks}")
        endif()
      endforeach()
    endforeach()
  endforeach()
endforeach()
