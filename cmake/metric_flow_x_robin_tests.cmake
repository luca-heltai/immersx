# These are real coupled application runs, not the unavailable-package stub.
if(NOT IMMERSX_WITH_METRIC_FLOW_X OR NOT DEAL_II_WITH_SUNDIALS OR
   NOT IMMERSX_PYTHON_EXECUTABLE)
  return()
endif()
find_program(_robin_mpiexec NAMES mpirun mpiexec)
set(_robin_ranks)
if(DEAL_II_WITH_UMFPACK)
  list(APPEND _robin_ranks 1)
endif()
if(DEAL_II_WITH_MPI AND DEAL_II_WITH_MUMPS AND _robin_mpiexec)
  list(APPEND _robin_ranks 2)
endif()
foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
  if(NOT _target MATCHES "^metric_flow_x_elastodynamics(_debug)?$")
    continue()
  endif()
  foreach(_ranks IN LISTS _robin_ranks)
    set(_name "ImmersX.Application.${_target}.${_ranks}Rank.RobinRCR")
    add_test(NAME "${_name}"
      COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/test_metric_flow_x_robin.py"
        --executable "$<TARGET_FILE:${_target}>"
        --parameters "${CMAKE_BINARY_DIR}/gtests/parameters/metric_flow_x_robin_equilibrium.prm"
        --output "${TEST_OUTPUT_DIR}/metric-flow-x-robin/${_target}/${_ranks}"
        --ranks "${_ranks}" --mpiexec "${_robin_mpiexec}")
    set_tests_properties("${_name}" PROPERTIES LABELS "application"
      TIMEOUT 115 PROCESSORS "${_ranks}"
      ENVIRONMENT "${IMMERSX_NUMERICAL_TEST_ENVIRONMENT}")
    if(_ranks GREATER 1)
      set_property(TEST "${_name}" APPEND PROPERTY LABELS "mpi")
    endif()
  endforeach()
endforeach()
