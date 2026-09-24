cmake_minimum_required(VERSION 3.18)

foreach(_required
    CORAL_EXECUTABLE
    PLUGIN
    GRAPH_SOURCE
    PARAMETERS_SOURCE
    WORKING_DIRECTORY)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

file(MAKE_DIRECTORY "${WORKING_DIRECTORY}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${GRAPH_SOURCE}" "${WORKING_DIRECTORY}/poisson_2d.json"
  RESULT_VARIABLE _copy_graph_result)
if(NOT _copy_graph_result EQUAL 0)
  message(FATAL_ERROR "Could not copy the Coral graph")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PARAMETERS_SOURCE}" "${WORKING_DIRECTORY}/poisson_2d.prm"
  RESULT_VARIABLE _copy_parameters_result)
if(NOT _copy_parameters_result EQUAL 0)
  message(FATAL_ERROR "Could not copy the Coral parameter file")
endif()

set(_environment
  "THREADS=1"
  "DYLD_LIBRARY_PATH=${CORAL_LIBRARY_DIRECTORY}"
  "LD_LIBRARY_PATH=${CORAL_LIBRARY_DIRECTORY}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env ${_environment}
          "${CORAL_EXECUTABLE}"
          -p "${PLUGIN}"
          run "${WORKING_DIRECTORY}/poisson_2d.json"
          --graph "${WORKING_DIRECTORY}/poisson_2d.dot"
  WORKING_DIRECTORY "${WORKING_DIRECTORY}"
  RESULT_VARIABLE _run_result)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR "Coral Poisson graph failed with ${_run_result}")
endif()

if(NOT EXISTS "${WORKING_DIRECTORY}/poisson_2d.dot")
  message(FATAL_ERROR "Coral did not write the graphviz output")
endif()
if(NOT EXISTS "${WORKING_DIRECTORY}/output/poisson_2d.pvd")
  message(FATAL_ERROR "Coral graph did not write the Poisson output")
endif()
