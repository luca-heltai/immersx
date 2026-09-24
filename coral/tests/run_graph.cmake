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

if(NOT DEFINED OUTPUT_FILE)
  set(OUTPUT_FILE output/poisson_2d.pvd)
endif()

file(MAKE_DIRECTORY "${WORKING_DIRECTORY}")
get_filename_component(_graph_name "${GRAPH_SOURCE}" NAME)
get_filename_component(_graph_stem "${GRAPH_SOURCE}" NAME_WE)
get_filename_component(_parameters_name "${PARAMETERS_SOURCE}" NAME)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${GRAPH_SOURCE}" "${WORKING_DIRECTORY}/${_graph_name}"
  RESULT_VARIABLE _copy_graph_result)
if(NOT _copy_graph_result EQUAL 0)
  message(FATAL_ERROR "Could not copy the Coral graph")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PARAMETERS_SOURCE}" "${WORKING_DIRECTORY}/${_parameters_name}"
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
          -p "${PLUGIN}" run "${WORKING_DIRECTORY}/${_graph_name}"
          --graph "${WORKING_DIRECTORY}/${_graph_stem}.dot"
  WORKING_DIRECTORY "${WORKING_DIRECTORY}"
  RESULT_VARIABLE _run_result)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR "Coral graph failed with ${_run_result}")
endif()

if(NOT EXISTS "${WORKING_DIRECTORY}/${_graph_stem}.dot")
  message(FATAL_ERROR "Coral did not write the graphviz output")
endif()
if(NOT EXISTS "${WORKING_DIRECTORY}/${OUTPUT_FILE}")
  message(FATAL_ERROR
    "Coral graph did not write the expected output '${OUTPUT_FILE}'")
endif()
