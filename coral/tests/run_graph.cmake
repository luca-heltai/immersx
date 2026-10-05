cmake_minimum_required(VERSION 3.18)

foreach(_required
    CORAL_EXECUTABLE
    PLUGIN
    GRAPH_SOURCE
    WORKING_DIRECTORY)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

if(NOT DEFINED EXPECTED_OUTPUTS)
  if(DEFINED OUTPUT_FILE)
    set(EXPECTED_OUTPUTS "${OUTPUT_FILE}")
  else()
    set(EXPECTED_OUTPUTS output/poisson_2d.pvd)
  endif()
endif()

file(REMOVE_RECURSE "${WORKING_DIRECTORY}")
file(MAKE_DIRECTORY "${WORKING_DIRECTORY}")
get_filename_component(_graph_name "${GRAPH_SOURCE}" NAME)
get_filename_component(_graph_stem "${GRAPH_SOURCE}" NAME_WE)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${GRAPH_SOURCE}" "${WORKING_DIRECTORY}/${_graph_name}"
  RESULT_VARIABLE _copy_graph_result)
if(NOT _copy_graph_result EQUAL 0)
  message(FATAL_ERROR "Could not copy the Coral graph")
endif()

if(DEFINED PARAMETERS_SOURCE)
  get_filename_component(_parameters_name "${PARAMETERS_SOURCE}" NAME)
endif()

if(DEFINED PARAMETER_REFERENCE)
  if(NOT DEFINED _parameters_name)
    message(FATAL_ERROR
      "PARAMETERS_SOURCE is required when PARAMETER_REFERENCE is used")
  endif()
  file(READ "${WORKING_DIRECTORY}/${_graph_name}" _graph)
  string(REPLACE "\"${_parameters_name}\""
                 "\"${PARAMETER_REFERENCE}\""
                 _graph
                 "${_graph}")
  file(WRITE "${WORKING_DIRECTORY}/${_graph_name}" "${_graph}")
endif()

if(DEFINED STRIP_PLUGIN_CONFIGURATION AND STRIP_PLUGIN_CONFIGURATION)
  file(READ "${WORKING_DIRECTORY}/${_graph_name}" _graph)
  string(FIND "${_graph}" "\"workflow\"" _workflow_position)
  if(_workflow_position LESS 0)
    message(FATAL_ERROR "The Coral graph does not contain a workflow")
  endif()
  string(LENGTH "${_graph}" _graph_length)
  math(EXPR _workflow_length "${_graph_length} - ${_workflow_position}")
  string(SUBSTRING "${_graph}" ${_workflow_position} ${_workflow_length}
                 _workflow)
  file(WRITE "${WORKING_DIRECTORY}/${_graph_name}" "{${_workflow}")
endif()

if(DEFINED PARAMETERS_SOURCE AND
   NOT (DEFINED SKIP_PARAMETER_COPY AND SKIP_PARAMETER_COPY))
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PARAMETERS_SOURCE}" "${WORKING_DIRECTORY}/${_parameters_name}"
    RESULT_VARIABLE _copy_parameters_result)
  if(NOT _copy_parameters_result EQUAL 0)
    message(FATAL_ERROR "Could not copy the Coral parameter file")
  endif()
endif()

if(DEFINED INPUT_FILES)
  foreach(_input_file IN LISTS INPUT_FILES)
    get_filename_component(_input_name "${_input_file}" NAME)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_input_file}" "${WORKING_DIRECTORY}/${_input_name}"
      RESULT_VARIABLE _copy_input_result)
    if(NOT _copy_input_result EQUAL 0)
      message(FATAL_ERROR "Could not copy Coral graph input '${_input_file}'")
    endif()
  endforeach()
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
string(REPLACE "," ";" _expected_outputs "${EXPECTED_OUTPUTS}")
foreach(_expected_output IN LISTS _expected_outputs)
  if(NOT EXISTS "${WORKING_DIRECTORY}/${_expected_output}")
    message(FATAL_ERROR
      "Coral graph did not write the expected output '${_expected_output}'")
  endif()
endforeach()

if(DEFINED EXPECTED_PARAMETER_FILE)
  set(_expected_parameter_file "${EXPECTED_PARAMETER_FILE}")
  if(NOT IS_ABSOLUTE "${_expected_parameter_file}")
    set(_expected_parameter_file
      "${WORKING_DIRECTORY}/${_expected_parameter_file}")
  endif()
  if(NOT EXISTS "${_expected_parameter_file}")
    message(FATAL_ERROR
      "Coral did not generate the expected parameter file "
      "'${_expected_parameter_file}'")
  endif()

  if(DEFINED EXPECTED_PARAMETER_MARKERS)
    file(READ "${_expected_parameter_file}" _generated_parameters)
    string(REPLACE "|" ";" _expected_markers
                   "${EXPECTED_PARAMETER_MARKERS}")
    foreach(_expected_marker IN LISTS _expected_markers)
      string(FIND "${_generated_parameters}" "${_expected_marker}"
             _marker_position)
      if(_marker_position LESS 0)
        message(FATAL_ERROR
          "Generated Coral parameter file does not contain expected default "
          "marker '${_expected_marker}'")
      endif()
    endforeach()
  endif()
endif()
