set(_command "${TEST_EXECUTABLE}" "${PARAMETER_FILE}")
if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "")
  if(NOT DEFINED MPI_RANKS)
    set(MPI_RANKS 2)
  endif()
  list(PREPEND _command "${MPIEXEC}" -np "${MPI_RANKS}")
endif()
if(NOT DEFINED APPLICATION_TIMEOUT)
  set(APPLICATION_TIMEOUT 110)
endif()
execute_process(COMMAND ${_command}
                RESULT_VARIABLE _result
                OUTPUT_VARIABLE _stdout
                ERROR_VARIABLE _stderr
                TIMEOUT ${APPLICATION_TIMEOUT})
message("${_stdout}${_stderr}")
if(NOT "${_result}" STREQUAL "0")
  message(FATAL_ERROR "MetricFlowX coupled application failed: ${_result}")
endif()
if(NOT _stdout MATCHES "accepted IDA steps: [1-9][0-9]*")
  message(FATAL_ERROR "MetricFlowX coupled application did not advance in time")
endif()

if(DEFINED EXPECTED_MUMPS_WORKSPACE AND
   NOT _stdout MATCHES "ICNTL\\(14\\)[^\r\n]*=[ \t]+${EXPECTED_MUMPS_WORKSPACE}([ \t\r\n]|$)")
  message(FATAL_ERROR
    "Native MUMPS did not report the requested workspace relaxation ${EXPECTED_MUMPS_WORKSPACE}")
endif()

foreach(_collection flow.pvd solid.pvd pressure_jump.pvd)
  set(_pvd "${OUTPUT_DIRECTORY}/${_collection}")
  file(STRINGS "${_pvd}" _records REGEX "<DataSet ")
  list(LENGTH _records _count)
  list(LENGTH EXPECTED_TIMES _expected_count)
  if(NOT _count EQUAL _expected_count)
    message(FATAL_ERROR "${_pvd}: expected ${_expected_count} output times, got ${_count}")
  endif()
  set(_index 0)
  foreach(_expected_time IN LISTS EXPECTED_TIMES)
    list(GET _records ${_index} _record)
    string(REGEX MATCH "timestep=\"([^\"]+)\"" _match "${_record}")
    if(NOT "${CMAKE_MATCH_1}" STREQUAL "${_expected_time}")
      message(FATAL_ERROR "${_pvd}: expected time ${_expected_time}, got ${_record}")
    endif()
    string(REGEX MATCH "file=\"([^\"]+)\"" _match "${_record}")
    get_filename_component(_directory "${_pvd}" DIRECTORY)
    if(NOT EXISTS "${_directory}/${CMAKE_MATCH_1}")
      message(FATAL_ERROR "${_pvd}: missing output file in ${_record}")
    endif()
    math(EXPR _index "${_index} + 1")
  endforeach()
endforeach()

if(DEFINED PARTICLE_CHECKER)
  string(REPLACE ";" "," _particle_times "${EXPECTED_TIMES}")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${PARTICLE_CHECKER}"
      --directory "${OUTPUT_DIRECTORY}" --times "${_particle_times}"
      --properties "${PARTICLE_PROPERTIES}" --nonzero-traction
    RESULT_VARIABLE _particle_result
    OUTPUT_VARIABLE _particle_stdout ERROR_VARIABLE _particle_stderr)
  message("${_particle_stdout}${_particle_stderr}")
  if(NOT "${_particle_result}" STREQUAL "0")
    message(FATAL_ERROR "Lifted wall particle checks failed: ${_particle_result}")
  endif()
endif()

if(REQUIRE_GENERATED_RESPONSE)
  if(NOT _stdout MATCHES "Vessel cells with a vertex outside the solid mesh: 0 of [1-9][0-9]*")
    message(FATAL_ERROR "Generated coupled network is empty or has external vertices")
  endif()
  foreach(_quantity "flow change" "solid displacement")
    string(REGEX MATCH "Generated coupled ${_quantity}: ([^ \r\n]+)" _match "${_stdout}")
    if(NOT _match OR NOT CMAKE_MATCH_1 GREATER 0)
      message(FATAL_ERROR "Generated coupled ${_quantity} must be positive")
    endif()
  endforeach()
endif()

if(REQUIRE_STEADY_INITIALIZATION)
  if(NOT _stdout MATCHES "Steady KINSOL post-gate scaled F max: [^;]+; target: [^;]+; finite/positive/zero-rate: 1" OR
     NOT _stdout MATCHES "Steady initialization KINSOL status: [0-9]+; iterations: [1-9][0-9]*")
    message(FATAL_ERROR "Missing accepted nontrivial four-field steady initialization")
  endif()
  foreach(_field displacement velocity flow multiplier)
    if(NOT _stdout MATCHES "Steady initial residual ${_field}: L2=[^;]+; max=[^;]+; scaled_inf=[^;]+; finite=1")
      message(FATAL_ERROR "Missing finite steady initial residual diagnostics for ${_field}")
    endif()
  endforeach()
endif()

if(REQUIRE_BACKWARD_EULER_CONSISTENCY)
  if(NOT _stdout MATCHES "Backward-Euler initial consistency pseudo-step = 0.001; alpha=1000" OR
     NOT _stdout MATCHES "Backward-Euler initial consistency attempt [0-9]+: eta=[^,]+, increment=[^,]+, predictor=(secant|previous|previous-fallback), KINSOL status=[0-9]+, iterations=[0-9]+, shifted scaled residual=[^ \r\n]+ accepted" OR
     NOT _stdout MATCHES "Backward-Euler KINSOL post-gate scaled F max: [^;]+; target: [^;]+; finite/positive/valid-rate: 1" OR
     NOT _stdout MATCHES "Backward-Euler initial consistency KINSOL status: [0-9]+; iterations: [1-9][0-9]*")
    message(FATAL_ERROR "Backward-Euler initial consistency did not pass its residual and state gates")
  endif()
  string(REGEX MATCH "Backward-Euler initial consistency rate inf: ([^ \r\n]+)"
         _rate_match "${_stdout}")
  if(NOT _rate_match OR NOT CMAKE_MATCH_1 GREATER 0)
    message(FATAL_ERROR "Backward-Euler initial consistency did not produce a nonzero initial rate")
  endif()
endif()

if(REQUIRE_STEADY_DIRECT AND
   NOT _stdout MATCHES "rows after second-order condensation")
  message(FATAL_ERROR "Steady initialization did not use the direct condensed solver")
endif()

if(REQUIRE_STEADY_ITERATIVE AND
   NOT _stdout MATCHES "IDAAdapter: (FGMRES|GMRES) converged")
  message(FATAL_ERROR "Steady initialization did not use the configured iterative solver")
endif()

if(REQUIRE_STEADY_MUMPS_BLR AND
   NOT _stdout MATCHES "MUMPS BLR preconditioner has [0-9]+ rows after second-order condensation")
  message(FATAL_ERROR "Steady initialization did not use the condensed MUMPS BLR preconditioner")
endif()

if(REQUIRE_STEADY_CONTINUATION AND
   NOT _stdout MATCHES "predictor=(secant|previous), KINSOL status=[0-9]+, iterations=[0-9]+, shifted scaled residual=[^ \r\n]+ accepted")
  message(FATAL_ERROR "Steady continuation did not accept a predictor stage")
endif()

if(REQUIRE_STEADY_PREVIOUS_PREDICTOR)
  if(NOT _stdout MATCHES
      "Steady continuation attempt ([2-9]|[1-9][0-9]+): eta=[^,]+, increment=[^,]+, predictor=previous, KINSOL status=[0-9]+, iterations=[0-9]+, shifted scaled residual=[^ \r\n]+ accepted")
    message(FATAL_ERROR "Steady continuation did not accept a previous-state-predicted stage")
  endif()
  if(_stdout MATCHES "predictor=secant")
    message(FATAL_ERROR "Previous-state predictor mode unexpectedly used secant extrapolation")
  endif()
endif()
