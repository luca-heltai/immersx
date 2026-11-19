set(_command "${TEST_EXECUTABLE}")
if(DEFINED MPIEXEC AND NOT MPIEXEC STREQUAL "")
  list(PREPEND _command "${MPIEXEC}" -np "${MPI_RANKS}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "GTEST_FILTER=LinearAdapter.BOTH_MumpsBlrWorkspaceReachesFactorization"
    ${_command}
  RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr
  TIMEOUT 110)
message("${_stdout}${_stderr}")
if(NOT "${_result}" STREQUAL "0")
  message(FATAL_ERROR "LinearAdapter MUMPS BLR workspace test failed: ${_result}")
endif()
if(NOT _stdout MATCHES "ICNTL\\(14\\)[^\r\n]*=[ \t]+73([ \t\r\n]|$)")
  message(FATAL_ERROR "Native BLR factorization did not report ICNTL(14)=73")
endif()
