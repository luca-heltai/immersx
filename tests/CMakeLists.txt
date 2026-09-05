INCLUDE_DIRECTORIES(${D2K_INCLUDE_DIRS})
DEAL_II_PICKUP_TESTS()

# Deal.II regression tests use the same primary categories and quick policy as
# GoogleTests.  These four fixtures exercise reusable numerical components;
# the MPI matrix-free transfer test is additionally marked with mpi.
get_property(_dealii_tests DIRECTORY PROPERTY TESTS)
foreach(_dealii_test ${_dealii_tests})
  set(_dealii_labels "integration;quick")
  if(_dealii_test MATCHES "matrix_free_transfers\.mpirun")
    list(APPEND _dealii_labels mpi)
  elseif(_dealii_test MATCHES "test_fourier")
    set(_dealii_labels "unit;quick")
  elseif(_dealii_test MATCHES "template")
    set(_dealii_labels "unit;quick")
  endif()
  set_tests_properties(${_dealii_test} PROPERTIES LABELS "${_dealii_labels}")
endforeach()
