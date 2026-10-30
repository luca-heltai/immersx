# Test the actual parameter-driven executable from an unrelated directory.
if(NOT IMMERSX_PYTHON_EXECUTABLE)
  return()
endif()
find_program(_vascular_mpiexec NAMES mpirun mpiexec)
set(_vascular_ranks 1)
if(_vascular_mpiexec)
  list(APPEND _vascular_ranks 2 4)
endif()
foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
  if(NOT _target MATCHES "^vascular_network(_debug)?$")
    continue()
  endif()
  foreach(_ranks IN LISTS _vascular_ranks)
    set(_name "ImmersX.Application.${_target}.${_ranks}Rank")
    add_test(NAME "${_name}"
      COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/test_vascular_network.py"
        --executable "$<TARGET_FILE:${_target}>"
        --parameters "${CMAKE_BINARY_DIR}/data/vascularization/standalone.prm"
        --output "${TEST_OUTPUT_DIR}/vascularization/${_target}/${_ranks}"
        --ranks "${_ranks}" --mpiexec "${_vascular_mpiexec}")
    set_tests_properties("${_name}" PROPERTIES LABELS "application"
      TIMEOUT 300 PROCESSORS "${_ranks}")
    if(_ranks GREATER 1)
      set_property(TEST "${_name}" APPEND PROPERTY LABELS "mpi")
    endif()
  endforeach()
endforeach()

# Keep complete native example runs below the build tree for figure publication.
foreach(_target IN LISTS IMMERSX_APPLICATION_TARGETS)
  if(NOT _target MATCHES "^vascular_network(_debug)?$")
    continue()
  endif()
  add_test(NAME "ImmersX.Application.${_target}.IllustratedExamples"
    COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/scripts/run_vascular_examples.py"
      --executable "$<TARGET_FILE:${_target}>"
      --parameters "${CMAKE_BINARY_DIR}/data/vascularization"
      --output-parent "${TEST_OUTPUT_DIR}/vascularization/examples/${_target}")
  set_tests_properties("ImmersX.Application.${_target}.IllustratedExamples"
    PROPERTIES LABELS "application" TIMEOUT 900)
endforeach()
if(_vascular_mpiexec AND TARGET vascular_network)
  add_test(NAME "ImmersX.Application.vascular_network.IllustratedExamples.MPI2"
    COMMAND "${IMMERSX_PYTHON_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/scripts/run_vascular_examples.py"
      --executable "$<TARGET_FILE:vascular_network>"
      --parameters "${CMAKE_BINARY_DIR}/data/vascularization"
      --output-parent "${TEST_OUTPUT_DIR}/vascularization/examples/vascular_network_mpi2"
      --ranks 2 --mpiexec "${_vascular_mpiexec}")
  set_tests_properties("ImmersX.Application.vascular_network.IllustratedExamples.MPI2"
    PROPERTIES LABELS "application;mpi" PROCESSORS 2 TIMEOUT 900)
endif()
