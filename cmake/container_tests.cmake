# The launcher contracts need neither a running daemon nor compiled applications.
if(ENABLE_GOOGLE_TESTING OR ENABLE_DEAL_II_APP_TESTING)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME ImmersX.Container.Launcher
      COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
        -s "${CMAKE_SOURCE_DIR}/tests/containers" -p "test_*.py")
    set_tests_properties(ImmersX.Container.Launcher PROPERTIES
      LABELS "unit;quick" TIMEOUT 60)
  endif()
endif()
