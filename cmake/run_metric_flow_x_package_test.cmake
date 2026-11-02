file(MAKE_DIRECTORY "${BINARY_DIR}/package")
configure_file(
  "${SOURCE_DIR}/cmake/tests/metric_flow_x/MetricFlowXConfig.cmake"
  "${BINARY_DIR}/package/MetricFlowXConfig.cmake" COPYONLY)
# Force a shared-library Makefile build so link.txt is the generated linker
# invocation, not a static archiver command or a target-property approximation.
# These fixtures assert library dependencies and do not execute their targets,
# so they need no runtime search paths.
execute_process(COMMAND "${CMAKE_COMMAND}"
  -S "${SOURCE_DIR}" -B "${BINARY_DIR}/consumer" -G "Unix Makefiles"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
  "-DDEAL_II_DIR=${DEAL_II_DIR}"
  "-DMetricFlowX_DIR=${BINARY_DIR}/package"
  "-DMETRIC_FIXTURE_PACKAGE=${PACKAGE}"
  -DIMMERSX_BUILD_STATIC=OFF -DIMMERSX_PORTABLE=OFF
  -DCMAKE_SKIP_RPATH=ON
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DIMMERSX_USE_CCACHE=ON -DIMMERSX_BUILD_CORAL_PLUGINS=OFF
  -DIMMERSX_ENABLE_METRIC_FLOW_X_PACKAGE_TESTS=OFF
  -DENABLE_GOOGLE_TESTING=OFF -DENABLE_DEAL_II_APP_TESTING=OFF
  RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
file(WRITE "${BINARY_DIR}/configure.log" "${_stdout}\n${_stderr}")
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Consumer configure failed: ${_stdout}\n${_stderr}")
endif()

file(READ "${BINARY_DIR}/consumer/include/immersx/config.h" _config)
if(EXPECT_ENABLED)
  if(NOT _config MATCHES "#define IMMERSX_WITH_METRIC_FLOW_X")
    message(FATAL_ERROR "Compatible package was disabled: ${_stdout}")
  endif()
elseif(_config MATCHES "#define IMMERSX_WITH_METRIC_FLOW_X")
  message(FATAL_ERROR "Incompatible package was enabled: ${_stdout}")
endif()

if(BUILD_TYPE STREQUAL "DebugRelease")
  set(_variants Debug Release)
else()
  set(_variants "${BUILD_TYPE}")
endif()
foreach(_variant IN LISTS _variants)
  if(_variant STREQUAL "Debug")
    set(_target immersx_lib_debug)
    set(_other Release)
  elseif(_variant STREQUAL "Release")
    set(_target immersx_lib)
    set(_other Debug)
  endif()
  file(READ "${BINARY_DIR}/consumer/CMakeFiles/${_target}.dir/link.txt" _link)
  string(FIND "${_link}" "libmetric_fixture_${_variant}." _metric)
  string(FIND "${_link}" "libmetric_fixture_${_other}." _wrong_metric)
  string(FIND "${_link}" "${DEAL_LIBRARY_${_variant}}" _deal)
  if(_deal EQUAL -1)
    message(FATAL_ERROR "${_target} does not link ${_variant} deal.II: ${_link}")
  endif()
  if(DEFINED DEAL_LIBRARY_${_other})
    string(FIND "${_link}" "${DEAL_LIBRARY_${_other}}" _wrong_deal)
    if(NOT _wrong_deal EQUAL -1)
      message(FATAL_ERROR "${_target} mixes deal.II variants: ${_link}")
    endif()
  endif()
  if(EXPECT_ENABLED)
    if(_metric EQUAL -1 OR NOT _wrong_metric EQUAL -1)
      message(FATAL_ERROR "${_target} has wrong MetricFlowX linkage: ${_link}")
    endif()
  elseif(_link MATCHES "libmetric_fixture_")
    message(FATAL_ERROR "Disabled package is still linked: ${_link}")
  endif()
endforeach()
message(STATUS "${PACKAGE}/${BUILD_TYPE}: verified ${_variants} library link commands")
