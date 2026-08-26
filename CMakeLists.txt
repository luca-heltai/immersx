## ---------------------------------------------------------------------
##
## Copyright (C) 2000 2020 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

##
# CMake script for large project, with one executable
# and a library of objects
#
# The structure of the directory is assumed to be: 
# ./source: containing implementations and one file with "int main()"
# ./include: containing all class declarations
# ./tests: containing pairs of files: test.cc/test.output to test
# ./gtests: containing gtest unit tests
##

cmake_minimum_required(VERSION 3.18)

project(ImmersX VERSION 0.3.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_main_target immersx)

# Public headers are rooted at include/immersx. Keep the source list
# recursive so new public subdirectories are visible to targets and installs.
file(GLOB_RECURSE _files CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/immersx/*.h")
file(GLOB _main_files CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cc")

option(ENABLE_COUPLED_PROBLEMS
  "Build applications coupled to the optional 1dsolver library." ON)
option(ENABLE_DEAL_II_APP_TESTING
  "Enable deal.II style tests in this application." ON)
option(ENABLE_GOOGLE_TESTING
  "Enable googletest style tests in this application." ON)
option(IMMERSX_BUILD_STATIC
  "Build the ImmersX library as a static archive." OFF)
option(IMMERSX_PORTABLE
  "Request statically linked executables for portable deployment." OFF)
option(IMMERSX_INSTALL_TESTS
  "Install test executables, fixtures, and test sources." ON)

set(IMMERSX_INSTALL_DOCDIR "${CMAKE_INSTALL_DATADIR}/doc/immersx"
  CACHE PATH "Installation directory for ImmersX documentation")

if(IMMERSX_PORTABLE)
  if(NOT IMMERSX_BUILD_STATIC)
    message(STATUS "IMMERSX_PORTABLE implies IMMERSX_BUILD_STATIC")
    set(IMMERSX_BUILD_STATIC ON)
  endif()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR
      "IMMERSX_PORTABLE currently supports Linux toolchains only. "
      "Use IMMERSX_BUILD_STATIC on other platforms and package its runtime "
      "dependencies normally.")
  endif()
endif()

find_package(deal.II 9.2 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF(${ENABLE_COUPLED_PROBLEMS})
  MESSAGE("-- Looking for 1dsolver library")
  SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
  SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a" ".lib" ".dll" ".dylib")
  FIND_LIBRARY(FVM_LIB NAMES 1dsolver HINTS ${CMAKE_CURRENT_SOURCE_DIR})
  IF(NOT FVM_LIB OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    MESSAGE("-- Could not find 1dsolver library -- disabling coupled problems")
    SET(ENABLE_COUPLED_PROBLEMS OFF)
  ELSE()
    MESSAGE("-- Found 1dsolver library -- enabling coupled problems")
    MESSAGE("-- FVM_LIB: ${FVM_LIB}")
  ENDIF()
ELSE()
  MESSAGE("-- Coupled problems disabled")
ENDIF()

FIND_PACKAGE(OpenMP)
IF(OpenMP_CXX_FOUND)
  MESSAGE(STATUS "OpenMP found for CXX: flags='${OpenMP_CXX_FLAGS}', libs='${OpenMP_CXX_LIBRARIES}'")
  IF(TARGET OpenMP::OpenMP_CXX)
    SET(_OPENMP_CXX_TARGET OpenMP::OpenMP_CXX)
  ELSE()
    SET(_OPENMP_CXX_COMPILE_OPTS "${OpenMP_CXX_FLAGS}")
    SET(_OPENMP_CXX_LINK_LIBS "${OpenMP_CXX_LIBRARIES}")
  ENDIF()
ELSE()
  MESSAGE(STATUS "OpenMP not found for CXX; building without OpenMP")
ENDIF()

INCLUDE(CheckCXXSourceCompiles)
IF(ENABLE_COUPLED_PROBLEMS AND FVM_LIB)
  SET(_FVM_RTTI_TEST_SOURCE
    "#include <typeinfo>\n"
    "#include <immersx/fvm/model1d.h>\n"
    "int main(){ Model1d m; const std::type_info &ti = typeid(m); (void)ti; return 0; }\n")
  SET(CMAKE_REQUIRED_INCLUDES
      "${CMAKE_CURRENT_SOURCE_DIR}/include")
  SET(CMAKE_REQUIRED_LIBRARIES "${FVM_LIB}")
  IF(OpenMP_CXX_FOUND AND OpenMP_CXX_FLAGS)
    SET(CMAKE_REQUIRED_FLAGS "${OpenMP_CXX_FLAGS}")
  ENDIF()
  CHECK_CXX_SOURCE_COMPILES("${_FVM_RTTI_TEST_SOURCE}" FVM_LIB_HAS_RTTI)
  UNSET(CMAKE_REQUIRED_INCLUDES)
  UNSET(CMAKE_REQUIRED_LIBRARIES)
  UNSET(CMAKE_REQUIRED_FLAGS)
  UNSET(_FVM_RTTI_TEST_SOURCE)
  IF(NOT FVM_LIB_HAS_RTTI)
    MESSAGE(WARNING
      "Found 1dsolver library but RTTI for Model1d is missing. "
      "Please rebuild lib1dsolver with RTTI enabled (no -fno-rtti). "
      "Disabling coupled problems.")
    SET(ENABLE_COUPLED_PROBLEMS OFF)
  ENDIF()
ENDIF()
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Test and tutorial inputs are configured into the build tree so that tests and
# documented commands never need to infer the location of source data from
# their current working directory.
SET(IMMERSX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
SET(IMMERSX_BINARY_DIR "${CMAKE_BINARY_DIR}")
SET(TEST_DATA_DIR "${IMMERSX_BINARY_DIR}/data")
SET(TEST_OUTPUT_DIR "${IMMERSX_BINARY_DIR}/test_output")

# Keep the input convention in one place. Every template below gtests/, tests/,
# data/, or tutorials/ is emitted at the same relative path below the build
# tree, with only the final .in suffix removed. @ONLY is intentional: parameter
# files may contain CMake-like ${...} text which must reach the test unchanged.
FILE(GLOB_RECURSE _immersx_test_input_templates CONFIGURE_DEPENDS
  "${IMMERSX_SOURCE_DIR}/gtests/*.in"
  "${IMMERSX_SOURCE_DIR}/tests/*.in"
  "${IMMERSX_SOURCE_DIR}/data/*.in"
  "${IMMERSX_SOURCE_DIR}/tutorials/*.in")

FOREACH(_template ${_immersx_test_input_templates})
  FILE(RELATIVE_PATH _relative_template
    "${IMMERSX_SOURCE_DIR}" "${_template}")
  STRING(REGEX REPLACE "\\.in$" "" _relative_output
    "${_relative_template}")
  GET_FILENAME_COMPONENT(_output_directory
    "${IMMERSX_BINARY_DIR}/${_relative_output}" DIRECTORY)
  FILE(MAKE_DIRECTORY "${_output_directory}")
  CONFIGURE_FILE("${_template}"
    "${IMMERSX_BINARY_DIR}/${_relative_output}" @ONLY)
ENDFOREACH()

# We add one library and one target for each type of deal.II library
# we found. If you compiled deal.II with both Release and Debug
# mode, this will generate both Release and Debug programs for you.
# The debug library and program are postfixed with ".g"
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix "_debug")
# SET(CMAKE_C_FLAGS -fopenmp)

SET(IMMERSX_LIBRARY_TARGETS)
SET(IMMERSX_APPLICATION_TARGETS)

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  #MESSAGE(${_build_type})
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      
      MESSAGE("-- Found ${_build_type} version of deal.II.")

      STRING(TOUPPER "${_build_type}" _BUILD_TYPE)

      # Setup library
      SET(_lib "${_main_target}_lib${_p}")
      MESSAGE("-- Configuring library ${_lib}")

      IF(IMMERSX_BUILD_STATIC)
        SET(_immersx_library_type STATIC)
      ELSE()
        SET(_immersx_library_type SHARED)
      ENDIF()
      ADD_LIBRARY(${_lib} ${_immersx_library_type} ${_files})
      SET_TARGET_PROPERTIES(${_lib} PROPERTIES EXPORT_NAME immersx)
      LIST(APPEND IMMERSX_LIBRARY_TARGETS ${_lib})
      TARGET_INCLUDE_DIRECTORIES(${_lib}
        PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
      #ADD_LIBRARY( STATIC )
      # Keep the source path private: exporting it would make the installed
      # package depend on this build tree's absolute path.
      TARGET_COMPILE_DEFINITIONS(${_lib} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
      IF(ENABLE_COUPLED_PROBLEMS)
        TARGET_COMPILE_DEFINITIONS(${_lib} PRIVATE ENABLE_COUPLED_PROBLEMS)
      ENDIF()

      TARGET_COMPILE_OPTIONS(${_lib} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>
      )
      
      DEAL_II_SETUP_TARGET(${_lib} ${_BUILD_TYPE})
      IF(OpenMP_CXX_FOUND)
        IF(DEFINED _OPENMP_CXX_TARGET)
          TARGET_LINK_LIBRARIES(${_lib} ${_OPENMP_CXX_TARGET})
        ELSE()
          IF(_OPENMP_CXX_COMPILE_OPTS)
            TARGET_COMPILE_OPTIONS(${_lib} PUBLIC ${_OPENMP_CXX_COMPILE_OPTS})
          ENDIF()
          IF(_OPENMP_CXX_LINK_LIBS)
            TARGET_LINK_LIBRARIES(${_lib} ${_OPENMP_CXX_LINK_LIBS})
          ENDIF()
        ENDIF()
      ENDIF()
      SET(TEST_LIBRARIES_${_BUILD_TYPE} ${_lib})

      
      FOREACH(_main ${_main_files})
        # Setup one executable per dimension/per build type
        get_filename_component(_main_we ${_main} NAME_WE)
        STRING(REPLACE "app_" "" _main_we ${_main_we})
        SET(_exe "${_main_we}${${_build_type}_postfix}")
        MESSAGE("-- Configuring executable ${_exe}")
        ADD_EXECUTABLE(${_exe} ${_main})
        LIST(APPEND IMMERSX_APPLICATION_TARGETS ${_exe})
        # TARGET_LINK_LIBRARIES(${_exe} ${_lib} 1dsolver)      
        TARGET_COMPILE_DEFINITIONS(${_exe} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        
        IF(ENABLE_COUPLED_PROBLEMS)
          TARGET_COMPILE_DEFINITIONS(${_exe} PRIVATE ENABLE_COUPLED_PROBLEMS)
          TARGET_INCLUDE_DIRECTORIES(${_exe} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
          TARGET_LINK_LIBRARIES(${_exe} ${_lib} ${FVM_LIB})
        ELSE()
          TARGET_LINK_LIBRARIES(${_exe} ${_lib})      
        ENDIF()

        TARGET_COMPILE_OPTIONS(${_exe} PRIVATE
          $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
          $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
          $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>
        )

        DEAL_II_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
        IF(IMMERSX_PORTABLE)
          TARGET_LINK_OPTIONS(${_exe} PRIVATE -static)
          SET_TARGET_PROPERTIES(${_exe} PROPERTIES
            LINK_SEARCH_START_STATIC TRUE
            LINK_SEARCH_END_STATIC TRUE)
        ENDIF()
        IF(OpenMP_CXX_FOUND)
          IF(DEFINED _OPENMP_CXX_TARGET)
            TARGET_LINK_LIBRARIES(${_exe} ${_OPENMP_CXX_TARGET})
          ELSE()
            IF(_OPENMP_CXX_COMPILE_OPTS)
              TARGET_COMPILE_OPTIONS(${_exe} PUBLIC ${_OPENMP_CXX_COMPILE_OPTS})
            ENDIF()
            IF(_OPENMP_CXX_LINK_LIBS)
              TARGET_LINK_LIBRARIES(${_exe} ${_OPENMP_CXX_LINK_LIBS})
            ENDIF()
          ENDIF()
        ENDIF()
    ENDFOREACH()
  ENDIF()
ENDFOREACH()

# Register CTest before adding test subdirectories so their add_test() calls
# are included in the top-level test manifest.
ENABLE_TESTING()

# Deal.II style testing
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests AND ENABLE_DEAL_II_APP_TESTING)
  ADD_SUBDIRECTORY(tests)
ENDIF()

# Google test style testing
IF(ENABLE_GOOGLE_TESTING)
  find_package(GTest)
  if (GTest_FOUND)
      ADD_SUBDIRECTORY(gtests)
  else ()
      MESSAGE(WARNING "GTest library not found. Tests will not be compiled!")
  endif ()
ENDIF()

# Installation and downstream package metadata.
IF(IMMERSX_LIBRARY_TARGETS)
  INSTALL(TARGETS ${IMMERSX_LIBRARY_TARGETS}
    EXPORT ImmersXTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  INSTALL(EXPORT ImmersXTargets
    FILE ImmersXTargets.cmake
    NAMESPACE ImmersX::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ImmersX)

  CONFIGURE_PACKAGE_CONFIG_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ImmersXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ImmersXConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ImmersX)
  WRITE_BASIC_PACKAGE_VERSION_FILE(
    ${CMAKE_CURRENT_BINARY_DIR}/ImmersXConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  INSTALL(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ImmersXConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ImmersXConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ImmersX)
ENDIF()

IF(IMMERSX_APPLICATION_TARGETS)
  INSTALL(TARGETS ${IMMERSX_APPLICATION_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
ENDIF()

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/immersx/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/immersx)
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/immersx/data)
INSTALL(FILES LICENSE.md README.md
  DESTINATION ${IMMERSX_INSTALL_DOCDIR})

IF(IMMERSX_INSTALL_TESTS)
  INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gtests/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/immersx/tests/gtests
    FILES_MATCHING
      PATTERN "*.cc"
      PATTERN "*.h"
      PATTERN "*.in"
      PATTERN "*.prm")
  INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/immersx/tests/tests
    FILES_MATCHING
      PATTERN "*.cc"
      PATTERN "*.h"
      PATTERN "*.in"
      PATTERN "*.output")
ENDIF()

# Documentation site
find_package(Doxygen QUIET)
find_program(SPHINX_EXECUTABLE NAMES sphinx-build)

if(DOXYGEN_FOUND AND SPHINX_EXECUTABLE)
  add_custom_target(doc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_doc.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation site with Doxygen and Sphinx"
    VERBATIM)
  add_custom_target(doc-site DEPENDS doc)
  message(STATUS "Configured documentation site target")
else()
  message(STATUS
    "Documentation site target disabled. Required tools: doxygen, sphinx-build.")
endif()

add_custom_target(indent ./scripts/indent WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
