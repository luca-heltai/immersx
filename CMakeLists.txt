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

# Set the name of the project and target
# If your application follows the structure above, you don't need to 
# specify anything else. 
SET(_main_target reduced_lagrange)

# Set the _main variable to empty if you don't want an executable
# but only a library
#SET(_main "")

############################################################
# Normally you shouldn't need to change anything below.
############################################################
# Declare all source files the target consists of:
file(GLOB _files source/*cc include/*.h)
file(GLOB _main_files apps/*cc)

INCLUDE_DIRECTORIES(include)

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FIND_PACKAGE(deal.II 9.2 REQUIRED
HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

OPTION(ENABLE_COUPLED_PROBLEMS "Enable coupled problems with 1d code." ON)

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

PROJECT(${_main_target})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
OPTION(ENABLE_DEAL_II_APP_TESTING 
"Enable deal.II style tests in this application." ON)
OPTION(ENABLE_GOOGLE_TESTING 
"Enable googletest style tests in this application." ON)

# We add one library and one target for each type of deal.II library
# we found. If you compiled deal.II with both Release and Debug
# mode, this will generate both Release and Debug programs for you.
# The debug library and program are postfixed with ".g"
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix "_debug")
SET(CMAKE_C_FLAGS -fopenmp)
SET(CMAKE_CXX_FLAGS -fopenmp)

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

      ADD_LIBRARY(${_lib} SHARED ${_files})
      #ADD_LIBRARY( STATIC )
      TARGET_COMPILE_DEFINITIONS(${_lib} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
      IF(ENABLE_COUPLED_PROBLEMS)
        TARGET_COMPILE_DEFINITIONS(${_lib} PUBLIC ENABLE_COUPLED_PROBLEMS)
      ENDIF()

      TARGET_COMPILE_OPTIONS(${_lib} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang>:-Wno-cpp>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-cpp>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-cpp>
      )
      
      DEAL_II_SETUP_TARGET(${_lib} ${_BUILD_TYPE})
      SET(TEST_LIBRARIES_${_BUILD_TYPE} ${_lib})

      
      FOREACH(_main ${_main_files})
        # Setup one executable per dimension/per build type
        get_filename_component(_main_we ${_main} NAME_WE)
        STRING(REPLACE "app_" "" _main_we ${_main_we})
        SET(_exe "${_main_we}${${_build_type}_postfix}")
        MESSAGE("-- Configuring executable ${_exe}")
        ADD_EXECUTABLE(${_exe} ${_main})
        # TARGET_LINK_LIBRARIES(${_exe} ${_lib} 1dsolver)      
        TARGET_COMPILE_DEFINITIONS(${_exe} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        
        IF(ENABLE_COUPLED_PROBLEMS)
          TARGET_COMPILE_DEFINITIONS(${_exe} PUBLIC ENABLE_COUPLED_PROBLEMS)
          TARGET_INCLUDE_DIRECTORIES(${_exe} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/fvm/)
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
    ENDFOREACH()
  ENDIF()
ENDFOREACH()

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

# Doxygen
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
    MESSAGE("-- Configured Doxygen")
  endif(DOXYGEN_FOUND)
endif()

add_custom_target(indent ./scripts/indent WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
ENABLE_TESTING()
