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
set(_target gtests)

file(GLOB _test_files *cc)

include(GoogleTest)

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")

    MESSAGE("-- Found ${_build_type} version of deal.II.")
    
    STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
    SET(_exe "${_target}${${_build_type}_postfix}")
    
    MESSAGE("-- Setting up ${_exe}")
    add_executable(${_exe} ${_test_files})

    TARGET_LINK_LIBRARIES(${_exe} ${TEST_LIBRARIES_${_BUILD_TYPE}} GTest::gtest_main)   
    DEAL_II_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
    include_directories(${GTEST_INCLUDE_DIRS})
    
    gtest_discover_tests(${_exe})
  ENDIF()
ENDFOREACH()