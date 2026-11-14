# Physical-tag-preserving import must not take ownership of an external session.
if(DEAL_II_WITH_GMSH_API OR DEAL_II_GMSH_WITH_API)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  if(TARGET ${DEAL_II_TARGET_RELEASE})
    set(CMAKE_REQUIRED_LIBRARIES ${DEAL_II_TARGET_RELEASE})
  else()
    set(CMAKE_REQUIRED_LIBRARIES ${DEAL_II_TARGET_DEBUG})
  endif()
  check_cxx_source_compiles(
    "#include <gmsh.h>\nint main() { return gmsh::isInitialized(); }"
    IMMERSX_GMSH_HAS_IS_INITIALIZED)
  cmake_pop_check_state()
  if(NOT IMMERSX_GMSH_HAS_IS_INITIALIZED)
    message(FATAL_ERROR
      "ImmersX requires a Gmsh C++ API providing gmsh::isInitialized() when "
      "deal.II enables its Gmsh API reader. This is needed to preserve external "
      "Gmsh sessions while importing physical tags. Rebuild deal.II against "
      "a compatible Gmsh installation, or use deal.II without Gmsh API support. "
      "See the CMake configure log for the failed API check.")
  endif()
endif()
