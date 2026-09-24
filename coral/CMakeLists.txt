cmake_minimum_required(VERSION 3.18)

set(_immersx_coral_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(_immersx_coral_install_dir "${CMAKE_INSTALL_LIBDIR}/immersx/coral")

foreach(_build_type ${_d2_build_types})
  if(NOT _build_type IN_LIST IMMERSX_REQUESTED_BUILD_TYPES)
    continue()
  endif()

  string(TOUPPER "${_build_type}" _build_type_upper)
  set(_postfix "${${_build_type}_postfix}")
  set(_immersx_library "immersx_lib${_postfix}")
  get_target_property(_coral_core_location
    coral::core IMPORTED_LOCATION_${_build_type_upper})
  if(NOT _coral_core_location)
    get_target_property(_coral_core_location
      coral::core IMPORTED_LOCATION)
  endif()
  if(_coral_core_location)
    get_filename_component(_coral_core_directory
      "${_coral_core_location}" DIRECTORY)
    set(_coral_test_environment
      "DYLD_LIBRARY_PATH=${_coral_core_directory}"
      "LD_LIBRARY_PATH=${_coral_core_directory}")
  endif()

  foreach(_spacedim RANGE 1 3)
    set(_plugin "coral_backend_immersx_${_spacedim}${_postfix}")
    add_library(${_plugin} MODULE
      source/plugin_${_spacedim}d.cc)
    if(_spacedim EQUAL 2)
      target_sources(${_plugin}
        PRIVATE source/coupled_poisson_2d.cc
                source/ida_elastodynamics_2d.cc)
    elseif(_spacedim EQUAL 3)
      target_sources(${_plugin}
        PRIVATE source/coupled_poisson_elasticity_3d.cc)
    endif()

    target_compile_features(${_plugin} PRIVATE cxx_std_17)
    target_include_directories(${_plugin}
      PRIVATE
        "${_immersx_coral_include_dir}"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/apps"
        "${CMAKE_BINARY_DIR}/include")
    target_link_libraries(${_plugin}
      ${_immersx_library}
      coral::core)
    DEAL_II_SETUP_TARGET(${_plugin} ${_build_type_upper})

    set_target_properties(${_plugin} PROPERTIES
      OUTPUT_NAME "coral_backend_immersx_${_spacedim}${_postfix}"
      INSTALL_RPATH_USE_LINK_PATH TRUE)
    if(APPLE)
      set_property(TARGET ${_plugin} APPEND PROPERTY
        BUILD_RPATH "$<TARGET_FILE_DIR:${_immersx_library}>")
      set_target_properties(${_plugin} PROPERTIES
        INSTALL_RPATH "@loader_path/../..")
    elseif(UNIX)
      set_property(TARGET ${_plugin} APPEND PROPERTY
        BUILD_RPATH "$<TARGET_FILE_DIR:${_immersx_library}>")
      set_target_properties(${_plugin} PROPERTIES
        INSTALL_RPATH "$ORIGIN/../..")
    endif()

    install(TARGETS ${_plugin}
      LIBRARY DESTINATION "${_immersx_coral_install_dir}"
      RUNTIME DESTINATION "${_immersx_coral_install_dir}")

    if(TARGET coral::coral)
      set(_registry
        "${CMAKE_CURRENT_BINARY_DIR}/registry_${_spacedim}d${_postfix}.json")
      add_test(NAME "ImmersX.Coral.Registry.${_spacedim}d${_postfix}"
        COMMAND ${CMAKE_COMMAND}
                -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                -DPLUGIN=$<TARGET_FILE:${_plugin}>
                -DREGISTRY=${_registry}
                -DSPACEDIM=${_spacedim}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_registry.cmake")
      set_tests_properties(
        "ImmersX.Coral.Registry.${_spacedim}d${_postfix}"
        PROPERTIES LABELS "unit;quick"
                   TIMEOUT 120
                   ENVIRONMENT "${_coral_test_environment}")

      if(IMMERSX_ENABLE_CORAL_GRAPH_TESTS AND _spacedim EQUAL 2)
        set(_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_poisson_2d${_postfix}")
        add_test(NAME "ImmersX.Coral.Graph.Poisson2D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/poisson_2d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/poisson_2d.prm
                  -DWORKING_DIRECTORY=${_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.Poisson2D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)

        set(_static_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_elastic_static_2d${_postfix}")
        add_test(NAME "ImmersX.Coral.Graph.ElasticStatic2D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/elastic_static_2d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/elastic_static_2d.prm
                  -DWORKING_DIRECTORY=${_static_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -DOUTPUT_FILE=output/elastic_static.pvd
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.ElasticStatic2D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)

        set(_elastodynamics_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_elastodynamics_2d${_postfix}")
        add_test(NAME "ImmersX.Coral.Graph.Elastodynamics2D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/elastodynamics_2d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/elastodynamics_2d.prm
                  -DWORKING_DIRECTORY=${_elastodynamics_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -DOUTPUT_FILE=output/elastodynamics.pvd
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.Elastodynamics2D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)

        if(DEAL_II_WITH_SUNDIALS)
          set(_ida_elastodynamics_graph_working_directory
            "${CMAKE_CURRENT_BINARY_DIR}/graph_ida_elastodynamics_2d${_postfix}")
          add_test(NAME "ImmersX.Coral.Graph.IDAElastodynamics2D${_postfix}"
            COMMAND ${CMAKE_COMMAND}
                    -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                    -DPLUGIN=$<TARGET_FILE:${_plugin}>
                    -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/ida_elastodynamics_2d.json
                    -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/ida_elastodynamics_2d.prm
                    -DWORKING_DIRECTORY=${_ida_elastodynamics_graph_working_directory}
                    -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                    -DOUTPUT_FILE=output/ida_elastodynamics.pvd
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
          set_tests_properties(
            "ImmersX.Coral.Graph.IDAElastodynamics2D${_postfix}"
            PROPERTIES LABELS "integration;quick;mpi"
                       TIMEOUT 300)
        endif()

        set(_coupled_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_coupled_poisson_2d${_postfix}")
        add_test(NAME "ImmersX.Coral.Graph.CoupledPoisson2D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/coupled_poisson_2d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/coupled_poisson_2d.prm
                  -DWORKING_DIRECTORY=${_coupled_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -DOUTPUT_FILE=output/coupled_multiplier.pvd
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.CoupledPoisson2D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)

        set(_fiber_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_fiber_reinforced_elastodynamics_2d${_postfix}")
        add_test(
          NAME "ImmersX.Coral.Graph.FiberReinforcedElastodynamics2D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/fiber_reinforced_elastodynamics_2d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/fiber_reinforced_elastodynamics_2d.prm
                  -DWORKING_DIRECTORY=${_fiber_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -DOUTPUT_FILE=output/velocity_multiplier.pvd
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.FiberReinforcedElastodynamics2D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)
      endif()

      if(IMMERSX_ENABLE_CORAL_GRAPH_TESTS AND _spacedim EQUAL 3)
        set(_coupled_elasticity_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_coupled_poisson_elasticity_3d${_postfix}")
        add_test(
          NAME "ImmersX.Coral.Graph.CoupledPoissonElasticity3D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/coupled_poisson_elasticity_3d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/coupled_poisson_elasticity_3d.prm
                  -DWORKING_DIRECTORY=${_coupled_elasticity_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -DOUTPUT_FILE=output/coupled_poisson_elasticity_diagnostics.txt
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.CoupledPoissonElasticity3D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)

        set(_fiber_3d_graph_working_directory
          "${CMAKE_CURRENT_BINARY_DIR}/graph_fiber_reinforced_elastodynamics_3d${_postfix}")
        add_test(
          NAME "ImmersX.Coral.Graph.FiberReinforcedElastodynamics3D${_postfix}"
          COMMAND ${CMAKE_COMMAND}
                  -DCORAL_EXECUTABLE=$<TARGET_FILE:coral::coral>
                  -DPLUGIN=$<TARGET_FILE:${_plugin}>
                  -DGRAPH_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/fiber_reinforced_elastodynamics_3d.json
                  -DPARAMETERS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/examples/fiber_reinforced_elastodynamics_3d.prm
                  -DWORKING_DIRECTORY=${_fiber_3d_graph_working_directory}
                  -DCORAL_LIBRARY_DIRECTORY=${_coral_core_directory}
                  -DOUTPUT_FILE=output/velocity_multiplier.pvd
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_graph.cmake")
        set_tests_properties(
          "ImmersX.Coral.Graph.FiberReinforcedElastodynamics3D${_postfix}"
          PROPERTIES LABELS "integration;quick;mpi"
                     TIMEOUT 300)
      endif()
    endif()
  endforeach()
endforeach()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/immersx/coral/examples")
