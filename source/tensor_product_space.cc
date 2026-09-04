// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/utilities.h>

#include <immersx/core/reduced_field_utils.h>
#include <immersx/coupling/immersed_repartitioner.h>
#include <immersx/coupling/legacy_inclusions.h>
#include <immersx/coupling/tensor_product_space.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>

#ifdef DEAL_II_WITH_VTK
#  include <immersx/io/vtk_utils.h>
#endif

namespace ImmersX
{
  namespace
  {
    template <int reduced_dim, int dim, int spacedim, int n_components>
    const ReferenceCrossSectionParameters<dim - reduced_dim,
                                          spacedim,
                                          n_components> &
    reference_section_parameters(
      const TensorProductSpaceParameters<reduced_dim,
                                         dim,
                                         spacedim,
                                         n_components> &par)
    {
      if (!par.inclusions_file.empty())
        {
          // P_d restricted to the unit circle contains 1+2d Fourier modes.
          // Choose the smallest polynomial degree that can contain the legacy
          // prefix, while preserving an explicitly configured larger degree.
          const auto required_degree = par.legacy_n_coefficients / 2;
          par.section.inclusion_degree =
            std::max(par.section.inclusion_degree, required_degree);

          if (par.section.selected_coefficients.empty())
            {
              const auto canonical_indices =
                LegacyInclusions::fourier_to_reference_indices(
                  par.legacy_n_coefficients,
                  n_components,
                  par.section.inclusion_degree,
                  spacedim);
              if (!par.legacy_selected_coefficients.empty())
                {
                  par.section.selected_coefficients.clear();
                  for (const auto legacy_index :
                       par.legacy_selected_coefficients)
                    {
                      AssertThrow(legacy_index < canonical_indices.size(),
                                  ExcIndexRange(legacy_index,
                                                0,
                                                canonical_indices.size()));
                      par.section.selected_coefficients.push_back(
                        canonical_indices[legacy_index]);
                    }
                }
              else
                par.section.selected_coefficients = canonical_indices;
            }
        }
      return par.section;
    }

    unsigned int
    quadrature_selector_order(const std::string &quadrature_type,
                              const unsigned int requested_n_q_points,
                              const unsigned int fe_degree)
    {
      return quadrature_type == "gauss" ?
               (requested_n_q_points == 0 ? 2 * fe_degree + 1 :
                                            requested_n_q_points) :
               0;
    }
  } // namespace

  template <int reduced_dim, int dim, int spacedim, int n_components>
  TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>::
    TensorProductSpaceParameters()
    : ParameterAcceptor("Representative domain")
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Quadrature type",
                  quadrature_type,
                  "1D quadrature family used by QuadratureSelector.",
                  this->prm,
                  Patterns::Selection(
                    QuadratureSelector<1>::get_quadrature_names()));
    add_parameter("Number of quadrature points", n_q_points);
    add_parameter("Number of quadrature repetitions",
                  n_quadrature_repetitions,
                  "How many times to repeat the quadrature formula.");
    add_parameter("Thickness", thickness);
    add_parameter("Input file fields", input_file_fields);
    add_parameter("Reduced grid name", reduced_grid_name);
    add_parameter(
      "Inclusions file",
      inclusions_file,
      "Legacy inclusion geometry input. It is converted immediately "
      "to the canonical tensor-product representation.");
    add_parameter("Data file",
                  data_file,
                  "Legacy per-record data input. It is imported as normal "
                  "representative properties.");
    add_parameter("Number of fourier coefficients", legacy_n_coefficients);
    add_parameter("Selection of Fourier coefficients",
                  legacy_selected_coefficients);
    add_parameter("Reference inclusion data", legacy_reference_inclusion_data);
  }

  // Constructor for TensorProductSpace
  template <int reduced_dim, int dim, int spacedim, int n_components>
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    TensorProductSpace(const TensorProductSpaceParameters<reduced_dim,
                                                          dim,
                                                          spacedim,
                                                          n_components> &par,
                       MPI_Comm mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , par(par)
    , reference_cross_section(reference_section_parameters(par))
    , triangulation(mpi_communicator)
    , fe(FE_Q<reduced_dim, spacedim>(par.fe_degree),
         reference_cross_section.n_selected_basis())
    , quadrature_formula(detail::make_tensor_product_quadrature<reduced_dim>(
        par.quadrature_type,
        quadrature_selector_order(par.quadrature_type,
                                  par.n_q_points,
                                  par.fe_degree),
        par.n_quadrature_repetitions))
    , dof_handler(triangulation)
    , properties_dh(triangulation)
  {}

  // Initialize the tensor product space
  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::initialize()
  {
    // Create the reduced grid and perform setup only if the triangulation is
    // empty
    if (triangulation.n_active_cells() == 0)
      {
        reference_cross_section.initialize();

        make_reduced_grid_and_properties();

        // Setup degrees of freedom
        setup_dofs();

        // Setup quadrature formulas
        compute_points_and_weights();
      }
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const ReferenceCrossSection<dim - reduced_dim, spacedim, n_components> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_reference_cross_section() const
  {
    return reference_cross_section;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    make_reduced_grid_and_properties()
  {
    // First create a serial triangulation and its serial property vector.
    Triangulation<reduced_dim, spacedim> serial_tria;
    DoFHandler<reduced_dim, spacedim>    serial_properties_dh(serial_tria);
    Vector<double>                       serial_properties;
    AssertThrow(par.reduced_grid_name.empty() || par.inclusions_file.empty(),
                ExcMessage(
                  "Reduced grid name and legacy Inclusions file cannot be "
                  "specified at the same time."));
    if (!par.inclusions_file.empty())
      {
        if constexpr (reduced_dim == 1 && dim == 2 && spacedim == 3)
          LegacyInclusions::read_3d(par.inclusions_file,
                                    par.data_file,
                                    par.legacy_n_coefficients,
                                    n_components,
                                    par.legacy_reference_inclusion_data,
                                    serial_tria,
                                    serial_properties_dh,
                                    serial_properties,
                                    properties_catalog);
        else
          AssertThrow(false,
                      ExcMessage(
                        "The legacy inclusion adapter supports only "
                        "ReducedCoupling<1,2,3> for positive-dimensional "
                        "representative domains."));
      }
    else
      {
#ifdef DEAL_II_WITH_VTK
        VTKUtils::read_vtk(par.reduced_grid_name,
                           serial_properties_dh,
                           serial_properties,
                           properties_catalog);
#else
        AssertThrow(false,
                    ExcMessage("Reading a reduced grid file requires a build "
                               "with VTK support."));
#endif
      }
    const bool legacy_radius_thickness =
      !par.inclusions_file.empty() && par.thickness == "0.01";
    std::string input_file_fields = par.input_file_fields;
    if (legacy_radius_thickness && input_file_fields != "*" &&
        input_file_fields.find("radius") == std::string::npos)
      input_file_fields =
        input_file_fields.empty() ? "radius" : input_file_fields + ",radius";
    properties_bindings =
      InputFieldSelector::resolve(input_file_fields, properties_catalog);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::cout << "Input fields exposed from "
                  << (par.inclusions_file.empty() ? par.reduced_grid_name :
                                                    par.inclusions_file)
                  << ":";
        if (properties_bindings.empty())
          std::cout << " (none)";
        for (const auto &binding : properties_bindings)
          {
            const auto &field = properties_catalog[binding.field_index];
            std::cout << "\n  " << binding.symbol_name << " <- "
                      << (field.association == FieldAssociation::point_data ?
                            "PointData" :
                            "CellData")
                      << " \"" << field.name << "\", component "
                      << binding.field_component << ", FE component "
                      << binding.fe_component;
          }
        std::cout << std::endl;
      }

    thickness_expression = legacy_radius_thickness ? "radius" : par.thickness;
    constant_thickness   = 0.01;
    try
      {
        std::size_t  parsed = 0;
        const double value  = std::stod(thickness_expression, &parsed);
        if (parsed == thickness_expression.size())
          {
            constant_thickness   = value;
            thickness_expression = "";
          }
      }
    catch (const std::invalid_argument &)
      {}
    catch (const std::out_of_range &)
      {
        AssertThrow(false,
                    ExcMessage("Thickness expression is out of range: '" +
                               par.thickness + "'."));
      }
    if (!thickness_expression.empty())
      {
        std::vector<std::string> symbols;
        symbols.reserve(properties_bindings.size());
        for (const auto &binding : properties_bindings)
          symbols.push_back(binding.symbol_name);
        thickness_evaluator.initialize({thickness_expression},
                                       symbols,
                                       {{"pi", numbers::PI},
                                        {"E", numbers::E}});
      }

    deallog << "Read reduced-grid input: "
            << (par.inclusions_file.empty() ? par.reduced_grid_name :
                                              par.inclusions_file)
            << ", properties norm: " << serial_properties.l2_norm()
            << std::endl;

    // The preprocessing hook may refine the serial reduced grid several times.
    // Keep the imported fields attached to the grid while that happens: a
    // SolutionTransfer prepared from the pre-refinement DoFHandler can then
    // rebuild the property vector on every new mesh produced by the hook.
    using SerialPropertiesTransfer =
      SolutionTransfer<reduced_dim, Vector<double>, spacedim>;
    std::shared_ptr<SerialPropertiesTransfer> properties_transfer;
    auto                                      pre_refinement_connection =
      serial_tria.signals.pre_refinement.connect([&]() {
        properties_transfer =
          std::make_shared<SerialPropertiesTransfer>(serial_properties_dh);
        properties_transfer->prepare_for_coarsening_and_refinement(
          serial_properties);
      });
    auto post_refinement_connection =
      serial_tria.signals.post_refinement.connect([&]() {
        serial_properties_dh.distribute_dofs(serial_properties_dh.get_fe());
        Vector<double> transferred_properties(serial_properties_dh.n_dofs());
        properties_transfer->interpolate(transferred_properties);
        serial_properties.swap(transferred_properties);
        properties_transfer.reset();
      });

    // Preprocess the serial triangulation
    preprocess_serial_triangulation(serial_tria);

    // Then make sure the partitioner is what the user wants
    set_partitioner(triangulation);

    // Once the triangulation is created, copy it to the distributed
    // triangulation
    triangulation.copy_triangulation(serial_tria);

    if (triangulation.n_locally_owned_active_cells() == 0)
      deallog << "Process "
              << Utilities::MPI::this_mpi_process(mpi_communicator)
              << " has no locally owned cells." << std::endl;

    properties_dh.distribute_dofs(serial_properties_dh.get_fe());

    properties.reinit(properties_dh.locally_owned_dofs(),
                      DoFTools::extract_locally_relevant_dofs(properties_dh),
                      mpi_communicator);

    if (serial_properties_dh.n_dofs() == properties_dh.n_dofs())
      {
        ReducedFieldUtils::serial_vector_to_distributed_vector(
          serial_properties_dh, properties_dh, serial_properties, properties);
      }
    else
      AssertThrow(
        false,
        ExcMessage("Imported reduced-grid properties do not match the prepared "
                   "reduced mesh DoF layout after refinement."));

    // Make sure we have ghost values
    properties.update_ghost_values();

    const auto &properties_fe = properties_dh.get_fe();
    const auto  block_indices =
      ReducedFieldUtils::get_block_indices(properties_fe);

    for (unsigned int i = 0; i < block_indices.size(); ++i)
      {
        const auto &name = properties_catalog[i].name;
        deallog << "Property name: " << name << ", block index: " << i
                << ", block size: " << block_indices.block_size(i)
                << ", block start: " << block_indices.block_start(i)
                << std::endl;
      }
    deallog << "Properties norm: " << properties.l2_norm() << std::endl;
    deallog << "Serial properties norm: " << serial_properties.l2_norm()
            << std::endl;
    AssertDimension(block_indices.total_size(), properties_fe.n_components());
    AssertDimension(block_indices.size(), properties_catalog.size());
  };

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const DoFHandler<reduced_dim, spacedim> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_dof_handler() const
  {
    return dof_handler;
  }

  /**
   * Return a vector of all quadrature points in the tensor product space that
   * are locally owned by the reduced domain.
   *
   * @return std::vector<Point<spacedim>>
   */
  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<Point<spacedim>> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_locally_owned_qpoints() const
  {
    const int n_local_qpoints = all_qpoints.size();
    const int global_qpoints =
      Utilities::MPI::sum(n_local_qpoints, mpi_communicator);

    AssertThrow(
      global_qpoints > 0,
      ExcMessage(
        "No quadrature points exist across all MPI ranks. You must call compute_points_and_weights() first"));
    return all_qpoints;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_locally_owned_weights() const
  {
    const int n_local_weights = all_weights.size();
    const int global_weights =
      Utilities::MPI::sum(n_local_weights, mpi_communicator);
    AssertThrow(global_weights > 0,
                ExcMessage(
                  "No weights exist across all MPI ranks. You must call"
                  " compute_points_and_weights() first"));
    return all_weights;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_locally_owned_mode_values() const
  {
    AssertThrow(all_mode_values.size() == all_qpoints.size(),
                ExcMessage("Tensor-product mode values are not initialized."));
    return all_mode_values;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<Point<spacedim>> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_locally_owned_reduced_qpoints() const
  {
    const int n_local_reduced_qpoints = reduced_qpoints.size();
    const int global_reduced_qpoints =
      Utilities::MPI::sum(n_local_reduced_qpoints, mpi_communicator);
    AssertThrow(
      global_reduced_qpoints > 0,
      ExcMessage(
        "No reduced quadrature points exist across all MPI ranks. You must call"
        " compute_points_and_weights() first"));
    return reduced_qpoints;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_locally_owned_reduced_weights() const
  {
    const int n_local_reduced_weights = reduced_weights.size();
    const int global_reduced_weights =
      Utilities::MPI::sum(n_local_reduced_weights, mpi_communicator);
    AssertThrow(global_reduced_weights > 0,
                ExcMessage(
                  "No reduced weights exist across all MPI ranks. You must call"
                  " compute_points_and_weights() first"));
    return reduced_weights;
  }


  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    update_local_dof_indices(
      const std::map<unsigned int, IndexSet> &remote_q_point_indices)
  {
    auto global_cell_indices =
      local_q_point_indices_to_global_cell_indices(remote_q_point_indices);
    std::map<
      unsigned int,
      std::map<types::global_cell_index, std::vector<types::global_dof_index>>>
      global_dof_indices;

    for (const auto &[proc, cell_indices] : global_cell_indices)
      for (const auto &id : cell_indices)
        global_dof_indices[proc][id] = global_cell_to_dof_indices[id];

    // Exchange the data with participating processors
    auto local_dof_indices =
      Utilities::MPI::some_to_some(mpi_communicator, global_dof_indices);
    // update global_cell_to_dof_indices
    for (const auto &[proc, cell_indices] : local_dof_indices)
      {
        global_cell_to_dof_indices.insert(cell_indices.begin(),
                                          cell_indices.end());
      }
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<types::global_dof_index> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_dof_indices(
    const types::global_cell_index cell_index) const
  {
    Assert(global_cell_to_dof_indices.find(cell_index) !=
             global_cell_to_dof_indices.end(),
           ExcMessage("Cell index not found in global cell to dof indices."));
    return global_cell_to_dof_indices.at(cell_index);
  }



  template <int reduced_dim, int dim, int spacedim, int n_components>
  std::tuple<unsigned int, unsigned int, unsigned int>
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    particle_id_to_cell_and_qpoint_indices(const unsigned int particle_id) const
  {
    AssertIndexRange(particle_id,
                     triangulation.n_global_active_cells() *
                       quadrature_formula.size() *
                       reference_cross_section.n_quadrature_points());
    const unsigned int cell_index =
      particle_id / (quadrature_formula.size() *
                     reference_cross_section.n_quadrature_points());
    const unsigned int qpoint_index_in_cell =
      (particle_id / reference_cross_section.n_quadrature_points()) %
      quadrature_formula.size();
    const unsigned int qpoint_index_in_section =
      particle_id % reference_cross_section.n_quadrature_points();

    AssertIndexRange(cell_index, triangulation.n_global_active_cells());
    AssertIndexRange(qpoint_index_in_cell, quadrature_formula.size());
    AssertIndexRange(qpoint_index_in_section,
                     reference_cross_section.n_quadrature_points());
    return std::make_tuple(cell_index,
                           qpoint_index_in_cell,
                           qpoint_index_in_section);
  }


  template <int reduced_dim, int dim, int spacedim, int n_components>
  std::tuple<unsigned int, unsigned int, unsigned int>
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    particle_id_to_representative_indices(const unsigned int qpoint_index) const
  {
    return particle_id_to_cell_and_qpoint_indices(qpoint_index);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    register_particle_id_mapping()
  {}

  template <int reduced_dim, int dim, int spacedim, int n_components>
  std::map<unsigned int, IndexSet>
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    local_q_point_indices_to_global_cell_indices(
      const std::map<unsigned int, IndexSet> &remote_q_point_indices) const
  {
    std::map<unsigned int, IndexSet> cell_indices;
    const IndexSet                  &owned_cells =
      triangulation.global_active_cell_index_partitioner()
        .lock()
        ->locally_owned_range();

    auto local_q_point_indices =
      Utilities::MPI::some_to_some(mpi_communicator, remote_q_point_indices);

    for (const auto &[proc, qpoint_indices] : local_q_point_indices)
      {
        IndexSet cell_indices_for_proc(triangulation.n_global_active_cells());
        for (const auto &qpoint_index : qpoint_indices)
          {
            const auto [cell_index, q_index, i] =
              particle_id_to_cell_and_qpoint_indices(qpoint_index);
            cell_indices_for_proc.add_index(
              owned_cells.nth_index_in_set(cell_index));
          }
        cell_indices_for_proc.compress();
        cell_indices[proc] = cell_indices_for_proc;
      }
    return cell_indices;
  }


  // Setup degrees of freedom for the tensor product space
  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::setup_dofs()
  {
    dof_handler.distribute_dofs(fe);
    // Additional setup can be done here if needed
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::vector<types::global_dof_index> dof_indices(
            fe.n_dofs_per_cell());
          cell->get_dof_indices(dof_indices);
          global_cell_to_dof_indices[cell->global_active_cell_index()] =
            dof_indices;
        }
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    locally_owned_qpoints() const
  {
    IndexSet locally_owned_cell_set =
      triangulation.global_active_cell_index_partitioner()
        .lock()
        ->locally_owned_range();

    // Now make a tensor product of the local indices with the total number of
    // quadrature points, and the number of quadrature points in the
    // cross-section
    const unsigned int n_qpoints_per_cell =
      reference_cross_section.n_quadrature_points() * quadrature_formula.size();

    IndexSet locally_owned_qpoints_set = locally_owned_cell_set.tensor_product(
      complete_index_set(n_qpoints_per_cell));

    return locally_owned_qpoints_set;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    locally_relevant_indices() const
  {
    IndexSet indices = triangulation.global_active_cell_index_partitioner()
                         .lock()
                         ->locally_owned_range();
    for (const auto &[cell_id, local_indices] : global_cell_to_dof_indices)
      indices.add_index(cell_id);
    indices.compress();
    return indices;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  auto
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_quadrature()
    const -> const Quadrature<reduced_dim> &
  {
    return quadrature_formula;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    compute_points_and_weights()
  {
    all_qpoints.clear();
    all_weights.clear();
    all_mode_values.clear();
    reduced_qpoints.clear();
    reduced_weights.clear();
    all_qpoints.reserve(triangulation.n_active_cells() *
                        quadrature_formula.size() *
                        reference_cross_section.n_quadrature_points());
    all_weights.reserve(triangulation.n_active_cells() *
                        quadrature_formula.size() *
                        reference_cross_section.n_quadrature_points());
    all_mode_values.reserve(triangulation.n_active_cells() *
                            quadrature_formula.size() *
                            reference_cross_section.n_quadrature_points());

    reduced_qpoints.reserve(triangulation.n_active_cells() *
                            quadrature_formula.size());
    reduced_weights.reserve(triangulation.n_active_cells() *
                            quadrature_formula.size());

    UpdateFlags flags =
      reduced_dim == 1 ?
        update_quadrature_points | update_JxW_values :
        update_quadrature_points | update_normal_vectors | update_JxW_values;

    FEValues<reduced_dim, spacedim> fev(fe, quadrature_formula, flags);



    ReducedFieldValues<reduced_dim, spacedim> field_values(properties_dh,
                                                           get_quadrature(),
                                                           properties,
                                                           properties_bindings);
    std::vector<double> bound_values(get_quadrature().size() *
                                     properties_bindings.size());
    std::vector<double> thickness_values(get_quadrature().size(),
                                         constant_thickness);

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fev.reinit(cell);
          const auto &qpoints = fev.get_quadrature_points();
          const auto &JxW     = fev.get_JxW_values();


          if (!properties_bindings.empty())
            field_values.extract(cell->as_dof_handler_iterator(properties_dh),
                                 bound_values);
          evaluate_thickness_values<reduced_dim, spacedim>(
            thickness_evaluator,
            thickness_expression.empty() ? std::string("constant") :
                                           thickness_expression,
            cell->as_dof_handler_iterator(properties_dh),
            qpoints,
            bound_values,
            constant_thickness,
            evaluation_time,
            thickness_values);

          reduced_qpoints.insert(reduced_qpoints.end(),
                                 qpoints.begin(),
                                 qpoints.end());
          for (const auto &w : JxW)
            reduced_weights.emplace_back(std::vector<double>(1, w));

          Tensor<1, spacedim> new_vertical;
          if constexpr (reduced_dim == 1)
            new_vertical = cell->vertex(1) - cell->vertex(0);

          for (const auto &q : fev.quadrature_point_indices())
            {
              const auto &qpoint = qpoints[q];
              if constexpr (reduced_dim == 2)
                new_vertical = fev.normal_vector(q);
              // [TODO] Make radius a function of the cell
              const auto lifted =
                detail::transform_representative_point<dim, spacedim>(
                  reference_cross_section,
                  par.section.selected_coefficients,
                  qpoint,
                  new_vertical,
                  fev.JxW(q),
                  thickness_values[q],
                  q,
                  {});
              for (const auto &point : lifted)
                {
                  all_qpoints.push_back(point.point);
                  all_weights.emplace_back(
                    std::vector<double>(1, point.weight));
                  all_mode_values.push_back(point.mode_values);
                }
            }
        }
  };

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const parallel::fullydistributed::Triangulation<reduced_dim, spacedim> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_triangulation() const
  {
    return triangulation;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  double
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_scaling(
    const unsigned int) const
  {
    return std::pow(constant_thickness, -((dim - reduced_dim) / 2.0));
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const LinearAlgebra::distributed::Vector<double> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_properties()
    const
  {
    return properties;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const DoFHandler<reduced_dim, spacedim> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_properties_dh() const
  {
    return properties_dh;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const FieldCatalog &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_properties_catalog() const
  {
    return properties_catalog;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<InputFieldBinding> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_properties_bindings() const
  {
    return properties_bindings;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::string &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_thickness_expression() const
  {
    return thickness_expression;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const SymbolicFieldEvaluator &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_thickness_evaluator() const
  {
    return thickness_evaluator;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  void
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::set_time(
    const double time)
  {
    evaluation_time = time;
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    n_representative_dofs() const
  {
    return dof_handler.n_dofs();
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    locally_owned_representative_dofs() const
  {
    return dof_handler.locally_owned_dofs();
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    locally_relevant_representative_dofs() const
  {
    return DoFTools::extract_locally_relevant_dofs(dof_handler);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    n_representative_entities() const
  {
    return triangulation.n_global_active_cells();
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    locally_owned_representative_entities() const
  {
    return triangulation.global_active_cell_index_partitioner()
      .lock()
      ->locally_owned_range();
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  const std::vector<types::global_dof_index> &
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    get_representative_dof_indices(types::global_dof_index entity_id) const
  {
    return get_dof_indices(entity_id);
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    n_representative_q_points_per_entity() const
  {
    return quadrature_formula.size();
  }

  template <int reduced_dim, int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
    n_representative_dofs_per_entity() const
  {
    return fe.n_dofs_per_cell();
  }

  template <int dim, int spacedim, int n_components>
  TensorProductSpace<0, dim, spacedim, n_components>::TensorProductSpace(
    const TensorProductSpaceParameters<0, dim, spacedim, n_components> &par,
    MPI_Comm mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , par(par)
    , reference_cross_section(reference_section_parameters(par))
    , point_cloud(par.point_cloud)
  {}

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::set_point_cloud(
    const PointCloud<spacedim> &new_point_cloud)
  {
    if (representative_handler_initialized)
      representative_particles.clear();
    representative_handler_initialized = false;
    n_global_representative_entities   = 0;
    source_entity_ids.clear();
    representative_properties.clear();
    entity_properties.clear();
    entity_thickness.clear();
    relevant_representative_entities = IndexSet();
    particle_id_to_representative.clear();
    all_qpoints.clear();
    all_weights.clear();
    all_mode_values.clear();
    reduced_qpoints.clear();
    reduced_weights.clear();
    section_measure.clear();
    point_cloud = new_point_cloud;
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::initialize()
  {
    prepare();
    if (!representative_handler_initialized)
      compute_points_and_weights();
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::prepare()
  {
    AssertThrow(par.reduced_grid_name.empty() || par.inclusions_file.empty(),
                ExcMessage(
                  "Reduced grid name and legacy Inclusions file cannot be "
                  "specified at the same time."));
    const bool file_input =
      !par.reduced_grid_name.empty() || !par.inclusions_file.empty();
    if (file_input)
      make_reduced_grid_and_properties();
    else if (point_cloud.points.empty())
      point_cloud = par.point_cloud;
    reference_cross_section.initialize();
    if (!file_input)
      make_reduced_grid_and_properties();
    AssertThrow(
      n_global_representative_entities > 0,
      ExcMessage(
        "A zero-dimensional representative domain requires at least one point."));
    representative_entity_to_dof_indices.clear();
    const unsigned int dofs_per_entity = n_representative_dofs_per_entity();
    for (unsigned int entity = 0; entity < n_global_representative_entities;
         ++entity)
      {
        auto &indices = representative_entity_to_dof_indices[entity];
        indices.resize(dofs_per_entity);
        for (unsigned int j = 0; j < dofs_per_entity; ++j)
          indices[j] = entity * dofs_per_entity + j;
      }
    relevant_representative_entities = locally_owned_representative_entities();
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::
    initialize_representative_particle_handler(
      const parallel::TriangulationBase<spacedim> &background_tria,
      const Mapping<spacedim>                     &mapping,
      const std::vector<std::vector<BoundingBox<spacedim>>>
        &global_bounding_boxes)
  {
    AssertThrow(n_global_representative_entities > 0, ExcNotInitialized());
    const unsigned int n_properties = properties_bindings.size() + spacedim + 1;
    representative_particles.initialize(background_tria, mapping, n_properties);
    std::vector<Point<spacedim>>     positions;
    std::vector<std::vector<double>> properties;
    positions.reserve(source_entity_ids.size());
    properties.reserve(source_entity_ids.size());
    for (unsigned int local = 0; local < source_entity_ids.size(); ++local)
      {
        positions.push_back(point_cloud.points[local]);
        properties.push_back(representative_properties[local]);
      }
    // Let ParticleHandler assign source-rank-prefix ids. This is deterministic
    // for both rank-local input and replicated input with an empty source rank,
    // while avoiding the collective restrictions of explicit-id insertion.
    representative_particles.insert_global_particles(positions,
                                                     global_bounding_boxes,
                                                     properties);
    representative_handler_initialized = true;
    relevant_representative_entities = locally_owned_representative_entities();

    // The particle handler is now the authoritative runtime store. In
    // particular, do not retain a replicated coordinate/property copy here.
    point_cloud.points.clear();
    point_cloud.properties.clear();
    representative_properties.clear();
    entity_properties.clear();
    entity_thickness.clear();
    source_entity_ids.clear();
  }

  template <int dim, int spacedim, int n_components>
  const Particles::ParticleHandler<spacedim> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_representative_particles() const
  {
    AssertThrow(representative_handler_initialized, ExcNotInitialized());
    return representative_particles;
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::
    make_reduced_grid_and_properties()
  {
    properties_catalog.clear();
    properties_bindings.clear();
    entity_properties.clear();
    entity_thickness.clear();

    if (!par.inclusions_file.empty())
      {
        if constexpr (dim == 1 && spacedim == 2)
          LegacyInclusions::read_2d(par.inclusions_file,
                                    par.data_file,
                                    par.legacy_n_coefficients,
                                    n_components,
                                    par.legacy_reference_inclusion_data,
                                    point_cloud);
        else
          AssertThrow(false,
                      ExcMessage("The legacy inclusion adapter supports only "
                                 "ReducedCoupling<0,1,2> for zero-dimensional "
                                 "representative domains."));
      }
    else if (!par.reduced_grid_name.empty())
      {
        PointCloud<spacedim> imported_cloud;
        const auto extension_pos = par.reduced_grid_name.find_last_of('.');
        const bool is_pvtu =
          extension_pos != std::string::npos &&
          par.reduced_grid_name.substr(extension_pos + 1) == "pvtu";
#ifdef DEAL_II_WITH_VTK
        if (is_pvtu)
          VTKUtils::read_vtk_point_cloud(
            par.reduced_grid_name,
            imported_cloud,
            Utilities::MPI::this_mpi_process(mpi_communicator),
            Utilities::MPI::n_mpi_processes(mpi_communicator));
        else
          VTKUtils::read_vtk_point_cloud(par.reduced_grid_name, imported_cloud);
#else
        (void)is_pvtu;
        AssertThrow(false,
                    ExcMessage("Reading a point-cloud file requires a build "
                               "with VTK support."));
#endif
        imported_cloud.distribution = is_pvtu ?
                                        PointCloudDistribution::rank_local :
                                        PointCloudDistribution::replicated;
        point_cloud                 = imported_cloud;
      }

    const unsigned int rank =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const auto local_count = point_cloud.points.size();
    const auto point_counts =
      Utilities::MPI::all_gather(mpi_communicator, local_count);
    const bool replicated =
      point_cloud.distribution == PointCloudDistribution::replicated;
    if (replicated)
      AssertThrow(
        std::all_of(point_counts.begin(),
                    point_counts.end(),
                    [&](const auto count) { return count == local_count; }),
        ExcMessage(
          "A replicated point cloud must provide the same number of points on "
          "every MPI rank; use rank_local for distributed input."));
    n_global_representative_entities = replicated ?
                                         local_count :
                                         std::accumulate(point_counts.begin(),
                                                         point_counts.end(),
                                                         std::size_t(0));
    AssertThrow(n_global_representative_entities > 0,
                ExcMessage("A zero-dimensional representative domain requires "
                           "at least one point."));

    properties_catalog = point_cloud.catalog;
    const auto catalog_sizes =
      Utilities::MPI::all_gather(mpi_communicator, properties_catalog.size());
    AssertThrow(std::all_of(catalog_sizes.begin(),
                            catalog_sizes.end(),
                            [&](const auto size) {
                              return size == properties_catalog.size();
                            }),
                ExcMessage(
                  "Point-cloud property schemas differ across MPI ranks."));
    AssertThrow(point_cloud.properties.size() <= properties_catalog.size(),
                ExcMessage(
                  "Point-cloud property arrays do not match the catalog."));
    point_cloud.properties.resize(properties_catalog.size());
    for (unsigned int field = 0; field < properties_catalog.size(); ++field)
      {
        const auto &descriptor    = properties_catalog[field];
        const auto  expected_size = local_count * descriptor.n_components;
        AssertThrow(point_cloud.properties[field].size() == expected_size,
                    ExcMessage("Point-cloud property '" + descriptor.name +
                               "' has an invalid number of local values."));

        const auto component_counts =
          Utilities::MPI::all_gather(mpi_communicator, descriptor.n_components);
        const auto associations = Utilities::MPI::all_gather(
          mpi_communicator, static_cast<unsigned int>(descriptor.association));
        const auto field_names =
          Utilities::MPI::all_gather(mpi_communicator, descriptor.name);
        AssertThrow(std::all_of(component_counts.begin(),
                                component_counts.end(),
                                [&](const auto value) {
                                  return value == descriptor.n_components;
                                }) &&
                      std::all_of(associations.begin(),
                                  associations.end(),
                                  [&](const auto value) {
                                    return value == static_cast<unsigned int>(
                                                      descriptor.association);
                                  }) &&
                      std::all_of(field_names.begin(),
                                  field_names.end(),
                                  [&](const auto &value) {
                                    return value == descriptor.name;
                                  }),
                    ExcMessage(
                      "Point-cloud property schemas differ across MPI ranks."));
      }

    source_entity_ids.clear();
    if (replicated)
      {
        if (rank == 0)
          {
            source_entity_ids.resize(local_count);
            std::iota(source_entity_ids.begin(), source_entity_ids.end(), 0u);
          }
        else
          source_entity_ids.clear();
      }
    else
      {
        const auto source_offset = std::accumulate(point_counts.begin(),
                                                   point_counts.begin() + rank,
                                                   std::size_t(0));
        source_entity_ids.resize(local_count);
        for (std::size_t i = 0; i < local_count; ++i)
          source_entity_ids[i] = source_offset + i;
      }

    const bool legacy_radius_thickness =
      !par.inclusions_file.empty() && par.thickness == "0.01";
    properties_bindings = InputFieldSelector::resolve(
      legacy_radius_thickness ?
        (par.input_file_fields.empty() || par.input_file_fields == "*" ?
           (par.input_file_fields.empty() ? "radius" : par.input_file_fields) :
           par.input_file_fields + ",radius") :
        par.input_file_fields,
      properties_catalog);
    entity_properties.resize(local_count,
                             std::vector<double>(properties_bindings.size()));
    for (unsigned int entity = 0; entity < local_count; ++entity)
      for (unsigned int binding = 0; binding < properties_bindings.size();
           ++binding)
        {
          const auto &selected = properties_bindings[binding];
          AssertIndexRange(selected.field_index, point_cloud.properties.size());
          const auto &field = properties_catalog[selected.field_index];
          AssertThrow(point_cloud.properties[selected.field_index].size() ==
                        point_cloud.points.size() * field.n_components,
                      ExcMessage("Point-cloud property '" + field.name +
                                 "' has an invalid number of values."));
          entity_properties[entity][binding] =
            point_cloud
              .properties[selected.field_index][entity * field.n_components +
                                                selected.field_component];
        }

    representative_properties.assign(
      local_count,
      std::vector<double>(properties_bindings.size() + spacedim + 1, 0.));
    for (unsigned int entity = 0; entity < local_count; ++entity)
      {
        std::copy(entity_properties[entity].begin(),
                  entity_properties[entity].end(),
                  representative_properties[entity].begin());
        representative_properties[entity][properties_bindings.size() +
                                          spacedim - 1] = 1.;
        for (unsigned int field = 0; field < properties_catalog.size(); ++field)
          if (properties_catalog[field].name == "orientation")
            {
              AssertThrow(
                properties_catalog[field].n_components == spacedim,
                ExcMessage(
                  "Point-cloud orientation must have spacedim components."));
              AssertIndexRange(field, point_cloud.properties.size());
              for (unsigned int d = 0; d < spacedim; ++d)
                representative_properties[entity][properties_bindings.size() +
                                                  d] =
                  point_cloud.properties[field][entity * spacedim + d];

              Tensor<1, spacedim> orientation;
              for (unsigned int d = 0; d < spacedim; ++d)
                {
                  orientation[d] =
                    representative_properties[entity]
                                             [properties_bindings.size() + d];
                  AssertThrow(std::isfinite(orientation[d]),
                              ExcMessage(
                                "Point-cloud orientation must be finite."));
                }
              AssertThrow(orientation.norm() > 0.,
                          ExcMessage(
                            "Point-cloud orientation must be non-zero."));
            }
      }

    thickness_expression = legacy_radius_thickness ? "radius" : par.thickness;
    constant_thickness   = 0.01;
    try
      {
        std::size_t  parsed = 0;
        const double value  = std::stod(thickness_expression, &parsed);
        if (parsed == thickness_expression.size())
          {
            constant_thickness = value;
            thickness_expression.clear();
          }
      }
    catch (const std::invalid_argument &)
      {}
    catch (const std::out_of_range &)
      {
        AssertThrow(false, ExcMessage("Thickness expression is out of range."));
      }
    if (thickness_expression.empty())
      AssertThrow(std::isfinite(constant_thickness) && constant_thickness > 0.,
                  ExcMessage("Thickness must be finite and positive."));
    if (!thickness_expression.empty())
      {
        const std::regex time_symbol("(^|[^A-Za-z0-9_])t([^A-Za-z0-9_]|$)");
        AssertThrow(
          !std::regex_search(thickness_expression, time_symbol),
          ExcMessage(
            "Time-dependent Thickness is unsupported for a 0D point cloud because lifted geometry cannot be recomputed coherently."));
        std::vector<std::string> symbols;
        symbols.reserve(properties_bindings.size());
        for (const auto &binding : properties_bindings)
          symbols.push_back(binding.symbol_name);
        thickness_evaluator.initialize({thickness_expression},
                                       symbols,
                                       {{"pi", numbers::PI},
                                        {"E", numbers::E}});
      }
    entity_thickness.resize(local_count, constant_thickness);
    if (!thickness_expression.empty())
      for (unsigned int entity = 0; entity < local_count; ++entity)
        {
          std::vector<double> value(1);
          thickness_evaluator.evaluate_into(point_cloud.points[entity],
                                            evaluation_time,
                                            entity_properties[entity],
                                            value);
          AssertThrow(
            std::isfinite(value[0]) && value[0] > 0.,
            ExcMessage(
              "Thickness must be finite and positive for every representative entity."));
          entity_thickness[entity] = value[0];
        }
    for (unsigned int entity = 0; entity < local_count; ++entity)
      representative_properties[entity].back() = entity_thickness[entity];

    // Replicated input is needed only while normalizing the source-local
    // metadata. Keep the source copy on rank zero and discard it elsewhere;
    // after particle insertion all ranks discard their remaining input copy.
    if (replicated && rank != 0)
      {
        point_cloud.points.clear();
        point_cloud.properties.clear();
        entity_properties.clear();
        representative_properties.clear();
        entity_thickness.clear();
      }
  }

  template <int dim, int spacedim, int n_components>
  const ReferenceCrossSection<dim, spacedim, n_components> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_reference_cross_section() const
  {
    return reference_cross_section;
  }

  template <int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<0, dim, spacedim, n_components>::
    n_representative_dofs_per_entity() const
  {
    return reference_cross_section.n_selected_basis();
  }

  template <int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<0, dim, spacedim, n_components>::n_representative_dofs()
    const
  {
    return n_representative_entities() * n_representative_dofs_per_entity();
  }

  template <int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<0, dim, spacedim, n_components>::
    locally_owned_representative_entities() const
  {
    if (!representative_handler_initialized)
      {
        IndexSet result(n_global_representative_entities);
        for (const auto entity : source_entity_ids)
          result.add_index(entity);
        result.compress();
        return result;
      }
    const auto n_entities =
      static_cast<std::size_t>(representative_particles.n_global_particles());
    IndexSet result(n_entities);
    for (const auto id : representative_particles.locally_owned_particle_ids())
      result.add_index(id);
    result.compress();
    return result;
  }

  template <int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<0, dim, spacedim, n_components>::
    locally_owned_representative_dofs() const
  {
    const auto         owned_entities = locally_owned_representative_entities();
    const unsigned int block          = n_representative_dofs_per_entity();
    IndexSet           result(n_representative_dofs());
    for (const auto entity : owned_entities)
      result.add_range(entity * block, (entity + 1) * block);
    result.compress();
    return result;
  }

  template <int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<0, dim, spacedim, n_components>::
    locally_relevant_representative_dofs() const
  {
    IndexSet   result(n_representative_dofs());
    const auto entities = relevant_representative_entities.size() ==
                              n_global_representative_entities ?
                            relevant_representative_entities :
                            locally_owned_representative_entities();
    const auto block    = n_representative_dofs_per_entity();
    for (const auto entity : entities)
      result.add_range(entity * block, (entity + 1) * block);
    result.compress();
    return result;
  }

  template <int dim, int spacedim, int n_components>
  unsigned int
  TensorProductSpace<0, dim, spacedim, n_components>::
    n_representative_entities() const
  {
    return n_global_representative_entities;
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<types::global_dof_index> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_representative_dof_indices(types::global_dof_index entity_id) const
  {
    AssertIndexRange(entity_id, n_global_representative_entities);
    return representative_entity_to_dof_indices.at(entity_id);
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<types::global_dof_index> &
  TensorProductSpace<0, dim, spacedim, n_components>::get_dof_indices(
    types::global_cell_index entity_id) const
  {
    return get_representative_dof_indices(entity_id);
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::
    compute_points_and_weights()
  {
    all_qpoints.clear();
    all_weights.clear();
    reduced_qpoints.clear();
    reduced_weights.clear();
    section_measure.clear();
    lifted_entity_ids.clear();
    lifted_section_indices.clear();
    const auto add_entity = [&](const unsigned int         entity_id,
                                const Point<spacedim>     &position,
                                const Tensor<1, spacedim> &orientation,
                                const double               thickness) {
      reduced_qpoints.push_back(position);
      reduced_weights.push_back({1.0});
      section_measure.push_back({reference_cross_section.measure(thickness)});
      const auto transformed =
        detail::transform_representative_point<dim, spacedim>(
          reference_cross_section,
          par.section.selected_coefficients,
          position,
          orientation,
          1.,
          thickness,
          0,
          {});
      for (const auto &point : transformed)
        {
          all_qpoints.push_back(point.point);
          all_weights.push_back({point.weight});
          all_mode_values.push_back(point.mode_values);
          lifted_entity_ids.push_back(entity_id);
          lifted_section_indices.push_back(point.section_qpoint);
        }
    };

    if (representative_handler_initialized)
      {
        for (const auto &particle : representative_particles)
          {
            const auto &particle_properties = particle.get_properties();
            AssertDimension(particle_properties.size(),
                            properties_bindings.size() + spacedim + 1);
            Tensor<1, spacedim> orientation;
            for (unsigned int d = 0; d < spacedim; ++d)
              orientation[d] =
                particle_properties[properties_bindings.size() + d];
            add_entity(particle.get_id(),
                       particle.get_location(),
                       orientation,
                       particle_properties[particle_properties.size() - 1]);
          }
      }
    else
      for (unsigned int local = 0; local < source_entity_ids.size(); ++local)
        {
          Tensor<1, spacedim> orientation;
          orientation[spacedim - 1] = 1.;
          if (representative_properties[local].size() >=
              properties_bindings.size() + spacedim + 1)
            for (unsigned int d = 0; d < spacedim; ++d)
              orientation[d] =
                representative_properties[local]
                                         [properties_bindings.size() + d];
          add_entity(source_entity_ids[local],
                     point_cloud.points[local],
                     orientation,
                     representative_properties[local].back());
        }
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<Point<spacedim>> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_locally_owned_qpoints() const
  {
    return all_qpoints;
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_locally_owned_weights() const
  {
    return all_weights;
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_locally_owned_mode_values() const
  {
    AssertThrow(all_mode_values.size() == all_qpoints.size(),
                ExcMessage("Tensor-product mode values are not initialized."));
    return all_mode_values;
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<Point<spacedim>> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_locally_owned_reduced_qpoints() const
  {
    return reduced_qpoints;
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_locally_owned_reduced_weights() const
  {
    return reduced_weights;
  }

  template <int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_locally_owned_section_measure() const
  {
    return section_measure;
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::
    register_particle_id_mapping()
  {
    all_lifted_entity_ids =
      Utilities::MPI::all_gather(mpi_communicator, lifted_entity_ids);
    all_lifted_section_indices =
      Utilities::MPI::all_gather(mpi_communicator, lifted_section_indices);
    particle_id_to_representative.clear();
    types::particle_index particle_id = 0;
    for (unsigned int rank = 0; rank < all_lifted_entity_ids.size(); ++rank)
      {
        AssertDimension(all_lifted_entity_ids[rank].size(),
                        all_lifted_section_indices[rank].size());
        for (unsigned int i = 0; i < all_lifted_entity_ids[rank].size(); ++i)
          particle_id_to_representative.emplace(
            particle_id++,
            std::make_tuple(all_lifted_entity_ids[rank][i],
                            0u,
                            all_lifted_section_indices[rank][i]));
      }
    AssertDimension(particle_id,
                    n_global_representative_entities *
                      reference_cross_section.n_quadrature_points());
  }

  template <int dim, int spacedim, int n_components>
  std::tuple<unsigned int, unsigned int, unsigned int>
  TensorProductSpace<0, dim, spacedim, n_components>::
    particle_id_to_representative_indices(unsigned int particle_id) const
  {
    const auto it = particle_id_to_representative.find(particle_id);
    AssertThrow(it != particle_id_to_representative.end(),
                ExcMessage(
                  "Particle id has no registered 0D representative mapping."));
    return it->second;
  }

  template <int dim, int spacedim, int n_components>
  std::tuple<unsigned int, unsigned int, unsigned int>
  TensorProductSpace<0, dim, spacedim, n_components>::
    particle_id_to_cell_and_qpoint_indices(unsigned int particle_id) const
  {
    return particle_id_to_representative_indices(particle_id);
  }

  template <int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<0, dim, spacedim, n_components>::locally_owned_qpoints()
    const
  {
    const unsigned int nsection = reference_cross_section.n_quadrature_points();
    IndexSet           result(n_global_representative_entities * nsection);
    const auto         owned = locally_owned_representative_entities();
    for (const auto entity : owned)
      result.add_range(entity * nsection, (entity + 1) * nsection);
    result.compress();
    return result;
  }

  template <int dim, int spacedim, int n_components>
  IndexSet
  TensorProductSpace<0, dim, spacedim, n_components>::locally_relevant_indices()
    const
  {
    return locally_owned_representative_entities();
  }

  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::update_local_dof_indices(
    const std::map<unsigned int, IndexSet> &remote_q_point_indices)
  {
    relevant_representative_entities = locally_owned_representative_entities();
    // insert_global_particles() returns, on each receiving rank, a map keyed by
    // the source rank. Its IndexSet contains indices in the source rank's local
    // qpoint vector. First send this map back to the source ranks so they can
    // translate those indices into representative entity ids. Then send the
    // translated ids to the receiving ranks, where they become relevant DoFs.
    const auto destinations_to_qpoints =
      Utilities::MPI::some_to_some(mpi_communicator, remote_q_point_indices);

    std::map<unsigned int, IndexSet> entities_by_destination;
    for (const auto &[destination_rank, qpoint_indices] :
         destinations_to_qpoints)
      {
        IndexSet entities(n_global_representative_entities);
        for (const auto qpoint : qpoint_indices)
          {
            AssertIndexRange(qpoint, lifted_entity_ids.size());
            entities.add_index(lifted_entity_ids[qpoint]);
          }
        entities.compress();
        entities_by_destination.emplace(destination_rank, std::move(entities));
      }

    const auto sources_to_entities =
      Utilities::MPI::some_to_some(mpi_communicator, entities_by_destination);
    for (const auto &[source_rank, entities] : sources_to_entities)
      for (const auto entity : entities)
        relevant_representative_entities.add_index(entity);

    relevant_representative_entities.compress();
  }

  template <int dim, int spacedim, int n_components>
  double
  TensorProductSpace<0, dim, spacedim, n_components>::get_scaling(
    unsigned int entity_id) const
  {
    AssertIndexRange(entity_id, n_global_representative_entities);
    return std::pow(get_entity_thickness(entity_id), -(dim / 2.0));
  }

  template <int dim, int spacedim, int n_components>
  const FieldCatalog &
  TensorProductSpace<0, dim, spacedim, n_components>::get_properties_catalog()
    const
  {
    return properties_catalog;
  }
  template <int dim, int spacedim, int n_components>
  const std::vector<std::vector<double>> &
  TensorProductSpace<0, dim, spacedim, n_components>::get_properties() const
  {
    return entity_properties;
  }
  template <int dim, int spacedim, int n_components>
  const std::vector<InputFieldBinding> &
  TensorProductSpace<0, dim, spacedim, n_components>::get_properties_bindings()
    const
  {
    return properties_bindings;
  }
  template <int dim, int spacedim, int n_components>
  const std::string &
  TensorProductSpace<0, dim, spacedim, n_components>::get_thickness_expression()
    const
  {
    return thickness_expression;
  }
  template <int dim, int spacedim, int n_components>
  const SymbolicFieldEvaluator &
  TensorProductSpace<0, dim, spacedim, n_components>::get_thickness_evaluator()
    const
  {
    return thickness_evaluator;
  }
  template <int dim, int spacedim, int n_components>
  void
  TensorProductSpace<0, dim, spacedim, n_components>::set_time(double time)
  {
    // Thickness expressions are validated to be time independent during
    // initialization. Updating the RHS time must therefore not invalidate a
    // fixed lifted geometry (e.g. Thickness = radius).
    evaluation_time = time;
  }

  template <int dim, int spacedim, int n_components>
  std::vector<double>
  TensorProductSpace<0, dim, spacedim, n_components>::
    get_entity_property_values(unsigned int entity_id) const
  {
    AssertIndexRange(entity_id, n_global_representative_entities);
    if (representative_handler_initialized)
      for (const auto &particle : representative_particles)
        if (particle.get_id() == entity_id)
          {
            const auto &values = particle.get_properties();
            return std::vector<double>(values.begin(),
                                       values.begin() +
                                         properties_bindings.size());
          }

    for (unsigned int local = 0; local < source_entity_ids.size(); ++local)
      if (source_entity_ids[local] == entity_id)
        return entity_properties[local];

    AssertThrow(false,
                ExcMessage("Representative entity is not locally available."));
    return {};
  }

  template <int dim, int spacedim, int n_components>
  double
  TensorProductSpace<0, dim, spacedim, n_components>::get_entity_thickness(
    unsigned int entity_id) const
  {
    AssertIndexRange(entity_id, n_global_representative_entities);
    if (representative_handler_initialized)
      for (const auto &particle : representative_particles)
        if (particle.get_id() == entity_id)
          {
            const auto &values = particle.get_properties();
            return values[values.size() - 1];
          }

    for (unsigned int local = 0; local < source_entity_ids.size(); ++local)
      if (source_entity_ids[local] == entity_id)
        return entity_thickness.empty() ? constant_thickness :
                                          entity_thickness[local];

    AssertThrow(false,
                ExcMessage("Representative entity is not locally available."));
    return constant_thickness;
  }

  template <int dim, int spacedim, int n_components>
  const Point<spacedim> &
  TensorProductSpace<0, dim, spacedim, n_components>::get_entity_position(
    unsigned int entity_id) const
  {
    AssertIndexRange(entity_id, n_global_representative_entities);
    if (representative_handler_initialized)
      for (const auto &particle : representative_particles)
        if (particle.get_id() == entity_id)
          return particle.get_location();

    for (unsigned int local = 0; local < source_entity_ids.size(); ++local)
      if (source_entity_ids[local] == entity_id)
        return point_cloud.points[local];

    AssertThrow(false,
                ExcMessage("Representative entity is not locally available."));
    return point_cloud.points.front();
  }

  template <int dim, int spacedim, int n_components>
  Tensor<1, spacedim>
  TensorProductSpace<0, dim, spacedim, n_components>::get_entity_orientation(
    unsigned int entity_id) const
  {
    Tensor<1, spacedim> result;
    result[spacedim - 1] = 1.;
    AssertIndexRange(entity_id, n_global_representative_entities);
    if (representative_handler_initialized)
      for (const auto &particle : representative_particles)
        if (particle.get_id() == entity_id)
          {
            const auto &values = particle.get_properties();
            for (unsigned int d = 0; d < spacedim; ++d)
              result[d] = values[properties_bindings.size() + d];
            AssertThrow(result.norm() > 0.,
                        ExcMessage(
                          "Point-cloud orientation must be non-zero."));
            return result;
          }

    for (unsigned int local = 0; local < source_entity_ids.size(); ++local)
      if (source_entity_ids[local] == entity_id)
        {
          for (unsigned int d = 0; d < spacedim; ++d)
            result[d] =
              representative_properties[local][properties_bindings.size() + d];
          AssertThrow(result.norm() > 0.,
                      ExcMessage("Point-cloud orientation must be non-zero."));
          return result;
        }

    AssertThrow(false,
                ExcMessage("Representative entity is not locally available."));
    return result;
  }

  template struct TensorProductSpaceParameters<0, 2, 2, 1>;
  template struct TensorProductSpaceParameters<0, 1, 3, 1>;
  template struct TensorProductSpaceParameters<0, 2, 3, 1>;
  template struct TensorProductSpaceParameters<0, 3, 3, 1>;
  template struct TensorProductSpaceParameters<0, 2, 2, 2>;
  template struct TensorProductSpaceParameters<0, 1, 2, 1>;
  template struct TensorProductSpaceParameters<0, 1, 2, 2>;
  template struct TensorProductSpaceParameters<1, 2, 2, 1>;
  template struct TensorProductSpaceParameters<1, 2, 3, 1>;
  template struct TensorProductSpaceParameters<1, 3, 3, 1>;
  template struct TensorProductSpaceParameters<2, 3, 3, 1>;

  template struct TensorProductSpaceParameters<1, 2, 2, 2>;
  template struct TensorProductSpaceParameters<1, 2, 3, 3>;
  template struct TensorProductSpaceParameters<1, 3, 3, 3>;
  template struct TensorProductSpaceParameters<2, 3, 3, 3>;

  template class TensorProductSpace<0, 2, 2, 1>;
  template class TensorProductSpace<0, 1, 3, 1>;
  template class TensorProductSpace<0, 2, 3, 1>;
  template class TensorProductSpace<0, 3, 3, 1>;
  template class TensorProductSpace<0, 2, 2, 2>;
  template class TensorProductSpace<0, 1, 2, 1>;
  template class TensorProductSpace<0, 1, 2, 2>;
  template class TensorProductSpace<1, 2, 2, 1>;
  template class TensorProductSpace<1, 2, 3, 1>;
  template class TensorProductSpace<1, 3, 3, 1>;
  template class TensorProductSpace<2, 3, 3, 1>;

  template class TensorProductSpace<1, 2, 2, 2>;
  template class TensorProductSpace<1, 2, 3, 3>;
  template class TensorProductSpace<1, 3, 3, 3>;
  template class TensorProductSpace<2, 3, 3, 3>;
} // namespace ImmersX
