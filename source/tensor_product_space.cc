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

#include "tensor_product_space.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/utilities.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <regex>

#include "immersed_repartitioner.h"

#ifdef DEAL_II_WITH_VTK

#  include "vtk_utils.h"

namespace
{
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
}

// Constructor for TensorProductSpace
template <int reduced_dim, int dim, int spacedim, int n_components>
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  TensorProductSpace(
    const TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>
            &par,
    MPI_Comm mpi_communicator)
  : mpi_communicator(mpi_communicator)
  , par(par)
  , reference_cross_section(par.section)
  , triangulation(mpi_communicator)
  , fe(FE_Q<reduced_dim, spacedim>(par.fe_degree),
       reference_cross_section.n_selected_basis())
  , quadrature_formula(QIterated<reduced_dim>(
      QuadratureSelector<1>(par.quadrature_type,
                            quadrature_selector_order(par.quadrature_type,
                                                      par.n_q_points,
                                                      par.fe_degree)),
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
  // First create a serial triangulation with the VTK file
  Triangulation<reduced_dim, spacedim> serial_tria;
  DoFHandler<reduced_dim, spacedim>    serial_properties_dh(serial_tria);
  Vector<double>                       serial_properties;
  VTKUtils::read_vtk(par.reduced_grid_name,
                     serial_properties_dh,
                     serial_properties,
                     properties_catalog);
  properties_names.clear();
  properties_names.reserve(properties_catalog.size());
  for (const auto &field : properties_catalog)
    properties_names.push_back(field.vtk_name);

  properties_bindings =
    InputFieldSelector::resolve(par.input_file_fields, properties_catalog);
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::cout << "Input fields exposed from " << par.reduced_grid_name << ":";
      if (properties_bindings.empty())
        std::cout << " (none)";
      for (const auto &binding : properties_bindings)
        {
          const auto &field = properties_catalog[binding.field_index];
          std::cout << "\n  " << binding.symbol_name << " <- "
                    << (field.association == VTKFieldAssociation::point_data ?
                          "PointData" :
                          "CellData")
                    << " \"" << field.vtk_name << "\", component "
                    << binding.vtk_component << ", FE component "
                    << binding.fe_component;
        }
      std::cout << std::endl;
    }

  thickness_expression = par.thickness;
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
                                     {{"pi", numbers::PI}, {"E", numbers::E}});
    }

  deallog << "Read VTK file: " << par.reduced_grid_name
          << ", properties norm: " << serial_properties.l2_norm() << std::endl;

  // The preprocessing hook may refine the serial reduced grid several times.
  // Keep the VTK fields attached to the grid while that happens: a
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
    deallog << "Process " << Utilities::MPI::this_mpi_process(mpi_communicator)
            << " has no locally owned cells." << std::endl;

  properties_dh.distribute_dofs(serial_properties_dh.get_fe());

  properties.reinit(properties_dh.locally_owned_dofs(),
                    DoFTools::extract_locally_relevant_dofs(properties_dh),
                    mpi_communicator);

  if (serial_properties_dh.n_dofs() == properties_dh.n_dofs())
    {
      VTKUtils::serial_vector_to_distributed_vector(serial_properties_dh,
                                                    properties_dh,
                                                    serial_properties,
                                                    properties);
    }
  else
    AssertThrow(false,
                ExcMessage("Imported VTK properties do not match the prepared "
                           "reduced mesh DoF layout after refinement."));

  // Make sure we have ghost values
  properties.update_ghost_values();

  const auto &properties_fe = properties_dh.get_fe();
  const auto  block_indices = VTKUtils::get_block_indices(properties_fe);

  for (unsigned int i = 0; i < block_indices.size(); ++i)
    {
      const auto &name = properties_names[i];
      deallog << "Property name: " << name << ", block index: " << i
              << ", block size: " << block_indices.block_size(i)
              << ", block start: " << block_indices.block_start(i) << std::endl;
    }
  deallog << "Properties norm: " << properties.l2_norm() << std::endl;
  deallog << "Serial properties norm: " << serial_properties.l2_norm()
          << std::endl;
  AssertDimension(block_indices.total_size(), properties_fe.n_components());
  AssertDimension(block_indices.size(), properties_names.size());
};

template <int reduced_dim, int dim, int spacedim, int n_components>
const DoFHandler<reduced_dim, spacedim> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_dof_handler()
  const
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
              ExcMessage("No weights exist across all MPI ranks. You must call"
                         " compute_points_and_weights() first"));
  return all_weights;
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
    particle_id /
    (quadrature_formula.size() * reference_cross_section.n_quadrature_points());
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
        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
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
  all_qpoints.reserve(triangulation.n_active_cells() *
                      quadrature_formula.size() *
                      reference_cross_section.n_quadrature_points());
  all_weights.reserve(triangulation.n_active_cells() *
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
            auto cross_section_qpoints =
              reference_cross_section.get_transformed_quadrature(
                qpoint, new_vertical, thickness_values[q]);

            all_qpoints.insert(all_qpoints.end(),
                               cross_section_qpoints.get_points().begin(),
                               cross_section_qpoints.get_points().end());

            for (const auto &w : cross_section_qpoints.get_weights())
              all_weights.emplace_back(std::vector<double>(1, w * fev.JxW(q)));
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
DoFHandler<reduced_dim, spacedim> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  get_properties_dh()
{
  return properties_dh;
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const std::vector<std::string> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  get_properties_names() const
{
  return properties_names;
}

template <int reduced_dim, int dim, int spacedim, int n_components>
std::vector<std::string> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  get_properties_names()
{
  return properties_names;
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const VTKFieldCatalog &
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
  , reference_cross_section(par.section)
  , point_cloud(par.point_cloud)
{}

template <int dim, int spacedim, int n_components>
void
TensorProductSpace<0, dim, spacedim, n_components>::set_point_cloud(
  const PointCloud<spacedim> &new_point_cloud)
{
  point_cloud = new_point_cloud;
}

template <int dim, int spacedim, int n_components>
void
TensorProductSpace<0, dim, spacedim, n_components>::initialize()
{
  const bool file_input = !par.reduced_grid_name.empty();
  if (file_input)
    make_reduced_grid_and_properties();
  else if (point_cloud.points.empty())
    point_cloud = par.point_cloud;
  AssertThrow(
    !point_cloud.points.empty(),
    ExcMessage(
      "A zero-dimensional representative domain requires at least one point."));
  reference_cross_section.initialize();
  if (!file_input)
    make_reduced_grid_and_properties();
  representative_entity_to_dof_indices.clear();
  const unsigned int dofs_per_entity = n_representative_dofs_per_entity();
  for (unsigned int entity = 0; entity < point_cloud.points.size(); ++entity)
    {
      auto &indices = representative_entity_to_dof_indices[entity];
      indices.resize(dofs_per_entity);
      for (unsigned int j = 0; j < dofs_per_entity; ++j)
        indices[j] = entity * dofs_per_entity + j;
    }
  compute_points_and_weights();
}

template <int dim, int spacedim, int n_components>
void
TensorProductSpace<0, dim, spacedim, n_components>::
  make_reduced_grid_and_properties()
{
  properties_names.clear();
  properties_catalog.clear();
  properties_bindings.clear();
  entity_properties.clear();
  entity_thickness.clear();

  if (!par.reduced_grid_name.empty())
    {
      PointCloud<spacedim> imported_cloud;
      VTKUtils::read_vtk_point_cloud(par.reduced_grid_name, imported_cloud);
      point_cloud = imported_cloud;
    }

  properties_catalog = point_cloud.catalog;
  properties_names   = point_cloud.property_names;
  if (properties_names.size() != properties_catalog.size())
    {
      properties_names.clear();
      properties_names.reserve(properties_catalog.size());
      for (const auto &field : properties_catalog)
        properties_names.push_back(field.vtk_name);
    }
  properties_bindings =
    InputFieldSelector::resolve(par.input_file_fields, properties_catalog);
  entity_properties.resize(point_cloud.points.size(),
                           std::vector<double>(properties_bindings.size()));
  for (unsigned int entity = 0; entity < point_cloud.points.size(); ++entity)
    for (unsigned int binding = 0; binding < properties_bindings.size();
         ++binding)
      {
        const auto &selected = properties_bindings[binding];
        AssertIndexRange(selected.field_index, point_cloud.properties.size());
        const auto &field = properties_catalog[selected.field_index];
        AssertThrow(point_cloud.properties[selected.field_index].size() ==
                      point_cloud.points.size() * field.n_components,
                    ExcMessage("Point-cloud property '" + field.vtk_name +
                               "' has an invalid number of values."));
        entity_properties[entity][binding] =
          point_cloud
            .properties[selected.field_index]
                       [entity * field.n_components + selected.vtk_component];
      }

  thickness_expression = par.thickness;
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
                                     {{"pi", numbers::PI}, {"E", numbers::E}});
    }
  entity_thickness.resize(point_cloud.points.size(), constant_thickness);
  if (!thickness_expression.empty())
    for (unsigned int entity = 0; entity < point_cloud.points.size(); ++entity)
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
  return point_cloud.points.size() * n_representative_dofs_per_entity();
}

template <int dim, int spacedim, int n_components>
IndexSet
TensorProductSpace<0, dim, spacedim, n_components>::
  locally_owned_representative_entities() const
{
  const unsigned int rank  = Utilities::MPI::this_mpi_process(mpi_communicator);
  const unsigned int nproc = Utilities::MPI::n_mpi_processes(mpi_communicator);
  IndexSet           result(point_cloud.points.size());
  for (unsigned int i = 0; i < point_cloud.points.size(); ++i)
    if (i % nproc == rank)
      result.add_index(i);
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
  IndexSet result(n_representative_dofs());
  result.add_range(0, n_representative_dofs());
  result.compress();
  return result;
}

template <int dim, int spacedim, int n_components>
unsigned int
TensorProductSpace<0, dim, spacedim, n_components>::n_representative_entities()
  const
{
  return point_cloud.points.size();
}

template <int dim, int spacedim, int n_components>
const std::vector<types::global_dof_index> &
TensorProductSpace<0, dim, spacedim, n_components>::
  get_representative_dof_indices(types::global_dof_index entity_id) const
{
  AssertIndexRange(entity_id, point_cloud.points.size());
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
TensorProductSpace<0, dim, spacedim, n_components>::compute_points_and_weights()
{
  all_qpoints.clear();
  all_weights.clear();
  reduced_qpoints.clear();
  reduced_weights.clear();
  section_measure.clear();
  const auto owned = locally_owned_representative_entities();
  for (const auto entity_id : owned)
    {
      reduced_qpoints.push_back(point_cloud.points[entity_id]);
      const double thickness = entity_thickness.empty() ?
                                 constant_thickness :
                                 entity_thickness[entity_id];
      reduced_weights.push_back({1.0});
      section_measure.push_back({reference_cross_section.measure(thickness)});
      const auto transformed =
        reference_cross_section.get_transformed_quadrature(
          point_cloud.points[entity_id],
          get_entity_orientation(entity_id),
          thickness);
      for (const auto q : transformed.get_points())
        all_qpoints.push_back(q);
      for (const auto weight : transformed.get_weights())
        all_weights.push_back({weight});
    }
}

template <int dim, int spacedim, int n_components>
const std::vector<Point<spacedim>> &
TensorProductSpace<0, dim, spacedim, n_components>::get_locally_owned_qpoints()
  const
{
  return all_qpoints;
}

template <int dim, int spacedim, int n_components>
const std::vector<std::vector<double>> &
TensorProductSpace<0, dim, spacedim, n_components>::get_locally_owned_weights()
  const
{
  return all_weights;
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
  const auto local_entities = [&]() {
    std::vector<unsigned int> result;
    const auto                owned = locally_owned_representative_entities();
    result.reserve(owned.n_elements());
    for (const auto entity : owned)
      result.push_back(entity);
    return result;
  }();
  const auto all_entities =
    Utilities::MPI::all_gather(mpi_communicator, local_entities);
  const unsigned int nsection = reference_cross_section.n_quadrature_points();
  particle_id_to_representative.clear();
  types::particle_index particle_id = 0;
  for (const auto &rank_entities : all_entities)
    for (const auto entity : rank_entities)
      for (unsigned int section_q = 0; section_q < nsection; ++section_q)
        particle_id_to_representative.emplace(
          particle_id++, std::make_tuple(entity, 0u, section_q));

  AssertDimension(particle_id, point_cloud.points.size() * nsection);
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
  IndexSet           result(point_cloud.points.size() * nsection);
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
  const std::map<unsigned int, IndexSet> &)
{}

template <int dim, int spacedim, int n_components>
double
TensorProductSpace<0, dim, spacedim, n_components>::get_scaling(
  unsigned int entity_id) const
{
  AssertIndexRange(entity_id, point_cloud.points.size());
  const double thickness =
    entity_thickness.empty() ? constant_thickness : entity_thickness[entity_id];
  return std::pow(thickness, -(dim / 2.0));
}

template <int dim, int spacedim, int n_components>
const std::vector<std::string> &
TensorProductSpace<0, dim, spacedim, n_components>::get_properties_names() const
{
  return properties_names;
}
template <int dim, int spacedim, int n_components>
std::vector<std::string> &
TensorProductSpace<0, dim, spacedim, n_components>::get_properties_names()
{
  return properties_names;
}
template <int dim, int spacedim, int n_components>
const VTKFieldCatalog &
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
  (void)time;
  AssertThrow(
    thickness_expression.empty(),
    ExcMessage(
      "Time-dependent Thickness is unsupported for a 0D point cloud because lifted geometry cannot be recomputed coherently."));
}

template <int dim, int spacedim, int n_components>
const std::vector<double> &
TensorProductSpace<0, dim, spacedim, n_components>::get_entity_property_values(
  unsigned int entity_id) const
{
  AssertIndexRange(entity_id, entity_properties.size());
  return entity_properties[entity_id];
}

template <int dim, int spacedim, int n_components>
double
TensorProductSpace<0, dim, spacedim, n_components>::get_entity_thickness(
  unsigned int entity_id) const
{
  AssertIndexRange(entity_id, point_cloud.points.size());
  return entity_thickness.empty() ? constant_thickness :
                                    entity_thickness[entity_id];
}

template <int dim, int spacedim, int n_components>
const Point<spacedim> &
TensorProductSpace<0, dim, spacedim, n_components>::get_entity_position(
  unsigned int entity_id) const
{
  AssertIndexRange(entity_id, point_cloud.points.size());
  return point_cloud.points[entity_id];
}

template <int dim, int spacedim, int n_components>
Tensor<1, spacedim>
TensorProductSpace<0, dim, spacedim, n_components>::get_entity_orientation(
  unsigned int entity_id) const
{
  AssertIndexRange(entity_id, point_cloud.points.size());
  Tensor<1, spacedim> result;
  result[spacedim - 1] = 1.;
  for (unsigned int field = 0; field < point_cloud.catalog.size(); ++field)
    if (point_cloud.catalog[field].vtk_name == "orientation")
      {
        AssertThrow(
          point_cloud.catalog[field].n_components == spacedim,
          ExcMessage("Point-cloud orientation must have spacedim components."));
        AssertIndexRange(field, point_cloud.properties.size());
        AssertThrow(point_cloud.properties[field].size() ==
                      point_cloud.points.size() * spacedim,
                    ExcMessage("Point-cloud orientation has invalid values."));
        for (unsigned int d = 0; d < spacedim; ++d)
          {
            result[d] = point_cloud.properties[field][entity_id * spacedim + d];
            AssertThrow(std::isfinite(result[d]),
                        ExcMessage("Point-cloud orientation must be finite."));
          }
        AssertThrow(result.norm() > 0.,
                    ExcMessage("Point-cloud orientation must be non-zero."));
        return result;
      }
  return result;
}

template struct TensorProductSpaceParameters<0, 2, 2, 1>;
template struct TensorProductSpaceParameters<0, 2, 2, 2>;
template struct TensorProductSpaceParameters<1, 2, 2, 1>;
template struct TensorProductSpaceParameters<1, 2, 3, 1>;
template struct TensorProductSpaceParameters<1, 3, 3, 1>;
template struct TensorProductSpaceParameters<2, 3, 3, 1>;

template struct TensorProductSpaceParameters<1, 2, 2, 2>;
template struct TensorProductSpaceParameters<1, 2, 3, 3>;
template struct TensorProductSpaceParameters<1, 3, 3, 3>;
template struct TensorProductSpaceParameters<2, 3, 3, 3>;

template class TensorProductSpace<0, 2, 2, 1>;
template class TensorProductSpace<0, 2, 2, 2>;
template class TensorProductSpace<1, 2, 2, 1>;
template class TensorProductSpace<1, 2, 3, 1>;
template class TensorProductSpace<1, 3, 3, 1>;
template class TensorProductSpace<2, 3, 3, 1>;

template class TensorProductSpace<1, 2, 2, 2>;
template class TensorProductSpace<1, 2, 3, 3>;
template class TensorProductSpace<1, 3, 3, 3>;
template class TensorProductSpace<2, 3, 3, 3>;

#endif // DEAL_II_WITH_VTK
