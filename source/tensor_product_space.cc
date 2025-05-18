// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include "tensor_product_space.h"

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/utilities.h>

#include "immersed_repartitioner.h"
#include "vtk_utils.h"

template <int reduced_dim, int dim, int spacedim, int n_components>
TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>::
  TensorProductSpaceParameters()
  : ParameterAcceptor("Representative domain")
{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Number of quadrature points", n_q_points);
  add_parameter("Thickness", thickness);
  add_parameter("Thickness field name", thickness_field_name);
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
  , quadrature_formula(par.n_q_points == 0 ? 2 * par.fe_degree + 1 :
                                             par.n_q_points)
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
                     properties_names);

  std::cout << "Read VTK file: " << par.reduced_grid_name
            << ", properties norm: " << serial_properties.l2_norm()
            << std::endl;

  // Then make sure the partitioner is what the user wants
  set_partitioner(triangulation);

  // Once the triangulation is created, copy it to the distributed
  // triangulation
  triangulation.copy_triangulation(serial_tria);

  if (triangulation.n_locally_owned_active_cells() == 0)
    std::cout << "Process "
              << Utilities::MPI::this_mpi_process(mpi_communicator)
              << " has no locally owned cells." << std::endl;

  properties_dh.distribute_dofs(serial_properties_dh.get_fe());

  AssertDimension(serial_properties_dh.n_dofs(), properties_dh.n_dofs());

  properties.reinit(properties_dh.locally_owned_dofs(),
                    DoFTools::extract_locally_relevant_dofs(properties_dh),
                    mpi_communicator);
  VTKUtils::serial_vector_to_distributed_vector(serial_properties_dh,
                                                properties_dh,
                                                serial_properties,
                                                properties);
  // Make sure we have ghost values
  properties.update_ghost_values();

  const auto &properties_fe = properties_dh.get_fe();
  const auto  block_indices = VTKUtils::get_block_indices(properties_fe);

  for (unsigned int i = 0; i < block_indices.size(); ++i)
    {
      const auto &name = properties_names[i];
      std::cout << "Property name: " << name << ", block index: " << i
                << ", block size: " << block_indices.block_size(i)
                << ", block start: " << block_indices.block_start(i)
                << std::endl;
    }
  std::cout << "Properties norm: " << properties.l2_norm() << std::endl;
  std::cout << "Serial properties norm: " << serial_properties.l2_norm()
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
  const -> const QGauss<reduced_dim> &
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



  const auto                     &properties_fe = properties_dh.get_fe();
  FEValues<reduced_dim, spacedim> properties_fe_values(properties_fe,
                                                       get_quadrature(),
                                                       update_values);

  // Find the index of the thickness field in the properties
  const unsigned int thickness_field_index =
    std::distance(properties_names.begin(),
                  std::find(properties_names.begin(),
                            properties_names.end(),
                            par.thickness_field_name));

  const auto thickness_start =
    thickness_field_index >= properties_names.size() ?
      numbers::invalid_unsigned_int :
      VTKUtils::get_block_indices(properties_fe)
        .block_start(thickness_field_index);

  FEValuesExtractors::Scalar thickness(thickness_start);

  // Initialize the thickness values with the constant thickness
  std::vector<double> thickness_values(get_quadrature().size(), par.thickness);

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fev.reinit(cell);
        const auto &qpoints = fev.get_quadrature_points();
        const auto &JxW     = fev.get_JxW_values();


        if (thickness_start != numbers::invalid_unsigned_int)
          {
            properties_fe_values.reinit(
              cell->as_dof_handler_iterator(this->properties_dh));
            properties_fe_values[thickness].get_function_values(
              properties, thickness_values);
          }

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
  return std::pow(par.thickness, -((dim - reduced_dim) / 2.0));
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



template struct TensorProductSpaceParameters<1, 2, 2, 1>;
template struct TensorProductSpaceParameters<1, 2, 3, 1>;
template struct TensorProductSpaceParameters<1, 3, 3, 1>;
template struct TensorProductSpaceParameters<2, 3, 3, 1>;

template struct TensorProductSpaceParameters<1, 2, 2, 2>;
template struct TensorProductSpaceParameters<1, 2, 3, 3>;
template struct TensorProductSpaceParameters<1, 3, 3, 3>;
template struct TensorProductSpaceParameters<2, 3, 3, 3>;

template class TensorProductSpace<1, 2, 2, 1>;
template class TensorProductSpace<1, 2, 3, 1>;
template class TensorProductSpace<1, 3, 3, 1>;
template class TensorProductSpace<2, 3, 3, 1>;

template class TensorProductSpace<1, 2, 2, 2>;
template class TensorProductSpace<1, 2, 3, 3>;
template class TensorProductSpace<1, 3, 3, 3>;
template class TensorProductSpace<2, 3, 3, 3>;