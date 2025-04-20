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

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/utilities.h>

template <int reduced_dim, int dim, int spacedim, int n_components>
TensorProductSpaceParameters<reduced_dim, dim, spacedim, n_components>::
  TensorProductSpaceParameters()
  : ParameterAcceptor("Tensor product space")
{
  enter_subsection("Representative domain");
  add_parameter("Refinement level", refinement_level);
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Radius", radius);
  leave_subsection();
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
  , quadrature_formula(2 * par.fe_degree + 1)
  , dof_handler(triangulation)
{
  make_reduced_grid = [](Triangulation<reduced_dim, spacedim> &tria) {
    GridGenerator::hyper_cube(tria, 0, 1);
  };
}

// Initialize the tensor product space
template <int reduced_dim, int dim, int spacedim, int n_components>
void
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::initialize()
{
  // Create the reduced grid
  make_reduced_grid(triangulation);
  triangulation.refine_global(par.refinement_level);

  // Setup degrees of freedom
  setup_dofs();
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const ReferenceCrossSection<dim - reduced_dim, spacedim, n_components> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  get_reference_cross_section() const
{
  return reference_cross_section;
}

template <int reduced_dim, int dim, int spacedim, int n_components>
const DoFHandler<reduced_dim, spacedim> &
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::get_dof_handler()
  const
{
  return dof_handler;
}


template <int reduced_dim, int dim, int spacedim, int n_components>
std::vector<Point<spacedim>>
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  get_locally_owned_qpoints_positions() const
{
  std::vector<Point<spacedim>> positions;
  positions.reserve(triangulation.n_active_cells() * quadrature_formula.size() *
                    reference_cross_section.n_quadrature_points());

  UpdateFlags flags = reduced_dim == 1 ?
                        update_quadrature_points :
                        update_quadrature_points | update_normal_vectors;

  FEValues<reduced_dim, spacedim> fev(fe, quadrature_formula, flags);

  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fev.reinit(cell);
        const auto         &qpoints = fev.get_quadrature_points();
        Tensor<1, spacedim> new_vertical;
        if constexpr (reduced_dim == 1)
          new_vertical = cell->vertex(1) - cell->vertex(0);

        for (const auto &q : fev.quadrature_point_indices())
          {
            const auto &qpoint = qpoints[q];
            if constexpr (reduced_dim == 2)
              new_vertical = fev.normal_vector(q);
            auto cross_section_qpoints =
              reference_cross_section.get_transformed_quadrature(qpoint,
                                                                 new_vertical,
                                                                 par.radius);
            positions.insert(positions.end(),
                             cross_section_qpoints.get_points().begin(),
                             cross_section_qpoints.get_points().end());
          }
      }
  return positions;
}



template <int reduced_dim, int dim, int spacedim, int n_components>
std::tuple<unsigned int, unsigned int, unsigned int>
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::
  qpoint_index_to_cell_and_qpoint_indices(const unsigned int qpoint_index) const
{
  AssertIndexRange(qpoint_index,
                   triangulation.n_active_cells() * quadrature_formula.size() *
                     reference_cross_section.n_quadrature_points());
  const unsigned int cell_index =
    qpoint_index /
    (quadrature_formula.size() * reference_cross_section.n_quadrature_points());
  const unsigned int qpoint_index_in_cell =
    (qpoint_index / reference_cross_section.n_quadrature_points()) %
    quadrature_formula.size();
  const unsigned int qpoint_index_in_section =
    qpoint_index % reference_cross_section.n_quadrature_points();
  return std::make_tuple(cell_index,
                         qpoint_index_in_cell,
                         qpoint_index_in_section);
}


// Setup degrees of freedom for the tensor product space
template <int reduced_dim, int dim, int spacedim, int n_components>
void
TensorProductSpace<reduced_dim, dim, spacedim, n_components>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  // Additional setup can be done here if needed
}

template struct TensorProductSpaceParameters<1, 2, 3, 1>;
template struct TensorProductSpaceParameters<1, 3, 3, 1>;
template struct TensorProductSpaceParameters<2, 3, 3, 1>;

template struct TensorProductSpaceParameters<1, 2, 3, 3>;
template struct TensorProductSpaceParameters<1, 3, 3, 3>;
template struct TensorProductSpaceParameters<2, 3, 3, 3>;

template class TensorProductSpace<1, 2, 3, 1>;
template class TensorProductSpace<1, 3, 3, 1>;
template class TensorProductSpace<2, 3, 3, 1>;

template class TensorProductSpace<1, 2, 3, 3>;
template class TensorProductSpace<1, 3, 3, 3>;
template class TensorProductSpace<2, 3, 3, 3>;