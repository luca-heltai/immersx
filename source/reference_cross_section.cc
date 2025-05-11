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

#include "reference_cross_section.h"

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/transformations.h>
#include <deal.II/physics/vector_relations.h>

#include <string>

template <int dim, int spacedim, int n_components>
ReferenceCrossSection<dim, spacedim, n_components>::ReferenceCrossSection(
  const ReferenceCrossSectionParameters<dim, spacedim, n_components> &par)
  : par(par)
  , polynomials(par.inclusion_degree)
  , quadrature_formula(2 * par.inclusion_degree + 1)
  , fe(FE_Q<dim, spacedim>(std::max(par.inclusion_degree, 1u)), n_components)
  , mapping(fe.degree)
  , dof_handler(triangulation)
{
  initialize();
}


template <int dim, int spacedim, int n_components>
void
ReferenceCrossSection<dim, spacedim, n_components>::make_grid()

{
  if (par.inclusion_type == "hyper_ball")
    {
      if constexpr (dim == 1 && spacedim == 3)
        {
          // 1D inclusion in 3D space. A disk.
          Triangulation<1, 2> tria_12;
          GridGenerator::hyper_sphere(tria_12);
          GridGenerator::flatten_triangulation(tria_12, triangulation);
          triangulation.set_all_manifold_ids(1);
          triangulation.set_manifold(1, PolarManifold<dim, spacedim>());
        }
      else if constexpr (dim == spacedim - 1)
        GridGenerator::hyper_sphere(triangulation);
      else
        GridGenerator::hyper_ball(triangulation);
    }
  else if (par.inclusion_type == "hyper_cube")
    GridGenerator::hyper_cube(triangulation, -1, 1);
  else
    {
      AssertThrow(false,
                  ExcMessage("Unknown inclusion type. Please check the "
                             "parameter file."));
    }
  triangulation.refine_global(par.refinement_level);
}


template <int dim, int spacedim, int n_components>
void
ReferenceCrossSection<dim, spacedim, n_components>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  sparsity_pattern.compress();
  mass_matrix.reinit(sparsity_pattern);
  MatrixTools::create_mass_matrix(dof_handler, quadrature_formula, mass_matrix);

  const auto n_global_q_points =
    triangulation.n_active_cells() * quadrature_formula.size();

  std::vector<Point<spacedim>> points;
  std::vector<double>          weights;
  points.reserve(n_global_q_points);
  weights.reserve(n_global_q_points);

  FEValues<dim, spacedim> fe_values(mapping,
                                    fe,
                                    quadrature_formula,
                                    update_quadrature_points |
                                      update_JxW_values);
  reference_measure = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      points.insert(points.end(),
                    fe_values.get_quadrature_points().begin(),
                    fe_values.get_quadrature_points().end());
      weights.insert(weights.end(),
                     fe_values.get_JxW_values().begin(),
                     fe_values.get_JxW_values().end());
      for (const auto &w : fe_values.get_JxW_values())
        reference_measure += w;
    }
  global_quadrature = Quadrature<spacedim>(points, weights);
}


template <int dim, int spacedim, int n_components>
void
ReferenceCrossSection<dim, spacedim, n_components>::compute_basis()
{
  basis_functions.resize(polynomials.n() * n_components,
                         Vector<double>(dof_handler.n_dofs()));
  for (unsigned int i = 0; i < basis_functions.size(); ++i)
    {
      const auto comp_i = i % n_components;
      const auto poly_i = i / n_components;
      VectorFunctionFromScalarFunctionObject<spacedim> function(
        [&](const Point<spacedim> &p) {
          return polynomials.compute_value(poly_i, p);
        },
        comp_i,
        n_components);
      VectorTools::interpolate(dof_handler, function, basis_functions[i]);
    }
  // Grahm-Schmidt orthogonalization
  std::vector<bool> is_zero(basis_functions.size(), false);
  for (unsigned int i = 0; i < basis_functions.size(); ++i)
    {
      for (unsigned int j = 0; j < i; ++j)
        {
          const auto coeff =
            mass_matrix.matrix_scalar_product(basis_functions[i],
                                              basis_functions[j]);
          basis_functions[i].sadd(1,
                                  -coeff / reference_measure,
                                  basis_functions[j]);
        }
      const auto coeff = mass_matrix.matrix_scalar_product(basis_functions[i],
                                                           basis_functions[i]);

      // If the basis functions are on a plane, and we are in 3d, they may be
      // zero
      if (coeff > 1e-10)
        basis_functions[i] *= std::sqrt(reference_measure) / std::sqrt(coeff);
      else
        is_zero[i] = true;
    }

  // Remove the zero basis functions
  for (int i = is_zero.size() - 1; i >= 0; --i)
    if (is_zero[i] == true)
      basis_functions.erase(basis_functions.begin() + i);

  // functions
  if (par.selected_coefficients.empty())
    {
      par.selected_coefficients.resize(basis_functions.size());
      std::iota(par.selected_coefficients.begin(),
                par.selected_coefficients.end(),
                0);
    }
  else
    {
      for (const auto &index : par.selected_coefficients)
        {
          AssertThrow(index < basis_functions.size(),
                      ExcIndexRange(index, 0, basis_functions.size()));
        }
    }

  selected_basis_functions.resize(par.selected_coefficients.size(),
                                  Vector<double>(dof_handler.n_dofs()));
  // Compute the selected basis functions as the rows of the inverse metric
  // times the basis functions
  for (unsigned int i = 0; i < par.selected_coefficients.size(); ++i)
    {
      selected_basis_functions[i] =
        basis_functions[par.selected_coefficients[i]];
    }

  // Now compute the Matrix of the selected basis functions evaluated on the
  // quadrature points
  basis_functions_on_qpoints.reinit(global_quadrature.size(),
                                    selected_basis_functions.size() *
                                      n_components);

  std::vector<Vector<double>> values(quadrature_formula.size(),
                                     Vector<double>(fe.n_components()));

  FEValues<dim, spacedim> fe_values(mapping,
                                    fe,
                                    quadrature_formula,
                                    update_values);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      const unsigned int shift =
        cell->global_active_cell_index() * quadrature_formula.size();

      for (unsigned int i = 0; i < selected_basis_functions.size(); ++i)
        {
          fe_values.get_function_values(selected_basis_functions[i], values);
          for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
            {
              for (unsigned int comp = 0; comp < n_components; ++comp)
                {
                  basis_functions_on_qpoints(q + shift,
                                             i * n_components + comp) =
                    values[q][comp];
                }
            }
        }
    }
}


template <int dim, int spacedim, int n_components>
double
ReferenceCrossSection<dim, spacedim, n_components>::measure(
  const double scale) const
{
  return reference_measure * std::pow(scale, dim);
};



template <int dim, int spacedim, int n_components>
auto
ReferenceCrossSection<dim, spacedim, n_components>::get_global_quadrature()
  const -> const Quadrature<spacedim> &
{
  return global_quadrature;
}



template <int dim, int spacedim, int n_components>
auto
ReferenceCrossSection<dim, spacedim, n_components>::get_basis_functions() const
  -> const std::vector<Vector<double>> &
{
  return selected_basis_functions;
}

template <int dim, int spacedim, int n_components>
const double &
ReferenceCrossSection<dim, spacedim, n_components>::shape_value(
  const unsigned int i,
  const unsigned int q,
  const unsigned int comp) const
{
  AssertIndexRange(i, selected_basis_functions.size());
  AssertIndexRange(q, global_quadrature.size());
  AssertIndexRange(comp, n_components);

  // Check that the matrix is not empty
  AssertThrow(!basis_functions_on_qpoints.empty(),
              ExcMessage(
                "The basis functions on quadrature points are empty."));
  return basis_functions_on_qpoints(q, i * n_components + comp);
};



template <int dim, int spacedim, int n_components>
auto
ReferenceCrossSection<dim, spacedim, n_components>::get_mass_matrix() const
  -> const SparseMatrix<double> &
{
  return mass_matrix;
}



template <int dim, int spacedim, int n_components>
auto
ReferenceCrossSection<dim, spacedim, n_components>::n_selected_basis() const
  -> unsigned int
{
  return selected_basis_functions.size();
}



template <int dim, int spacedim, int n_components>
unsigned int
ReferenceCrossSection<dim, spacedim, n_components>::max_n_basis() const
{
  return basis_functions.size();
}

template <int dim, int spacedim, int n_components>
ReferenceCrossSectionParameters<dim, spacedim, n_components>::
  ReferenceCrossSectionParameters()
  : ParameterAcceptor("Cross section")
{
  add_parameter("Maximum inclusion degree", inclusion_degree);
  add_parameter(
    "Selected indices",
    selected_coefficients,
    "This allows one to select a subset of the components of the "
    "basis functions used for the representation of the data "
    "(boundary data or forcing data). Notice that these indices are "
    "w.r.t. to the total number of components of the problem, that "
    "is, dimension of the space P^{inclusion_degree} number of Fourier coefficients x number of vector "
    "components. In particular any entry of this list must be in "
    "the set [0,inclusion_degree*n_components). ");
  add_parameter("Inclusion type", inclusion_type);
  add_parameter("Refinement level", refinement_level);
}

template <int dim, int spacedim, int n_components>
Quadrature<spacedim>
ReferenceCrossSection<dim, spacedim, n_components>::get_transformed_quadrature(
  const Point<spacedim>     &new_origin,
  const Tensor<1, spacedim> &new_vertical,
  const double               scale) const
{
  AssertThrow(new_vertical.norm() > 0,
              ExcMessage("The new vertical direction must be non-zero."));

  Tensor<2, spacedim> rotation;
  Tensor<1, spacedim> vertical;
  vertical[spacedim - 1] = 1;
  if constexpr (spacedim == 3)
    {
      Tensor<1, spacedim> axis      = cross_product_3d(vertical, new_vertical);
      const double        axis_norm = axis.norm();
      if (axis_norm < 1e-10)
        {
          // The two vectors are parallel, no rotation needed
          for (unsigned int i = 0; i < spacedim; ++i)
            rotation[i][i] = 1;
        }
      else
        {
          axis /= axis_norm;
          double angle = Physics::VectorRelations::signed_angle(vertical,
                                                                new_vertical,
                                                                axis);
          rotation =
            Physics::Transformations::Rotations::rotation_matrix_3d(axis,
                                                                    angle);
        }
    }
  else if constexpr (spacedim == 2)
    {
      double angle = Physics::VectorRelations::angle(vertical, new_vertical);
      rotation = Physics::Transformations::Rotations::rotation_matrix_2d(angle);
    }

  // Create transformed quadrature by applying the mapping to points and weights
  std::vector<Point<spacedim>> transformed_points(global_quadrature.size());
  std::vector<double>          transformed_weights(global_quadrature.size());

  const auto &original_points  = global_quadrature.get_points();
  const auto &original_weights = global_quadrature.get_weights();

  // Calculate scale factor for weights
  const double weight_scale_factor = std::pow(scale, dim);

  for (unsigned int q = 0; q < original_points.size(); ++q)
    {
      transformed_points[q] =
        rotation * (scale * original_points[q]) + new_origin;

      // Scale the weight by scale^dim
      transformed_weights[q] = original_weights[q] * weight_scale_factor;
    }

  return Quadrature<spacedim>(transformed_points, transformed_weights);
}


template <int dim, int spacedim, int n_components>
unsigned int
ReferenceCrossSection<dim, spacedim, n_components>::n_quadrature_points() const
{
  return global_quadrature.size();
}

template <int dim, int spacedim, int n_components>
void
ReferenceCrossSection<dim, spacedim, n_components>::initialize()
{
  dof_handler.clear();
  triangulation.clear();
  make_grid();
  setup_dofs();
  compute_basis();
}



// Scalar case
template class ReferenceCrossSection<1, 2, 1>;
template class ReferenceCrossSection<1, 3, 1>;

template class ReferenceCrossSection<2, 2, 1>;
template class ReferenceCrossSection<2, 3, 1>;
template class ReferenceCrossSection<3, 3, 1>;

// Vector case
template class ReferenceCrossSection<1, 2, 2>;
template class ReferenceCrossSection<1, 3, 3>;

template class ReferenceCrossSection<2, 2, 2>;
template class ReferenceCrossSection<2, 3, 3>;
template class ReferenceCrossSection<3, 3, 3>;

// Explicit instantiations for ReferenceCrossSectionParameters
// Scalar case
template struct ReferenceCrossSectionParameters<1, 2, 1>;
template struct ReferenceCrossSectionParameters<1, 3, 1>;

template struct ReferenceCrossSectionParameters<2, 2, 1>;
template struct ReferenceCrossSectionParameters<2, 3, 1>;
template struct ReferenceCrossSectionParameters<3, 3, 1>;

// Vector case
template struct ReferenceCrossSectionParameters<1, 2, 2>;
template struct ReferenceCrossSectionParameters<1, 3, 3>;

template struct ReferenceCrossSectionParameters<2, 2, 2>;
template struct ReferenceCrossSectionParameters<2, 3, 3>;
template struct ReferenceCrossSectionParameters<3, 3, 3>;
