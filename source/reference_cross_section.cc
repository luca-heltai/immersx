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
  , fe(FE_Q<dim, spacedim>(par.inclusion_degree), n_components)
  , mapping(par.inclusion_degree)
  , dof_handler(triangulation)
{
  make_grid();
  setup_dofs();
  compute_basis();
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
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      points.insert(points.end(),
                    fe_values.get_quadrature_points().begin(),
                    fe_values.get_quadrature_points().end());
      weights.insert(weights.end(),
                     fe_values.get_JxW_values().begin(),
                     fe_values.get_JxW_values().end());
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
          basis_functions[i].sadd(1, -coeff, basis_functions[j]);
        }
      const auto coeff = mass_matrix.matrix_scalar_product(basis_functions[i],
                                                           basis_functions[i]);

      // If the basis functions are on a plane, and we are in 3d, they may be
      // zero
      if (coeff > 1e-10)
        basis_functions[i] /= std::sqrt(coeff);
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
}



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
  : ParameterAcceptor("Reference inclusion")
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
  Tensor<1, spacedim> new_vertical_normalized =
    new_vertical / new_vertical.norm();
  vertical[spacedim - 1] = 1;
  if constexpr (spacedim == 3)
    {
      Tensor<1, spacedim> axis =
        cross_product_3d(vertical, new_vertical_normalized);
      double angle =
        Physics::VectorRelations::signed_angle(vertical,
                                               new_vertical_normalized,
                                               axis);
      rotation =
        Physics::Transformations::Rotations::rotation_matrix_3d(axis, angle);
    }
  else if constexpr (spacedim == 2)
    {
      double angle =
        Physics::VectorRelations::angle(vertical, new_vertical_normalized);
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
  return quadrature_formula.size();
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
