#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "reference_cross_section.h"


template <int dim, int spacedim, int n_components>
ReferenceCrossSection<dim, spacedim, n_components>::ReferenceCrossSection(
  const ReferenceCrossSectionParameters<dim, spacedim, n_components> &par)
  : par(par)
  , polynomials(par.inclusion_degree)
  , quadrature_formula(2 * par.inclusion_degree + 1)
  , fe(FE_Q<dim, spacedim>(par.inclusion_degree), n_components)
  , dof_handler(triangulation)
{
  for (const auto &index : par.selected_coefficients)
    {
      AssertThrow(index < polynomials.n() * n_components,
                  ExcMessage(
                    "One of the selected index is out of range. It should be "
                    "in [0, inclusion_degree*n_components)."));
    }
  make_grid();
  setup_dofs();
  compute_basis();
}


template <int dim, int spacedim, int n_components>
void
ReferenceCrossSection<dim, spacedim, n_components>::make_grid()

{
  if constexpr (dim == 1)
    GridGenerator::hyper_cube(triangulation, -1, 1);
  else
    {
      if (par.inclusion_type == "hyper_ball")
        {
          if constexpr (dim == 2 && spacedim == 3)
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

  std::vector<Point<spacedim>> points(n_global_q_points);
  std::vector<double>          weights(n_global_q_points);
  FEValues<dim, spacedim>      fe_values(
    fe, quadrature_formula, update_quadrature_points | update_JxW_values);
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
      basis_functions[i] /= std::sqrt(coeff);
    }

  // functions
  if (par.selected_coefficients.empty())
    {
      par.selected_coefficients.resize(basis_functions.size());
      std::iota(par.selected_coefficients.begin(),
                par.selected_coefficients.end(),
                0);
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
  return n_components * polynomials.n();
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
template class ReferenceCrossSectionParameters<1, 2, 1>;
template class ReferenceCrossSectionParameters<1, 3, 1>;

template class ReferenceCrossSectionParameters<2, 2, 1>;
template class ReferenceCrossSectionParameters<2, 3, 1>;
template class ReferenceCrossSectionParameters<3, 3, 1>;

// Vector case
template class ReferenceCrossSectionParameters<1, 2, 2>;
template class ReferenceCrossSectionParameters<1, 3, 3>;

template class ReferenceCrossSectionParameters<2, 2, 2>;
template class ReferenceCrossSectionParameters<2, 3, 3>;
template class ReferenceCrossSectionParameters<3, 3, 3>;
