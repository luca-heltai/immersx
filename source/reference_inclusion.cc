#include "reference_inclusion.h"

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


template <int dim, int spacedim, int n_components>
ReferenceInclusion<dim, spacedim, n_components>::ReferenceInclusion(
  const ReferenceInclusionParameters<dim, spacedim, n_components> &par)
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
}


template <int dim, int spacedim, int n_components>
void
ReferenceInclusion<dim, spacedim, n_components>::make_grid()

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
ReferenceInclusion<dim, spacedim, n_components>::setup_dofs()
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
ReferenceInclusion<dim, spacedim, n_components>::compute_basis()
{
  std::vector<Vector<double>> basis_functions(
    polynomials.n() * n_components, Vector<double>(dof_handler.n_dofs()));
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
  // Metric and inverse metric
  FullMatrix<double> G(basis_functions.size(), basis_functions.size());
  FullMatrix<double> Ginv(basis_functions.size(), basis_functions.size());
  for (unsigned int i = 0; i < basis_functions.size(); ++i)
    for (unsigned int j = 0; j < basis_functions.size(); ++j)
      G(i, j) = mass_matrix.matrix_scalar_product(basis_functions[i],
                                                  basis_functions[j]);
  G.invert(Ginv);
  selected_basis_functions.resize(par.selected_coefficients.size(),
                                  Vector<double>(dof_handler.n_dofs()));
  // Compute the selected basis functions as the rows of the inverse metric
  // times the basis functions
  for (unsigned int i = 0; i < par.selected_coefficients.size(); ++i)
    {
      const auto index = par.selected_coefficients[i];
      for (unsigned int j = 0; j < basis_functions.size(); ++j)
        for (unsigned int k = 0; k < dof_handler.n_dofs(); ++k)
          {
            selected_basis_functions[i][k] +=
              Ginv(index, j) * basis_functions[j][k];
          }
    }
}

template <int dim, int spacedim, int n_components>
ReferenceInclusionParameters<dim, spacedim, n_components>::
  ReferenceInclusionParameters()
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
template class ReferenceInclusion<1, 2, 1>;
template class ReferenceInclusion<1, 3, 1>;

template class ReferenceInclusion<2, 2, 1>;
template class ReferenceInclusion<2, 3, 1>;
template class ReferenceInclusion<3, 3, 1>;

// Vector case
template class ReferenceInclusion<1, 2, 2>;
template class ReferenceInclusion<1, 3, 3>;

template class ReferenceInclusion<2, 2, 2>;
template class ReferenceInclusion<2, 3, 3>;
template class ReferenceInclusion<3, 3, 3>;

// Explicit instantiations for ReferenceInclusionParameters
// Scalar case
template class ReferenceInclusionParameters<1, 2, 1>;
template class ReferenceInclusionParameters<1, 3, 1>;

template class ReferenceInclusionParameters<2, 2, 1>;
template class ReferenceInclusionParameters<2, 3, 1>;
template class ReferenceInclusionParameters<3, 3, 1>;

// Vector case
template class ReferenceInclusionParameters<1, 2, 2>;
template class ReferenceInclusionParameters<1, 3, 3>;

template class ReferenceInclusionParameters<2, 2, 2>;
template class ReferenceInclusionParameters<2, 3, 3>;
template class ReferenceInclusionParameters<3, 3, 3>;
