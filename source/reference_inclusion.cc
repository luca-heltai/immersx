#include "reference_inclusion.h"

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>


template <int dim, int spacedim, int n_components>
ReferenceInclusion<dim, spacedim, n_components>::ReferenceInclusion(
  const ReferenceInclusionParameters<dim, spacedim, n_components> &par)
  : par(par)
  , quadrature_formula(par.n_q_points)
  , quadrature_formula_1d(par.n_q_points)
  , fe(FE_DGQArbitraryNodes<dim, spacedim>(quadrature_formula_1d), n_components)
  , dof_handler(triangulation)
{
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
      switch (par.inclusion_type)
        {
          case ReferenceInclusionParameters<dim, spacedim, n_components>::
            InclusionType::disk:
            if constexpr (dim == 2 && spacedim == 3)
              GridGenerator::hyper_sphere(triangulation);
            else
              GridGenerator::hyper_ball(triangulation);
            break;
          case ReferenceInclusionParameters<dim, spacedim, n_components>::
            InclusionType::ball:
            GridGenerator::hyper_ball(triangulation);
            break;
          case ReferenceInclusionParameters<dim, spacedim, n_components>::
            InclusionType::cube:
            GridGenerator::hyper_cube(triangulation, -1, 1);
            break;
          default:
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
