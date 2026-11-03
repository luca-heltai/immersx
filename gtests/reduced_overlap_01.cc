#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparsity_tools.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/coupling/reduced_coupling.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

#ifdef DEAL_II_WITH_VTK
namespace
{
  template <int reduced_dim, int dim, int spacedim>
  void
  check_sampled_gram(ReducedCoupling<reduced_dim, dim, spacedim, 1> &coupling,
                     const Point<spacedim>                          &lower,
                     const Point<spacedim>                          &upper)
  {
    const auto owned    = coupling.locally_owned_representative_dofs();
    const auto relevant = coupling.locally_relevant_representative_dofs();
    DynamicSparsityPattern sparsity(relevant);
    if constexpr (reduced_dim == 0)
      for (const auto entity : coupling.locally_owned_representative_entities())
        {
          const auto &indices = coupling.get_representative_dof_indices(entity);
          for (const auto i : indices)
            for (const auto j : indices)
              sparsity.add(i, j);
        }
    else
      DoFTools::make_sparsity_pattern(coupling.get_dof_handler(),
                                      sparsity,
                                      coupling.get_coupling_constraints(),
                                      false);
    SparsityTools::distribute_sparsity_pattern(sparsity,
                                               owned,
                                               MPI_COMM_WORLD,
                                               relevant);
    ImmersXLA::MPI::SparseMatrix total, inside, outside, expected_inside,
      expected_outside;
    for (auto *matrix :
         {&total, &inside, &outside, &expected_inside, &expected_outside})
      matrix->reinit(owned, owned, sparsity, MPI_COMM_WORLD);
    coupling.assemble_coupling_mass_matrix(total);
    coupling.assemble_overlap_mass_matrix(inside);
    coupling.assemble_outside_mass_matrix(outside);
    const auto            &points  = coupling.get_locally_owned_qpoints();
    const auto            &weights = coupling.get_locally_owned_weights();
    const auto            &modes   = coupling.get_locally_owned_mode_values();
    const auto             owned_points = coupling.locally_owned_qpoints();
    ImmersXLA::MPI::Vector rhs(owned, MPI_COMM_WORLD),
      expected_rhs(owned, MPI_COMM_WORLD);
    coupling.assemble_reduced_rhs(rhs);
    std::uint64_t local_inside = 0, local_outside = 0;
    double        local_cross_mode = 0.;
    for (unsigned int q = 0; q < points.size(); ++q)
      {
        bool found = true;
        for (unsigned int d = 0; d < spacedim; ++d)
          found = found && points[q][d] >= lower[d] - 1.e-10 &&
                  points[q][d] <= upper[d] + 1.e-10;
        EXPECT_EQ(coupling.source_point_membership()[q], found);
        found ? ++local_inside : ++local_outside;
        const auto [entity, representative_q, section_q] =
          coupling.particle_id_to_representative_indices(
            owned_points.nth_index_in_set(q));
        (void)section_q;
        const auto                indices = coupling.get_dof_indices(entity);
        std::vector<double>       basis(indices.size());
        std::vector<unsigned int> component(indices.size());
        for (unsigned int i = 0; i < indices.size(); ++i)
          {
            if constexpr (reduced_dim == 0)
              {
                component[i] = i;
                basis[i]     = modes[q][i];
              }
            else
              {
                const auto &fe = coupling.get_dof_handler().get_fe();
                component[i]   = fe.system_to_component_index(i).first;
                basis[i]       = fe.shape_value(i,
                                          coupling.get_quadrature().point(
                                            representative_q)) *
                           modes[q][component[i]];
              }
          }
        const std::vector<double> imposed_coefficients{1., 2., -0.5};
        double                    imposed = 0.;
        for (unsigned int c = 0; c < imposed_coefficients.size(); ++c)
          imposed += imposed_coefficients[c] * modes[q][c];
        auto &expected = found ? expected_inside : expected_outside;
        for (unsigned int i = 0; i < indices.size(); ++i)
          {
            if (found)
              expected_rhs[indices[i]] += weights[q][0] * basis[i] * imposed;
            for (unsigned int j = 0; j < indices.size(); ++j)
              {
                const double entry = weights[q][0] * basis[i] * basis[j];
                expected.add(indices[i], indices[j], entry);
                if (!found && component[i] == 0 && component[j] != 0)
                  local_cross_mode += std::abs(entry);
              }
          }
      }
    expected_inside.compress(VectorOperation::add);
    expected_outside.compress(VectorOperation::add);
    expected_rhs.compress(VectorOperation::add);
    EXPECT_GT(Utilities::MPI::sum(local_inside, MPI_COMM_WORLD), 0u);
    EXPECT_GT(Utilities::MPI::sum(local_outside, MPI_COMM_WORLD), 0u);
    EXPECT_GT(Utilities::MPI::sum(local_cross_mode, MPI_COMM_WORLD), 1.e-5);
    inside.add(-1., expected_inside);
    outside.add(-1., expected_outside);
    EXPECT_LT(inside.frobenius_norm(), 1.e-12);
    EXPECT_LT(outside.frobenius_norm(), 1.e-12);
    total.add(-1., expected_inside);
    total.add(-1., expected_outside);
    EXPECT_LT(total.frobenius_norm(), 1.e-12);
    rhs -= expected_rhs;
    EXPECT_LT(rhs.l2_norm(), 1.e-12);
  }
} // namespace

TEST(ReducedCoupling, BOTH_OverlapSamplesRetainCrossModeGram)
{
  ParameterAcceptor::clear();
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  const Point<3> lower(0.3, 0.5, 0.), upper(0.8, 1., 1.);
  GridGenerator::hyper_rectangle(background, lower, upper);
  background.refine_global(1);
  ReducedCouplingParameters<1, 2, 3, 1> parameters;
  auto &space = parameters.tensor_product_space_parameters;
  space.reduced_grid_name =
    TestPaths::data_filename("tests/simplex_interior_line_3d.vtk");
  space.thickness                                       = "0.1";
  space.n_q_points                                      = 3;
  space.section.inclusion_degree                        = 1;
  space.section.selected_coefficients                   = {0, 1, 2};
  parameters.coupling_rhs_expressions                   = {"1", "2", "-0.5"};
  parameters.refinement_parameters.max_refinement_level = 0;
  ReducedCoupling<1, 2, 3, 1> coupling(background, parameters);
  coupling.initialize();
  check_sampled_gram(coupling, lower, upper);
}

TEST(ReducedCoupling0D, BOTH_ExteriorCentersRetainSourceSamples)
{
  ParameterAcceptor::clear();
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  background.refine_global(1);
  ReducedCouplingParameters<0, 1, 2, 1> parameters;
  auto &space                    = parameters.tensor_product_space_parameters;
  space.thickness                = "0.25";
  space.section.inclusion_degree = 1;
  space.section.selected_coefficients                   = {0, 1, 2};
  space.point_cloud.points                              = {Point<2>(-0.1, 0.5),
                                                           Point<2>(2., 2.),
                                                           Point<2>(0.5, 0.5)};
  parameters.coupling_rhs_expressions                   = {"1", "2", "-0.5"};
  parameters.refinement_parameters.max_refinement_level = 0;
  ReducedCoupling<0, 1, 2, 1> coupling(background, parameters);
  coupling.initialize();
  EXPECT_EQ(coupling.n_representative_entities(), 3u);
  EXPECT_EQ(Utilities::MPI::sum(
              coupling.locally_owned_representative_entities().n_elements(),
              MPI_COMM_WORLD),
            3u);
  check_sampled_gram(coupling, Point<2>(), Point<2>(1., 1.));
}
#endif
