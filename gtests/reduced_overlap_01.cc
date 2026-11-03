#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparsity_tools.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/coupling/inclusions.h>
#include <immersx/coupling/reduced_coupling.h>

#include <cmath>
#include <cstdint>

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
    ImmersXLA::MPI::Vector prescribed(owned, MPI_COMM_WORLD),
      expected_prescribed(owned, MPI_COMM_WORLD);
    coupling.assemble_prescribed_rhs(prescribed);
    const auto data = coupling.source_modal_data_values();
    for (const auto &[entity, values] : data)
      {
        (void)entity;
        for (const auto &coefficients : values)
          EXPECT_EQ(coefficients, (std::vector<double>{1., 2., -0.5}));
      }
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
            expected_prescribed[indices[i]] +=
              weights[q][0] * basis[i] * imposed;
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
    expected_prescribed.compress(VectorOperation::add);
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
    prescribed -= expected_prescribed;
    EXPECT_LT(prescribed.l2_norm(), 1.e-12);
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

namespace
{
  void
  check_inclusion_overlap(const bool cluster)
  {
    ParameterAcceptor::clear();
    parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(background, 0., 1.);
    background.refine_global(1);
    Inclusions<2> inclusions(2);
    inclusions.set_n_q_points(32);
    inclusions.set_n_coefficients(3);
    inclusions.inclusions            = {{-0.1, 0.5, 0.25},
                                        {2., 2., 0.25},
                                        {0.35, 0.35, 0.25},
                                        {0.4, 0.4, 0.25}};
    inclusions.cluster_with_segments = cluster;
    inclusions.setup_inclusions_particles(background);
    IndexSet owned;
    if (cluster)
      {
        EXPECT_EQ(inclusions.n_global_segments(), 3u);
        const auto partitioning = Utilities::MPI::create_ascending_partitioning(
          MPI_COMM_WORLD, inclusions.n_local_segments());
        owned = partitioning[Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]
                  .tensor_product(complete_index_set(6));
      }
    else
      owned =
        Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,
                                                               4)
          .tensor_product(complete_index_set(6));
    IndexSet relevant = owned;
    for (const auto inclusion : inclusions.locally_owned_source_inclusions())
      {
        const auto indices = inclusions.get_dof_indices(inclusion * 32);
        relevant.add_indices(indices.begin(), indices.end());
      }
    relevant.compress();
    DynamicSparsityPattern graph(relevant);
    for (const auto inclusion : inclusions.locally_owned_source_inclusions())
      {
        const auto indices = inclusions.get_dof_indices(inclusion * 32);
        for (const auto i : indices)
          for (const auto j : indices)
            graph.add(i, j);
      }
    SparsityTools::distribute_sparsity_pattern(graph,
                                               owned,
                                               MPI_COMM_WORLD,
                                               relevant);
    ImmersXLA::MPI::SparseMatrix total, overlap, outside, expected_overlap,
      expected_outside;
    for (auto *matrix :
         {&total, &overlap, &outside, &expected_overlap, &expected_outside})
      matrix->reinit(owned, owned, graph, MPI_COMM_WORLD);
    inclusions.assemble_coupling_mass_matrix(total);
    inclusions.assemble_overlap_mass_matrix(overlap);
    inclusions.assemble_outside_mass_matrix(outside);
    const auto    source      = inclusions.locally_owned_source_points();
    std::uint64_t found_count = 0;
    for (const auto inclusion : inclusions.locally_owned_source_inclusions())
      {
        const auto points  = inclusions.get_current_support_points(inclusion);
        const auto indices = inclusions.get_dof_indices(inclusion * 32);
        for (unsigned int q = 0; q < 32; ++q)
          {
            const auto &point = points[q];
            const bool found = point[0] >= -1.e-10 && point[0] <= 1. + 1.e-10 &&
                               point[1] >= -1.e-10 && point[1] <= 1. + 1.e-10;
            EXPECT_EQ(
              inclusions.source_point_membership()[source.index_within_set(
                inclusion * 32 + q)],
              found);
            found_count += found;
            const double              theta = 2. * numbers::PI * q / 32.;
            const std::vector<double> basis{1.,
                                            1.,
                                            std::cos(theta),
                                            std::cos(theta),
                                            std::sin(theta),
                                            std::sin(theta)};
            const double              weight = 2. * numbers::PI * 0.25 / 32.;
            auto &expected = found ? expected_overlap : expected_outside;
            for (unsigned int i = 0; i < 6; ++i)
              for (unsigned int j = 0; j < 6; ++j)
                if (i % 2 == j % 2)
                  expected.add(indices[i],
                               indices[j],
                               weight * basis[i] * basis[j]);
          }
      }
    expected_overlap.compress(VectorOperation::add);
    expected_outside.compress(VectorOperation::add);
    overlap.add(-1., expected_overlap);
    outside.add(-1., expected_outside);
    total.add(-1., expected_overlap);
    total.add(-1., expected_outside);
    EXPECT_LT(overlap.frobenius_norm(), 1.e-12);
    EXPECT_LT(outside.frobenius_norm(), 1.e-12);
    EXPECT_LT(total.frobenius_norm(), 1.e-12);
    EXPECT_EQ(Utilities::MPI::sum(found_count, MPI_COMM_WORLD),
              inclusions.inclusions_as_particles.n_global_particles());
    EXPECT_EQ(Utilities::MPI::sum(source.n_elements(), MPI_COMM_WORLD), 128u);
  }
} // namespace

TEST(InclusionsBasis2, BOTH_OverlapGramRetainsExteriorSourcePoints)
{
  check_inclusion_overlap(false);
}

TEST(InclusionsBasis2, BOTH_ExteriorCenterSegmentsHaveOwners)
{
  check_inclusion_overlap(true);
}

#endif
