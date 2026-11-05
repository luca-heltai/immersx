#include <deal.II/base/function_lib.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/vector_tools.h>

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
TEST(ReducedCoupling0D, BOTH_BoundaryMembershipMatchesLocatedParticles)
{
  ParameterAcceptor::clear();
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);
  ReducedCouplingParameters<0, 1, 2, 1> parameters;
  auto &space                    = parameters.tensor_product_space_parameters;
  space.thickness                = "0.25";
  space.section.inclusion_degree = 0;
  space.section.refinement_level = 0;
  space.section.quadrature_type  = "gauss_lobatto";
  space.section.n_q_points       = 2;
  const ReferenceCrossSection<1, 2, 1> section(space.section);
  Tensor<1, 2>                         vertical;
  vertical[1] = 1.;
  const auto offset =
    section.get_transformed_quadrature(Point<2>(), vertical, 0.25).point(0);
  for (const double shift : {-1.e-7, -1.e-11, -1.e-13, 1.e-13, 1.e-7})
    space.point_cloud.points.emplace_back(shift - offset[0], 0.5 - offset[1]);
  parameters.coupling_rhs_expressions                   = {"1"};
  parameters.refinement_parameters.max_refinement_level = 0;
  ReducedCoupling<0, 1, 2, 1> coupling(background, parameters);
  coupling.initialize();
  const auto source = coupling.locally_owned_qpoints();
  IndexSet   located(source.size());
  for (const auto &particle : coupling.get_particles())
    {
      const auto [entity, representative_q, section_q] =
        coupling.particle_id_to_representative_indices(particle.get_id());
      (void)representative_q;
      located.add_index(
        entity * coupling.get_reference_cross_section().n_quadrature_points() +
        section_q);
    }
  for (const auto &on_rank :
       Utilities::MPI::all_gather(MPI_COMM_WORLD, located))
    located.add_indices(on_rank);
  unsigned int near_boundary = 0;
  for (const auto &point : coupling.get_locally_owned_qpoints())
    near_boundary += std::abs(point[0]) < 2.e-7;
  EXPECT_GE(Utilities::MPI::sum(near_boundary, MPI_COMM_WORLD), 5u);
  unsigned int local_q = 0;
  for (const auto q : source)
    EXPECT_EQ(coupling.source_point_membership()[local_q++],
              located.is_element(q));
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

namespace
{
  void
  check_affine_append_with_empty_ranks(const bool insert_input)
  {
    ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
    ParameterAcceptor::clear();
    const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_rectangle(background,
                                              std::vector<unsigned int>{2, 1},
                                              Point<2>(),
                                              Point<2>(1., 1.));
    Point<2> center;
    if (rank == 1)
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            center = cell->center();
            break;
          }
    center = Utilities::MPI::broadcast(MPI_COMM_WORLD, center, 1);
    ReducedCouplingParameters<0, 2, 2, 1> parameters;
    auto &space                    = parameters.tensor_product_space_parameters;
    space.thickness                = "0.05";
    space.section.inclusion_degree = 1;
    space.section.selected_coefficients                   = {0, 1, 2};
    space.point_cloud.points                              = {center};
    parameters.coupling_rhs_expressions                   = {"0", "0", "0"};
    parameters.refinement_parameters.max_refinement_level = 0;
    ReducedCoupling<0, 2, 2, 1> coupling(background, parameters);
    coupling.initialize();
    const auto owned_lambda = coupling.locally_owned_representative_dofs();
    EXPECT_EQ(owned_lambda.n_elements(), rank == 0 ? 3u : 0u);
    EXPECT_EQ(coupling.get_particles().n_locally_owned_particles() == 0,
              rank == 0);
    FE_Q<2>       fe(1);
    DoFHandler<2> dh(background);
    dh.distribute_dofs(fe);
    AffineConstraints<double> constraints(
      dh.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dh));
    VectorTools::interpolate_boundary_values(dh,
                                             0,
                                             Functions::ConstantFunction<2>(1.),
                                             constraints);
    constraints.close();
    DynamicSparsityPattern graph(dh.n_dofs(),
                                 coupling.n_representative_dofs(),
                                 DoFTools::extract_locally_relevant_dofs(dh));
    coupling.assemble_coupling_sparsity(graph, dh, constraints);
    SparsityTools::distribute_sparsity_pattern(
      graph,
      dh.locally_owned_dofs(),
      MPI_COMM_WORLD,
      DoFTools::extract_locally_relevant_dofs(dh));
    ImmersXLA::MPI::SparseMatrix matrix;
    matrix.reinit(dh.locally_owned_dofs(), owned_lambda, graph, MPI_COMM_WORLD);
    ImmersXLA::MPI::Vector rhs(owned_lambda, MPI_COMM_WORLD);
    for (const auto i : owned_lambda)
      if (insert_input)
        rhs[i] = 3. + i;
      else
        {
          const double value = 3. + i;
          rhs.add(1, &i, &value);
        }
    if (insert_input)
      rhs.compress(VectorOperation::insert);
    // Pending additions remain valid on the rank without particles.
    std::vector<double> integrals(3);
    const auto         &weights = coupling.get_locally_owned_weights();
    const auto         &modes   = coupling.get_locally_owned_mode_values();
    for (unsigned int q = 0; q < weights.size(); ++q)
      for (unsigned int i = 0; i < 3; ++i)
        integrals[i] += weights[q][0] * modes[q][i];
    for (auto &integral : integrals)
      integral = Utilities::MPI::sum(integral, MPI_COMM_WORLD);
    coupling.assemble_multiplier_load_matrix(matrix, dh, constraints, rhs);
    EXPECT_LT(matrix.frobenius_norm(), 1.e-13);
    for (const auto i : owned_lambda)
      EXPECT_NEAR(rhs[i], 3. + i - integrals[i], 1.e-12);
  }

} // namespace

TEST(ReducedCoupling0D, MPI_AffineAppendPreservesInsertRhsWithEmptyRanks)
{
  check_affine_append_with_empty_ranks(true);
}

TEST(ReducedCoupling0D, MPI_AffineAppendPreservesAddRhsWithEmptyRanks)
{
  check_affine_append_with_empty_ranks(false);
}

#endif
