// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_lib.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/domain.h>
#include <immersx/core/known_term.h>
#include <immersx/core/lifted_integration_support.h>
#include <immersx/core/weak_term.h>
#include <immersx/coupling/immersed_repartitioner.h>
#include <immersx/coupling/particle_coupling.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct SquareOnSimplex
  {
    struct Evaluation
    {
      double              value;
      std::vector<double> derivatives;
    };
    Evaluation
    evaluate(const std::vector<double> &values) const
    {
      return {values.at(0) * values.at(0), {2. * values.at(0)}};
    }
  };

  template <int dim>
  void
  check_simplex_integrals()
  {
    Triangulation<dim> cubes, tria;
    GridGenerator::hyper_cube(cubes);
    GridGenerator::convert_hypercube_to_simplex_mesh(cubes, tria);
    FE_SimplexP<dim> fe(1);
    MappingFE<dim>   mapping(fe);
    DoFHandler<dim>  dh(tria);
    dh.distribute_dofs(fe);
    AffineConstraints<double> constraints;
    constraints.close();
    StateLayout layout;
    const auto  space  = fe_space(dh, mapping, constraints);
    const auto  source = space.field(layout, "source");
    const auto  target = space.field(layout, "target");
    using VectorType   = Vector<double>;
    using MatrixType   = SparseMatrix<double>;
    using Model        = SemiDiscreteModel<VectorType, MatrixType>;
    VectorType state(dh.n_dofs()), zero(dh.n_dofs());
    state = 1.;
    StateView<VectorType> view(layout, 0.);
    view.bind(source.id(), state);
    view.bind(target.id(), zero);
    const EvaluationContext<VectorType> context(0., view);
    auto                                residual = [&](const auto &term) {
      Model                                       model;
      SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
      term.add(builder);
      VectorType result(dh.n_dofs());
      model.evaluate_row(target.id(), context, result);
      return result;
    };
    EXPECT_NEAR(residual(weak_term(value(source), test(target))).mean_value() *
                  dh.n_dofs(),
                1.,
                1.e-12);
    EXPECT_NEAR(residual(weak_term(value(source), test(target)).on_boundary(0))
                    .mean_value() *
                  dh.n_dofs(),
                2. * dim,
                1.e-12);
    Functions::ConstantFunction<dim> two(2.);
    EXPECT_NEAR(residual(known_term(known_function<dim>(two), test(target)))
                    .mean_value() *
                  dh.n_dofs(),
                2.,
                1.e-12);
    EXPECT_NEAR(residual(known_term(known_function<dim>(two), test(target))
                           .on_boundary(0))
                    .mean_value() *
                  dh.n_dofs(),
                4. * dim,
                1.e-12);
    EXPECT_NEAR(
      residual(known_term(frozen(source, state), test(target))).mean_value() *
        dh.n_dofs(),
      1.,
      1.e-12);
    EXPECT_NEAR(
      residual(known_term(frozen(source, state), test(target)).on_boundary(0))
          .mean_value() *
        dh.n_dofs(),
      2. * dim,
      1.e-12);

    Model                                       nonlinear;
    SemidiscreteBuilder<VectorType, MatrixType> builder(layout, nonlinear);
    weak_term(transform(value(source), SquareOnSimplex{}), test(target))
      .add(builder);
    VectorType result(dh.n_dofs()), action(dh.n_dofs());
    nonlinear.evaluate_row(target.id(), context, result);
    EXPECT_NEAR(result.mean_value() * dh.n_dofs(), 1., 1.e-12);
    nonlinear.state_operator(target.id(), source.id(), context)
      .vmult(action, state);
    EXPECT_NEAR(action.mean_value() * dh.n_dofs(), 2., 1.e-12);

    std::vector<Point<dim>> points(dh.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dh, points);
    for (unsigned int i = 0; i < state.size(); ++i)
      state[i] = points[i][0];
    result = residual(weak_term(gradient(source), gradient(test(target))));
    EXPECT_NEAR(state * result, 1., 1.e-12);

    state = 1.;
    Model                                       constrained;
    SemidiscreteBuilder<VectorType, MatrixType> constraint_builder(layout,
                                                                   constrained);
    const auto                                  fields =
      make_constraint(weak_term(value(source), test(space.field("lambda"))))
        .add(constraint_builder);
    StateView<VectorType> constraint_view(layout, 0.);
    constraint_view.bind(source.id(), state);
    constraint_view.bind(fields.multiplier.id(), state);
    const EvaluationContext<VectorType> constraint_context(0., constraint_view);
    constrained.evaluate_row(fields.multiplier.id(),
                             constraint_context,
                             result);
    EXPECT_NEAR(result.mean_value() * dh.n_dofs(), 1., 1.e-12);
    const auto metric =
      constrained.multiplier_metric(fields.multiplier.id(), constraint_context);
    ASSERT_TRUE(metric.has_value());
    metric->matrix()->vmult(action, state);
    EXPECT_NEAR(action.mean_value() * dh.n_dofs(), 1., 1.e-12);
  }

  template <int dim>
  void
  check_simplex_domain()
  {
    ParameterAcceptor::clear();
    Triangulation<dim> cubes, coarse;
    GridGenerator::hyper_cube(cubes);
    GridGenerator::convert_hypercube_to_simplex_mesh(cubes, coarse);
    DomainParameters<dim> parameters("/Tests/SimplexDomain/");
    parameters.triangulation_type = "fullydistributed";
    parameters.initial_refinement = 1;
    Domain<dim> domain(parameters);
    domain.prepare_grid(coarse);
    const auto expected_cells = coarse.n_active_cells() * (1u << dim);
    domain.finalize_grid();
    EXPECT_EQ(domain.triangulation().n_global_active_cells(), expected_cells);
    FiniteElementSpace<dim> space(domain.triangulation(),
                                  std::make_unique<FE_SimplexP<dim>>(1));
    const auto              view = space.view();
    FEValues<dim>           values(view.mapping(),
                         space.finite_element(),
                         QGaussSimplex<dim>(2),
                         update_JxW_values);
    double                  local_measure = 0.;
    for (const auto &cell : space.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell);
          for (const auto q : values.quadrature_point_indices())
            local_measure += values.JxW(q);
        }
    EXPECT_NEAR(Utilities::MPI::sum(local_measure, MPI_COMM_WORLD), 1., 1.e-12);

    ParticleCouplingParameters<dim> particle_parameters(
      "/Tests/SimplexParticles/");
    ParticleCoupling<dim> particles(particle_parameters);
    particles.initialize_particle_handler(domain.triangulation());
    std::vector<Point<dim>>          positions;
    std::vector<std::vector<double>> properties;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        Point<dim> point;
        for (unsigned int d = 0; d < dim; ++d)
          point[d] = 0.2 + 0.1 * d;
        positions.push_back(point);
        properties.push_back({1.});
      }
    particles.insert_points(positions, properties);
    EXPECT_EQ(particles.get_particles().n_global_particles(), 1u);

    Triangulation<1, dim> line;
    GridGenerator::hyper_cube(line, 0.2, 0.8);
    Tensor<1, dim> shift;
    for (unsigned int d = 1; d < dim; ++d)
      shift[d] = 0.3;
    GridTools::shift(shift, line);
    DomainParameters<1, dim> line_parameters("/Tests/SimplexImmersed/");
    line_parameters.initial_refinement = 1;
    Domain<1, dim> immersed(line_parameters);
    immersed.prepare_grid(line);
    const ImmersedRepartitioner<1, dim> policy(domain.triangulation());
    immersed.finalize_grid(policy);
    EXPECT_EQ(immersed.triangulation().n_global_active_cells(), 2u);
  }
} // namespace

TEST(WeakTerm, SimplexTriangleIntegrals)
{
  check_simplex_integrals<2>();
}

TEST(WeakTerm, SimplexTetrahedronIntegrals)
{
  check_simplex_integrals<3>();
}

TEST(Domain, BOTH_SimplexTriangleRefinementAndSpaceMapping)
{
  check_simplex_domain<2>();
}

TEST(Domain, BOTH_SimplexTetrahedronRefinementAndSpaceMapping)
{
  check_simplex_domain<3>();
}

TEST(Domain, BOTH_RejectsMixedReferenceCells)
{
  ParameterAcceptor::clear();
  Triangulation<2>            mixed;
  const std::vector<Point<2>> vertices = {
    {0., 0.}, {1., 0.}, {2., 0.}, {0., 1.}, {1., 1.}, {2., 1.}};
  std::vector<CellData<2>> cells = {CellData<2>(4),
                                    CellData<2>(3),
                                    CellData<2>(3)};
  cells[0].vertices              = {0, 1, 3, 4};
  cells[1].vertices              = {1, 2, 4};
  cells[2].vertices              = {2, 5, 4};
  mixed.create_triangulation(vertices, cells, SubCellData());
  ASSERT_EQ(mixed.get_reference_cells().size(), 2u);
  DomainParameters<2> parameters("/Tests/MixedDomain/");
  parameters.triangulation_type = "fullydistributed";
  parameters.initial_refinement = 0;
  Domain<2> domain(parameters);
  EXPECT_THROW(domain.prepare_grid(mixed), ExceptionBase);
  const IntegrationSupport<2> support(mixed);
  EXPECT_THROW(support.triangulation(), ExceptionBase);
}

TEST(LiftedIntegrationSupport, BOTH_TriangularSheetVolume)
{
  ParameterAcceptor::clear();
  Triangulation<2, 3> cubes, coarse;
  GridGenerator::hyper_cube(cubes);
  GridGenerator::convert_hypercube_to_simplex_mesh(cubes, coarse);
  DomainParameters<2, 3> parameters("/Tests/SimplexSheetDomain/");
  parameters.triangulation_type = "fullydistributed";
  parameters.initial_refinement = 0;
  Domain<2, 3> domain(parameters);
  domain.prepare_grid(coarse);
  domain.finalize_grid();
  TensorProductLift<2, 3, 3> lift("/Tests/SimplexSheetLift/");
  lift.section.inclusion_type        = "hyper_cube";
  lift.section.inclusion_degree      = 0;
  lift.section.refinement_level      = 0;
  lift.section.selected_coefficients = {0};
  lift.section.n_q_points            = 2;
  lift.representative_n_q_points     = 2;
  lift.representative_n_repetitions  = 2;
  lift.thickness                     = "0.25";
  const auto support                 = domain.lift(lift);
  double     local_measure           = 0.;
  for (const auto &point : support.locally_owned_quadrature_points())
    {
      local_measure += point.weight;
      EXPECT_GE(point.point[0], -1.e-12);
      EXPECT_LE(point.point[0], 1. + 1.e-12);
      EXPECT_GE(point.point[1], -1.e-12);
      EXPECT_LE(point.point[1], 1. + 1.e-12);
      EXPECT_LE(std::abs(point.point[2]), 0.25 + 1.e-12);
    }
  // The interval cross-section is [-1,1]: volume = area * 2 * thickness.
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, MPI_COMM_WORLD), 0.5, 1.e-12);
}
