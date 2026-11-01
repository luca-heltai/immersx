#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/lifted_integration_support.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/state.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/core/weak_term.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct LiftedLineFixture
  {
    LiftedLineFixture()
      : domain_parameters("/Lifted line domain/")
      , lift("/Lifted line geometry/")
      , domain(domain_parameters)
    {
      domain_parameters.name_of_grid       = "hyper_cube";
      domain_parameters.arguments_for_grid = "-1: 1: false";
      domain_parameters.initial_refinement = 0;
      domain.make_grid();

      lift.section.inclusion_degree      = 0;
      lift.section.inclusion_type        = "hyper_cube";
      lift.section.refinement_level      = 0;
      lift.section.selected_coefficients = {0};
      lift.section.n_q_points            = 2;
      lift.representative_n_q_points     = 2;
      lift.thickness                     = "0.25";
    }

    DomainParameters<1, 3>     domain_parameters;
    TensorProductLift<1, 2, 3> lift;
    Domain<1, 3>               domain;
  };
} // namespace

TEST(LiftedIntegrationSupport, BOTH_UsesPhysicalQuadratureWithoutAField)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  const auto        support = fixture.domain.lift(fixture.lift);
  static_assert(
    is_physical_quadrature_support<decltype(support)>::value,
    "Lifted support must satisfy the physical point-range contract.");

  double                  local_measure = 0.;
  std::set<std::uint64_t> local_ids;
  for (std::size_t q = 0; q < support.n_locally_owned_quadrature_points(); ++q)
    {
      const auto &point = support.locally_owned_quadrature_points()[q];
      local_measure += point.weight;
      EXPECT_TRUE(local_ids.insert(point.stable_id).second);
      EXPECT_EQ(point.owner_rank,
                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
      EXPECT_NEAR(support.normal(q).norm(), 1., 1.e-12);
    }
  const std::vector<std::uint64_t> original_ids(local_ids.begin(),
                                                local_ids.end());

  const auto global_measure =
    Utilities::MPI::sum(local_measure, fixture.domain.communicator());
  EXPECT_NEAR(global_measure, 1., 1.e-10);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  support.n_locally_owned_quadrature_points()),
                                fixture.domain.communicator()),
            4u);

  // The descriptor is non-owning and changes take effect after rebuild().
  fixture.lift.thickness = "0.5";
  support.rebuild();
  local_measure = 0.;
  std::vector<std::uint64_t> rebuilt_ids;
  for (const auto &point : support.locally_owned_quadrature_points())
    {
      local_measure += point.weight;
      rebuilt_ids.push_back(point.stable_id);
    }
  std::sort(rebuilt_ids.begin(), rebuilt_ids.end());
  EXPECT_EQ(rebuilt_ids, original_ids);
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, fixture.domain.communicator()),
              2.,
              1.e-10);
}

TEST(LiftedIntegrationSupport, RejectsInvalidThickness)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  fixture.lift.thickness = "0";
  EXPECT_THROW(fixture.domain.lift(fixture.lift), ExceptionBase);
}

TEST(LiftedIntegrationSupport, BOTH_OwnershipAndIdsArePartitionIndependent)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> domain_parameters("/Distributed lifted line domain/");
  domain_parameters.name_of_grid       = "hyper_cube";
  domain_parameters.arguments_for_grid = "-1: 1: false";
  domain_parameters.initial_refinement = 2;
  Domain<1, 3> domain(domain_parameters);
  domain.make_grid();

  TensorProductLift<1, 2, 3> lift("/Distributed lifted line geometry/");
  lift.section.inclusion_degree      = 0;
  lift.section.inclusion_type        = "hyper_cube";
  lift.section.refinement_level      = 0;
  lift.section.selected_coefficients = {0};
  lift.section.n_q_points            = 2;
  lift.representative_n_q_points     = 2;
  lift.thickness                     = "0.25";
  const auto support                 = domain.lift(lift);

  const auto rank = Utilities::MPI::this_mpi_process(domain.communicator());
  std::vector<std::uint64_t> local_ids;
  for (const auto &point : support.locally_owned_quadrature_points())
    {
      EXPECT_EQ(point.owner_rank, rank);
      local_ids.push_back(point.stable_id);
    }
  const auto gathered_ids =
    Utilities::MPI::all_gather(domain.communicator(), local_ids);
  std::vector<std::uint64_t> all_ids;
  for (const auto &rank_ids : gathered_ids)
    all_ids.insert(all_ids.end(), rank_ids.begin(), rank_ids.end());
  std::sort(all_ids.begin(), all_ids.end());
  ASSERT_EQ(all_ids.size(), 16u);
  EXPECT_TRUE(std::adjacent_find(all_ids.begin(), all_ids.end()) ==
              all_ids.end());
}

TEST(LiftedIntegrationSupport, BOTH_IntegratesFrozenNonconstantThickness)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  const auto        support =
    fixture.domain.lift(fixture.lift, [](const Point<3> &point) {
      return 0.25 + 0.1 * point[0];
    });

  double local_measure = 0.;
  for (const auto &point : support.locally_owned_quadrature_points())
    local_measure += point.weight;
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, fixture.domain.communicator()),
              1.,
              1.e-10);
}

TEST(LiftedIntegrationSupport,
     BOTH_RotatedRepresentativeLineUsesPhysicalGeometry)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> domain_parameters("/Rotated lifted line domain/");
  domain_parameters.name_of_grid =
    TestPaths::data_filename("tests/one_cylinder_rotation_z.vtk");
  domain_parameters.initial_refinement = 0;
  Domain<1, 3> domain(domain_parameters);
  domain.make_grid();

  TensorProductLift<1, 2, 3> lift("/Rotated lifted line geometry/");
  lift.section.inclusion_degree      = 0;
  lift.section.inclusion_type        = "hyper_cube";
  lift.section.refinement_level      = 0;
  lift.section.selected_coefficients = {0};
  lift.section.n_q_points            = 2;
  lift.representative_n_q_points     = 2;
  lift.thickness                     = "0.25";
  const auto support                 = domain.lift(lift);

  double      local_measure = 0.;
  const auto &points        = support.locally_owned_quadrature_points();
  for (std::size_t q = 0; q < points.size(); ++q)
    {
      const auto &point = points[q];
      local_measure += point.weight;
      EXPECT_NEAR(point.representative_point[0], 0., 1.e-12);
      EXPECT_NEAR(point.representative_point[1], 0., 1.e-12);
      EXPECT_NEAR(support.normal(q).norm(), 1., 1.e-12);
    }
  EXPECT_NEAR(Utilities::MPI::sum(local_measure, domain.communicator()),
              2.,
              1.e-10);
}

TEST(LiftedIntegrationSupport, BOTH_AssemblesBulkMassOnPhysicalQuadrature)
{
  ParameterAcceptor::clear();
  LiftedLineFixture fixture;
  const auto        support = fixture.domain.lift(fixture.lift);

  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(1);
  FE_Q<3>       source_element(1);
  FE_Q<3>       target_element(2);
  DoFHandler<3> source_dof_handler(background);
  DoFHandler<3> target_dof_handler(background);
  source_dof_handler.distribute_dofs(source_element);
  target_dof_handler.distribute_dofs(target_element);
  const auto source_relevant =
    DoFTools::extract_locally_relevant_dofs(source_dof_handler);
  const auto target_relevant =
    DoFTools::extract_locally_relevant_dofs(target_dof_handler);
  AffineConstraints<double> source_constraints;
  source_constraints.reinit(source_dof_handler.locally_owned_dofs(),
                            source_relevant);
  DoFTools::make_hanging_node_constraints(source_dof_handler,
                                          source_constraints);
  source_constraints.close();
  AffineConstraints<double> target_constraints;
  target_constraints.reinit(target_dof_handler.locally_owned_dofs(),
                            target_relevant);
  DoFTools::make_hanging_node_constraints(target_dof_handler,
                                          target_constraints);
  target_constraints.close();
  const auto source_space = fe_space(source_dof_handler,
                                     StaticMappingQ1<3>::mapping,
                                     source_constraints,
                                     source_relevant);
  const auto target_space = fe_space(target_dof_handler,
                                     StaticMappingQ1<3>::mapping,
                                     target_constraints,
                                     target_relevant);

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  StateLayout layout;
  const auto  source = source_space.field(layout, "bulk-source");
  const auto  target = target_space.field(layout, "bulk-target");
  Model       model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  (value(source) * test(target) * support).add(builder);

  Vector state;
  state.reinit(source.locally_owned_dofs(), MPI_COMM_WORLD);
  state = 1.;
  state.compress(VectorOperation::insert);
  StateView<Vector> state_view(layout, 0.);
  state_view.bind(source.field_id(), state);
  const EvaluationContext<Vector> context(0., state_view);
  Vector                          residual;
  residual.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  model.evaluate_row(target.field_id(), context, residual);

  const auto matrix =
    model.state_matrix_operator(target.field_id(), source.field_id(), context);
  ASSERT_TRUE(matrix.has_value());
  Vector action;
  action.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  matrix->view.vmult(action, state);
  action -= residual;
  EXPECT_LT(action.l2_norm(), 1.e-12);

  double local_measure = 0.;
  for (const auto &point : support.locally_owned_quadrature_points())
    local_measure += point.weight;
  const auto global_measure =
    Utilities::MPI::sum(local_measure, MPI_COMM_WORLD);
  EXPECT_NEAR(global_measure, 1., 1.e-10);
  Vector ones;
  ones.reinit(target.locally_owned_dofs(), MPI_COMM_WORLD);
  ones = 1.;
  ones.compress(VectorOperation::insert);
  EXPECT_NEAR(residual * ones, global_measure, 1.e-10);
}

TEST(LiftedIntegrationSupport, BOTH_SourceBulkLiftSamplesPhysicalPoints)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> representative_parameters(
    "/Lifted source-bulk domain/");
  representative_parameters.name_of_grid       = "hyper_cube";
  representative_parameters.arguments_for_grid = "-1: 1: false";
  representative_parameters.initial_refinement = 2;
  Domain<1, 3> representative_domain(representative_parameters);
  representative_domain.make_grid();

  TensorProductLift<1, 2, 3, 3> vector_lift("/Lifted source-bulk geometry/");
  vector_lift.section.inclusion_degree      = 0;
  vector_lift.section.inclusion_type        = "hyper_cube";
  vector_lift.section.selected_coefficients = {0};
  vector_lift.section.n_q_points            = 2;
  vector_lift.representative_n_q_points     = 2;
  vector_lift.thickness                     = "0.25";
  const auto support = representative_domain.lift(vector_lift);

  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(1);
  FESystem<3>   bulk_element(FE_Q<3>(2), 3);
  DoFHandler<3> bulk_dof_handler(background);
  bulk_dof_handler.distribute_dofs(bulk_element);
  AffineConstraints<double> bulk_constraints;
  bulk_constraints.close();
  const auto bulk_space =
    fe_space(bulk_dof_handler, StaticMappingQ1<3>::mapping, bulk_constraints);

  FE_Q<1, 3>       line_element(1);
  DoFHandler<1, 3> line_dof_handler(representative_domain.triangulation());
  line_dof_handler.distribute_dofs(line_element);
  AffineConstraints<double> line_constraints;
  line_constraints.close();
  const auto line_space = fe_space(line_dof_handler,
                                   StaticMappingQ1<1, 3>::mapping,
                                   line_constraints);

  StateLayout layout;
  const auto  bulk =
    bulk_space.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line = line_space.field(layout, "line");

  using Vector = ImmersXLA::MPI::Vector;
  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<Vector, Matrix>;
  Model                               model;
  SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
  const auto lifted_line = lift(value(line), vector_lift);
  weak_term(value(bulk), test(lifted_line)).add(builder);

  const auto lift_points = lifted_line.lifted_points();
  ASSERT_FALSE(lift_points.empty());
  const auto direction = lift_points.front().mode_vector;
  ASSERT_NEAR(direction.norm(), 1., 1.e-12);

  class TransverseQuadratic : public Function<3>
  {
  public:
    explicit TransverseQuadratic(const Tensor<1, 3> &direction)
      : Function<3>(3)
      , direction_(direction)
    {}

    double
    value(const Point<3>    &point,
          const unsigned int component = 0) const override
    {
      return direction_[component] *
             (1. + point[1] * point[1] + point[2] * point[2]);
    }

  private:
    const Tensor<1, 3> direction_;
  } function(direction);

  Vector bulk_values;
  bulk_values.reinit(bulk.locally_owned_dofs(), MPI_COMM_WORLD);
  VectorTools::interpolate(bulk_dof_handler, function, bulk_values);

  StateView<Vector> state_view(layout, 0.);
  state_view.bind(bulk.field_id(), bulk_values);
  const EvaluationContext<Vector> context(0., state_view);
  const auto                      matrix =
    model.state_matrix_operator(line.field_id(), bulk.field_id(), context);
  ASSERT_TRUE(matrix.has_value());

  Vector line_ones;
  line_ones.reinit(line.locally_owned_dofs(), MPI_COMM_WORLD);
  line_ones = 1.;
  line_ones.compress(VectorOperation::insert);
  Vector coupled_values;
  coupled_values.reinit(line.locally_owned_dofs(), MPI_COMM_WORLD);
  matrix->view.vmult(coupled_values, bulk_values);
  const double coupled_integral = line_ones * coupled_values;

  // Independently integrate the exact bulk polynomial at the support's
  // physical points. The representative curve has zero transverse values.
  double      local_physical_integral       = 0.;
  double      local_representative_integral = 0.;
  const auto &support_points = support.locally_owned_quadrature_points();
  ASSERT_EQ(lift_points.size(), support_points.size());
  for (std::size_t q = 0; q < support_points.size(); ++q)
    {
      const auto &point        = support_points[q];
      const auto &lifted_point = lift_points[q];
      EXPECT_EQ(point.stable_id, lifted_point.stable_id);
      EXPECT_NEAR(point.weight, lifted_point.weight, 1.e-12);
      EXPECT_NEAR(direction * lifted_point.mode_vector, 1., 1.e-12);
      local_physical_integral += point.weight *
                                 (direction * lifted_point.mode_vector) *
                                 (1. + point.point[1] * point.point[1] +
                                  point.point[2] * point.point[2]);
      local_representative_integral +=
        point.weight * (direction * lifted_point.mode_vector) *
        (1. + point.representative_point[1] * point.representative_point[1] +
         point.representative_point[2] * point.representative_point[2]);
    }
  const double physical_integral =
    Utilities::MPI::sum(local_physical_integral, MPI_COMM_WORLD);
  const double representative_integral =
    Utilities::MPI::sum(local_representative_integral, MPI_COMM_WORLD);
  EXPECT_GT(physical_integral, representative_integral + 1.e-6);
  EXPECT_NEAR(coupled_integral, physical_integral, 1.e-10);
}

TEST(LiftedIntegrationSupport,
     BOTH_SelectedVectorModesPreserveAmplitudeAndRegistration)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  FiniteElementSpace<1, 3>      space(fixture.domain.triangulation(),
                                 std::make_unique<FE_Q<1, 3>>(1));
  const auto                    view = space.view();
  TensorProductLift<1, 2, 3, 3> geometry("/Selected vector modes/");
  geometry.thickness                     = "0.1";
  geometry.section.inclusion_degree      = 1;
  geometry.section.selected_coefficients = {1, 4, 7};
  const auto observable =
    lift(value(view.field("coefficient")), geometry).with_id(FieldId(42));
  ReferenceCrossSection<1, 3, 3> section(geometry.section);
  Tensor<1, 3>                   tangent;
  tangent[0]           = 1.;
  unsigned int nonunit = 0;
  for (unsigned int slot = 0; slot < 3; ++slot)
    {
      const auto selected = observable.mode(slot);
      EXPECT_EQ(selected.source().id(), FieldId(42));
      ASSERT_EQ(selected.dependencies().size(), 1u);
      EXPECT_EQ(selected.dependencies().front(), FieldId(42));
      for (const auto &point : selected.lifted_points())
        {
          const auto expected =
            section.get_transformed_mode_values(point.section_qpoint, tangent);
          ASSERT_EQ(point.mode_values.size(), 3u);
          for (unsigned int d = 0; d < 3; ++d)
            {
              EXPECT_NEAR(point.mode_vector[d], expected[3 * slot + d], 1.e-14);
              EXPECT_NEAR(point.mode_values[d], expected[3 * slot + d], 1.e-14);
            }
          if (std::abs(point.mode_vector.norm() - 1.) > 1.e-4)
            ++nonunit;
        }
    }
  EXPECT_GT(Utilities::MPI::sum(nonunit, MPI_COMM_WORLD), 0u);
  EXPECT_THROW(observable.mode(3), ExceptionBase);
}

TEST(LiftedIntegrationSupport,
     BOTH_RejectsChangedConstraintStructureCollectively)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  TensorProductLift<1, 2, 3, 3> geometry("/Constraint structure guard/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, -2., 2.);
  mesh.refine_global(1);
  FiniteElementSpace<3> bulk_space(mesh,
                                   std::make_unique<FESystem<3>>(FE_Q<3>(1),
                                                                 3));
  auto                 &constraints = bulk_space.constraints();
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  if (bulk_space.locally_relevant_dofs().is_element(0))
    constraints.add_line(0);
  constraints.close();
  const auto  bulk_view = bulk_space.view();
  const auto  line_view = line_space.view();
  StateLayout layout;
  const auto  bulk =
    bulk_view.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line = line_view.field(layout, "line");
  using V         = ImmersXLA::MPI::Vector;
  using M         = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(bulk), test(lift(value(line), geometry).mode(0)))
    .add(builder);
  V state, residual;
  detail::initialize_weak_vector(state,
                                 bulk.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  state = 1.;
  StateView<V> view(layout, 0.);
  view.bind(bulk.id(), state);
  EvaluationContext<V> context(0., view);
  EXPECT_NO_THROW(model.evaluate_row(line.id(), context, residual));
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  constraints.close();
  EXPECT_THROW(model.evaluate_row(line.id(), context, residual), ExceptionBase);
}

TEST(LiftedIntegrationSupport, BOTH_VectorLiftHangingNodesAndAffineSourceShift)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  TensorProductLift<1, 2, 3, 3> geometry("/Hanging affine vector lift/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, -1., 1.);
  mesh.refine_global(1);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  FiniteElementSpace<3> bulk_space(mesh,
                                   std::make_unique<FESystem<3>>(FE_Q<3>(1),
                                                                 3));
  auto                 &constraints = bulk_space.constraints();
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  DoFTools::make_hanging_node_constraints(bulk_space.dof_handler(),
                                          constraints);
  EXPECT_GT(Utilities::MPI::sum(constraints.n_constraints(), MPI_COMM_WORLD),
            0u);
  Functions::ConstantFunction<3> exact(std::vector<double>{0.1, -0.2, 0.3});
  VectorTools::interpolate_boundary_values(bulk_space.dof_handler(),
                                           0,
                                           exact,
                                           constraints);
  constraints.close();
  const auto  bulk_view = bulk_space.view();
  const auto  line_view = line_space.view();
  StateLayout layout;
  const auto  bulk =
    bulk_view.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line   = line_view.field(layout, "line");
  const auto lifted = lift(value(line), geometry).mode(0);
  using V           = ImmersXLA::MPI::Vector;
  using M           = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(bulk), test(lifted)).add(builder);
  V state, residual, expected;
  detail::initialize_weak_vector(state,
                                 bulk.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(expected,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  VectorTools::interpolate(bulk.dof_handler(), exact, state);
  constraints.distribute(state);
  StateView<V> view(layout, 0.);
  view.bind(bulk.id(), state);
  EvaluationContext<V> context(0., view);
  model.evaluate_row(line.id(), context, residual);
  // Integrate the exact physical constant, including contributions whose
  // source shape functions have been eliminated by inhomogeneous constraints.
  Tensor<1, 3> constant;
  constant[0] = 0.1;
  constant[1] = -0.2;
  constant[2] = 0.3;
  for (const auto &point : lifted.lifted_points())
    {
      dealii::Vector<double> local(point.dof_indices.size());
      for (unsigned int i = 0; i < local.size(); ++i)
        local[i] = point.weight * point.source_basis_values[i] *
                   (constant * point.mode_vector);
      line.constraints().distribute_local_to_global(local,
                                                    point.dof_indices,
                                                    expected);
    }
  expected.compress(VectorOperation::add);
  EXPECT_GT(expected.l2_norm(), 1.e-5);
  residual -= expected;
  EXPECT_LT(residual.l2_norm(), 1.e-12);
}

TEST(LiftedIntegrationSupport,
     BOTH_VectorLiftAutomaticAffineUpdatesAndStructureGuard)
{
  ParameterAcceptor::clear();
  LiftedLineFixture             fixture;
  TensorProductLift<1, 2, 3, 3> geometry("/Updated vector affine lift/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  parallel::distributed::Triangulation<3> mesh(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(mesh, -2., 2.);
  mesh.refine_global(1);
  FiniteElementSpace<3> bulk_space(mesh,
                                   std::make_unique<FESystem<3>>(FE_Q<3>(1),
                                                                 3));
  auto                 &constraints = bulk_space.constraints();
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  for (const auto i : bulk_space.locally_relevant_dofs())
    constraints.add_line(i);
  constraints.close();
  const auto  bulk_view = bulk_space.view();
  const auto  line_view = line_space.view();
  StateLayout layout;
  const auto  bulk =
    bulk_view.field(layout, "bulk", FEValuesExtractors::Vector(0));
  const auto line   = line_view.field(layout, "line");
  const auto lifted = lift(value(line), geometry).mode(0);
  using V           = ImmersXLA::MPI::Vector;
  using M           = ImmersXLA::MPI::SparseMatrix;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  weak_term(value(bulk), test(lifted)).add(builder);
  V state, residual, expected;
  detail::initialize_weak_vector(state,
                                 bulk.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(residual,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  detail::initialize_weak_vector(expected,
                                 line.locally_owned_dofs(),
                                 MPI_COMM_WORLD);
  state = 0.;
  StateView<V> view(layout, 0.);
  view.bind(bulk.id(), state);
  EvaluationContext<V> context(0., view);
  model.evaluate_row(line.id(), context, residual);
  EXPECT_DOUBLE_EQ(residual.l2_norm(), 0.);
  const auto matrix_operator =
    model.state_matrix_operator(line.id(), bulk.id(), context);
  ASSERT_TRUE(matrix_operator.has_value());
  const auto *matrix = matrix_operator->source_matrix();
  ASSERT_NE(matrix, nullptr);
  for (const double amplitude : {1., -0.5, 0., 2.})
    {
      expected = 0.;
      Tensor<1, 3> constant;
      constant[0] = 0.1 * amplitude;
      constant[1] = -0.2 * amplitude;
      constant[2] = 0.3 * amplitude;
      for (const auto &cell : bulk.dof_handler().active_cell_iterators())
        if (!cell->is_artificial())
          {
            std::vector<types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              constraints.set_inhomogeneity(
                indices[i],
                constant[cell->get_fe().system_to_component_index(i).first]);
          }
      EXPECT_NO_THROW(model.evaluate_row(line.id(), context, residual));
      for (const auto &point : lifted.lifted_points())
        {
          dealii::Vector<double> local(point.dof_indices.size());
          for (unsigned int i = 0; i < local.size(); ++i)
            local[i] = point.weight * point.source_basis_values[i] *
                       (constant * point.mode_vector);
          line.constraints().distribute_local_to_global(local,
                                                        point.dof_indices,
                                                        expected);
        }
      expected.compress(VectorOperation::add);
      if (amplitude != 0.)
        EXPECT_GT(expected.l2_norm(), 1.e-5);
      residual -= expected;
      EXPECT_LT(residual.l2_norm(), 1.e-12);
      const auto updated_operator =
        model.state_matrix_operator(line.id(), bulk.id(), context);
      ASSERT_TRUE(updated_operator.has_value());
      EXPECT_EQ(updated_operator->source_matrix(), matrix);
    }
  // Offsets may change, but an already condensed matrix cannot follow a new
  // set of constraint masters or constrained DoFs.
  constraints.clear();
  constraints.reinit(bulk_space.locally_owned_dofs(),
                     bulk_space.locally_relevant_dofs());
  constraints.close();
  EXPECT_THROW(model.evaluate_row(line.id(), context, residual), ExceptionBase);
}

namespace
{
  template <int dim, int spacedim>
  void
  check_volume_lift()
  {
    ParameterAcceptor::clear();
    DomainParameters<dim, spacedim> parameters("/Normal volume source/");
    parameters.arguments_for_grid = "-1: 1: false";
    Domain<dim, spacedim> domain(parameters);
    domain.make_grid();
    if constexpr (dim == 2)
      GridTools::transform(
        [](const Point<3> &p) {
          return Point<3>(p[0], p[1] / std::sqrt(2.), p[1] / std::sqrt(2.));
        },
        domain.triangulation());
    TensorProductLift<dim, spacedim, spacedim> lift("/Normal volume geometry/");
    lift.section.inclusion_type        = "hyper_cube";
    lift.section.inclusion_degree      = 0;
    lift.section.refinement_level      = 0;
    lift.section.selected_coefficients = {0};
    lift.section.n_q_points            = 2;
    lift.representative_n_q_points     = 2;
    lift.thickness                     = "0.25";
    const auto support                 = domain.lift(lift);
    double     measure                 = 0.;
    for (const auto &p : support.locally_owned_quadrature_points())
      {
        measure += p.weight;
        const auto offset = p.point - p.representative_point;
        EXPECT_NEAR(offset[0], 0., 1.e-12);
        if constexpr (dim == 2)
          {
            EXPECT_NEAR(offset[1] + offset[2], 0., 1.e-12);
          }
      }
    EXPECT_NEAR(Utilities::MPI::sum(measure, domain.communicator()),
                std::pow(2., dim) * std::pow(0.5, spacedim - dim),
                1.e-11);
    FiniteElementSpace<dim, spacedim> space(
      domain.triangulation(), std::make_unique<FE_Q<dim, spacedim>>(1));
    auto       view       = space.view();
    const auto observable = make_lift(value(view.field("scalar")), lift);
    double     observable_measure = 0.;
    for (const auto &p : observable.lifted_points())
      {
        observable_measure += p.weight;
        EXPECT_NEAR(std::accumulate(p.basis_values.begin(),
                                    p.basis_values.end(),
                                    0.),
                    1.,
                    1.e-12);
      }
    EXPECT_NEAR(Utilities::MPI::sum(observable_measure, domain.communicator()),
                std::pow(2., dim) * std::pow(0.5, spacedim - dim),
                1.e-11);
  }
} // namespace

TEST(LiftedIntegrationSupport, BOTH_NormalVolumeLifts)
{
  check_volume_lift<1, 2>();
  check_volume_lift<1, 3>();
  check_volume_lift<2, 3>();
}

#ifdef DEAL_II_WITH_SUNDIALS
TEST(LiftedIntegrationSupport, BOTH_ElastodynamicsMovingDirichlet)
{
  ParameterAcceptor::clear();
  LiftedLineFixture           fixture;
  ElastodynamicsParameters<3> parameters;
  ElastodynamicsSolver<3>     problem(parameters);
  initialize_parameters_from_string(R"(
    subsection Elastodynamics
      subsection Functions
        subsection Displacement boundary
          set Function expression = t^3; -2*t^3; 3*t^3
        end
        subsection Velocity boundary
          set Function expression = 3*t^2; -6*t^2; 9*t^2
        end
      end
    end
  )");
  parameters.domain_parameters.name_of_grid       = "hyper_cube";
  parameters.domain_parameters.arguments_for_grid = "-2: 2: false";
  parameters.domain_parameters.initial_refinement = 0;
  parameters.dirichlet_ids                        = {0};
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  TensorProductLift<1, 2, 3, 3> geometry("/Moving Dirichlet lift/");
  geometry.thickness                     = "0.1";
  geometry.section.selected_coefficients = {0};
  FiniteElementSpace<1, 3> line_space(fixture.domain.triangulation(),
                                      std::make_unique<FE_Q<1, 3>>(1));
  using V = ImmersXLA::MPI::Vector;
  using G = ImmersXLA::MPI::BlockVector;
  IDAAdapter<V, G> adapter(parameters.time_parameters,
                           parameters.ida_parameters,
                           MPI_COMM_WORLD);
  const auto       solid     = adapter.add(problem);
  const auto       line_view = line_space.view();
  const auto       line      = adapter.add([&](auto &builder) {
    const auto id = builder.algebraic_field("trace",
                                            line_space.locally_owned_dofs(),
                                            line_space.locally_relevant_dofs());
    return line_view.field(id, "trace");
  });
  const auto       lifted    = lift(value(line.fields()), geometry).mode(0);
  adapter.add([&](auto &builder) {
    weak_term(value(solid.fields().displacement), test(lifted)).add(builder);
    return line.fields();
  });
  auto state     = adapter.make_state();
  auto state_dot = adapter.make_state();
  auto residual  = adapter.make_state();
  V    expected;
  detail::initialize_weak_vector(expected,
                                 line_space.locally_owned_dofs(),
                                 MPI_COMM_WORLD);

  // Every bulk DoF is prescribed. The trace must recover the physical value
  // from the current boundary offsets, even with a zero execution vector.
  for (const double time : {0.2, 0.4, 0.1, 0., 0.2})
    {
      adapter.solver().setup_jacobian(time, state, state_dot, 1.);
      EXPECT_DOUBLE_EQ(parameters.displacement_boundary.get_time(), time);
      EXPECT_DOUBLE_EQ(parameters.velocity_boundary.get_time(), time);
      for (const auto index : problem.locally_owned_dofs())
        EXPECT_TRUE(problem.constraints().is_constrained(index));
      adapter.solver().residual(time, state, state_dot, residual);
      EXPECT_DOUBLE_EQ(problem.displacement().l2_norm(), 0.);
      EXPECT_DOUBLE_EQ(problem.velocity().l2_norm(), 0.);
      expected = 0.;
      Tensor<1, 3> exact;
      exact[0] = time * time * time;
      exact[1] = -2. * time * time * time;
      exact[2] = 3. * time * time * time;
      for (const auto &point : lifted.lifted_points())
        {
          dealii::Vector<double> local(point.dof_indices.size());
          for (unsigned int i = 0; i < local.size(); ++i)
            local[i] = point.weight * point.source_basis_values[i] *
                       (exact * point.mode_vector);
          line.fields().constraints().distribute_local_to_global(
            local, point.dof_indices, expected);
        }
      expected.compress(VectorOperation::add);
      if (time != 0.)
        EXPECT_GT(expected.l2_norm(), 1.e-6);
      V error;
      error.reinit(expected);
      error = adapter.field(residual, line.fields());
      error -= expected;
      EXPECT_LT(error.l2_norm(), 1.e-12);
    }
}
#endif

TEST(LiftedIntegrationSupport,
     NonlinearRepresentativeJacobianUsesExecutionIndices)
{
  ParameterAcceptor::clear();
  Triangulation<1, 3> mesh;
  GridGenerator::hyper_cube(mesh, -1., 1.);
  FESystem<1, 3>   source_fe(FE_DGQ<1, 3>(1), 2);
  FE_DGQ<1, 3>     target_fe(2);
  DoFHandler<1, 3> source_dofs(mesh), target_dofs(mesh);
  source_dofs.distribute_dofs(source_fe);
  target_dofs.distribute_dofs(target_fe);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto source_space =
    fe_space(source_dofs, StaticMappingQ1<1, 3>::mapping, constraints);
  const auto target_space =
    fe_space(target_dofs, StaticMappingQ1<1, 3>::mapping, constraints);
  IndexSet owned(3);
  owned.add_range(0, 3);
  using V = ImmersXLA::MPI::Vector;
  using M = ImmersXLA::MPI::SparseMatrix;
  StateLayout               layout;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  const auto                source_id = builder.field("u", owned, owned, owned);
  const auto target_id = builder.algebraic_field("lambda", owned, owned);
  std::vector<types::global_dof_index> mapping(
    source_dofs.n_dofs(), std::numeric_limits<types::global_dof_index>::max());
  std::vector<types::global_dof_index> native(source_fe.n_dofs_per_cell());
  source_dofs.begin_active()->get_dof_indices(native);
  unsigned int active = 0;
  for (unsigned int j = 0; j < native.size(); ++j)
    if (source_fe.system_to_component_index(j).first == 0)
      mapping[native[j]] = active++ == 0 ? 2 : 0;
  ASSERT_EQ(active, 2u);
  const auto u =
    source_space.field(source_id, "u").reindexed("u", owned, owned, mapping);
  const auto target = target_space.field(target_id, "lambda")
                        .reindexed("lambda", owned, owned, {2, 1, 0});
  TensorProductLift<1, 2, 3, 3> geometry("/Reindexed representative lift/");
  geometry.section.inclusion_degree      = 0;
  geometry.section.inclusion_type        = "hyper_cube";
  geometry.section.refinement_level      = 0;
  geometry.section.selected_coefficients = {0};
  geometry.section.n_q_points            = 2;
  geometry.representative_n_q_points     = 3;
  geometry.thickness                     = "0.25";
  weak_term(lift(value(u) * value(u), geometry), test(target)).add(builder);

  V state, direction, action, plus_residual, minus_residual;
  for (auto *vector :
       {&state, &direction, &action, &plus_residual, &minus_residual})
    vector->reinit(owned, MPI_COMM_SELF);
  state[0] = 2.5;
  state[1] = 99.; // An extra unknown that is not represented by this FE field.
  state[2] = 1.5;
  state.compress(VectorOperation::insert);
  StateView<V> view(layout, 0.);
  view.bind(source_id, state);
  const EvaluationContext<V> context(0., view);
  const auto                 jacobian =
    model.state_matrix_operator(target_id, source_id, context);
  ASSERT_TRUE(jacobian.has_value());
  direction    = 0.;
  direction[1] = 1.;
  direction.compress(VectorOperation::insert);
  jacobian->view.vmult(action, direction);
  EXPECT_LT(action.l2_norm(), 1.e-14);

  direction[0] = .3;
  direction[1] = 0.;
  direction[2] = -.2;
  direction.compress(VectorOperation::insert);
  jacobian->view.vmult(action, direction);
  EXPECT_GT(action.l2_norm(), 1.e-8);
  constexpr double epsilon = 1.e-6;
  auto             plus    = state;
  auto             minus   = state;
  plus.add(epsilon, direction);
  minus.add(-epsilon, direction);
  StateView<V> plus_view(layout, 0.), minus_view(layout, 0.);
  plus_view.bind(source_id, plus);
  minus_view.bind(source_id, minus);
  model.evaluate_row(target_id,
                     EvaluationContext<V>(0., plus_view),
                     plus_residual);
  model.evaluate_row(target_id,
                     EvaluationContext<V>(0., minus_view),
                     minus_residual);
  plus_residual -= minus_residual;
  plus_residual *= .5 / epsilon;
  plus_residual -= action;
  EXPECT_LT(plus_residual.l2_norm(), 1.e-7 * action.l2_norm() + 1.e-12);
}


TEST(LiftedIntegrationSupport, BOTH_NonlinearHpFieldsUseActiveCellElements)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> parameters("/hp line domain/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "-1:1:false";
  parameters.initial_refinement = 2;
  Domain<1, 3> domain(parameters);
  domain.make_grid();
  hp::FECollection<1, 3> elements;
  elements.push_back(FESystem<1, 3>(FE_DGQ<1, 3>(1), 1, FE_Nothing<1, 3>(), 1));
  elements.push_back(FESystem<1, 3>(FE_DGQ<1, 3>(1), 1, FE_DGQ<1, 3>(0), 1));
  DoFHandler<1, 3> dofs(domain.triangulation());
  for (const auto &cell : dofs.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(cell->center()[0] > 0. ? 1 : 0);
  dofs.distribute_dofs(elements);
  AffineConstraints<double> constraints;
  constraints.close();
  const auto space =
    fe_space(dofs, StaticMappingQ1<1, 3>::mapping, constraints);
  using V = ImmersXLA::MPI::Vector;
  using M = ImmersXLA::MPI::SparseMatrix;
  StateLayout               layout;
  SemiDiscreteModel<V, M>   model;
  SemidiscreteBuilder<V, M> builder(layout, model);
  const auto                id         = builder.field("hp state",
                                space.locally_owned_dofs(),
                                space.locally_relevant_dofs(),
                                space.locally_owned_dofs());
  const auto                u          = space.field(id, "u");
  const auto                expression = value(u) * value(u);
  const auto                target     = test(u);
  weak_term(expression, target).add(builder);
  TensorProductLift<1, 2, 3> geometry("/hp line lift/");
  geometry.section.inclusion_degree      = 0;
  geometry.section.inclusion_type        = "hyper_cube";
  geometry.section.refinement_level      = 0;
  geometry.section.selected_coefficients = {0};
  geometry.section.n_q_points            = 2;
  geometry.representative_n_q_points     = 3;
  geometry.thickness                     = "0.25";
  const auto lifted                      = lift(expression, geometry);
  const auto lifted_id                   = builder.algebraic_field("lifted row",
                                                 space.locally_owned_dofs(),
                                                 space.locally_relevant_dofs());
  weak_term(lifted, test(u.with_id(lifted_id))).add(builder);

  const auto component = DoFTools::extract_dofs(
    dofs, elements.component_mask(FEValuesExtractors::Scalar(0)));
  V owned, state, direction, residual, action, expected, lifted_expected;
  for (auto *v :
       {&owned, &direction, &residual, &action, &expected, &lifted_expected})
    v->reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
  owned = 9000.; // The auxiliary component must not enter the scalar square.
  for (const auto i : space.locally_owned_dofs())
    if (component.is_element(i))
      owned[i] = 2.;
  owned.compress(VectorOperation::insert);
  state.reinit(space.locally_owned_dofs(),
               space.locally_relevant_dofs(),
               MPI_COMM_WORLD);
  state = owned;
  state.update_ghost_values();
  StateView<V> view(layout, 0.);
  view.bind(id, state);
  const EvaluationContext<V> context(0., view);
  expected = lifted_expected = 0.;
  for (const auto &cell : dofs.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto                          &fe = cell->get_fe();
        std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(indices);
        for (unsigned int j = 0; j < indices.size(); ++j)
          if (fe.system_to_component_index(j).first == 0)
            expected[indices[j]] = 4. * cell->measure() / 2.;
        for (const auto &point : lifted.lifted_points())
          if (point.source_cell_id == cell->id())
            {
              const auto unit = space.mapping().transform_real_to_unit_cell(
                cell, point.representative_point);
              for (unsigned int j = 0; j < indices.size(); ++j)
                lifted_expected[indices[j]] +=
                  4. * fe.shape_value_component(j, unit, 0) * point.weight;
            }
      }
  expected.compress(VectorOperation::insert);
  lifted_expected.compress(VectorOperation::add);
  for (const auto row : {id, lifted_id})
    {
      const auto &oracle = row == id ? expected : lifted_expected;
      model.evaluate_row(row, context, residual);
      residual -= oracle;
      EXPECT_LT(residual.l2_norm(), 1.e-12);
      direction = 0.;
      for (const auto i : space.locally_owned_dofs())
        if (component.is_element(i))
          direction[i] = 1.;
      direction.compress(VectorOperation::insert);
      const auto jacobian = model.state_matrix_operator(row, id, context);
      ASSERT_TRUE(jacobian.has_value());
      jacobian->view.vmult(action, direction);
      action -= oracle; // d(u^2)/du = u^2 = 4 at u = 2.
      EXPECT_LT(action.l2_norm(), 1.e-12);
      direction = 1.;
      for (const auto i : space.locally_owned_dofs())
        if (component.is_element(i))
          direction[i] = 0.;
      direction.compress(VectorOperation::insert);
      jacobian->view.vmult(action, direction);
      EXPECT_LT(action.l2_norm(), 1.e-14);
    }
}
