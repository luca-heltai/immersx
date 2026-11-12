// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#include <deal.II/base/config.h>

#include <deal.II/base/function_parser.h>

#include <gtest/gtest.h>
#include <immersx/config.h>

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)
#  include <immersx/algebra/mumps_preconditioner.h>
#  include <immersx/core/overlap_integration_support.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/grid/lift_co_refinement.h>
#  include <immersx/io/lifted_particle_output.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastic_static_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_wall.h>

#  include "test_paths.h"

namespace
{
  using namespace dealii;
  using namespace ImmersX;
  using Flow         = MetricFlowX::BloodFlowSystem<1, 3>;
  using FieldVector  = MetricFlowX::VectorType;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  template <typename Check>
  void
  with_coupled_problem(Check      check,
                       const bool pressure_feedback = true,
                       const bool co_refine         = false)
  {
    ParameterAcceptor::clear();
    Flow                   flow(MPI_COMM_WORLD);
    TimeIntervalParameters time("/", false);
    IDAParameters ida("/MetricFlowSystem<1, 3>/IDA parameters/", &time);
    ElasticStaticParameters<3>    parameters("/Elastic static/");
    TensorProductLift<1, 2, 3, 3> wall_lift("/MetricFlowX vessel wall lift/");
    FiniteElementSpaceParameters<1, 3> multiplier_parameters(
      "/MetricFlowX multiplier space/", "auto");
    ParameterAcceptor coupling("/MetricFlowX coupling/");
    double            pressure_tolerance = 0.;
    coupling.add_parameter("Wall pressure absolute tolerance",
                           pressure_tolerance);
    LiftedParticleOutputParameters particle_output(
      "/MetricFlowX coupling/Particle output/");
    const auto input = TestPaths::parameter_path(
      "gtests/parameters/metric_flow_x_elastostatic.prm");
    initialize_parameters(input);
    flow.initialize_params(
      input,
      TestPaths::output_path(
        "metric-flow-x-elastostatic/last_used_parameters.prm")
        .string());
    if (!pressure_feedback)
      {
        // Extra algebraic fields change IDA's weighted error norm. Tight,
        // matching controls resolve the independent adaptive trajectories.
        ida.absolute_tolerance = 1.e-11;
        ida.relative_tolerance = 1.e-9;
      }
    if (co_refine)
      {
        time.final_time                                 = 0.0002;
        time.output_time_interval                       = 0.0001;
        pressure_tolerance                              = 1.e-7;
        parameters.domain_parameters.triangulation_type = "fullydistributed";
      }
    ElasticStaticProblem<3> solid(parameters);
    solid.use_fixed_stiffness();
    DomainParameters<1, 3> flow_domain_parameters("/Co-refinement test flow/");
    flow_domain_parameters.name_of_grid =
      TestPaths::data_filename("metric_flow_x/single_vessel_centered.vtk");
    flow_domain_parameters.read_vtk_fields = true;
    Domain<1, 3> flow_domain(flow_domain_parameters, flow.triangulation());
    if (co_refine)
      {
        RefinementParameters<1> refinement("/Co-refinement test/");
        refinement.refinement_strategy  = "both";
        refinement.lower_factor         = 0.000001;
        refinement.upper_factor         = 0.001;
        refinement.max_refinement_level = 1;
        flow.setup([&](const auto &coarse, const unsigned int level) {
          flow_domain_parameters.initial_refinement = level;
          flow_domain.prepare_grid(coarse);
          solid.make_grid([&](auto &solid_domain) {
            solid_domain.prepare_grid();
            auto &mesh = solid_domain.refinement_triangulation();
            mesh.set_mesh_smoothing(
              Triangulation<3>::limit_level_difference_at_vertices);
            mesh.begin_active()->set_refine_flag();
            mesh.execute_coarsening_and_refinement();
            const auto report = co_refine_lifted_domains(
              solid_domain,
              flow_domain,
              wall_lift,
              refinement,
              [&](const CellId &cell, const Point<1> &unit) {
                const double area =
                  flow_domain.prepared_field_value("a0", cell, unit);
                EXPECT_NEAR(area, 3.0605e-4, 1.e-14);
                return std::sqrt(area / numbers::PI);
              },
              2);
            EXPECT_GT(report.fluid_cells, report.initial_fluid_cells);
            EXPECT_EQ(report.section_level, 2u);
          });
        });
        solid.setup_fe();
        EXPECT_EQ(Utilities::MPI::max(
                    solid.triangulation().has_hanging_nodes() ? 1u : 0u,
                    MPI_COMM_WORLD),
                  1u);
        EXPECT_GT(solid.constraints().n_constraints(), 0u);
      }
    else
      {
        flow.setup([&](const auto &coarse, const unsigned int level) {
          flow_domain_parameters.initial_refinement = level;
          flow_domain.prepare_grid(coarse);
          flow_domain.finalize_grid();
        });
        solid.setup();
      }
    multiplier_parameters.finite_element =
      "FE_DGQ(" + std::to_string(flow.finite_element().base_element(0).degree) +
      ")";
    FiniteElementSpace<1, 3> multiplier_space(flow.triangulation(),
                                              multiplier_parameters);
    const auto               lambda_view = multiplier_space.view();
    const auto               lambda      = lambda_view.field("lambda");
    const auto               a0          = flow_domain.field("a0");
    const auto               reference   = a0.space();
    FieldVector              reference_area;
    reference_area.reinit(reference.locally_owned_dofs(),
                          reference.locally_relevant_dofs(),
                          MPI_COMM_WORLD);
    reference_area = a0.coefficients();
    reference_area.update_ghost_values();
    if (co_refine)
      {
        FEValues<1, 3>      values(reference.mapping(),
                              reference.finite_element(),
                              QGauss<1>(2),
                              update_values);
        std::vector<double> areas(2);
        for (const auto &cell : reference.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              values.reinit(cell);
              values[a0.field().extractor()].get_function_values(reference_area,
                                                                 areas);
              for (const auto area : areas)
                {
                  EXPECT_NEAR(area, 3.0605e-4, 1.e-14);
                }
            }
      }
    const auto radius = sqrt(frozen(a0.field(), reference_area) / numbers::PI);
    const auto reference_wall = lift(value(lambda), wall_lift, radius);
    GridTools::Cache<3> solid_cache(solid.triangulation(), solid.mapping());
    const auto          supports =
      make_overlap_supports(IntegrationSupport<1, 3>(flow.triangulation()),
                            lambda.mapping(),
                            reference_wall.representative_quadrature(),
                            reference_wall.lifted_points(),
                            solid_cache);
    const auto [projection, metric] =
      make_metric_flow_x_pressure_projection(lambda, supports.overlap);
    Adapter adapter(time, ida, MPI_COMM_WORLD);
    adapter.set_stop_at_output_times();
    adapter.set_schur_primal_coupling();
    const auto solid_fields =
      adapter.add(elastic_static_semidiscrete(solid), "solid");
    auto flow_description = metric_flow_x(flow);
#  ifdef DEAL_II_WITH_MUMPS
    flow_description.preconditioner_factory = [](const auto &matrix,
                                                 const auto &prototype) {
      return make_mumps_preconditioner<FieldVector>(matrix,
                                                    prototype,
                                                    MPI_COMM_WORLD);
    };
#  endif
    const auto flow_fields = adapter.add(flow_description, "flow");
    const auto wall        = adapter.add(
      metric_flow_x_wall_constraint(flow,
                                    solid_fields.fields().displacement,
                                    flow_fields.fields(),
                                    lambda,
                                    wall_lift,
                                    radius,
                                    supports,
                                    projection,
                                    metric,
                                    solid_fields.fields().displacement,
                                           {solid_fields.fields().displacement.id(),
                                            flow_fields.fields().state},
                                    pressure_feedback),
      "wall");
    if (pressure_tolerance > 0.)
      adapter.set_absolute_tolerance(wall.fields().multiplier,
                                     pressure_tolerance);
    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    flow.initialize_state(adapter.field(state, flow_fields.fields().state),
                          time.initial_time);
    flow.initialize_state_derivative(adapter.field(state_dot,
                                                   flow_fields.fields().state),
                                     time.initial_time);
    check(flow,
          solid,
          adapter,
          solid_fields,
          flow_fields,
          wall,
          state,
          state_dot,
          time,
          ida);
  }
  void
  check_transient(const bool pressure_feedback, const bool co_refine = false)
  {
    with_coupled_problem(
      [pressure_feedback](auto       &flow,
                          auto       &solid,
                          auto       &adapter,
                          const auto &solid_fields,
                          const auto &flow_fields,
                          const auto &wall,
                          auto       &state,
                          auto       &state_dot,
                          const auto &time,
                          const auto &ida) {
        auto initial_flow = adapter.field(state, flow_fields.fields().state);
        unsigned int observations = 0;
        double       last_time    = time.initial_time;
        adapter.set_output_step([&](const double        t,
                                    const GlobalVector &y,
                                    const GlobalVector &ydot,
                                    const unsigned int) {
          auto residual = adapter.make_state();
          adapter.solver().residual(t, y, ydot, residual);
          EXPECT_LT(adapter.field(residual, solid_fields.fields().displacement)
                      .l2_norm(),
                    1.e-7);
          EXPECT_LT(adapter.field(residual, wall.fields().multiplier).l2_norm(),
                    1.e-11);
          if (!pressure_feedback)
            {
              auto native_residual = flow.make_state();
              flow.assemble_residual(
                t,
                adapter.field(y, flow_fields.fields().state),
                adapter.field(ydot, flow_fields.fields().state),
                native_residual);
              native_residual -=
                adapter.field(residual, flow_fields.fields().state);
              EXPECT_LT(native_residual.l2_norm(), 1.e-13);
            }
          if (t == time.initial_time)
            {
              // The zero solid guess cannot balance the nonzero initial load.
              // IDA must correct it while retaining differential flow data.
              EXPECT_GT(
                adapter.field(y, solid_fields.fields().displacement).l2_norm(),
                1.e-13);
              EXPECT_GT(adapter.field(y, wall.fields().multiplier).l2_norm(),
                        1.e-8);
              const auto &corrected_flow =
                adapter.field(y, flow_fields.fields().state);
              for (const auto index : flow.differential_dofs())
                {
                  EXPECT_NEAR(corrected_flow[index],
                              initial_flow[index],
                              1.e-14);
                }
            }
          last_time = t;
          ++observations;
        });
        auto constraints = std::make_shared<GlobalVector>(adapter.make_state());
        auto &flow_constraints =
          adapter.field(*constraints, flow_fields.fields().state);
        for (const auto component :
             {Flow::Component::area, Flow::Component::area_trace})
          for (const auto index : flow.component_dofs(component))
            flow_constraints[index] = 2.;
        constraints->compress(VectorOperation::insert);
        adapter.solver().get_constraint_vector =
          [constraints]() -> GlobalVector & { return *constraints; };
        const auto steps = adapter.solve(state, state_dot);
        EXPECT_GT(steps, 0u);
        EXPECT_EQ(solid.stiffness_assemblies(), 1u);
        EXPECT_GE(observations, 3u);
        EXPECT_NEAR(last_time, time.final_time, 1.e-12);
        auto changed_flow = adapter.field(state, flow_fields.fields().state);
        changed_flow -= initial_flow;
        EXPECT_GT(changed_flow.l2_norm(), 1.e-9);
        EXPECT_GT(
          adapter.field(state, solid_fields.fields().displacement).l2_norm(),
          1.e-13);
        EXPECT_GT(adapter.field(state, wall.fields().multiplier).l2_norm(),
                  1.e-8);
        for (const auto index : flow.component_dofs(Flow::Component::area))
          if (flow.locally_owned_dofs().is_element(index))
            {
              EXPECT_GT(adapter.field(state, flow_fields.fields().state)[index],
                        0.);
            }
        if (!pressure_feedback)
          {
            // Independent native-flow execution can choose a different adaptive
            // grid.
            Adapter native_adapter(time, ida, MPI_COMM_WORLD);
            native_adapter.set_stop_at_output_times();
            auto native_description = metric_flow_x(flow);
#  ifdef DEAL_II_WITH_MUMPS
            native_description.preconditioner_factory =
              [](const auto &matrix, const auto &prototype) {
                return make_mumps_preconditioner<FieldVector>(matrix,
                                                              prototype,
                                                              MPI_COMM_WORLD);
              };
#  endif
            const auto native_fields =
              native_adapter.add(native_description, "native-flow");
            auto native_state = native_adapter.make_state();
            auto native_dot   = native_adapter.make_state();
            flow.initialize_state(
              native_adapter.field(native_state, native_fields.fields().state),
              time.initial_time);
            flow.initialize_state_derivative(
              native_adapter.field(native_dot, native_fields.fields().state),
              time.initial_time);
            const auto native_steps =
              native_adapter.solve(native_state, native_dot);
            EXPECT_GT(native_steps, 0u);
            auto difference = adapter.field(state, flow_fields.fields().state);
            difference -=
              native_adapter.field(native_state, native_fields.fields().state);
            EXPECT_LT(difference.l2_norm(),
                      2.e-4 * changed_flow.l2_norm() + 2.e-7);
          }
      },
      pressure_feedback,
      co_refine);
  }
} // namespace

TEST(MetricFlowXElastostatic, BOTH_AlgebraicSolidTractionAndCoupledJacobian)
{
#  ifndef DEAL_II_WITH_VTK
  GTEST_SKIP() << "VTK input requires deal.II with VTK";
#  else
  with_coupled_problem([](auto       &flow,
                          auto       &solid,
                          auto       &adapter,
                          const auto &solid_fields,
                          const auto &flow_fields,
                          const auto &wall,
                          auto       &state,
                          auto       &state_dot,
                          const auto &,
                          const auto &) {
    const auto u      = solid_fields.fields().displacement;
    const auto lambda = wall.fields().multiplier;
    ASSERT_EQ(state.n_blocks(), 3u);
    IndexSet expected(state.size());
    for (const auto index : flow.differential_dofs())
      expected.add_index(solid.locally_owned_dofs().size() + index);
    expected.compress();
    EXPECT_EQ(adapter.differential_components(), expected);

    auto baseline = adapter.make_state();
    auto forced   = adapter.make_state();
    adapter.solver().residual(0., state, state_dot, baseline);
    EXPECT_GT(adapter.field(baseline, u).l2_norm(), 1.e-10);
    EXPECT_LT(adapter.field(baseline, lambda).l2_norm(), 1.e-14);
    EXPECT_LT(adapter.field(baseline, flow_fields.fields().state).l2_norm(),
              1.e-10);
    adapter.field(state, lambda) = 0.5;
    adapter.solver().residual(0., state, state_dot, forced);
    forced -= baseline;
    EXPECT_GT(adapter.field(forced, u).l2_norm(), 1.e-10);
    EXPECT_LT(adapter.field(forced, lambda).l2_norm(), 1.e-14);
    // A positive traction does positive work against an outward radial test.
    FunctionParser<3> radial_test(3);
    radial_test.initialize("x,y,z", "0;y;z", {});
    FieldVector radial;
    radial.reinit(solid.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(solid.mapping(),
                             solid.dof_handler(),
                             radial_test,
                             radial);
    solid.constraints().set_zero(radial);
    EXPECT_GT(radial * adapter.field(forced, u), 1.e-15);
    // The native constant-pressure oracle fixes the sign independently of
    // the wall helper and its observable pressure provider.
    auto native_pressure = flow.make_state();
    native_pressure      = 0.;
    flow.add_external_pressure_residual(
      0.,
      adapter.field(state, flow_fields.fields().state),
      [](const auto &) { return -0.5; },
      native_pressure);
    auto pressure_difference =
      adapter.field(forced, flow_fields.fields().state);
    pressure_difference -= native_pressure;
    EXPECT_LT(pressure_difference.l2_norm(),
              1.e-10 * native_pressure.l2_norm() + 1.e-13);
    for (const auto index : flow.component_dofs(Flow::Component::area))
      if (flow.locally_owned_dofs().is_element(index))
        {
          EXPECT_NEAR(adapter.field(forced, flow_fields.fields().state)[index],
                      0.,
                      1.e-14);
        }

    auto  direction      = adapter.make_state();
    auto &flow_direction = adapter.field(direction, flow_fields.fields().state);
    auto &flow_state     = adapter.field(state, flow_fields.fields().state);
    for (const auto index : flow.component_dofs(Flow::Component::area))
      if (flow.locally_owned_dofs().is_element(index))
        {
          flow_state[index] *= 1.1;
          flow_direction[index] = 0.1 * flow_state[index];
        }
    for (const auto component :
         {Flow::Component::velocity, Flow::Component::velocity_trace})
      for (const auto index : flow.component_dofs(component))
        if (flow.locally_owned_dofs().is_element(index))
          {
            flow_state[index]     = 0.02 * (1 + index % 3);
            flow_direction[index] = 0.01 * (1 + index % 2);
          }
    flow_state.compress(VectorOperation::insert);
    flow_direction.compress(VectorOperation::insert);
    for (const auto index : u.locally_owned_dofs())
      if (!u.constraints().is_constrained(index))
        adapter.field(direction, u)[index] = 0.001;
    adapter.field(direction, u).compress(VectorOperation::insert);
    adapter.field(direction, lambda) = 0.2;
    constexpr double alpha           = 10.;
    adapter.solver().setup_jacobian(0.001, state, state_dot, alpha);
    auto action = adapter.make_state();
    adapter.current_jacobian().vmult(action, direction);
    constexpr double epsilon   = 1.e-5;
    auto             plus      = state;
    auto             minus     = state;
    auto             dot_plus  = state_dot;
    auto             dot_minus = state_dot;
    plus.add(epsilon, direction);
    minus.add(-epsilon, direction);
    dot_plus.add(alpha * epsilon, direction);
    dot_minus.add(-alpha * epsilon, direction);
    adapter.solver().residual(0.001, plus, dot_plus, forced);
    adapter.solver().residual(0.001, minus, dot_minus, baseline);
    forced -= baseline;
    forced *= 0.5 / epsilon;
    forced -= action;
    for (unsigned int block = 0; block < forced.n_blocks(); ++block)
      EXPECT_LT(forced.block(block).l2_norm(),
                1.e-6 * action.block(block).l2_norm() + 1.e-13)
        << "block " << block;

    // No solid or multiplier rate may enter any equation.
    auto rate_only                   = adapter.make_state();
    adapter.field(rate_only, u)      = 1.;
    adapter.field(rate_only, lambda) = 1.;
    auto unchanged                   = adapter.make_state();
    adapter.solver().residual(0.001, state, state_dot, baseline);
    state_dot += rate_only;
    adapter.solver().residual(0.001, state, state_dot, unchanged);
    unchanged -= baseline;
    EXPECT_LT(unchanged.l2_norm(), 1.e-14);
  });
#  endif
}

TEST(MetricFlowXElastostatic,
     BOTH_OneWayTransientMatchesNativeFlowAndPreservesWallConstraint)
{
#  ifndef DEAL_II_WITH_VTK
  GTEST_SKIP() << "VTK input requires deal.II with VTK";
#  else
  check_transient(false);
#  endif
}

TEST(MetricFlowXElastostatic,
     BOTH_TwoWayTransientPreservesEquilibriumAndWallConstraint)
{
#  ifndef DEAL_II_WITH_VTK
  GTEST_SKIP() << "VTK input requires deal.II with VTK";
#  else
  check_transient(true);
#  endif
}

TEST(MetricFlowXElastostatic,
     BOTH_CoRefinedTransientPreservesEquilibriumAndWallConstraint)
{
#  ifndef DEAL_II_WITH_VTK
  GTEST_SKIP() << "VTK input requires deal.II with VTK";
#  else
  check_transient(true, true);
#  endif
}

TEST(MetricFlowXElastostatic,
     BOTH_MumpsInverseRedistributesNoncontiguousOwnership)
{
#  ifndef DEAL_II_WITH_MUMPS
  GTEST_SKIP() << "MUMPS block inverse requires deal.II with MUMPS";
#  else
  constexpr unsigned int size = 8;
  const auto ranks            = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ASSERT_LE(ranks, size);
  IndexSet owned(size), relevant(size);
  relevant.add_range(0, size);
  for (unsigned int i = rank; i < size; i += ranks)
    owned.add_index(i);
  owned.compress();
  DynamicSparsityPattern pattern(relevant);
  for (const auto i : owned)
    {
      pattern.add(i, i);
      if (i > 0)
        pattern.add(i, i - 1);
      if (i + 1 < size)
        pattern.add(i, i + 1);
    }
  SparsityTools::distribute_sparsity_pattern(pattern,
                                             owned,
                                             MPI_COMM_WORLD,
                                             relevant);
  ImmersXLA::MPI::SparseMatrix matrix;
  matrix.reinit(owned, owned, pattern, MPI_COMM_WORLD);
  for (const auto i : owned)
    {
      matrix.set(i, i, 4.);
      if (i > 0)
        matrix.set(i, i - 1, -0.5);
      if (i + 1 < size)
        matrix.set(i, i + 1, 1.);
    }
  matrix.compress(VectorOperation::insert);
  const std::function<void(FieldVector &, bool)> prototype =
    [owned](auto &v, const bool) { v.reinit(owned, MPI_COMM_WORLD); };
  auto inverse =
    make_mumps_preconditioner<FieldVector>(matrix, prototype, MPI_COMM_WORLD);
  FieldVector exact, rhs, result, error;
  for (auto *v : {&exact, &rhs, &result, &error})
    v->reinit(owned, MPI_COMM_WORLD);
  for (const auto i : owned)
    exact[i] = 1. + 0.1 * i;
  exact.compress(VectorOperation::insert);
  matrix.vmult(rhs, exact);
  inverse.vmult(result, rhs);
  error = result;
  error -= exact;
  EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
  inverse.vmult_add(result, rhs);
  error = result;
  error.add(-2., exact);
  EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
  matrix.Tvmult(rhs, exact);
  inverse.Tvmult(result, rhs);
  error = result;
  error -= exact;
  EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
  inverse.Tvmult_add(result, rhs);
  error = result;
  error.add(-2., exact);
  EXPECT_LT(error.l2_norm(), 1.e-12 * exact.l2_norm());
#  endif
}
#endif
