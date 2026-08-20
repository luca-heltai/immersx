// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/solver_gmres.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "linear_algebra.h"
#include "representation.h"
#include "semidiscrete_pde_models.h"
#include "sundials_ida_adapter.h"
#include "test_paths.h"
#include "utils.h"

using namespace dealii;

#ifdef DEAL_II_WITH_SUNDIALS

namespace
{
  using FieldVector  = LA::MPI::Vector;
  using GlobalVector = LA::MPI::BlockVector;

  template <typename GlobalVectorType>
  void
  solve_action(const ImmersX::JacobianAction<GlobalVectorType> &action,
               const GlobalVectorType                          &rhs,
               GlobalVectorType                                &dst,
               const double                                     tolerance)
  {
    SolverControl control(500, std::max(1.e-12, tolerance), false);
    SolverFGMRES<GlobalVectorType> solver(control);
    dst = 0.;
    solver.solve(action.as_linear_operator(rhs),
                 dst,
                 rhs,
                 PreconditionIdentity());
  }

  IndexSet
  shifted_indices(const IndexSet   &indices,
                  const std::size_t offset,
                  const std::size_t global_size)
  {
    IndexSet shifted(global_size);
    for (const auto index : indices)
      shifted.add_index(offset + index);
    shifted.compress();
    return shifted;
  }

  void
  configure_elastodynamics(ElastodynamicsParameters<2> &parameters)
  {
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("distributed-ida/elastodynamics");
    parameters.output_frequency   = 0;
    parameters.initial_refinement = 0;
    parameters.final_time         = 0.02;
    parameters.time_step          = 0.01;
    parameters.number_of_steps    = 2;
    parameters.solver_control.set_reduction(1.e-10);
    parameters.solver_control.set_tolerance(1.e-12);
  }

  void
  configure_stokes(NavierStokesParameters<2> &parameters)
  {
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("distributed-ida/stokes");
    parameters.output_frequency        = 0;
    parameters.initial_refinement      = 0;
    parameters.include_convective_term = false;
    parameters.final_time              = 0.02;
    parameters.number_of_time_steps    = 2;
    parameters.inner_solver_max_steps  = 300;
    parameters.inner_solver_tolerance  = 1.e-10;
  }
} // namespace


TEST(DistributedIDA, MPI_BlockFieldLayoutBindsWithoutCopy) // NOLINT
{
  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int n    = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor pressure_descriptor;
  pressure_descriptor.name          = "fluid.pressure";
  pressure_descriptor.time_role     = ImmersX::TimeRole::algebraic;
  const auto               pressure = layout.add_field(pressure_descriptor);
  ImmersX::FieldDescriptor velocity_descriptor;
  velocity_descriptor.name = "fluid.velocity";
  const auto velocity      = layout.add_field(velocity_descriptor);

  const std::size_t pressure_size = 3 * n;
  const std::size_t velocity_size = 2 * n;
  IndexSet          pressure_owned(pressure_size);
  IndexSet          velocity_owned(velocity_size);
  pressure_owned.add_range(3 * rank, 3 * rank + 3);
  velocity_owned.add_range(2 * rank, 2 * rank + 2);
  pressure_owned.compress();
  velocity_owned.compress();

  GlobalVector global;
  global.reinit({pressure_owned, velocity_owned}, MPI_COMM_WORLD);
  global.block(0) = 4.;
  global.block(1) = 7.;

  ImmersX::BlockFieldLayout<FieldVector, GlobalVector> field_layout(layout);
  field_layout.add_field(velocity, 1);
  field_layout.add_field(pressure, 0);
  field_layout.set_block_distribution(0, pressure_size, pressure_owned);
  field_layout.set_block_distribution(1, velocity_size, velocity_owned);

  ImmersX::StateView<FieldVector> state(layout, 0.);
  field_layout.bind_state(state, global);
  EXPECT_EQ(&state.field(velocity, 0.), &global.block(1));
  EXPECT_EQ(&state.field(pressure, 0.), &global.block(0));

  GlobalVector residual;
  residual.reinit({pressure_owned, velocity_owned}, MPI_COMM_WORLD);
  residual = 0.;
  ImmersX::ResidualAccumulator<FieldVector> accumulator(layout);
  field_layout.bind_residual(accumulator, residual);
  accumulator.field(velocity) = 11.;
  accumulator.field(pressure) = 13.;
  EXPECT_DOUBLE_EQ(residual.block(1).l2_norm(),
                   std::sqrt(static_cast<double>(velocity_size)) * 11.);
  EXPECT_DOUBLE_EQ(residual.block(0).l2_norm(),
                   std::sqrt(static_cast<double>(pressure_size)) * 13.);

  const auto differential = field_layout.differential_components();
  EXPECT_EQ(differential.size(), pressure_size + velocity_size);
  EXPECT_EQ(differential.n_elements(), velocity_owned.n_elements());
  for (const auto index : velocity_owned)
    EXPECT_TRUE(differential.is_element(pressure_size + index));
}


TEST(DistributedIDA, MPI_ElastodynamicsBlockIDA) // NOLINT
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  configure_elastodynamics(parameters);
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();

  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();

  ImmersX::ElastodynamicsSemiDiscreteModel<2> semantic(problem);
  using FieldLayout = ImmersX::BlockFieldLayout<FieldVector, GlobalVector>;
  FieldLayout field_layout(semantic.layout());
  field_layout.add_field(semantic.velocity_field(), 1);
  field_layout.add_field(semantic.displacement_field(), 0);
  field_layout.set_block_distribution(0,
                                      problem.locally_owned_dofs().size(),
                                      problem.locally_owned_dofs());
  field_layout.set_block_distribution(1,
                                      problem.locally_owned_dofs().size(),
                                      problem.locally_owned_dofs());

  const auto n = problem.locally_owned_dofs().size();
  ImmersX::DifferentialAlgebraicMetadata metadata(semantic.layout(), 2 * n);
  metadata.add_field(semantic.displacement_field(),
                     shifted_indices(problem.locally_owned_dofs(), 0, 2 * n));
  metadata.add_field(semantic.velocity_field(),
                     shifted_indices(problem.locally_owned_dofs(), n, 2 * n));

  GlobalVector                state;
  GlobalVector                state_dot;
  const std::vector<IndexSet> partitions = {problem.locally_owned_dofs(),
                                            problem.locally_owned_dofs()};
  state.reinit(partitions, MPI_COMM_WORLD);
  state_dot.reinit(partitions, MPI_COMM_WORLD);
  state.block(0) = problem.displacement();
  state.block(1) = problem.velocity();
  state_dot      = 0.;

  // Compare the semantic residual with the native M/K/D operators for a
  // nonzero externally supplied state.  The model must not read the solver's
  // accepted state while evaluating this residual.
  GlobalVector evaluation_state;
  GlobalVector evaluation_derivative;
  evaluation_state.reinit(partitions, MPI_COMM_WORLD);
  evaluation_derivative.reinit(partitions, MPI_COMM_WORLD);
  evaluation_state.block(0)      = 0.25;
  evaluation_state.block(1)      = 0.5;
  evaluation_derivative.block(0) = 0.75;
  evaluation_derivative.block(1) = 1.25;

  GlobalVector semantic_residual;
  semantic_residual.reinit(partitions, MPI_COMM_WORLD);
  semantic_residual = 0.;
  ImmersX::StateView<FieldVector> evaluation_view(semantic.layout(), 0.);
  ImmersX::StateView<FieldVector> derivative_view(semantic.layout(), 0.);
  field_layout.bind_state(evaluation_view, evaluation_state);
  field_layout.bind_state(derivative_view, evaluation_derivative);
  const ImmersX::EvaluationContext<FieldVector> evaluation(0.,
                                                           evaluation_view,
                                                           &derivative_view);
  ImmersX::ResidualAccumulator<FieldVector>     residual_accumulator(
    semantic.layout());
  field_layout.bind_residual(residual_accumulator, semantic_residual);
  semantic.model().evaluate(evaluation, residual_accumulator);

  FieldVector expected_displacement;
  FieldVector expected_velocity;
  FieldVector product;
  FieldVector force;
  expected_displacement.reinit(problem.locally_owned_dofs());
  expected_velocity.reinit(problem.locally_owned_dofs());
  product.reinit(problem.locally_owned_dofs());
  problem.body_force_at_time(0., force);
  problem.mass_matrix().vmult(expected_displacement,
                              evaluation_derivative.block(0));
  problem.mass_matrix().vmult(product, evaluation_state.block(1));
  expected_displacement -= product;
  problem.mass_matrix().vmult(expected_velocity,
                              evaluation_derivative.block(1));
  problem.stiffness_matrix().vmult(product, evaluation_state.block(0));
  expected_velocity += product;
  problem.damping_matrix().vmult(product, evaluation_state.block(1));
  expected_velocity += product;
  expected_velocity -= force;
  for (const auto index : problem.locally_owned_dofs())
    {
      if (problem.constraints().is_constrained(index))
        expected_displacement(index) = 0.;
      if (problem.velocity_constraints().is_constrained(index))
        expected_velocity(index) = 0.;
    }
  GlobalVector residual_difference;
  residual_difference.reinit(partitions, MPI_COMM_WORLD);
  residual_difference.block(0) = semantic_residual.block(0);
  residual_difference.block(0) -= expected_displacement;
  residual_difference.block(1) = semantic_residual.block(1);
  residual_difference.block(1) -= expected_velocity;
  EXPECT_NEAR(residual_difference.l2_norm(), 0., 1.e-11);

  SundialsIDAResidualAdapter<FieldVector, GlobalVector> adapter(
    semantic.model(),
    field_layout,
    metadata,
    [partitions](GlobalVector &vector) {
      vector.reinit(partitions, MPI_COMM_WORLD);
    },
    solve_action<GlobalVector>);

  dealii::SUNDIALS::IDA<GlobalVector>::AdditionalData data;
  data.initial_time                  = 0.;
  data.final_time                    = 0.02;
  data.initial_step_size             = 0.005;
  data.output_period                 = 0.02;
  data.absolute_tolerance            = 1.e-9;
  data.relative_tolerance            = 1.e-9;
  data.maximum_non_linear_iterations = 20;
  data.ic_type    = dealii::SUNDIALS::IDA<GlobalVector>::AdditionalData::none;
  data.reset_type = dealii::SUNDIALS::IDA<GlobalVector>::AdditionalData::none;

  dealii::SUNDIALS::IDA<GlobalVector> ida(data, MPI_COMM_WORLD);
  adapter.connect(ida);
  ida.setup_jacobian(0., state, state_dot, 1.);

  GlobalVector increment;
  increment.reinit(partitions, MPI_COMM_WORLD);
  increment.block(0) = 1.;
  increment.block(1) = 2.;
  GlobalVector jacobian_increment;
  jacobian_increment.reinit(partitions, MPI_COMM_WORLD);
  adapter.current_jacobian_action().vmult(jacobian_increment, increment);

  GlobalVector expected_jacobian;
  expected_jacobian.reinit(partitions, MPI_COMM_WORLD);
  problem.mass_matrix().vmult(expected_jacobian.block(0), increment.block(0));
  problem.mass_matrix().vmult(product, increment.block(1));
  expected_jacobian.block(0) -= product;
  problem.mass_matrix().vmult(expected_jacobian.block(1), increment.block(1));
  problem.stiffness_matrix().vmult(product, increment.block(0));
  expected_jacobian.block(1) += product;
  problem.damping_matrix().vmult(product, increment.block(1));
  expected_jacobian.block(1) += product;
  for (const auto index : problem.locally_owned_dofs())
    {
      if (problem.constraints().is_constrained(index))
        expected_jacobian.block(0)(index) = 0.;
      if (problem.velocity_constraints().is_constrained(index))
        expected_jacobian.block(1)(index) = 0.;
    }
  GlobalVector jacobian_difference = jacobian_increment;
  jacobian_difference -= expected_jacobian;
  EXPECT_NEAR(jacobian_difference.l2_norm(), 0., 1.e-11);

  const unsigned int steps = ida.solve_dae(state, state_dot);
  EXPECT_GT(steps, 0u);
  EXPECT_TRUE(problem.state_is_finite());
  EXPECT_TRUE(std::isfinite(state.l2_norm()));
  EXPECT_TRUE(std::isfinite(state_dot.l2_norm()));

  GlobalVector final_residual;
  final_residual.reinit(partitions, MPI_COMM_WORLD);
  ida.residual(data.final_time, state, state_dot, final_residual);
  EXPECT_LT(final_residual.l2_norm(), 1.e-6);

  for (unsigned int step = 0; step < 2; ++step)
    problem.advance_one_timestep();
  GlobalVector standalone_difference = state;
  standalone_difference.block(0) -= problem.displacement();
  standalone_difference.block(1) -= problem.velocity();
  EXPECT_LT(standalone_difference.l2_norm(), 1.e-6);
}


TEST(DistributedIDA, MPI_UnsteadyStokesBlockIDAAndJacobian) // NOLINT
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  configure_stokes(parameters);
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();

  NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_system();

  ImmersX::NavierStokesSemiDiscreteModel<2> semantic(problem);
  using FieldLayout = ImmersX::BlockFieldLayout<FieldVector, GlobalVector>;
  FieldLayout field_layout(semantic.layout());
  field_layout.add_field(semantic.pressure_field(), 1);
  field_layout.add_field(semantic.velocity_field(), 0);
  field_layout.set_block_distribution(
    0,
    problem.locally_owned_dofs_by_block()[0].size(),
    problem.locally_owned_dofs_by_block()[0]);
  field_layout.set_block_distribution(
    1,
    problem.locally_owned_dofs_by_block()[1].size(),
    problem.locally_owned_dofs_by_block()[1]);

  const auto n_velocity = problem.locally_owned_dofs_by_block()[0].size();
  const auto n_pressure = problem.locally_owned_dofs_by_block()[1].size();
  const auto n_total    = n_velocity + n_pressure;
  ImmersX::DifferentialAlgebraicMetadata metadata(semantic.layout(), n_total);
  metadata.add_field(semantic.velocity_field(),
                     shifted_indices(problem.locally_owned_dofs_by_block()[0],
                                     0,
                                     n_total));
  metadata.add_field(semantic.pressure_field(),
                     shifted_indices(problem.locally_owned_dofs_by_block()[1],
                                     n_velocity,
                                     n_total));

  const std::vector<IndexSet> partitions = {
    problem.locally_owned_dofs_by_block()[0],
    problem.locally_owned_dofs_by_block()[1]};
  GlobalVector state;
  GlobalVector state_dot;
  state.reinit(partitions, MPI_COMM_WORLD);
  state_dot.reinit(partitions, MPI_COMM_WORLD);
  state     = 0.;
  state_dot = 0.;

  SundialsIDAResidualAdapter<FieldVector, GlobalVector> adapter(
    semantic.model(),
    field_layout,
    metadata,
    [partitions](GlobalVector &vector) {
      vector.reinit(partitions, MPI_COMM_WORLD);
    },
    solve_action<GlobalVector>);

  EXPECT_EQ(metadata.differential_components().n_elements(),
            problem.locally_owned_dofs_by_block()[0].n_elements());
  for (const auto index : problem.locally_owned_dofs_by_block()[1])
    EXPECT_FALSE(
      metadata.differential_components().is_element(n_velocity + index));

  dealii::SUNDIALS::IDA<GlobalVector>::AdditionalData data;
  data.initial_time                  = 0.;
  data.final_time                    = 0.02;
  data.initial_step_size             = 0.005;
  data.output_period                 = 0.02;
  data.absolute_tolerance            = 1.e-9;
  data.relative_tolerance            = 1.e-9;
  data.maximum_non_linear_iterations = 20;
  data.ic_type    = dealii::SUNDIALS::IDA<GlobalVector>::AdditionalData::none;
  data.reset_type = dealii::SUNDIALS::IDA<GlobalVector>::AdditionalData::none;

  dealii::SUNDIALS::IDA<GlobalVector> ida(data, MPI_COMM_WORLD);
  adapter.connect(ida);
  ida.setup_jacobian(0., state, state_dot, 2.);

  GlobalVector increment;
  increment.reinit(partitions, MPI_COMM_WORLD);
  increment.block(0) = 1.;
  increment.block(1) = 2.;
  GlobalVector action_value;
  action_value.reinit(partitions, MPI_COMM_WORLD);
  adapter.current_jacobian_action().vmult(action_value, increment);

  GlobalVector expected;
  expected.reinit(partitions, MPI_COMM_WORLD);
  problem.continuous_operator().vmult(expected, increment);
  FieldVector velocity_mass_product;
  velocity_mass_product.reinit(increment.block(0));
  problem.velocity_mass_matrix().vmult(velocity_mass_product,
                                       increment.block(0));
  velocity_mass_product *= 2. * problem.density();
  expected.block(0) += velocity_mass_product;
  for (const auto index : problem.locally_owned_dofs_by_block()[0])
    if (problem.constraints().is_constrained(index))
      expected.block(0)(index) = 0.;
  for (const auto index : problem.locally_owned_dofs_by_block()[1])
    if (problem.constraints().is_constrained(n_velocity + index))
      expected.block(1)(index) = 0.;
  GlobalVector difference = action_value;
  difference -= expected;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);

  const unsigned int steps = ida.solve_dae(state, state_dot);
  EXPECT_GT(steps, 0u);
  EXPECT_TRUE(std::isfinite(state.block(0).l2_norm()));
  EXPECT_TRUE(std::isfinite(state.block(1).l2_norm()));

  GlobalVector final_residual;
  final_residual.reinit(partitions, MPI_COMM_WORLD);
  ida.residual(data.final_time, state, state_dot, final_residual);
  EXPECT_LT(final_residual.l2_norm(), 1.e-6);

  for (unsigned int step = 0; step < 2; ++step)
    problem.advance_one_timestep();
  GlobalVector standalone_difference = state;
  standalone_difference.block(0) -= problem.solution().block(0);
  standalone_difference.block(1) -= problem.solution().block(1);
  EXPECT_LT(standalone_difference.l2_norm(), 1.e-6);
}


TEST(DistributedIDA,
     MPI_NavierStokesVelocityRepresentationDependsOnlyOnVelocity)
{
  ParameterAcceptor::clear();
  NavierStokesParameters<2> parameters;
  configure_stokes(parameters);
  parameters.initial_refinement = 1;
  initialize_parameters();
  ParameterAcceptor::parse_all_parameters();

  NavierStokesSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor velocity_descriptor;
  velocity_descriptor.name          = "fluid.velocity";
  velocity_descriptor.time_role     = ImmersX::TimeRole::differential;
  velocity_descriptor.locally_owned = problem.locally_owned_dofs_by_block()[0];
  const auto velocity = layout.add_field(std::move(velocity_descriptor));
  ImmersX::FieldDescriptor pressure_descriptor;
  pressure_descriptor.name      = "fluid.pressure";
  pressure_descriptor.time_role = ImmersX::TimeRole::algebraic;
  const auto pressure = layout.add_field(std::move(pressure_descriptor));

  ImmersX::RepresentationMetadata metadata;
  metadata.dependencies = {velocity};
  using Representation  = VectorFiniteElementRepresentation<2>;
  Representation representation(problem.triangulation(),
                                problem.dof_handler(),
                                problem.locally_owned_dofs(),
                                problem.locally_relevant_dofs(),
                                problem.constraints(),
                                problem.mapping(),
                                problem.velocity_extractor(),
                                metadata);

  EXPECT_EQ(representation.dependencies().size(), 1u);
  EXPECT_EQ(representation.dependencies()[0], velocity);
  EXPECT_NE(velocity, pressure);

  const auto points =
    representation.locally_owned_quadrature_points(QGauss<2>(2));
  double velocity_norm = 0.;
  double pressure_norm = 0.;
  for (const auto &point : points)
    for (unsigned int i = 0; i < point.basis_values.size(); ++i)
      if (problem.finite_element().system_to_component_index(i).first < 2)
        velocity_norm += point.basis_values[i].norm();
      else
        pressure_norm += point.basis_values[i].norm();
  velocity_norm = Utilities::MPI::sum(velocity_norm, MPI_COMM_WORLD);
  pressure_norm = Utilities::MPI::sum(pressure_norm, MPI_COMM_WORLD);
  EXPECT_GT(velocity_norm, 1.e-12);
  EXPECT_NEAR(pressure_norm, 0., 1.e-14);
}

#else

TEST(DistributedIDA, AdapterIsConditionallyUnavailable) // NOLINT
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}

#endif
