// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <cmath>

#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  void
  configure_problem(FiberReinforcedElastodynamicsParameters<2> &parameters,
                    const bool                                  forcing)
  {
    parameters.output_directory = TestPaths::output_directory(
      forcing ? "fiber-reinforced-forced" : "fiber-reinforced-zero");
    parameters.output_frequency = 0;

    initialize_parameters_from_string(std::string(R"(
        set dimension       = 2
        set space dimension = 2
        subsection Fiber Reinforced Elastodynamics
          subsection Time integration
            set Final time   = 0.02
            set Time step    = 0.01
            set Number of time steps = 2
          end
          subsection Matrix Elastodynamics
            set Initial refinement = 1
            set Dirichlet boundary ids = 0,1,2,3
            subsection Grid generation
              set Grid generator           = hyper_cube
              set Grid generator arguments = -1: 1: false
            end
            subsection Functions
              subsection Body force
                set Function expression = )") +
                                      (forcing ? "0; 1" : "0; 0") +
                                      R"(
                set Variable names      = x,y,t
              end
              subsection Displacement boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Velocity boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial displacement
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial velocity
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
            end
          end
          subsection Fiber Elastodynamics
            set Initial refinement = 1
            set Dirichlet boundary ids =
            subsection Grid generation
              set Grid generator           = subdivided_hyper_rectangle
              set Grid generator arguments = 3, 1: -0.6, -0.1: 0.6, 0.1: true
            end
            subsection Material
              set Density     = 2.0
              set Lame mu     = 2.0
              set Lame lambda = 2.0
            end
            subsection Functions
              subsection Body force
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Displacement boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Velocity boundary
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial displacement
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
              subsection Initial velocity
                set Function expression = 0; 0
                set Variable names      = x,y,t
              end
            end
          end
        end
      )");
  }
} // namespace


TEST(FiberReinforcedElastodynamics, ZeroPreservationAndCompatibility)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, false);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.run();

  EXPECT_EQ(driver.time_step_number(), 2u);
  EXPECT_NEAR(driver.current_time(), 0.02, 1.e-14);
  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_NEAR(driver.matrix_problem().displacement().l2_norm(), 0., 1.e-12);
  EXPECT_NEAR(driver.matrix_problem().velocity().l2_norm(), 0., 1.e-12);
  EXPECT_NEAR(driver.fiber_problem().displacement().l2_norm(), 0., 1.e-12);
  EXPECT_NEAR(driver.fiber_problem().velocity().l2_norm(), 0., 1.e-12);
  EXPECT_LT(driver.residuals().matrix_velocity, 1.e-10);
  EXPECT_LT(driver.residuals().fiber_velocity, 1.e-10);
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-10);
  EXPECT_LT(driver.residuals().displacement_compatibility, 1.e-10);
}


TEST(FiberReinforcedElastodynamics, CoupledResidualAndExcessResponse)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.setup();
  EXPECT_GT(driver.interaction().coupling_matrix().frobenius_norm(), 1.e-12);
  EXPECT_GT(driver.interaction().pairing_matrix().frobenius_norm(), 1.e-12);
  driver.set_initial_conditions();
  driver.advance_one_timestep();
  driver.advance_one_timestep();

  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_LT(driver.residuals().matrix_velocity, 1.e-8);
  EXPECT_LT(driver.residuals().fiber_velocity, 1.e-8);
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-8);
  EXPECT_LT(driver.residuals().displacement_compatibility, 1.e-8);
  EXPECT_TRUE(std::isfinite(driver.fiber_excess_elastic_energy()));
  EXPECT_GT(driver.matrix_only_displacement_difference(), 1.e-12);
  EXPECT_GT(driver.fiber_problem().displacement().l2_norm(), 1.e-12);
}


TEST(FiberReinforcedElastodynamics, MPI_FiveFieldResidualAndJacobian)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.setup();
  driver.set_initial_conditions();

  const auto previous_matrix_displacement =
    driver.matrix_problem().displacement();
  const auto previous_matrix_velocity = driver.matrix_problem().velocity();
  const auto previous_fiber_displacement =
    driver.fiber_problem().displacement();
  const auto previous_fiber_velocity = driver.fiber_problem().velocity();
  driver.advance_one_timestep();

  const auto  &matrix                      = driver.matrix_problem();
  const auto  &fiber                       = driver.fiber_problem();
  const double dt                          = driver.current_time();
  const auto   current_matrix_displacement = matrix.displacement();
  const auto   current_matrix_velocity     = matrix.velocity();
  const auto   current_fiber_displacement  = fiber.displacement();
  const auto   current_fiber_velocity      = fiber.velocity();
  const auto   current_multiplier          = driver.multiplier();

  StateLayout layout;
  const auto  matrix_fields = register_elastodynamics_fields(
    layout, matrix, "matrix", HistoryGroupId(101));
  const auto fiber_fields =
    register_elastodynamics_fields(layout, fiber, "fiber", HistoryGroupId(202));
  const auto interaction_fields = driver.interaction().register_fields(
    layout, matrix_fields.velocity, fiber_fields.velocity, "fiber_coupling");

  SemiDiscreteModel<LA::MPI::Vector> model;
  add_elastodynamics_terms(model, matrix, matrix_fields);
  add_elastodynamics_terms(model, fiber, fiber_fields);
  driver.interaction().add_semidiscrete_terms(model, interaction_fields);

  LA::MPI::Vector matrix_displacement     = current_matrix_displacement;
  LA::MPI::Vector matrix_velocity         = current_matrix_velocity;
  LA::MPI::Vector fiber_displacement      = current_fiber_displacement;
  LA::MPI::Vector fiber_velocity          = current_fiber_velocity;
  LA::MPI::Vector multiplier              = current_multiplier;
  LA::MPI::Vector matrix_displacement_dot = matrix_displacement;
  LA::MPI::Vector matrix_velocity_dot     = matrix_velocity;
  LA::MPI::Vector fiber_displacement_dot  = fiber_displacement;
  LA::MPI::Vector fiber_velocity_dot      = fiber_velocity;
  matrix_displacement_dot -= previous_matrix_displacement;
  matrix_velocity_dot -= previous_matrix_velocity;
  fiber_displacement_dot -= previous_fiber_displacement;
  fiber_velocity_dot -= previous_fiber_velocity;
  matrix_displacement_dot *= 1. / dt;
  matrix_velocity_dot *= 1. / dt;
  fiber_displacement_dot *= 1. / dt;
  fiber_velocity_dot *= 1. / dt;

  StateView<LA::MPI::Vector> state(layout, driver.current_time());
  state.bind(matrix_fields.displacement, matrix_displacement);
  state.bind(matrix_fields.velocity, matrix_velocity);
  state.bind(fiber_fields.displacement, fiber_displacement);
  state.bind(fiber_fields.velocity, fiber_velocity);
  state.bind(interaction_fields.multiplier, multiplier);
  StateView<LA::MPI::Vector> derivative(layout, driver.current_time());
  derivative.bind(matrix_fields.displacement, matrix_displacement_dot);
  derivative.bind(matrix_fields.velocity, matrix_velocity_dot);
  derivative.bind(fiber_fields.displacement, fiber_displacement_dot);
  derivative.bind(fiber_fields.velocity, fiber_velocity_dot);
  EvaluationContext<LA::MPI::Vector> evaluation(driver.current_time(),
                                                state,
                                                &derivative);

  LA::MPI::Vector matrix_displacement_residual(matrix_displacement);
  LA::MPI::Vector matrix_velocity_residual(matrix_velocity);
  LA::MPI::Vector fiber_displacement_residual(fiber_displacement);
  LA::MPI::Vector fiber_velocity_residual(fiber_velocity);
  LA::MPI::Vector multiplier_residual(multiplier);
  matrix_displacement_residual = 0.;
  matrix_velocity_residual     = 0.;
  fiber_displacement_residual  = 0.;
  fiber_velocity_residual      = 0.;
  multiplier_residual          = 0.;
  ResidualAccumulator<LA::MPI::Vector> residual(layout);
  residual.bind(matrix_fields.displacement, matrix_displacement_residual);
  residual.bind(matrix_fields.velocity, matrix_velocity_residual);
  residual.bind(fiber_fields.displacement, fiber_displacement_residual);
  residual.bind(fiber_fields.velocity, fiber_velocity_residual);
  residual.bind(interaction_fields.multiplier, multiplier_residual);
  model.evaluate(evaluation, residual);

  EXPECT_LT(matrix_displacement_residual.l2_norm(), 1.e-10);
  EXPECT_NEAR(matrix_velocity_residual.l2_norm(),
              driver.residuals().matrix_velocity,
              1.e-10);
  EXPECT_LT(fiber_displacement_residual.l2_norm(), 1.e-10);
  EXPECT_NEAR(fiber_velocity_residual.l2_norm(),
              driver.residuals().fiber_velocity,
              1.e-10);
  EXPECT_NEAR(multiplier_residual.l2_norm(),
              driver.residuals().velocity_constraint,
              1.e-10);
  EXPECT_LT(matrix_velocity_residual.l2_norm(), 1.e-8);
  EXPECT_LT(fiber_velocity_residual.l2_norm(), 1.e-8);
  EXPECT_LT(multiplier_residual.l2_norm(), 1.e-8);

  auto constant = [](const LA::MPI::Vector &prototype, const double value) {
    LA::MPI::Vector result(prototype);
    result = value;
    return result;
  };
  const auto delta_matrix_displacement = constant(matrix_displacement, 0.25);
  const auto delta_matrix_velocity     = constant(matrix_velocity, -0.5);
  const auto delta_fiber_displacement  = constant(fiber_displacement, 0.75);
  const auto delta_fiber_velocity      = constant(fiber_velocity, 1.25);
  const auto delta_multiplier          = constant(multiplier, -0.25);
  StateView<LA::MPI::Vector> increment(layout, driver.current_time());
  increment.bind(matrix_fields.displacement, delta_matrix_displacement);
  increment.bind(matrix_fields.velocity, delta_matrix_velocity);
  increment.bind(fiber_fields.displacement, delta_fiber_displacement);
  increment.bind(fiber_fields.velocity, delta_fiber_velocity);
  increment.bind(interaction_fields.multiplier, delta_multiplier);
  const double                          alpha = 1.75;
  LinearizationContext<LA::MPI::Vector> linearization(evaluation, 1., alpha);

  matrix_displacement_residual = 0.;
  matrix_velocity_residual     = 0.;
  fiber_displacement_residual  = 0.;
  fiber_velocity_residual      = 0.;
  multiplier_residual          = 0.;
  ResidualAccumulator<LA::MPI::Vector> jacobian(layout);
  jacobian.bind(matrix_fields.displacement, matrix_displacement_residual);
  jacobian.bind(matrix_fields.velocity, matrix_velocity_residual);
  jacobian.bind(fiber_fields.displacement, fiber_displacement_residual);
  jacobian.bind(fiber_fields.velocity, fiber_velocity_residual);
  jacobian.bind(interaction_fields.multiplier, multiplier_residual);
  model.add_jacobian_action(linearization, increment, jacobian);

  auto expected_jacobian = [](const auto  &problem,
                              const auto  &delta_displacement,
                              const auto  &delta_velocity,
                              const double derivative_weight,
                              auto        &displacement_row,
                              auto        &velocity_row) {
    LA::MPI::Vector product(displacement_row);
    problem.mass_matrix().vmult(displacement_row, delta_displacement);
    displacement_row *= derivative_weight;
    problem.mass_matrix().vmult(product, delta_velocity);
    displacement_row -= product;
    problem.mass_matrix().vmult(velocity_row, delta_velocity);
    velocity_row *= derivative_weight;
    problem.stiffness_matrix().vmult(product, delta_displacement);
    velocity_row += product;
    problem.damping_matrix().vmult(product, delta_velocity);
    velocity_row += product;
    for (const auto index : displacement_row.locally_owned_elements())
      if (problem.constraints().is_constrained(index))
        displacement_row(index) = 0.;
    for (const auto index : velocity_row.locally_owned_elements())
      if (problem.velocity_constraints().is_constrained(index))
        velocity_row(index) = 0.;
  };
  LA::MPI::Vector expected_matrix_displacement(matrix_displacement);
  LA::MPI::Vector expected_matrix_velocity(matrix_velocity);
  LA::MPI::Vector expected_fiber_displacement(fiber_displacement);
  LA::MPI::Vector expected_fiber_velocity(fiber_velocity);
  expected_matrix_displacement = 0.;
  expected_matrix_velocity     = 0.;
  expected_fiber_displacement  = 0.;
  expected_fiber_velocity      = 0.;
  expected_jacobian(matrix,
                    delta_matrix_displacement,
                    delta_matrix_velocity,
                    alpha,
                    expected_matrix_displacement,
                    expected_matrix_velocity);
  expected_jacobian(fiber,
                    delta_fiber_displacement,
                    delta_fiber_velocity,
                    alpha,
                    expected_fiber_displacement,
                    expected_fiber_velocity);
  LA::MPI::Vector expected_multiplier(multiplier);
  expected_multiplier = 0.;
  LA::MPI::Vector product_matrix(expected_matrix_velocity);
  LA::MPI::Vector product_fiber(expected_fiber_velocity);
  LA::MPI::Vector product_multiplier(expected_multiplier);
  driver.interaction().coupling_matrix().vmult(product_matrix,
                                               delta_multiplier);
  expected_matrix_velocity += product_matrix;
  driver.interaction().pairing_matrix().Tvmult(product_fiber, delta_multiplier);
  expected_fiber_velocity -= product_fiber;
  driver.interaction().coupling_matrix().Tvmult(product_multiplier,
                                                delta_matrix_velocity);
  expected_multiplier += product_multiplier;
  driver.interaction().pairing_matrix().vmult(product_multiplier,
                                              delta_fiber_velocity);
  expected_multiplier -= product_multiplier;

  auto difference = matrix_displacement_residual;
  difference -= expected_matrix_displacement;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
  difference = matrix_velocity_residual;
  difference -= expected_matrix_velocity;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
  difference = fiber_displacement_residual;
  difference -= expected_fiber_displacement;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
  difference = fiber_velocity_residual;
  difference -= expected_fiber_velocity;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
  difference = multiplier_residual;
  difference -= expected_multiplier;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-10);
}


TEST(FiberReinforcedElastodynamics, MPI_CoupledTransient)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  configure_problem(parameters, true);
  FiberReinforcedElastodynamics<2> driver(parameters);
  driver.run();

  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-8);
  EXPECT_LT(driver.residuals().displacement_compatibility, 1.e-8);
}


TEST(FiberReinforcedElastodynamics, ThreeDimensionalSmoke)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<3> parameters;
  parameters.output_directory =
    TestPaths::output_directory("fiber-reinforced-3d");
  parameters.output_frequency                     = 0;
  parameters.matrix_parameters.initial_refinement = 0;
  parameters.matrix_parameters.dirichlet_ids      = {0, 1, 2, 3, 4, 5};
  parameters.fiber_parameters.initial_refinement  = 0;
  parameters.fiber_parameters.dirichlet_ids.clear();

  initialize_parameters_from_string(R"(
    set dimension       = 3
    set space dimension = 3
    subsection Fiber Reinforced Elastodynamics
      subsection Time integration
        set Time step          = 0.01
        set Number of time steps = 1
      end
      subsection Matrix Elastodynamics
        subsection Grid generation
          set Grid generator           = hyper_cube
          set Grid generator arguments = -1: 1: false
        end
      end
      subsection Fiber Elastodynamics
        subsection Grid generation
          set Grid generator           = subdivided_hyper_rectangle
          set Grid generator arguments = 2, 1, 1: -0.6, -0.1, -0.1: 0.6, 0.1, 0.1: true
        end
      end
    end
  )");

  FiberReinforcedElastodynamics<3> driver(parameters);
  driver.run();
  EXPECT_TRUE(driver.matrix_problem().state_is_finite());
  EXPECT_TRUE(driver.fiber_problem().state_is_finite());
  EXPECT_EQ(driver.time_step_number(), 1u);
  EXPECT_LT(driver.residuals().velocity_constraint, 1.e-10);
}
