// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/algebra/vector_lagrange_multiplier_interaction.h>
#include <immersx/core/representation.h>

#include <cmath>
#include <vector>

using namespace ImmersX;
using namespace dealii;

namespace
{
  struct VectorSpace
  {
    explicit VectorSpace(const MPI_Comm     communicator,
                         const unsigned int refinement = 0)
      : triangulation(communicator)
      , dof_handler(triangulation)
      , constraints()
    {
      GridGenerator::hyper_cube(triangulation, -1., 1.);
      triangulation.refine_global(refinement);
      dof_handler.distribute_dofs(fe);
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
      constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
    }

    FESystem<2>                             fe{FE_Q<2>(1), 2};
    parallel::distributed::Triangulation<2> triangulation;
    DoFHandler<2>                           dof_handler;
    IndexSet                                locally_owned_dofs;
    IndexSet                                locally_relevant_dofs;
    AffineConstraints<double>               constraints;
  };

  using Representation = VectorFiniteElementRepresentation<2, 2>;
  using Interaction =
    VectorLagrangeMultiplierInteraction<Representation, Representation>;
  using VectorType = ImmersXLA::MPI::Vector;

  Representation
  make_representation(const VectorSpace &space)
  {
    return Representation(space.triangulation,
                          space.dof_handler,
                          space.locally_owned_dofs,
                          space.locally_relevant_dofs,
                          space.constraints);
  }

  VectorType
  constant_vector(const VectorSpace &space,
                  const double       x_value,
                  const double       y_value)
  {
    VectorType values(space.locally_owned_dofs, MPI_COMM_WORLD);
    values = 0.;
    std::vector<types::global_dof_index> indices(space.fe.n_dofs_per_cell());
    for (const auto &cell : space.dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (values.locally_owned_elements().is_element(indices[i]))
              values(indices[i]) =
                space.fe.system_to_component_index(i).first == 0 ? x_value :
                                                                   y_value;
        }
    values.compress(VectorOperation::insert);
    return values;
  }

  VectorType
  affine_vector(const VectorSpace &space)
  {
    VectorType values(space.locally_owned_dofs, MPI_COMM_WORLD);
    values = 0.;
    MappingQ1<2> mapping;
    const auto  &support_points = space.fe.get_unit_support_points();
    std::vector<types::global_dof_index> indices(space.fe.n_dofs_per_cell());
    for (const auto &cell : space.dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (values.locally_owned_elements().is_element(indices[i]))
              {
                const auto point =
                  mapping.transform_unit_to_real_cell(cell, support_points[i]);
                const auto component =
                  space.fe.system_to_component_index(i).first;
                values(indices[i]) = component == 0 ? point[0] + 2. * point[1] :
                                                      -point[0] + point[1];
              }
        }
    values.compress(VectorOperation::insert);
    return values;
  }

  std::vector<unsigned int>
  dof_components(const VectorSpace &space)
  {
    std::vector<unsigned int> components(space.dof_handler.n_dofs(), 0);
    std::vector<types::global_dof_index> indices(space.fe.n_dofs_per_cell());
    for (const auto &cell : space.dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            components[indices[i]] =
              space.fe.system_to_component_index(i).first;
        }
    return components;
  }
} // namespace


TEST(VectorLagrangeMultiplierInteraction, MatchingPairingAndDimensions)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  VectorSpace matrix_space(MPI_COMM_WORLD);
  VectorSpace fiber_space(MPI_COMM_WORLD);
  auto        matrix_representation = make_representation(matrix_space);
  auto        fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;

  Interaction interaction(matrix_representation,
                          fiber_representation,
                          search_parameters);
  interaction.assemble();

  EXPECT_EQ(interaction.coupling_matrix().m(),
            matrix_space.dof_handler.n_dofs());
  EXPECT_EQ(interaction.coupling_matrix().n(),
            fiber_space.dof_handler.n_dofs());
  EXPECT_EQ(interaction.pairing_matrix().m(), fiber_space.dof_handler.n_dofs());
  EXPECT_EQ(interaction.pairing_matrix().n(), fiber_space.dof_handler.n_dofs());
  EXPECT_GT(interaction.coupling_matrix().frobenius_norm(), 1.e-12);
  EXPECT_GT(interaction.pairing_matrix().frobenius_norm(), 1.e-12);

  for (types::global_dof_index i = 0; i < matrix_space.dof_handler.n_dofs();
       ++i)
    for (types::global_dof_index j = 0; j < fiber_space.dof_handler.n_dofs();
         ++j)
      EXPECT_NEAR(interaction.coupling_matrix().el(i, j),
                  interaction.pairing_matrix().el(i, j),
                  1.e-12);
  EXPECT_TRUE(interaction.assembly_is_current());
}


TEST(VectorLagrangeMultiplierInteraction, ComponentIndependence)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  VectorSpace matrix_space(MPI_COMM_WORLD);
  VectorSpace fiber_space(MPI_COMM_WORLD);
  auto        matrix_representation = make_representation(matrix_space);
  auto        fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;
  Interaction                   interaction(matrix_representation,
                          fiber_representation,
                          search_parameters);
  interaction.assemble();

  const auto matrix_components = dof_components(matrix_space);
  const auto fiber_components  = dof_components(fiber_space);

  for (types::global_dof_index i = 0; i < matrix_space.dof_handler.n_dofs();
       ++i)
    for (types::global_dof_index j = 0; j < fiber_space.dof_handler.n_dofs();
         ++j)
      if (matrix_components[i] != fiber_components[j])
        EXPECT_NEAR(interaction.coupling_matrix().el(i, j), 0., 1.e-12);
}


TEST(VectorLagrangeMultiplierInteraction, NonmatchingConstantReproduction)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  VectorSpace matrix_space(MPI_COMM_WORLD);
  VectorSpace fiber_space(MPI_COMM_WORLD);
  fiber_space.triangulation.clear();
  GridGenerator::subdivided_hyper_rectangle(fiber_space.triangulation,
                                            {4, 1},
                                            Point<2>(-.6, -.1),
                                            Point<2>(.6, .1),
                                            true);
  fiber_space.dof_handler.distribute_dofs(fiber_space.fe);
  fiber_space.constraints.clear();
  fiber_space.locally_owned_dofs = fiber_space.dof_handler.locally_owned_dofs();
  fiber_space.locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(fiber_space.dof_handler);
  fiber_space.constraints.reinit(fiber_space.locally_owned_dofs,
                                 fiber_space.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(fiber_space.dof_handler,
                                          fiber_space.constraints);
  fiber_space.constraints.close();

  auto matrix_representation = make_representation(matrix_space);
  auto fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;
  Interaction                   interaction(matrix_representation,
                          fiber_representation,
                          search_parameters);
  interaction.assemble();

  const auto matrix_values = constant_vector(matrix_space, 1., 2.);
  const auto fiber_values  = constant_vector(fiber_space, 1., 2.);
  std::vector<const VectorType *> states{&matrix_values, &fiber_values};
  VectorType                      residual;
  interaction.constraint_equation().residual(states, residual);

  EXPECT_TRUE(std::isfinite(residual.l2_norm()));
  EXPECT_LT(residual.l2_norm(), 1.e-11);
  EXPECT_GT(interaction.coupling_matrix().n_nonzero_elements(), 0u);
  EXPECT_GT(interaction.pairing_matrix().n_nonzero_elements(), 0u);
}


TEST(VectorLagrangeMultiplierInteraction, MatchingAffineReproduction)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  VectorSpace matrix_space(MPI_COMM_WORLD, 1);
  VectorSpace fiber_space(MPI_COMM_WORLD, 1);
  auto        matrix_representation = make_representation(matrix_space);
  auto        fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;
  Interaction                   interaction(matrix_representation,
                          fiber_representation,
                          search_parameters);
  interaction.assemble();

  const auto                      matrix_values = affine_vector(matrix_space);
  const auto                      fiber_values  = affine_vector(fiber_space);
  std::vector<const VectorType *> states{&matrix_values, &fiber_values};
  VectorType                      residual;
  interaction.constraint_equation().residual(states, residual);

  EXPECT_LT(residual.l2_norm(), 1.e-11);
}


TEST(VectorLagrangeMultiplierInteraction, MPI_SemanticResidualAndJacobian)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);

  VectorSpace matrix_space(MPI_COMM_WORLD, 1);
  VectorSpace fiber_space(MPI_COMM_WORLD);
  auto        matrix_representation = make_representation(matrix_space);
  auto        fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;
  Interaction                   interaction(matrix_representation,
                          fiber_representation,
                          search_parameters);
  interaction.assemble();

  StateLayout     layout;
  FieldDescriptor first_descriptor;
  first_descriptor.name             = "matrix.velocity";
  first_descriptor.time_role        = TimeRole::differential;
  first_descriptor.locally_owned    = matrix_space.locally_owned_dofs;
  first_descriptor.locally_relevant = matrix_space.locally_relevant_dofs;
  const auto first                  = layout.add_field(first_descriptor);

  FieldDescriptor second_descriptor;
  second_descriptor.name             = "fiber.velocity";
  second_descriptor.time_role        = TimeRole::differential;
  second_descriptor.locally_owned    = fiber_space.locally_owned_dofs;
  second_descriptor.locally_relevant = fiber_space.locally_relevant_dofs;
  const auto second                  = layout.add_field(second_descriptor);

  const auto fields =
    interaction.register_fields(layout, first, second, "fiber_coupling");
  EXPECT_EQ(layout.field(fields.multiplier).time_role, TimeRole::algebraic);
  EXPECT_EQ(layout.field(fields.multiplier).locally_owned,
            interaction.multiplier_locally_owned_dofs());

  SemiDiscreteModel<VectorType> model;
  interaction.add_semidiscrete_terms(model, fields);

  const auto first_values  = constant_vector(matrix_space, 1., 2.);
  const auto second_values = constant_vector(fiber_space, 3., 4.);
  VectorType lambda(interaction.multiplier_locally_owned_dofs(),
                    MPI_COMM_WORLD);
  lambda = 0.25;

  StateView<VectorType> state(layout, 0.);
  state.bind(first, first_values);
  state.bind(second, second_values);
  state.bind(fields.multiplier, lambda);
  EvaluationContext<VectorType> evaluation(0., state);

  VectorType first_residual(matrix_space.locally_owned_dofs, MPI_COMM_WORLD);
  VectorType second_residual(fiber_space.locally_owned_dofs, MPI_COMM_WORLD);
  VectorType lambda_residual(interaction.multiplier_locally_owned_dofs(),
                             MPI_COMM_WORLD);
  first_residual  = 0.;
  second_residual = 0.;
  lambda_residual = 0.;
  ResidualAccumulator<VectorType> residual(layout);
  residual.bind(first, first_residual);
  residual.bind(second, second_residual);
  residual.bind(fields.multiplier, lambda_residual);
  model.evaluate(evaluation, residual);

  VectorType expected_first(first_residual);
  VectorType expected_second(second_residual);
  VectorType expected_lambda(lambda_residual);
  expected_first  = 0.;
  expected_second = 0.;
  expected_lambda = 0.;
  interaction.coupling_matrix().vmult(expected_first, lambda);
  interaction.pairing_matrix().Tvmult(expected_second, lambda);
  expected_second *= -1.;
  interaction.coupling_matrix().Tvmult(expected_lambda, first_values);
  VectorType product(lambda_residual);
  interaction.pairing_matrix().vmult(product, second_values);
  expected_lambda -= product;

  VectorType difference(first_residual);
  difference -= expected_first;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = second_residual;
  difference -= expected_second;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = lambda_residual;
  difference -= expected_lambda;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);

  const auto first_increment  = constant_vector(matrix_space, 2., 1.);
  const auto second_increment = constant_vector(fiber_space, -1., 3.);
  VectorType lambda_increment(lambda);
  lambda_increment = 0.5;
  StateView<VectorType> increment(layout, 0.);
  increment.bind(first, first_increment);
  increment.bind(second, second_increment);
  increment.bind(fields.multiplier, lambda_increment);
  LinearizationContext<VectorType> linearization(evaluation, 2., 7.);

  first_residual  = 0.;
  second_residual = 0.;
  lambda_residual = 0.;
  ResidualAccumulator<VectorType> jacobian_residual(layout);
  jacobian_residual.bind(first, first_residual);
  jacobian_residual.bind(second, second_residual);
  jacobian_residual.bind(fields.multiplier, lambda_residual);
  model.add_jacobian_action(linearization, increment, jacobian_residual);

  expected_first  = 0.;
  expected_second = 0.;
  expected_lambda = 0.;
  interaction.coupling_matrix().vmult(expected_first, lambda_increment);
  interaction.pairing_matrix().Tvmult(expected_second, lambda_increment);
  expected_second *= -1.;
  interaction.coupling_matrix().Tvmult(expected_lambda, first_increment);
  interaction.pairing_matrix().vmult(product, second_increment);
  expected_lambda -= product;
  expected_first *= 2.;
  expected_second *= 2.;
  expected_lambda *= 2.;

  difference = first_residual;
  difference -= expected_first;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = second_residual;
  difference -= expected_second;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
  difference = lambda_residual;
  difference -= expected_lambda;
  EXPECT_NEAR(difference.l2_norm(), 0., 1.e-11);
}


TEST(VectorLagrangeMultiplierInteraction, MPI_NonmatchingDistributedSearch)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);

  VectorSpace matrix_space(MPI_COMM_WORLD, 2);
  VectorSpace fiber_space(MPI_COMM_WORLD);
  fiber_space.triangulation.clear();
  GridGenerator::subdivided_hyper_rectangle(fiber_space.triangulation,
                                            {4, 2},
                                            Point<2>(-.6, -.1),
                                            Point<2>(.6, .1),
                                            true);
  fiber_space.dof_handler.distribute_dofs(fiber_space.fe);
  fiber_space.constraints.clear();
  fiber_space.locally_owned_dofs = fiber_space.dof_handler.locally_owned_dofs();
  fiber_space.locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(fiber_space.dof_handler);
  fiber_space.constraints.reinit(fiber_space.locally_owned_dofs,
                                 fiber_space.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(fiber_space.dof_handler,
                                          fiber_space.constraints);
  fiber_space.constraints.close();

  auto matrix_representation = make_representation(matrix_space);
  auto fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;
  Interaction                   interaction(matrix_representation,
                          fiber_representation,
                          search_parameters);
  interaction.assemble();

  EXPECT_EQ(interaction.coupling_matrix().m(),
            matrix_space.dof_handler.n_dofs());
  EXPECT_EQ(interaction.pairing_matrix().m(), fiber_space.dof_handler.n_dofs());
  EXPECT_TRUE(interaction.assembly_is_current());
}
