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
#include <immersx/core/constraint.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/representation.h>
#include <immersx/core/state.h>

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


TEST(VectorLagrangeMultiplierInteraction, UnifiedConstraintMatchesBlocks)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  VectorSpace matrix_space(MPI_COMM_WORLD);
  VectorSpace fiber_space(MPI_COMM_WORLD);
  auto        matrix_representation = make_representation(matrix_space);
  auto        fiber_representation  = make_representation(fiber_space);
  ParticleCouplingParameters<2> search_parameters;

  Interaction legacy(matrix_representation,
                     fiber_representation,
                     search_parameters);
  legacy.assemble();

  DoFHandler<2> multiplier_dh(fiber_space.triangulation);
  multiplier_dh.distribute_dofs(fiber_space.fe);
  const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
  const auto multiplier_relevant =
    DoFTools::extract_locally_relevant_dofs(multiplier_dh);
  AffineConstraints<double> multiplier_constraints;
  multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
  DoFTools::make_hanging_node_constraints(multiplier_dh,
                                          multiplier_constraints);
  multiplier_constraints.close();

  const auto  matrix_view     = fe_space(matrix_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    matrix_space.constraints);
  const auto  fiber_view      = fe_space(fiber_space.dof_handler,
                                   StaticMappingQ1<2>::mapping,
                                   fiber_space.constraints);
  const auto  multiplier_view = fe_space(multiplier_dh,
                                        StaticMappingQ1<2>::mapping,
                                        multiplier_constraints,
                                        multiplier_relevant);
  StateLayout layout;
  const auto  matrix =
    matrix_view.field(layout, "matrix", FEValuesExtractors::Vector(0));
  const auto fiber =
    fiber_view.field(layout, "fiber", FEValuesExtractors::Vector(0));
  const auto lambda =
    multiplier_view.field("lambda", FEValuesExtractors::Vector(0));

  using Matrix = ImmersXLA::MPI::SparseMatrix;
  using Model  = SemiDiscreteModel<VectorType, Matrix>;
  Model                                   model;
  SemidiscreteBuilder<VectorType, Matrix> builder(layout, model);
  const auto fields = make_constraint(weak_term(value(matrix), lambda) -
                                      weak_term(value(fiber), lambda))
                        .add(builder);

  ASSERT_EQ(multiplier_dh.n_dofs(), fiber_space.dof_handler.n_dofs());
  VectorType matrix_state = constant_vector(matrix_space, 1., 2.);
  VectorType fiber_state  = constant_vector(fiber_space, 1., 2.);
  VectorType lambda_state(multiplier_owned, MPI_COMM_WORLD);
  lambda_state = 0.;
  lambda_state.compress(VectorOperation::insert);
  StateView<VectorType> state_view(layout, 0.);
  state_view.bind(matrix.field_id(), matrix_state);
  state_view.bind(fiber.field_id(), fiber_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<VectorType> context(0., state_view);

  const auto matrix_block =
    model.state_matrix_operator(fields.multiplier, matrix.field_id(), context);
  const auto fiber_block =
    model.state_matrix_operator(fields.multiplier, fiber.field_id(), context);
  ASSERT_TRUE(matrix_block.has_value());
  ASSERT_TRUE(fiber_block.has_value());
  const auto matrix_operator = matrix_block->matrix();
  const auto fiber_operator  = fiber_block->matrix();
  for (types::global_dof_index i = 0; i < multiplier_dh.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < matrix_space.dof_handler.n_dofs();
         ++j)
      EXPECT_NEAR(matrix_operator->el(i, j),
                  legacy.coupling_matrix().el(j, i),
                  1.e-12);
  for (types::global_dof_index i = 0; i < multiplier_dh.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < fiber_space.dof_handler.n_dofs();
         ++j)
      EXPECT_NEAR(fiber_operator->el(i, j),
                  -legacy.pairing_matrix().el(i, j),
                  1.e-12);
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
