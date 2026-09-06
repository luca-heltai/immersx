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

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/constraint.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>

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

  using VectorType = ImmersXLA::MPI::Vector;
  using MatrixType = ImmersXLA::MPI::SparseMatrix;
  using Model      = SemiDiscreteModel<VectorType, MatrixType>;

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

  void
  initialize_constraints(const DoFHandler<2>       &dof_handler,
                         const IndexSet            &locally_owned,
                         const IndexSet            &locally_relevant,
                         AffineConstraints<double> &constraints)
  {
    constraints.reinit(locally_owned, locally_relevant);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
  }
} // namespace


TEST(Constraint, IndependentVectorMultiplierSharedGeometry)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);

  VectorSpace   source_space(MPI_COMM_WORLD);
  FESystem<2>   multiplier_fe(FE_Q<2>(2), 2);
  DoFHandler<2> multiplier_dh(source_space.triangulation);
  multiplier_dh.distribute_dofs(multiplier_fe);
  const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
  const auto multiplier_relevant =
    DoFTools::extract_locally_relevant_dofs(multiplier_dh);
  AffineConstraints<double> multiplier_constraints;
  initialize_constraints(multiplier_dh,
                         multiplier_owned,
                         multiplier_relevant,
                         multiplier_constraints);

  const auto source_view     = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto multiplier_view = fe_space(multiplier_dh,
                                        StaticMappingQ1<2>::mapping,
                                        multiplier_constraints,
                                        multiplier_relevant);

  StateLayout layout;
  const auto  source =
    source_view.field(layout, "source", FEValuesExtractors::Vector(0));
  const auto lambda =
    multiplier_view.field("lambda", FEValuesExtractors::Vector(0));

  Model                                       model;
  SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
#ifdef IMMERSX_WEAK_TERM_TESTING
  const auto preparations = detail::weak_term_nonmatching_preparations.load();
#endif
  const auto fields =
    make_constraint(weak_term(value(source), test(lambda))).add(builder);
#ifdef IMMERSX_WEAK_TERM_TESTING
  EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(), preparations);
#endif

  VectorType source_state = constant_vector(source_space, 1., -2.);
  VectorType lambda_state(multiplier_owned, MPI_COMM_WORLD);
  lambda_state = 0.;
  lambda_state.compress(VectorOperation::insert);

  StateView<VectorType> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<VectorType> context(0., state_view);

  const auto multiplier_block =
    model.state_matrix_operator(fields.multiplier, source.field_id(), context);
  const auto source_block =
    model.state_matrix_operator(source.field_id(), fields.multiplier, context);
  ASSERT_TRUE(multiplier_block.has_value());
  ASSERT_TRUE(source_block.has_value());
  ASSERT_EQ(multiplier_block->matrix()->m(), multiplier_dh.n_dofs());
  ASSERT_EQ(multiplier_block->matrix()->n(), source_space.dof_handler.n_dofs());
  ASSERT_EQ(source_block->matrix()->m(), source_space.dof_handler.n_dofs());
  ASSERT_EQ(source_block->matrix()->n(), multiplier_dh.n_dofs());
  EXPECT_GT(multiplier_block->matrix()->n_nonzero_elements(), 0u);

  VectorType multiplier_action(multiplier_owned, MPI_COMM_WORLD);
  VectorType source_reaction(source_space.locally_owned_dofs, MPI_COMM_WORLD);
  multiplier_block->view.vmult(multiplier_action, source_state);
  source_block->view.vmult(source_reaction, lambda_state);
  EXPECT_NEAR(multiplier_action * lambda_state,
              source_state * source_reaction,
              1.e-11);
}


TEST(Constraint, MPI_VectorNonmatchingGeometry)
{
  ASSERT_GE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);

  VectorSpace source_space(MPI_COMM_WORLD, 2);
  VectorSpace multiplier_space(MPI_COMM_WORLD);
  const auto  source_view     = fe_space(source_space.dof_handler,
                                    StaticMappingQ1<2>::mapping,
                                    source_space.constraints);
  const auto  multiplier_view = fe_space(multiplier_space.dof_handler,
                                        StaticMappingQ1<2>::mapping,
                                        multiplier_space.constraints,
                                        multiplier_space.locally_relevant_dofs);

  StateLayout layout;
  const auto  source =
    source_view.field(layout, "source", FEValuesExtractors::Vector(0));
  const auto lambda =
    multiplier_view.field("lambda", FEValuesExtractors::Vector(0));

  Model                                       model;
  SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
#ifdef IMMERSX_WEAK_TERM_TESTING
  const auto preparations = detail::weak_term_nonmatching_preparations.load();
#endif
  const auto fields =
    make_constraint(weak_term(value(source), test(lambda))).add(builder);
#ifdef IMMERSX_WEAK_TERM_TESTING
  EXPECT_EQ(detail::weak_term_nonmatching_preparations.load(),
            preparations + 1);
#endif

  VectorType source_state(source_space.locally_owned_dofs, MPI_COMM_WORLD);
  source_state = 1.;
  source_state.compress(VectorOperation::insert);
  VectorType lambda_state(multiplier_space.locally_owned_dofs, MPI_COMM_WORLD);
  lambda_state = 0.;
  lambda_state.compress(VectorOperation::insert);
  StateView<VectorType> state_view(layout, 0.);
  state_view.bind(source.field_id(), source_state);
  state_view.bind(fields.multiplier, lambda_state);
  const EvaluationContext<VectorType> context(0., state_view);

  const auto multiplier_block =
    model.state_matrix_operator(fields.multiplier, source.field_id(), context);
  ASSERT_TRUE(multiplier_block.has_value());
  EXPECT_EQ(multiplier_block->matrix()->m(),
            multiplier_space.dof_handler.n_dofs());
  EXPECT_EQ(multiplier_block->matrix()->n(), source_space.dof_handler.n_dofs());
  EXPECT_GT(multiplier_block->matrix()->n_nonzero_elements(), 0u);
}
