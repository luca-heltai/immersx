// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/algebra/lagrange_multiplier_interaction.h>

#include <cmath>
#include <vector>

using namespace ImmersX;
#include <immersx/algebra/lagrange_multiplier_schur_solver.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/tensor_product_space.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include "test_paths.h"


using namespace dealii;


#ifdef DEAL_II_WITH_VTK

TEST(TensorProductCoupling, ReducedLineToCylindricalSurface) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<3>    bulk_parameters("/Bulk Poisson/");
  PoissonParameters<1, 3> reduced_parameters("/Reduced Poisson/");
  TensorProductSpaceParameters<1, 2, 3, 1> tensor_parameters;
  ParticleCouplingParameters<3>            search_parameters;

  initialize_parameters_from_string(
    ImmersX::TestPaths::expand_configured_paths(R"(
    subsection Bulk Poisson
      set FE degree                   = 1
      set Initial refinement          = 2
      set Dirichlet boundary ids      = 0
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = 0: 1: false
      end
      subsection Right hand side
        set Function expression = 0
        set Variable names      = x,y,z,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,z,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 200
          set Reduction  = 1.e-12
          set Tolerance  = 1.e-14
          set Log result = false
        end
      end
    end
    subsection Reduced Poisson
      set FE degree                   = 1
      set Initial refinement          = 0
      set Dirichlet boundary ids      = 0
      subsection Grid generation
        set Grid generator           = @TEST_DATA_DIR@/tests/one_cylinder.vtk
        set Grid generator arguments = unused
      end
      subsection Right hand side
        set Function expression = 1
        set Variable names      = x,y,z,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,z,t
      end
      subsection Solver
        subsection Control
          set Max steps  = 200
          set Reduction  = 1.e-12
          set Tolerance  = 1.e-14
          set Log result = false
        end
      end
    end
  )"));

  bulk_parameters.output_directory =
    ImmersX::TestPaths::output_directory("tensor-product-cylindrical");
  reduced_parameters.output_directory =
    ImmersX::TestPaths::output_directory("tensor-product-cylindrical");
  reduced_parameters.name_of_grid =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  reduced_parameters.arguments_for_grid = "unused";

  tensor_parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  tensor_parameters.thickness                = "0.2";
  tensor_parameters.n_q_points               = 2;
  tensor_parameters.section.refinement_level = 1;

  PoissonSolver<3>    bulk_problem(bulk_parameters);
  PoissonSolver<1, 3> reduced_problem(reduced_parameters);

  bulk_problem.make_grid();
  bulk_problem.setup_fe();
  bulk_problem.setup_system();
  bulk_problem.assemble_system();

  reduced_problem.make_grid();
  reduced_problem.setup_fe();
  reduced_problem.setup_system();
  reduced_problem.assemble_system();

  TensorProductSpace<1, 2, 3, 1> tensor_space(tensor_parameters);
  tensor_space.initialize();

  IdentityRepresentation<3, 3> bulk_representation(
    bulk_problem.triangulation(),
    bulk_problem.dof_handler(),
    bulk_problem.locally_owned_dofs(),
    bulk_problem.locally_relevant_dofs(),
    bulk_problem.constraints());
  IdentityRepresentation<1, 3> reduced_representation(
    reduced_problem.triangulation(),
    reduced_problem.dof_handler(),
    reduced_problem.locally_owned_dofs(),
    reduced_problem.locally_relevant_dofs(),
    reduced_problem.constraints());
  auto cylindrical_representation =
    make_tensor_product_representation(reduced_representation, tensor_space);
  EXPECT_FALSE(
    cylindrical_representation.locally_owned_quadrature_points(QGauss<2>(2))
      .empty());

  LagrangeMultiplierInteraction<IdentityRepresentation<3, 3>,
                                TensorProductRepresentation<1, 2, 3, 1>>
    interaction(bulk_representation,
                cylindrical_representation,
                search_parameters);
  interaction.assemble();
  ASSERT_EQ(interaction.constraint_equation().contributions_view().size(), 2u);
  ASSERT_TRUE(interaction.constraint_equation().has_multiplier_metric());

  // The cylindrical representation is a reusable non-owning view.  A second
  // independent interaction can consume it without changing either existing
  // problem or the first interaction.
  PoissonSolver<3> second_bulk_problem(bulk_parameters);
  second_bulk_problem.make_grid();
  second_bulk_problem.setup_fe();
  second_bulk_problem.setup_system();
  second_bulk_problem.assemble_system();

  IdentityRepresentation<3, 3> second_bulk_representation(
    second_bulk_problem.triangulation(),
    second_bulk_problem.dof_handler(),
    second_bulk_problem.locally_owned_dofs(),
    second_bulk_problem.locally_relevant_dofs(),
    second_bulk_problem.constraints());
  LagrangeMultiplierInteraction<IdentityRepresentation<3, 3>,
                                TensorProductRepresentation<1, 2, 3, 1>>
    second_interaction(second_bulk_representation,
                       cylindrical_representation,
                       search_parameters);
  second_interaction.assemble();

  EXPECT_EQ(&interaction.second_representation(), &cylindrical_representation);
  EXPECT_EQ(&second_interaction.second_representation(),
            &cylindrical_representation);
  EXPECT_NE(&interaction.constraint_equation(),
            &second_interaction.constraint_equation());
  ASSERT_EQ(
    second_interaction.constraint_equation().contributions_view().size(), 2u);
  EXPECT_NE(
    interaction.constraint_equation().contributions_view()[0].matrix,
    second_interaction.constraint_equation().contributions_view()[0].matrix);
  EXPECT_GT(interaction.coupling_matrix().frobenius_norm(), 1.e-12);
  EXPECT_GT(second_interaction.coupling_matrix().frobenius_norm(), 1.e-12);

  using MatrixType = ImmersXLA::MPI::SparseMatrix;
  using VectorType = ImmersXLA::MPI::Vector;
  using AMGType    = ImmersXLA::MPI::PreconditionAMG;
  using SchurSolver =
    LagrangeMultiplierSchurSolver<MatrixType, VectorType, AMGType>;

  SchurSolver solver(bulk_problem.system_matrix(),
                     reduced_problem.system_matrix(),
                     interaction.coupling_matrix(),
                     interaction.multiplier_mass_matrix(),
                     bulk_problem.locally_owned_dofs(),
                     reduced_problem.locally_owned_dofs(),
                     interaction.multiplier_locally_owned_dofs(),
                     MPI_COMM_WORLD);

  VectorType bulk_solution;
  VectorType reduced_solution;
  VectorType multiplier;
  solver.solve(bulk_solution,
               reduced_solution,
               multiplier,
               bulk_problem.system_rhs(),
               reduced_problem.system_rhs());
  bulk_problem.set_solution(bulk_solution);
  reduced_problem.set_solution(reduced_solution);

  VectorType bulk_residual;
  VectorType reduced_residual;
  VectorType constraint_residual;
  VectorType temporary;
  bulk_residual.reinit(bulk_problem.locally_owned_dofs(), MPI_COMM_WORLD);
  reduced_residual.reinit(reduced_problem.locally_owned_dofs(), MPI_COMM_WORLD);
  constraint_residual.reinit(interaction.multiplier_locally_owned_dofs(),
                             MPI_COMM_WORLD);
  temporary.reinit(reduced_problem.locally_owned_dofs(), MPI_COMM_WORLD);

  bulk_problem.system_matrix().vmult(bulk_residual, bulk_problem.solution());
  interaction.coupling_matrix().vmult_add(bulk_residual, multiplier);
  bulk_residual -= bulk_problem.system_rhs();

  reduced_problem.system_matrix().vmult(reduced_residual,
                                        reduced_problem.solution());
  interaction.multiplier_mass_matrix().Tvmult(temporary, multiplier);
  reduced_residual -= temporary;
  reduced_residual -= reduced_problem.system_rhs();

  const std::vector<const VectorType *> states{&bulk_problem.solution(),
                                               &reduced_problem.solution()};
  interaction.constraint_equation().residual(states, constraint_residual);

  EXPECT_EQ(cylindrical_representation.representative_dimension, 1u);
  EXPECT_EQ(cylindrical_representation.support_dimension, 2u);
  EXPECT_EQ(
    cylindrical_representation.dof_handler().get_triangulation().dimension, 1u);
  EXPECT_EQ(interaction.multiplier_mass_matrix().m(), reduced_problem.n_dofs());
  EXPECT_TRUE(bulk_problem.solution_is_finite());
  EXPECT_TRUE(reduced_problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(multiplier.l2_norm()));
  EXPECT_GT(bulk_problem.solution_l2_norm(), 1.e-12);
  EXPECT_GT(reduced_problem.solution_l2_norm(), 1.e-12);
  EXPECT_LT(bulk_residual.l2_norm(), 1.e-8);
  EXPECT_LT(reduced_residual.l2_norm(), 1.e-8);
  EXPECT_LT(constraint_residual.l2_norm(), 1.e-8);
}

#endif // DEAL_II_WITH_VTK
