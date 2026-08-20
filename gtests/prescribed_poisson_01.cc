// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>
#include <immersx/algebra/lagrange_multiplier_constraint_solver.h>

#include <cmath>
#include <vector>

using namespace ImmersX;
#include <immersx/algebra/lagrange_multiplier_interaction.h>
#include <immersx/core/representation.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>


using namespace dealii;


TEST(PrescribedPoisson, ReducedPoissonReplacement) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<2>          bulk_parameters("/Bulk Poisson/");
  ParticleCouplingParameters<2> search_parameters;

  initialize_parameters_from_string(R"(
    subsection Bulk Poisson
      set FE degree                   = 1
      set Initial refinement          = 2
      set Dirichlet boundary ids      = 0
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 0
        set Variable names      = x,y,t
      end
      subsection Dirichlet boundary conditions
        set Function expression = 0
        set Variable names      = x,y,t
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
  )");

  PoissonSolver<2> bulk_problem(bulk_parameters);
  bulk_problem.make_grid();
  bulk_problem.setup_fe();
  bulk_problem.setup_system();
  bulk_problem.assemble_system();

  // This is a representation-only line mesh.  It is not a second Problem and
  // it has no PDE operator or state.  Its coefficients are the prescribed
  // target data for the immersed constraint.
  Triangulation<1, 2>      serial_line;
  std::vector<Point<2>>    line_vertices{Point<2>(-0.75, 0.13),
                                      Point<2>(0.75, 0.13)};
  std::vector<CellData<1>> line_cells(1);
  line_cells[0].vertices[0] = 0;
  line_cells[0].vertices[1] = 1;
  SubCellData line_subcells;
  serial_line.create_triangulation(line_vertices, line_cells, line_subcells);

  parallel::fullydistributed::Triangulation<1, 2> line_tria(MPI_COMM_WORLD);
  line_tria.copy_triangulation(serial_line);

  FE_DGQ<1, 2>     line_fe(0);
  DoFHandler<1, 2> line_dh(line_tria);
  line_dh.distribute_dofs(line_fe);
  const auto line_owned    = line_dh.locally_owned_dofs();
  const auto line_relevant = DoFTools::extract_locally_relevant_dofs(line_dh);
  AffineConstraints<double> line_constraints;
  line_constraints.reinit(line_owned, line_relevant);
  line_constraints.close();

  IdentityRepresentation<2, 2> bulk_representation(
    bulk_problem.triangulation(),
    bulk_problem.dof_handler(),
    bulk_problem.locally_owned_dofs(),
    bulk_problem.locally_relevant_dofs(),
    bulk_problem.constraints());
  IdentityRepresentation<1, 2> line_representation(
    line_tria, line_dh, line_owned, line_relevant, line_constraints);

  LagrangeMultiplierInteraction<IdentityRepresentation<2, 2>,
                                IdentityRepresentation<1, 2>>
    interaction(bulk_representation, line_representation, search_parameters);
  interaction.assemble();

  ImmersXLA::MPI::Vector prescribed_coefficients;
  prescribed_coefficients.reinit(line_owned, MPI_COMM_WORLD);
  prescribed_coefficients = 1.;
  PrescribedFieldDatum<IdentityRepresentation<1, 2>> datum(
    line_representation, prescribed_coefficients);

  const auto constraint_equation =
    interaction.prescribed_constraint_equation(datum);
  const auto &constraint_rhs = constraint_equation.rhs();
  ASSERT_EQ(constraint_equation.contributions_view().size(), 1u);
  ASSERT_TRUE(constraint_equation.has_multiplier_metric());
  ASSERT_GT(interaction.coupling_matrix().frobenius_norm(), 1.e-12);
  ASSERT_GT(interaction.multiplier_mass_matrix().frobenius_norm(), 1.e-12);
  ASSERT_GT(constraint_rhs.l2_norm(), 1.e-12);

  using MatrixType = ImmersXLA::MPI::SparseMatrix;
  using VectorType = ImmersXLA::MPI::Vector;
  using AMGType    = ImmersXLA::MPI::PreconditionAMG;
  using Solver =
    LagrangeMultiplierConstraintSolver<MatrixType, VectorType, AMGType>;

  Solver constrained_solver(bulk_problem.system_matrix(),
                            interaction.coupling_matrix(),
                            bulk_problem.locally_owned_dofs(),
                            interaction.multiplier_locally_owned_dofs(),
                            MPI_COMM_WORLD);

  VectorType bulk_solution;
  VectorType multiplier;
  constrained_solver.solve(bulk_solution,
                           multiplier,
                           bulk_problem.system_rhs(),
                           constraint_rhs);
  bulk_problem.set_solution(bulk_solution);

  VectorType bulk_residual;
  VectorType constraint_residual;
  bulk_residual.reinit(bulk_problem.locally_owned_dofs(), MPI_COMM_WORLD);
  constraint_residual.reinit(interaction.multiplier_locally_owned_dofs(),
                             MPI_COMM_WORLD);
  bulk_problem.system_matrix().vmult(bulk_residual, bulk_problem.solution());
  interaction.coupling_matrix().vmult_add(bulk_residual, multiplier);
  bulk_residual -= bulk_problem.system_rhs();
  const std::vector<const VectorType *> states{&bulk_problem.solution()};
  constraint_equation.residual(states, constraint_residual);

  EXPECT_TRUE(bulk_problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(multiplier.l2_norm()));
  EXPECT_GT(bulk_problem.solution_l2_norm(), 1.e-12);
  EXPECT_LT(bulk_residual.l2_norm(), 1.e-8);
  EXPECT_LT(constraint_residual.l2_norm(), 1.e-8);
}
