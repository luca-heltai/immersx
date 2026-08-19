// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "lagrange_multiplier_interaction.h"
#include "lagrange_multiplier_schur_solver.h"
#include "poisson.h"
#include "representation.h"
#include "utils.h"


using namespace dealii;


TEST(CoupledPoisson, MPI_RepresentationDrivenSchurSolve) // NOLINT
{
  ParameterAcceptor::clear();

  PoissonParameters<2>          bulk_parameters("/Bulk Poisson/");
  PoissonParameters<1, 2>       embedded_parameters("/Embedded Poisson/");
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
    subsection Embedded Poisson
      set FE degree                   = 1
      set Initial refinement          = 2
      set Dirichlet boundary ids      = 0,1
      subsection Grid generation
        set Grid generator           = hyper_cube
        set Grid generator arguments = -1: 1: false
      end
      subsection Right hand side
        set Function expression = 1
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

  const auto output_directory =
    (std::filesystem::temp_directory_path() / "immersx_coupled_poisson_01")
      .string();
  bulk_parameters.output_directory     = output_directory;
  embedded_parameters.output_directory = output_directory;
  bulk_parameters.output_name          = "coupled_bulk";
  embedded_parameters.output_name      = "coupled_embedded";

  PoissonSolver<2>    bulk_problem(bulk_parameters);
  PoissonSolver<1, 2> embedded_problem(embedded_parameters);

  // Keep this first proving ground fixed-mesh. Adaptivity and representation
  // transfer are deliberately outside this prototype.
  bulk_problem.make_grid();
  bulk_problem.setup_fe();
  bulk_problem.setup_system();
  bulk_problem.assemble_system();

  embedded_problem.make_grid();
  embedded_problem.setup_fe();
  embedded_problem.setup_system();
  embedded_problem.assemble_system();

  IdentityRepresentation<2, 2> bulk_representation(
    bulk_problem.triangulation(),
    bulk_problem.dof_handler(),
    bulk_problem.locally_owned_dofs(),
    bulk_problem.locally_relevant_dofs(),
    bulk_problem.constraints());
  IdentityRepresentation<1, 2> embedded_representation(
    embedded_problem.triangulation(),
    embedded_problem.dof_handler(),
    embedded_problem.locally_owned_dofs(),
    embedded_problem.locally_relevant_dofs(),
    embedded_problem.constraints());

  LagrangeMultiplierInteraction<IdentityRepresentation<2, 2>,
                                IdentityRepresentation<1, 2>>
    interaction(bulk_representation,
                embedded_representation,
                search_parameters);
  interaction.assemble();
  ASSERT_EQ(interaction.constraint_equation().contributions_view().size(), 2u);
  ASSERT_TRUE(interaction.constraint_equation().has_multiplier_metric());

  using MatrixType = ImmersXLA::MPI::SparseMatrix;
  using VectorType = ImmersXLA::MPI::Vector;
  using AMGType    = ImmersXLA::MPI::PreconditionAMG;
  using SchurSolver =
    LagrangeMultiplierSchurSolver<MatrixType, VectorType, AMGType>;

  SchurSolver solver(bulk_problem.system_matrix(),
                     embedded_problem.system_matrix(),
                     interaction.coupling_matrix(),
                     interaction.multiplier_mass_matrix(),
                     bulk_problem.locally_owned_dofs(),
                     embedded_problem.locally_owned_dofs(),
                     interaction.multiplier_locally_owned_dofs(),
                     MPI_COMM_WORLD);

  VectorType bulk_solution;
  VectorType embedded_solution;
  VectorType multiplier;
  solver.solve(bulk_solution,
               embedded_solution,
               multiplier,
               bulk_problem.system_rhs(),
               embedded_problem.system_rhs());

  bulk_problem.set_solution(bulk_solution);
  embedded_problem.set_solution(embedded_solution);

  VectorType bulk_residual;
  VectorType embedded_residual;
  VectorType multiplier_residual;
  VectorType temporary;
  bulk_residual.reinit(bulk_problem.locally_owned_dofs(), MPI_COMM_WORLD);
  embedded_residual.reinit(embedded_problem.locally_owned_dofs(),
                           MPI_COMM_WORLD);
  multiplier_residual.reinit(interaction.multiplier_locally_owned_dofs(),
                             MPI_COMM_WORLD);
  temporary.reinit(embedded_problem.locally_owned_dofs(), MPI_COMM_WORLD);

  bulk_problem.system_matrix().vmult(bulk_residual, bulk_problem.solution());
  interaction.coupling_matrix().vmult_add(bulk_residual, multiplier);
  bulk_residual -= bulk_problem.system_rhs();

  embedded_problem.system_matrix().vmult(embedded_residual,
                                         embedded_problem.solution());
  interaction.multiplier_mass_matrix().Tvmult(temporary, multiplier);
  embedded_residual -= temporary;
  embedded_residual -= embedded_problem.system_rhs();

  const std::vector<const VectorType *> states{&bulk_problem.solution(),
                                               &embedded_problem.solution()};
  interaction.constraint_equation().residual(states, multiplier_residual);

  EXPECT_TRUE(bulk_problem.solution_is_finite());
  EXPECT_TRUE(embedded_problem.solution_is_finite());
  EXPECT_TRUE(std::isfinite(multiplier.l2_norm()));
  EXPECT_GT(bulk_problem.solution_l2_norm(), 1.e-12);
  EXPECT_GT(embedded_problem.solution_l2_norm(), 1.e-12);
  EXPECT_LT(bulk_residual.l2_norm(), 1.e-8);
  EXPECT_LT(embedded_residual.l2_norm(), 1.e-8);
  EXPECT_LT(multiplier_residual.l2_norm(), 1.e-8);

  // The externally computed states remain valid Poisson states, including
  // ghost updates used by the existing output path.
  bulk_problem.output_results();
  embedded_problem.output_results();
}
