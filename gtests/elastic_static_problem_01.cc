// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>

#include <cmath>
#include <filesystem>

#include "test_paths.h"

TEST(ElasticStaticProblem, ParameterParsingAndIsolation)
{
  using namespace ImmersX;

  dealii::ParameterAcceptor::clear();
  ElasticStaticParameters<2> left("/Left/Elastic static/");
  ElasticStaticParameters<2> right("/Right/Elastic static/");

  initialize_parameters_from_string(R"(
subsection Left
  subsection Elastic static
    set FE degree = 2
    set Dirichlet boundary ids = 0, 2
    set Neumann boundary ids = 1
    subsection Functions
      subsection Right hand side
        set Function expression = 1; 2
      end
      subsection Dirichlet boundary conditions
        set Function expression = 3; 4
      end
      subsection Neumann boundary conditions
        set Function expression = 5; 6
      end
    end
  end
end
subsection Right
  subsection Elastic static
    set FE degree = 1
    set Dirichlet boundary ids = 3
    set Neumann boundary ids = 0
    subsection Functions
      subsection Right hand side
        set Function expression = 7; 8
      end
      subsection Dirichlet boundary conditions
        set Function expression = 9; 10
      end
      subsection Neumann boundary conditions
        set Function expression = 11; 12
      end
    end
  end
end
  )");

  EXPECT_EQ(left.fe_degree, 2U);
  EXPECT_EQ(right.fe_degree, 1U);
  EXPECT_EQ(left.dirichlet_ids, (std::set<dealii::types::boundary_id>{0, 2}));
  EXPECT_EQ(right.dirichlet_ids, (std::set<dealii::types::boundary_id>{3}));
  EXPECT_EQ(left.neumann_ids, (std::set<dealii::types::boundary_id>{1}));
  EXPECT_EQ(right.neumann_ids, (std::set<dealii::types::boundary_id>{0}));

  dealii::Point<2> point(0.25, 0.5);
  EXPECT_EQ(left.rhs.value(point, 0), 1.);
  EXPECT_EQ(right.rhs.value(point, 0), 7.);
  EXPECT_EQ(left.bc.value(point, 1), 4.);
  EXPECT_EQ(right.bc.value(point, 1), 10.);
  EXPECT_EQ(left.neumann_bc.value(point, 0), 5.);
  EXPECT_EQ(right.neumann_bc.value(point, 0), 11.);
}

TEST(ElasticStaticProblem, MaterialPropertiesById)
{
  using namespace ImmersX;

  dealii::ParameterAcceptor::clear();
  ElasticStaticParameters<2> parameters;

  initialize_parameters_from_string(R"(
subsection Elastic static
  subsection Material properties
    set Material tags by material id = 1: stiff
    subsection default
      set Lame mu = 2
      set Lame lambda = 3
    end
    subsection stiff
      set Lame mu = 5
      set Lame lambda = 7
    end
  end
end
  )");

  EXPECT_DOUBLE_EQ(parameters.get_material_properties(0).Lame_mu, 2.);
  EXPECT_DOUBLE_EQ(parameters.get_material_properties(0).Lame_lambda, 3.);
  EXPECT_DOUBLE_EQ(parameters.get_material_properties(1).Lame_mu, 5.);
  EXPECT_DOUBLE_EQ(parameters.get_material_properties(1).Lame_lambda, 7.);
}

TEST(ElasticStaticProblem, FileMeshSetup)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  const auto grid_file = ImmersX::TestPaths::source_path(
    "data/elasticity/geometry/circle_1hole.msh");
  ImmersX::initialize_parameters_from_string(
    "subsection Elastic static\n"
    "  set Initial refinement = 0\n"
    "  set Dirichlet boundary ids =\n"
    "  set Neumann boundary ids =\n"
    "  subsection Grid generation\n"
    "    set Domain type = file\n"
    "    set Grid generator = " +
    grid_file.string() +
    "\n"
    "    set Grid generator arguments =\n"
    "\n"
    "  end\n"
    "end\n");

  Problem problem(parameters);
  problem.setup();

  EXPECT_GT(problem.triangulation().n_global_active_cells(), 0U);
}

TEST(ElasticStaticProblem, LinearAdapterSolve)
{
  using Problem      = ImmersX::ElasticStaticProblem<3, 3>;
  using FieldVector  = typename Problem::VectorType;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::LinearAdapter<FieldVector, GlobalVector>;

  ImmersX::ElasticStaticParameters<3> parameters;
  Problem                             problem(parameters);
  problem.setup();
  problem.set_forcing(
    dealii::Functions::ConstantFunction<3>(std::vector<double>{0., 0., 1.}));

  Adapter    adapter(MPI_COMM_WORLD,
                  [](const auto &operator_view,
                     const auto &rhs,
                     auto       &solution) {
                    dealii::SolverControl          control(1000, 1.e-10);
                    dealii::SolverCG<GlobalVector> solver(control);
                    dealii::PreconditionIdentity   preconditioner;
                    solver.solve(operator_view, solution, rhs, preconditioner);
                  });
  const auto fields = adapter.add(problem, "elasticity");

  auto state = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().displacement));

  EXPECT_TRUE(std::isfinite(problem.solution().l2_norm()));
  EXPECT_GT(problem.solution().l2_norm(), 0.);

  GlobalVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-8);
}

TEST(ElasticStaticProblem, ParsedBodyForceAndNonzeroDirichlet)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 1
  set Dirichlet boundary ids = 0
  subsection Functions
    subsection Right hand side
      set Function expression = 1; 2
    end
    subsection Dirichlet boundary conditions
      set Function expression = 1; 0
    end
  end
end
  )");

  Problem problem(parameters);
  problem.setup();

  bool has_nonzero_dirichlet = false;
  for (const auto &line : problem.constraints().get_lines())
    has_nonzero_dirichlet |= std::abs(line.inhomogeneity) > 1.e-14;

  EXPECT_TRUE(has_nonzero_dirichlet);
  EXPECT_GT(problem.forcing().l2_norm(), 0.);
}

TEST(ElasticStaticProblem, ParsedNeumannTraction)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 1
  set Dirichlet boundary ids =
  set Neumann boundary ids = 0
  subsection Functions
    subsection Dirichlet boundary conditions
      set Function expression = 0; 0
    end
    subsection Neumann boundary conditions
      set Function expression = 3; 0
    end
  end
end
  )");

  Problem problem(parameters);
  problem.setup();

  EXPECT_EQ(parameters.neumann_ids, (std::set<dealii::types::boundary_id>{0}));
  EXPECT_DOUBLE_EQ(parameters.neumann_bc.value(dealii::Point<2>(0.5, 1.), 0),
                   3.);
  unsigned int boundary_zero_faces = 0;
  for (const auto &cell : problem.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int face = 0; face < cell->n_faces(); ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 0)
          ++boundary_zero_faces;
  EXPECT_GT(boundary_zero_faces, 0U);
  EXPECT_GT(problem.forcing().l2_norm(), 0.);
}

TEST(ElasticStaticProblem, IndependentResidualOracle)
{
  using Problem      = ImmersX::ElasticStaticProblem<2>;
  using FieldVector  = typename Problem::VectorType;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::LinearAdapter<FieldVector, GlobalVector>;

  dealii::ParameterAcceptor::clear();
  ImmersX::ElasticStaticParameters<2> parameters;
  ImmersX::initialize_parameters_from_string("");
  Problem problem(parameters);
  problem.setup();
  problem.set_forcing(
    dealii::Functions::ConstantFunction<2>(std::vector<double>{0., 1.}));

  Adapter    adapter(MPI_COMM_WORLD,
                  [](const auto &operator_view,
                     const auto &rhs,
                     auto       &solution) {
                    dealii::SolverControl          control(1000, 1.e-10);
                    dealii::SolverCG<GlobalVector> solver(control);
                    dealii::PreconditionIdentity   preconditioner;
                    solver.solve(operator_view, solution, rhs, preconditioner);
                  });
  const auto fields = adapter.add(problem, "elastic-static-oracle");
  auto       state  = adapter.make_state();
  adapter.solve(state);
  problem.set_solution(adapter.field(state, fields.fields().displacement));

  FieldVector residual;
  residual.reinit(problem.solution());
  problem.stiffness_operator().vmult(residual, problem.solution());
  residual -= problem.forcing();
  EXPECT_LT(residual.l2_norm(), 1.e-8);
}

class ElasticStaticBackendTest : public ::testing::TestWithParam<std::string>
{};

TEST_P(ElasticStaticBackendTest, MPI_BackendSetupAndOutput)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  const auto output_directory =
    ImmersX::TestPaths::output_directory("elastic-static/" + GetParam());
  ImmersX::initialize_parameters_from_string("subsection Elastic static\n"
                                             "  set Initial refinement = 1\n"
                                             "  set Output directory = " +
                                             output_directory +
                                             "\n"
                                             "  subsection Grid generation\n"
                                             "    set Triangulation type = " +
                                             GetParam() +
                                             "\n"
                                             "  end\n"
                                             "end\n");

  Problem problem(parameters);
  problem.setup();

  EXPECT_GT(problem.triangulation().n_active_cells(), 0U);
  typename Problem::VectorType probe;
  probe.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  probe = 1.;
  typename Problem::VectorType image;
  image.reinit(probe);
  problem.stiffness_operator().vmult(image, probe);
  EXPECT_TRUE(std::isfinite(image.l2_norm()));

  problem.set_solution(probe);

  problem.output_results();
  MPI_Barrier(MPI_COMM_WORLD);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      ASSERT_TRUE(std::filesystem::exists(output_directory));
      bool has_vtk_output = false;
      for (const auto &entry :
           std::filesystem::directory_iterator(output_directory))
        has_vtk_output |= entry.path().extension() == ".vtu" ||
                          entry.path().extension() == ".pvtu";
      EXPECT_TRUE(has_vtk_output);
    }
}

INSTANTIATE_TEST_SUITE_P(TriangulationBackends,
                         ElasticStaticBackendTest,
                         ::testing::Values("distributed", "fullydistributed"));
