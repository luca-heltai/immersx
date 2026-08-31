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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_rhs.templates.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <sstream>
#include <vector>

#include "test_paths.h"

namespace
{
  void
  check_distributed_refinement()
  {
    using Parameters = ImmersX::ElasticStaticParameters<2>;
    using Problem    = ImmersX::ElasticStaticProblem<2>;

    dealii::ParameterAcceptor::clear();
    Parameters parameters;
    ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 0
  subsection Grid generation
    set Triangulation type = distributed
  end
end
)");

    Problem problem(parameters);
    problem.setup();
    const auto n_cells_before = problem.triangulation().n_global_active_cells();
    const auto n_dofs_before  = problem.dof_handler().n_dofs();
    Problem::VectorType previous_solution;
    previous_solution.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    previous_solution = 1.;
    problem.set_solution(previous_solution);

    problem.refine_global();

    EXPECT_GT(problem.triangulation().n_global_active_cells(), n_cells_before);
    EXPECT_GT(problem.dof_handler().n_dofs(), n_dofs_before);
    EXPECT_EQ(problem.forcing().locally_owned_elements(),
              problem.locally_owned_dofs());
    EXPECT_EQ(problem.solution().locally_owned_elements(),
              problem.locally_owned_dofs());
    EXPECT_EQ(problem.solution().l2_norm(), 0.);

    Problem::VectorType probe;
    probe.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    probe = 1.;
    Problem::VectorType image;
    image.reinit(probe);
    problem.stiffness_operator().vmult(image, probe);
    EXPECT_TRUE(std::isfinite(image.l2_norm()));
  }

  void
  check_fullydistributed_refinement()
  {
    using Parameters = ImmersX::ElasticStaticParameters<2>;
    using Problem    = ImmersX::ElasticStaticProblem<2>;

    dealii::ParameterAcceptor::clear();
    Parameters parameters;
    ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 0
  subsection Grid generation
    set Triangulation type = fullydistributed
  end
end
)");

    Problem problem(parameters);
    problem.setup();

    EXPECT_THROW(problem.refine_global(), dealii::ExceptionBase);
  }
} // namespace

TEST(ElasticStaticProblem, ParameterParsingAndOverrides)
{
  using namespace ImmersX;

  dealii::ParameterAcceptor::clear();
  ElasticStaticParameters<2> parameters;

  const auto parameter_text = R"(
subsection Elastic static
  set Dirichlet boundary ids = 0, 3
  set Neumann boundary ids = 1, 2
  set Rhs material ids = 4, 7
  subsection Functions
    subsection Right hand side
      set Function expression = 1; 2
    end
    subsection Right hand side 4
      set Function expression = x; 0
    end
    subsection Right hand side 7
      set Function expression = 0; y
    end
    subsection Dirichlet boundary conditions
      set Function expression = 0; 0
    end
    subsection Dirichlet boundary conditions 0
      set Function expression = x; 0
    end
    subsection Dirichlet boundary conditions 3
      set Function expression = 0; y
    end
    subsection Neumann boundary conditions
      set Function expression = 0; 0
    end
    subsection Neumann boundary conditions 1
      set Function expression = 2; 0
    end
    subsection Neumann boundary conditions 2
      set Function expression = 0; 3
    end
    subsection Exact solution
      set Function expression = x; y
    end
  end
end
  )";

  initialize_parameters_from_string(parameter_text);

  const dealii::Point<2> point(0.25, 0.5);
  EXPECT_EQ(parameters.get_dirichlet_bc(0).value(point, 0), 0.25);
  EXPECT_EQ(parameters.get_dirichlet_bc(3).value(point, 1), 0.5);
  EXPECT_EQ(parameters.get_neumann_bc(1).value(point, 0), 2.);
  EXPECT_EQ(parameters.get_neumann_bc(2).value(point, 1), 3.);
  EXPECT_EQ(parameters.get_rhs(4).value(point, 0), 0.25);
  EXPECT_EQ(parameters.get_rhs(7).value(point, 1), 0.5);
  EXPECT_EQ(parameters.get_dirichlet_bc(1).value(point, 0), 0.);
  EXPECT_EQ(parameters.get_neumann_bc(3).value(point, 0), 0.);
  EXPECT_EQ(parameters.get_rhs(0).value(point, 0), 1.);
  EXPECT_EQ(parameters.get_rhs(0).value(point, 1), 2.);
  EXPECT_EQ(parameters.exact_solution.value(point, 0), 0.25);
  EXPECT_EQ(parameters.exact_solution.value(point, 1), 0.5);

  initialize_parameters_from_string(parameter_text);

  EXPECT_EQ(parameters.get_dirichlet_bc(0).value(point, 0), 0.25);
  EXPECT_EQ(parameters.get_neumann_bc(2).value(point, 1), 3.);
  EXPECT_EQ(parameters.get_rhs(7).value(point, 1), 0.5);
}

TEST(ElasticStaticProblem, ParsesGeneratedConvergenceParameterFixture)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  ImmersX::initialize_parameters(
    ImmersX::TestPaths::parameter_path("gtests/parameters/elastic_static.prm"));

  const dealii::Point<2> point(0.25, 0.5);
  EXPECT_EQ(parameters.initial_refinement, 0U);
  EXPECT_EQ(parameters.n_refinement_cycles, 2U);
  EXPECT_EQ(parameters.dirichlet_ids,
            (std::set<dealii::types::boundary_id>{0, 2, 3}));
  EXPECT_EQ(parameters.neumann_ids, (std::set<dealii::types::boundary_id>{1}));
  EXPECT_EQ(parameters.triangulation_type, "distributed");
  EXPECT_EQ(parameters.get_dirichlet_bc(0).value(point, 0), 0.25);
  EXPECT_EQ(parameters.get_dirichlet_bc(2).value(point, 1), 0.5);
  EXPECT_EQ(parameters.get_neumann_bc(1).value(point, 1), 1.);
  EXPECT_EQ(parameters.rhs.value(point, 0), 1.);
  EXPECT_EQ(parameters.exact_solution.value(point, 1), 0.5);
  EXPECT_EQ(parameters.get_material_properties(1).Lame_mu, 5.);
  EXPECT_EQ(parameters.get_material_properties(1).Lame_lambda, 7.);
}

TEST(ElasticStaticProblem, ExactSolutionAndErrorObservation)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  Parameters other_parameters("/Other elastic static/");
  ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 1
  set Dirichlet boundary ids =
  subsection Functions
    subsection Exact solution
      set Function expression = x; y
    end
  end
end

subsection Other elastic static
  subsection Functions
    subsection Exact solution
      set Function expression = x; y
    end
  end
end
  )");

  const dealii::Point<2> point(0.25, 0.5);
  EXPECT_DOUBLE_EQ(parameters.exact_solution.value(point, 0), 0.25);
  EXPECT_DOUBLE_EQ(parameters.exact_solution.value(point, 1), 0.5);
  EXPECT_DOUBLE_EQ(other_parameters.exact_solution.value(point, 0), 0.25);
  EXPECT_DOUBLE_EQ(other_parameters.exact_solution.value(point, 1), 0.5);

  Problem problem(parameters);
  problem.setup();

  Problem::VectorType interpolated;
  interpolated.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  dealii::VectorTools::interpolate(problem.dof_handler(),
                                   parameters.exact_solution,
                                   interpolated);
  interpolated.compress(dealii::VectorOperation::insert);
  problem.set_solution(interpolated);

  dealii::ParsedConvergenceTable table(
    std::vector<std::string>(2, "displacement"),
    {{dealii::VectorTools::L2_norm, dealii::VectorTools::H1_seminorm}});
  problem.compute_error(table);

  std::ostringstream table_output;
  table.output_table(table_output);
  EXPECT_NE(table_output.str().find("displacement_L2_norm"), std::string::npos);
  EXPECT_NE(table_output.str().find("displacement_H1_seminorm"),
            std::string::npos);

  const dealii::Functions::IdentityFunction<2> exact_displacement;
  dealii::Vector<double>                       difference_per_cell(
    problem.triangulation().n_global_active_cells());
  dealii::VectorTools::integrate_difference(problem.dof_handler(),
                                            interpolated,
                                            exact_displacement,
                                            difference_per_cell,
                                            dealii::QGauss<2>(3),
                                            dealii::VectorTools::L2_norm);
  const auto l2_error =
    dealii::VectorTools::compute_global_error(problem.triangulation(),
                                              difference_per_cell,
                                              dealii::VectorTools::L2_norm);

  dealii::VectorTools::integrate_difference(problem.dof_handler(),
                                            interpolated,
                                            exact_displacement,
                                            difference_per_cell,
                                            dealii::QGauss<2>(3),
                                            dealii::VectorTools::H1_seminorm);
  const auto h1_seminorm_error =
    dealii::VectorTools::compute_global_error(problem.triangulation(),
                                              difference_per_cell,
                                              dealii::VectorTools::H1_seminorm);

  EXPECT_LT(l2_error, 1.e-12);
  EXPECT_LT(h1_seminorm_error, 1.e-12);
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
  const auto &adapter_solution =
    adapter.field(state, fields.fields().displacement);
  problem.set_solution(adapter_solution);

  FieldVector accepted_difference;
  accepted_difference.reinit(problem.solution());
  accepted_difference = problem.solution();
  accepted_difference -= adapter_solution;
  EXPECT_LT(accepted_difference.l2_norm(), 1.e-12);

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

TEST(ElasticStaticProblem, BoundarySpecificDirichletConstraints)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 1
  set Dirichlet boundary ids = 0, 3
  subsection Grid generation
    set Grid generator arguments = 0: 1: true
  end
  subsection Functions
    subsection Dirichlet boundary conditions
      set Function expression = 0; 0
    end
    subsection Dirichlet boundary conditions 0
      set Function expression = x; 0
    end
    subsection Dirichlet boundary conditions 3
      set Function expression = 0; y
    end
  end
end
  )");

  Problem problem(parameters);
  problem.setup();

  const auto support_points =
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<2>(),
                                                 problem.dof_handler());
  std::map<dealii::types::global_dof_index, unsigned int> component_by_dof;
  std::vector<dealii::types::global_dof_index>            local_dof_indices(
    problem.dof_handler().get_fe().n_dofs_per_cell());
  for (const auto &cell : problem.dof_handler().active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        component_by_dof.emplace(
          local_dof_indices[i],
          problem.dof_handler().get_fe().system_to_component_index(i).first);
    }

  bool found_boundary_zero_x  = false;
  bool found_boundary_three_y = false;
  for (const auto &line : problem.constraints().get_lines())
    {
      const auto &point     = support_points.at(line.index);
      const auto  component = component_by_dof.at(line.index);
      if (component == 0 && std::abs(point[0]) < 1.e-12 && point[1] > 1.e-12 &&
          point[1] < 1. - 1.e-12)
        {
          EXPECT_NEAR(line.inhomogeneity, point[0], 1.e-12);
          EXPECT_GT(std::abs(line.inhomogeneity - point[1]), 1.e-12);
          found_boundary_zero_x = true;
        }
      if (component == 1 && std::abs(point[1] - 1.) < 1.e-12 &&
          point[0] > 1.e-12 && point[0] < 1. - 1.e-12)
        {
          EXPECT_NEAR(line.inhomogeneity, point[1], 1.e-12);
          EXPECT_GT(std::abs(line.inhomogeneity - point[0]), 1.e-12);
          found_boundary_three_y = true;
        }
    }

  EXPECT_TRUE(found_boundary_zero_x);
  EXPECT_TRUE(found_boundary_three_y);
}

TEST(ElasticStaticProblem, BoundarySpecificNeumannOracle)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  ImmersX::initialize_parameters_from_string(R"(
subsection Elastic static
  set Initial refinement = 1
  set Dirichlet boundary ids =
  set Neumann boundary ids = 1, 3
  subsection Grid generation
    set Grid generator arguments = 0: 1: true
  end
  subsection Functions
    subsection Right hand side
      set Function expression = 0; 0
    end
    subsection Neumann boundary conditions
      set Function expression = 0; 0
    end
    subsection Neumann boundary conditions 1
      set Function expression = 2; 0
    end
    subsection Neumann boundary conditions 3
      set Function expression = 0; 3
    end
  end
end
  )");

  Problem problem(parameters);
  problem.setup();

  Problem::VectorType expected;
  expected.reinit(problem.forcing());
  expected = 0.;
  for (const auto boundary_id : parameters.neumann_ids)
    {
      Problem::VectorType contribution;
      contribution.reinit(problem.forcing());
      contribution = 0.;
      dealii::VectorTools::create_boundary_right_hand_side(
        problem.dof_handler(),
        dealii::QGauss<1>(2),
        parameters.get_neumann_bc(boundary_id),
        contribution,
        {boundary_id});
      contribution *= parameters.get_neumann_bc(boundary_id).scale(0.);
      expected += contribution;
    }
  expected.compress(dealii::VectorOperation::add);

  expected -= problem.forcing();
  EXPECT_LT(expected.linfty_norm(), 1.e-11);
}

TEST(ElasticStaticProblem, MaterialSpecificRHS)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  const auto mesh_file = ImmersX::TestPaths::source_path(
    "data/tests/elastic_static_two_materials.msh");
  ImmersX::initialize_parameters_from_string(
    "subsection Elastic static\n"
    "  set Initial refinement = 0\n"
    "  set Dirichlet boundary ids =\n"
    "  set Neumann boundary ids =\n"
    "  set Rhs material ids = 0, 1\n"
    "  subsection Grid generation\n"
    "    set Domain type = file\n"
    "    set Grid generator = " +
    mesh_file.string() +
    "\n"
    "    set Grid generator arguments =\n"
    "  end\n"
    "  subsection Functions\n"
    "    subsection Right hand side\n"
    "      set Function expression = 3; 4\n"
    "    end\n"
    "    subsection Right hand side 0\n"
    "      set Function expression = 1; 0\n"
    "    end\n"
    "    subsection Right hand side 1\n"
    "      set Function expression = 0; 2\n"
    "    end\n"
    "  end\n"
    "end\n");

  Problem problem(parameters);
  problem.setup();

  Problem::VectorType expected;
  expected.reinit(problem.forcing());
  expected = 0.;
  const dealii::QGauss<2> quadrature(2);
  dealii::FEValues<2>     fe_values(problem.dof_handler().get_fe(),
                                quadrature,
                                dealii::update_values |
                                  dealii::update_JxW_values);
  std::vector<dealii::types::global_dof_index> local_dof_indices(
    problem.dof_handler().get_fe().n_dofs_per_cell());
  for (const auto &cell : problem.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        ASSERT_LT(cell->material_id(), 2U);
        const std::array<double, 2> rhs = cell->material_id() == 0 ?
                                            std::array<double, 2>{{1., 0.}} :
                                            std::array<double, 2>{{0., 2.}};
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        dealii::Vector<double> cell_rhs(local_dof_indices.size());
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
            {
              const auto component = problem.dof_handler()
                                       .get_fe()
                                       .system_to_component_index(i)
                                       .first;
              cell_rhs(i) += fe_values.shape_value_component(i, q, component) *
                             rhs[component] * fe_values.JxW(q);
            }
        expected.add(local_dof_indices, cell_rhs);
      }
  expected.compress(dealii::VectorOperation::add);

  expected -= problem.forcing();
  EXPECT_LT(expected.linfty_norm(), 1.e-11);
  EXPECT_DOUBLE_EQ(parameters.get_rhs(2).value(dealii::Point<2>(), 0), 3.);
  EXPECT_DOUBLE_EQ(parameters.get_rhs(2).value(dealii::Point<2>(), 1), 4.);
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

TEST(ElasticStaticProblem, DistributedRefinement)
{
  check_distributed_refinement();
}

TEST(ElasticStaticProblem, FullyDistributedRefinement)
{
  check_fullydistributed_refinement();
}

TEST(ElasticStaticProblem, MPI_DistributedRefinement)
{
  check_distributed_refinement();
}

TEST(ElasticStaticProblem, MPI_FullyDistributedRefinement)
{
  check_fullydistributed_refinement();
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

  problem.output_results(0);
  MPI_Barrier(MPI_COMM_WORLD);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      ASSERT_TRUE(std::filesystem::exists(output_directory));
      EXPECT_TRUE(
        std::filesystem::exists(output_directory + "/elastic_static.pvd"));
      bool has_vtk_output = false;
      for (const auto &entry :
           std::filesystem::directory_iterator(output_directory))
        has_vtk_output |= (entry.path().stem() == "elastic_static_0" &&
                           (entry.path().extension() == ".vtu" ||
                            entry.path().extension() == ".pvtu"));
      EXPECT_TRUE(has_vtk_output);
    }
}

TEST(ElasticStaticProblem, MPI_DistributedOutputRecordsEveryRefinementCycle)
{
  using Parameters = ImmersX::ElasticStaticParameters<2>;
  using Problem    = ImmersX::ElasticStaticProblem<2>;

  dealii::ParameterAcceptor::clear();
  Parameters parameters;
  const auto output_directory =
    ImmersX::TestPaths::output_directory("elastic-static/distributed-cycles");
  ImmersX::initialize_parameters_from_string(
    "subsection Elastic static\n"
    "  set Initial refinement = 0\n"
    "  set Output directory = " +
    output_directory +
    "\n"
    "  subsection Grid generation\n"
    "    set Triangulation type = distributed\n"
    "  end\n"
    "end\n");

  Problem problem(parameters);
  problem.setup();
  problem.output_results(0);
  problem.refine_global();
  problem.output_results(1);

  MPI_Barrier(MPI_COMM_WORLD);
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      ASSERT_TRUE(std::filesystem::exists(output_directory));
      EXPECT_TRUE(
        std::filesystem::exists(output_directory + "/elastic_static.pvd"));
      for (const auto cycle : {0U, 1U})
        {
          const auto filename =
            output_directory + "/elastic_static_" + std::to_string(cycle);
          EXPECT_TRUE(std::filesystem::exists(filename + ".vtu") ||
                      std::filesystem::exists(filename + ".pvtu"));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TriangulationBackends,
                         ElasticStaticBackendTest,
                         ::testing::Values("distributed", "fullydistributed"));
