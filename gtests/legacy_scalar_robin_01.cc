// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_parser.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/laplacian.h>
#include <immersx/physics/reduced_poisson.h>

#include <set>
#include <sstream>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  template <int dim>
  std::string
  scalar_robin_parameters(const std::string &section,
                          const bool         mixed,
                          const bool         simplex     = false,
                          const double       global_beta = 2.)
  {
    const std::string exact     = dim == 2 ? "1+x+2*y" : "1+x+2*y+3*z";
    const std::string variables = dim == 2 ? "x,y,t" : "x,y,z,t";
    const std::string beta =
      global_beta == 0. ? "0" : std::to_string(global_beta) + "+x";
    const std::string flux =
      "if(x<1.e-8,-1,0)+if(x>1-1.e-8,1,0)+"
      "if(y<1.e-8,-2,0)+if(y>1-1.e-8,2,0)" +
      std::string(dim == 3 ? "+if(z<1.e-8,-3,0)+if(z>1-1.e-8,3,0)" : "");
    std::ostringstream p;
    p << "subsection " << section
      << "\n set Dirichlet boundary ids = " << (mixed ? "0" : "")
      << "\n set Robin boundary ids = " << (mixed ? "1,2,3" : "0")
      << "\n subsection "
      << (section == "Immersed Problem" ? "Domain" : "Grid generation")
      << "\n set Triangulation type = "
      << (simplex ? "fullydistributed" : "distributed")
      << "\n set Initial refinement = " << (simplex ? "0" : "1")
      << "\n set Grid generator = "
      << (simplex ?
            TestPaths::data_filename(dim == 2 ? "tests/simplex_square.msh" :
                                                "tests/simplex_cube.msh") :
            "hyper_cube")
      << "\n set Grid generator arguments = 0:1:" << (mixed ? "true" : "false")
      << "\n end\n subsection Dirichlet boundary conditions\n"
      << " set Variable names = " << variables
      << "\n set Function expression = " << exact
      << "\n end\n subsection Functions\n subsection Robin coefficient\n"
      << " set Variable names = " << variables
      << "\n set Function expression = " << beta
      << "\n end\n subsection Robin boundary conditions\n"
      << " set Variable names = " << variables
      << "\n set Function expression = (" << flux << ")+(" << beta << ")*("
      << exact << ")"
      << "\n end\n";
    if (mixed)
      p << "subsection Robin coefficient 1\n set Function constants = a=3\n set Function expression = a+y\n end\n"
        << "subsection Robin boundary conditions 1\n set Function expression = 1+(3+y)*("
        << exact << ")\n end\n";
    p << "end\n";
    if (section == "Immersed Problem")
      p << "subsection Immersed inclusions\n set Inclusions =\n set Inclusions file =\n subsection Boundary data\n set Function expression = 0\n end\n end\n";
    p << "end\n";
    return p.str();
  }

  template <int dim, typename Problem>
  void
  check_scalar_solution(const Problem &problem, const bool mixed)
  {
    FunctionParser<dim> exact;
    exact.initialize(dim == 2 ? "x,y" : "x,y,z",
                     dim == 2 ? "1+x+2*y" : "1+x+2*y+3*z",
                     {});
    auto expected = problem.bulk_values();
    expected      = 0.;
    VectorTools::interpolate(
      compatibility::linear_mapping<dim>(
        problem.bulk_dof_handler().get_fe().reference_cell()),
      problem.bulk_dof_handler(),
      exact,
      expected);
    auto error = problem.bulk_values();
    error -= expected;
    EXPECT_LT(error.l2_norm(), 1.e-9 * std::max(1., expected.l2_norm()));
    auto residual = problem.bulk_rhs();
    problem.assembled_bulk().vmult(residual, expected);
    residual -= problem.bulk_rhs();
    // Only free physical rows are equations; constrained rows store diagonals.
    if (mixed)
      {
        const auto boundary =
          DoFTools::extract_boundary_dofs(problem.bulk_dof_handler(),
                                          ComponentMask(),
                                          {0});
        for (const auto i : residual.locally_owned_elements())
          if (boundary.is_element(i))
            residual[i] = 0.;
        residual.compress(VectorOperation::insert);
      }
    EXPECT_LT(residual.l2_norm(),
              1.e-10 * std::max(1., problem.bulk_rhs().l2_norm()));
  }

  template <int dim, typename Parameters>
  void
  set_accuracy(Parameters &p)
  {
    p.inner_control.set_max_steps(1000);
    p.inner_control.set_tolerance(1.e-13);
    p.inner_control.set_reduction(1.e-13);
    p.outer_control.set_max_steps(1000);
    p.outer_control.set_tolerance(1.e-13);
    p.outer_control.set_reduction(1.e-13);
  }

  template <int dim>
  void
  check_laplacian_robin(const bool mixed, const bool simplex = false)
  {
    ParameterAcceptor::clear();
    ProblemParameters<dim> p;
    PoissonProblem<dim>    problem(p);
    initialize_parameters_from_string(
      scalar_robin_parameters<dim>("Immersed Problem", mixed, simplex));
    set_accuracy<dim>(p);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    problem.assemble_poisson_system();
    problem.assemble_coupling();
    problem.solve();
    check_scalar_solution<dim>(problem, mixed);
    if (mixed)
      for (const double beta : {4., 0.})
        {
          initialize_parameters_from_string(scalar_robin_parameters<dim>(
            "Immersed Problem", true, false, beta));
          problem.assemble_poisson_system();
          problem.assemble_coupling();
          problem.solve();
          check_scalar_solution<dim>(problem, true);
          EXPECT_NEAR(p.robin_coefficient_for(1).value(Point<dim>()),
                      3.,
                      1.e-14);
          EXPECT_NEAR(p.robin_coefficient_for(2).value(Point<dim>()),
                      beta,
                      1.e-14);
        }
  }

  template <int dim>
  void
  check_reduced_robin(const bool mixed, const bool simplex = false)
  {
    ParameterAcceptor::clear();
    ReducedPoissonParameters<dim, 0> p;
    initialize_parameters_from_string(
      scalar_robin_parameters<dim>("Reduced Poisson", mixed, simplex));
    auto      &coupling = p.reduced_coupling_parameters;
    auto      &tensor   = coupling.tensor_product_space_parameters;
    Point<dim> exterior;
    for (unsigned int d = 0; d < dim; ++d)
      exterior[d] = 2.;
    tensor.point_cloud.points.push_back(exterior);
    tensor.thickness                                            = ".05";
    tensor.n_q_points                                           = 3;
    tensor.section.inclusion_degree                             = 0;
    tensor.section.selected_coefficients                        = {0};
    coupling.refinement_parameters.space_pre_refinement_cycles  = 0;
    coupling.refinement_parameters.space_post_refinement_cycles = 0;
    coupling.refinement_parameters.max_refinement_level         = 0;
    coupling.coupling_rhs_expressions                           = {"0"};
    set_accuracy<dim>(p);
    ReducedPoisson<2, dim, 0> problem(p);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    problem.assemble_poisson_system();
    problem.assemble_coupling_system();
    problem.solve();
    EXPECT_EQ(problem.coupling_matrix_frobenius_norm(), 0.);
    EXPECT_GT(problem.assembled_outside_mass().frobenius_norm(), 0.);
    EXPECT_LT(problem.multiplier_solution_l2_norm(), 1.e-12);
    check_scalar_solution<dim>(problem, mixed);
    if (mixed)
      for (const double beta : {4., 0.})
        {
          initialize_parameters_from_string(
            scalar_robin_parameters<dim>("Reduced Poisson", true, false, beta));
          problem.assemble_poisson_system();
          problem.assemble_coupling_system();
          problem.solve();
          check_scalar_solution<dim>(problem, true);
          EXPECT_NEAR(p.robin_coefficient_for(1).value(Point<dim>()),
                      3.,
                      1.e-14);
          EXPECT_NEAR(p.robin_coefficient_for(2).value(Point<dim>()),
                      beta,
                      1.e-14);
        }
  }
} // namespace
TEST(LaplacianValidation, BOTH_RobinManufacturedAllBoundaryDimensions)
{
  check_laplacian_robin<2>(false);
  check_laplacian_robin<3>(false);
}
TEST(LaplacianValidation, BOTH_RobinNonzeroDirichletAndPerIdReparse)
{
  check_laplacian_robin<2>(true);
}
TEST(LaplacianValidation, BOTH_RobinSimplexManufactured)
{
  check_laplacian_robin<2>(false, true);
  check_laplacian_robin<3>(false, true);
}
TEST(ReducedPoissonValidation, BOTH_RobinManufacturedAllBoundaryDimensions)
{
  check_reduced_robin<2>(false);
  check_reduced_robin<3>(false);
}
TEST(ReducedPoissonValidation, BOTH_RobinNonzeroDirichletAndPerIdReparse)
{
  check_reduced_robin<2>(true);
}
TEST(ReducedPoissonValidation, BOTH_RobinSimplexManufactured)
{
  check_reduced_robin<2>(false, true);
  check_reduced_robin<3>(false, true);
}

namespace
{
  template <typename Parameters, typename Problem>
  void
  check_scalar_invalid_data(Parameters        &parameters,
                            Problem           &problem,
                            const std::string &section,
                            const std::string &expression,
                            const bool         invalid_load)
  {
    initialize_parameters_from_string(
      scalar_robin_parameters<2>(section, false) + "subsection " + section +
      "\n subsection " +
      (section == "Immersed Problem" ? "Domain" : "Grid generation") +
      "\n set Initial refinement = 0\n end\n subsection Functions\n subsection " +
      (invalid_load ? "Robin boundary conditions" : "Robin coefficient") +
      "\n set Function expression = " + expression + "\n end\n end\n end\n");
    problem.make_grid();
    problem.setup_fe();
    parameters.dirichlet_ids = {0};
    EXPECT_THROW(problem.setup_dofs(), ExceptionBase);
    parameters.dirichlet_ids.clear();
    problem.setup_dofs();
    // One cell leaves an empty MPI rank: data errors must be collective.
    EXPECT_THROW(problem.assemble_poisson_system(), ExceptionBase);
  }
} // namespace
TEST(LaplacianValidation, BOTH_RobinRejectsInvalidDataCollectively)
{
  for (const auto &entry :
       {std::pair<std::string, bool>{"if(x<1.e-8,-1,2)", false},
        {"1/(x-x)", false},
        {"1/(y-y)", true}})
    {
      ParameterAcceptor::clear();
      ProblemParameters<2> p;
      PoissonProblem<2>    problem(p);
      check_scalar_invalid_data(
        p, problem, "Immersed Problem", entry.first, entry.second);
    }
}
TEST(ReducedPoissonValidation, BOTH_RobinRejectsInvalidDataCollectively)
{
  for (const auto &entry :
       {std::pair<std::string, bool>{"if(x<1.e-8,-1,2)", false},
        {"1/(x-x)", false},
        {"1/(y-y)", true}})
    {
      ParameterAcceptor::clear();
      ReducedPoissonParameters<2, 0> p;
      p.reduced_coupling_parameters.tensor_product_space_parameters.point_cloud
        .points.emplace_back(2., 2.);
      p.reduced_coupling_parameters.refinement_parameters
        .space_pre_refinement_cycles = 0;
      p.reduced_coupling_parameters.refinement_parameters
        .space_post_refinement_cycles = 0;
      p.reduced_coupling_parameters.refinement_parameters.max_refinement_level =
        0;
      ReducedPoisson<2, 2, 0> problem(p);
      check_scalar_invalid_data(
        p, problem, "Reduced Poisson", entry.first, entry.second);
    }
}
