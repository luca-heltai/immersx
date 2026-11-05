// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_lib.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/fiber_reinforced_elastodynamics.h>
#include <immersx/physics/static_elasticity_term.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "test_paths.h"
#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#endif

using namespace ImmersX;
using namespace dealii;

namespace
{
  std::string
  components(const unsigned int count, const std::string &factor)
  {
    std::string result;
    for (unsigned int d = 0; d < count; ++d)
      result += (d ? ";" : "") + std::to_string(d + 1) + "*(" + factor + ")";
    return result;
  }

  template <int dim, int spacedim>
  void
  dynamic_translation(const std::string &strategy, const bool simplex = false)
  {
    ParameterAcceptor::clear();
    ElastodynamicsParameters<dim, spacedim> parameters;
    parameters.domain_parameters.initial_refinement = 0;
    parameters.domain_parameters.arguments_for_grid = "0:1:false";
    parameters.domain_parameters.triangulation_type = "fullydistributed";
    if (simplex)
      parameters.domain_parameters.name_of_grid = TestPaths::data_filename(
        dim == 2 ? "tests/simplex_square.msh" : "tests/simplex_cube.msh");
    parameters.dirichlet_ids.clear();
    parameters.robin_ids                                = {0, 1};
    parameters.fixed_step_parameters.time_step          = .025;
    parameters.fixed_step_parameters.number_of_steps    = 2;
    parameters.fixed_step_parameters.time_step_strategy = strategy;
    parameters.solver_control.set_tolerance(1.e-12);
    parameters.solver_control.set_reduction(1.e-12);
    initialize_parameters_from_string(
      "subsection Elastodynamics\nsubsection Material properties\nsubsection default\n"
      "set Density = 2\nset Lame mu = 3\nset Lame lambda = 4\n"
      "set Rayleigh alpha = .3\nset Rayleigh beta = .1\nset Viscosity eta = .2\n"
      "end\nend\nsubsection Functions\nsubsection Robin coefficient\n"
      "set Function expression = 2+x+t\nend\nsubsection Robin boundary\n"
      "set Function expression = " +
      components(spacedim, "(2+x+t)*t") +
      "\nend\nsubsection Body force\nset Function expression = " +
      components(spacedim, ".6") +
      "\nend\nsubsection Initial velocity\nset Function expression = " +
      components(spacedim, "1") + "\nend\nend\nend\n");
    ElastodynamicsSolver<dim, spacedim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();
    typename ElastodynamicsSolver<dim, spacedim>::VectorType expected, error;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    error.reinit(expected);
    // The material volume stiffness annihilates translation. The face spring
    // is the only nonzero stiffness action, including at the previous time.
    for (unsigned int step = 0; step < 2; ++step)
      {
        problem.advance_one_timestep();
        std::vector<double> values(spacedim);
        for (unsigned int d = 0; d < spacedim; ++d)
          values[d] = (d + 1) * problem.current_time();
        Functions::ConstantFunction<spacedim> displacement(values);
        VectorTools::interpolate(problem.mapping(),
                                 problem.dof_handler(),
                                 displacement,
                                 expected);
        error = problem.displacement();
        error -= expected;
        EXPECT_LT(error.linfty_norm(), 2.e-10);
        for (unsigned int d = 0; d < spacedim; ++d)
          values[d] = d + 1;
        Functions::ConstantFunction<spacedim> velocity(values);
        VectorTools::interpolate(problem.mapping(),
                                 problem.dof_handler(),
                                 velocity,
                                 expected);
        error = problem.velocity();
        error -= expected;
        EXPECT_LT(error.linfty_norm(), 2.e-9);
      }
    const auto accepted      = problem.displacement();
    const auto accepted_time = problem.current_time();
    typename ElastodynamicsSolver<dim, spacedim>::VectorType ones, image;
    ones.reinit(expected);
    image.reinit(expected);
    ones = 1.;
    for (const auto time : {.0, .4, .4, .1, .0})
      {
        const auto count = problem.assembly_statistics().robin_operators;
        problem.update_boundary_operators(time);
        const auto updated = problem.assembly_statistics().robin_operators;
        problem.update_boundary_operators(time);
        EXPECT_EQ(problem.assembly_statistics().robin_operators, updated);
        EXPECT_GE(updated, count);
        problem.stiffness_matrix().vmult(image, ones);
        // Integral of beta over the unit cube boundary is 2*dim*(2.5+t).
        EXPECT_NEAR(ones * image, spacedim * 2. * dim * (2.5 + time), 2.e-10);
      }
    error = problem.displacement();
    error -= accepted;
    EXPECT_DOUBLE_EQ(error.l2_norm(), 0.);
    EXPECT_DOUBLE_EQ(problem.current_time(), accepted_time);
    EXPECT_EQ(problem.time_step_number(), 2u);
  }
} // namespace

TEST(RobinModern, BOTH_BackwardEulerTranslationAllDimensions)
{
  dynamic_translation<1, 1>("backward_euler");
  dynamic_translation<1, 2>("backward_euler");
  dynamic_translation<1, 3>("backward_euler");
  dynamic_translation<2, 2>("backward_euler");
  dynamic_translation<2, 3>("backward_euler");
  dynamic_translation<3, 3>("backward_euler");
}

TEST(RobinModern, BOTH_TrapezoidalTranslationAllDimensions)
{
  dynamic_translation<1, 1>("trapezoidal");
  dynamic_translation<1, 2>("trapezoidal");
  dynamic_translation<1, 3>("trapezoidal");
  dynamic_translation<2, 2>("trapezoidal");
  dynamic_translation<2, 3>("trapezoidal");
  dynamic_translation<3, 3>("trapezoidal");
}

namespace
{
  template <int dim, int spacedim>
  void
  static_translation(const bool simplex = false)
  {
    ParameterAcceptor::clear();
    ElasticStaticParameters<dim, spacedim> parameters;
    parameters.domain_parameters.initial_refinement = 0;
    parameters.domain_parameters.arguments_for_grid = "0:1:true";
    parameters.domain_parameters.triangulation_type = "fullydistributed";
    if (simplex)
      parameters.domain_parameters.name_of_grid = TestPaths::data_filename(
        dim == 2 ? "tests/simplex_square.msh" : "tests/simplex_cube.msh");
    parameters.dirichlet_ids.clear();
    for (unsigned int id = 0; id < 2 * dim; ++id)
      parameters.robin_ids.insert(id);
    initialize_parameters_from_string(
      "subsection Elastic static\nsubsection Functions\nsubsection Robin coefficient\n"
      "set Function expression = 2+x\nend\nsubsection Robin boundary conditions\n"
      "set Function expression = " +
      components(spacedim, "2+x") +
      "\nend\nsubsection Robin coefficient 1\nset Function expression = 3+x\nend\n"
      "subsection Robin boundary conditions 1\nset Function expression = " +
      components(spacedim, "3+x") + "\nend\nend\nend\n");
    SCOPED_TRACE(::testing::Message() << "dim=" << dim << ", spacedim="
                                      << spacedim << ", simplex=" << simplex);
    ElasticStaticProblem<dim, spacedim> problem(parameters);
    problem.setup();
    {
      SCOPED_TRACE("native");
      ASSERT_NO_THROW(problem.solve());
    }
    typename ElasticStaticProblem<dim, spacedim>::VectorType expected,
      difference;
    expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    std::vector<double> values(spacedim);
    for (unsigned int d = 0; d < spacedim; ++d)
      values[d] = d + 1;
    Functions::ConstantFunction<spacedim> exact(values);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             exact,
                             expected);
    difference = problem.solution();
    difference -= expected;
    EXPECT_LT(difference.linfty_norm(), 1.e-9);
    auto ones = expected, image = expected;
    ones = 1.;
    problem.stiffness_operator().vmult(image, ones);
    // Boundary id 1 changes beta by one on a unit face. This distinguishes
    // per-id assembly from a consistent but incorrect global beta/g fallback.
    EXPECT_NEAR(ones * image,
                spacedim * (5. * dim + (simplex ? 0. : 1.)),
                2.e-10);

    LinearSolverParameters                                             control;
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
      control, MPI_COMM_WORLD);
    const auto fields = adapter.add(problem);
    auto       state  = adapter.make_state();
    {
      SCOPED_TRACE("contributor");
      ASSERT_NO_THROW(adapter.solve(state));
    }
    difference = adapter.field(state, fields.fields().displacement);
    difference -= expected;
    EXPECT_LT(difference.linfty_norm(), 1.e-9);
    auto space = finite_element_space_view(problem);
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
               expression_adapter(control, MPI_COMM_WORLD);
    const auto field =
      expression_adapter.add(algebraic_field(vector_field(space, "u")), "solid")
        .fields();
    expression_adapter.add(static_elasticity_term(field, parameters),
                           "elasticity");
    expression_adapter.add_preconditioner(
      field.id(), [](const auto &matrix, const auto &reinit) {
        return make_amg_preconditioner(matrix, reinit);
      });
    auto expression_state = expression_adapter.make_state();
    {
      SCOPED_TRACE("expression");
      ASSERT_NO_THROW(expression_adapter.solve(expression_state));
    }
    difference = expression_adapter.field(expression_state, field);
    difference -= expected;
    EXPECT_LT(difference.linfty_norm(), 1.e-9);
  }
} // namespace

TEST(RobinModern, BOTH_StaticTranslationAllDimensions)
{
  static_translation<1, 1>();
  static_translation<1, 2>();
  static_translation<1, 3>();
  static_translation<2, 2>();
  static_translation<2, 3>();
  static_translation<3, 3>();
}

#ifdef DEAL_II_WITH_SUNDIALS
TEST(RobinModern, BOTH_AdapterCandidateTimesAndJacobian)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  parameters.domain_parameters.initial_refinement = 1;
  parameters.domain_parameters.arguments_for_grid = "0:1:false";
  parameters.dirichlet_ids.clear();
  parameters.robin_ids = {0};
  initialize_parameters_from_string(R"(
subsection Elastodynamics
 subsection Material properties
  subsection default
   set Density = 2
   set Lame mu = 3
   set Lame lambda = 4
   set Rayleigh alpha = .3
   set Rayleigh beta = .1
   set Viscosity eta = .2
  end
 end
 subsection Functions
  subsection Robin coefficient
   set Function expression = 2+x+t
  end
  subsection Robin boundary
   set Function expression = (2+x+t)*t;2*(2+x+t)*t
  end
  subsection Body force
   set Function expression = .6;1.2
  end
 end
end
)");
  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();
  using Vector = ElastodynamicsSolver<2>::VectorType;
  using Global = ImmersXLA::MPI::BlockVector;
  IDAAdapter<Vector, Global> adapter(parameters.time_parameters,
                                     parameters.ida_parameters,
                                     MPI_COMM_WORLD);
  const auto                 fields = adapter.add(problem, "solid");
  auto state = adapter.make_state(), derivative = adapter.make_state(),
       residual  = adapter.make_state();
  auto direction = adapter.make_state(), action = adapter.make_state(),
       perturbed                                         = adapter.make_state();
  adapter.field(direction, fields.fields().displacement) = 1.;
  constexpr double epsilon                               = 1.e-6;
  for (const double time : {0., .4, .4, .1, 0.})
    {
      Functions::ConstantFunction<2> displacement(
        std::vector<double>{time, 2 * time});
      Functions::ConstantFunction<2> velocity(std::vector<double>{1., 2.});
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               displacement,
                               adapter.field(state,
                                             fields.fields().displacement));
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               velocity,
                               adapter.field(state, fields.fields().velocity));
      adapter.field(derivative, fields.fields().displacement) =
        adapter.field(state, fields.fields().velocity);
      adapter.solver().residual(time, state, derivative, residual);
      EXPECT_LT(residual.l2_norm(), 2.e-11);
      adapter.solver().setup_jacobian(time, state, derivative, 2.);
      adapter.current_jacobian().vmult(action, direction);
      auto trial = state;
      trial.add(epsilon, direction);
      auto trial_derivative = derivative;
      trial_derivative.add(2. * epsilon, direction);
      adapter.solver().residual(time, trial, trial_derivative, perturbed);
      perturbed -= residual;
      perturbed /= epsilon;
      perturbed -= action;
      EXPECT_LT(perturbed.l2_norm(), 2.e-7);
      const auto &load = adapter.field(action, fields.fields().velocity);
      Vector      ones;
      ones.reinit(load);
      ones = 1.;
      EXPECT_NEAR(ones * load, 8. * (2.5 + time), 2.e-10);
    }
  EXPECT_DOUBLE_EQ(problem.current_time(), 0.);
  EXPECT_EQ(problem.time_step_number(), 0u);
  EXPECT_DOUBLE_EQ(problem.displacement().l2_norm(), 0.);
}
#endif

TEST(RobinModern, BOTH_RejectsInvalidDataAndOverlappingIdsCollectively)
{
  for (const auto &data :
       std::vector<std::pair<std::string, std::string>>{{"x-.25", "0;0"},
                                                        {"exp(1000*x)", "0;0"},
                                                        {"2", "sqrt(x-.5);0"},
                                                        {"2", "exp(1000*x);0"}})
    {
      ParameterAcceptor::clear();
      ElastodynamicsParameters<2> parameters;
      initialize_parameters_from_string(
        "subsection Elastodynamics\nset Dirichlet boundary ids =\nset Robin boundary ids = 0\n"
        "subsection Domain\nset Initial refinement = 1\nset Grid generator arguments = 0:1:false\nend\n"
        "subsection Functions\nsubsection Robin coefficient\nset Function expression = " +
        data.first +
        "\nend\nsubsection Robin boundary\nset Function expression = " +
        data.second + "\nend\nend\nend\n");
      ElastodynamicsSolver<2> problem(parameters);
      problem.make_grid();
      problem.setup_fe();
      problem.setup_system();
      EXPECT_THROW(
        {
          problem.assemble_operators();
          problem.set_initial_conditions();
        },
        dealii::ExceptionBase);
    }
  {
    ParameterAcceptor::clear();
    ElastodynamicsParameters<2> parameters;
    initialize_parameters_from_string(
      "subsection Elastodynamics\nset Robin boundary ids = 0\nend\n");
    ElastodynamicsSolver<2> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    EXPECT_THROW(problem.setup_system(), dealii::ExceptionBase);
  }
}

TEST(RobinModern, BOTH_ConstantCoefficientReusesOperatorWithTimeDependentLoad)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  initialize_parameters_from_string(R"(
subsection Elastodynamics
 set Dirichlet boundary ids =
 set Robin boundary ids = 0
 subsection Domain
  set Initial refinement = 1
  set Grid generator arguments = 0:1:false
 end
 subsection Functions
  subsection Robin coefficient
   set Function expression = 2
  end
  subsection Robin boundary
   set Function expression = t;2*t
  end
 end
end
)");
  ElastodynamicsSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  const auto count = problem.assembly_statistics().robin_operators;
  ElastodynamicsSolver<2>::VectorType force, ones;
  ones.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  ones = 1.;
  for (const double time : {0., .1, .3, .1})
    {
      problem.update_boundary_operators(time);
      problem.body_force_at_time(time, force);
      EXPECT_EQ(problem.assembly_statistics().robin_operators, count);
      EXPECT_NEAR(ones * force, 12. * time, 2.e-12);
    }
}

TEST(RobinModern, BOTH_CoupledFiberBackwardEulerRefreshesBothBoundaryOperators)
{
  ParameterAcceptor::clear();
  FiberReinforcedElastodynamicsParameters<2> parameters;
  std::string input = "subsection Fiber Reinforced Elastodynamics\n";
  input +=
    "subsection Time interval\nset Final time = .05\nend\nsubsection Fixed step\nset Time step = .025\nset Number of time steps = 2\nend\n";
  for (const std::string part :
       {"Matrix Elastodynamics", "Fiber Elastodynamics"})
    input +=
      "subsection " + part +
      "\nset Dirichlet boundary ids =\nset Robin boundary ids = 0,1\n"
      "subsection Domain\nset Initial refinement = 1\nset Grid generator arguments = " +
      (part == "Matrix Elastodynamics" ? "-1:1:false" : "-.6:.6:false") +
      "\nend\nsubsection Material properties\nsubsection default\n"
      "set Density = 2\nset Lame mu = 3\nset Lame lambda = 4\n"
      "set Rayleigh alpha = 0\nset Rayleigh beta = .1\nset Viscosity eta = .2\nend\nend\n"
      "subsection Functions\nsubsection Robin coefficient\nset Function expression = 2+x+t\nend\n"
      "subsection Robin boundary\nset Function expression = (2+x+t)*t;2*(2+x+t)*t\nend\n"
      "subsection Initial velocity\nset Function expression = 1;2\nend\nend\nend\n";
  input += "end\n";
  initialize_parameters_from_string(input);
  FiberReinforcedElastodynamics<2> problem(parameters);
  problem.setup();
  problem.set_initial_conditions();
  const auto check = [](const auto &solid, const double time) {
    using Problem = std::decay_t<decltype(solid)>;
    typename Problem::VectorType expected, difference;
    expected.reinit(solid.locally_owned_dofs(), MPI_COMM_WORLD);
    Functions::ConstantFunction<2> exact(std::vector<double>{time, 2 * time});
    VectorTools::interpolate(solid.mapping(),
                             solid.dof_handler(),
                             exact,
                             expected);
    difference = solid.displacement();
    difference -= expected;
    EXPECT_LT(difference.linfty_norm(), 2.e-8);
  };
  for (unsigned int step = 0; step < 2; ++step)
    {
      problem.advance_one_timestep();
      check(problem.matrix_problem(), (step + 1) * .025);
      check(problem.fiber_problem(), (step + 1) * .025);
      EXPECT_LT(problem.residuals().matrix_velocity, 2.e-8);
      EXPECT_LT(problem.residuals().fiber_velocity, 2.e-8);
      EXPECT_LT(problem.residuals().velocity_constraint, 2.e-8);
    }
}

TEST(RobinModern, BOTH_SimplexStaticAndTransient)
{
  static_translation<2, 2>(true);
  static_translation<2, 3>(true);
  static_translation<3, 3>(true);
  dynamic_translation<2, 2>("backward_euler", true);
  dynamic_translation<2, 3>("backward_euler", true);
  dynamic_translation<3, 3>("backward_euler", true);
  dynamic_translation<2, 2>("trapezoidal", true);
  dynamic_translation<2, 3>("trapezoidal", true);
  dynamic_translation<3, 3>("trapezoidal", true);
}
