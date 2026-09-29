// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <cmath>
#include <string>
#include <vector>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  using Problem      = ElastodynamicsSolver<2>;
  using FieldVector  = Problem::VectorType;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  struct Measurements
  {
    double       l2_displacement;
    double       h1_displacement;
    double       l2_velocity;
    double       h;
    unsigned int dofs;
    unsigned int cells;
  };

  Measurements
  measure_case(const std::string &parameter_file,
               const unsigned int refinement,
               const bool         check_trace = false)
  {
    ParameterAcceptor::clear();
    ElastodynamicsParameters<2> parameters;
    Problem                     problem(parameters);
    initialize_parameters(parameter_file);
    parameters.domain_parameters.initial_refinement = refinement;

    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    problem.set_initial_conditions();

    Adapter    adapter(parameters.time_parameters,
                    parameters.ida_parameters,
                    MPI_COMM_WORLD);
    const auto fields    = adapter.add(problem, "elastodynamics");
    auto       state     = adapter.make_state();
    auto       state_dot = adapter.make_state();
    initialize_elastodynamics_adapter_state(
      adapter, fields, problem, state, state_dot);
    adapter.solve(state, state_dot);

    constexpr double final_time = 0.13;
    parameters.exact_solution.set_time(final_time);
    parameters.velocity_boundary.set_time(final_time);
    const QGauss<2> quadrature(4);
    Vector<double>  errors(problem.triangulation().n_active_cells());
    VectorTools::integrate_difference(problem.dof_handler(),
                                      state.block(0),
                                      parameters.exact_solution,
                                      errors,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double l2_displacement = errors.l2_norm();
    VectorTools::integrate_difference(problem.dof_handler(),
                                      state.block(0),
                                      parameters.exact_solution,
                                      errors,
                                      quadrature,
                                      VectorTools::H1_norm);
    const double h1_displacement = errors.l2_norm();
    VectorTools::integrate_difference(problem.dof_handler(),
                                      state.block(1),
                                      parameters.velocity_boundary,
                                      errors,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double l2_velocity = errors.l2_norm();

    if (check_trace)
      {
        parameters.displacement_boundary.set_time(final_time);
        AffineConstraints<double> final_constraints;
        final_constraints.reinit(problem.locally_owned_dofs(),
                                 problem.locally_relevant_dofs());
        DoFTools::make_hanging_node_constraints(problem.dof_handler(),
                                                final_constraints);
        VectorTools::interpolate_boundary_values(
          problem.dof_handler(),
          0,
          parameters.displacement_boundary,
          final_constraints);
        final_constraints.close();
        for (const auto index : problem.locally_owned_dofs())
          {
            if (final_constraints.is_constrained(index) &&
                final_constraints.get_constraint_entries(index)->empty())
              {
                EXPECT_NEAR(state.block(0)(index),
                            final_constraints.get_inhomogeneity(index),
                            1.e-7);
              }
          }
      }

    std::cout << "MMS refinement " << refinement
              << ": h=" << problem.triangulation().begin_active()->diameter()
              << ", dofs=" << problem.n_dofs() << ", L2(u)=" << l2_displacement
              << ", H1(u)=" << h1_displacement << ", L2(v)=" << l2_velocity
              << std::endl;

    return {l2_displacement,
            h1_displacement,
            l2_velocity,
            problem.triangulation().begin_active()->diameter(),
            static_cast<unsigned int>(problem.n_dofs()),
            problem.triangulation().n_active_cells()};
  }

  std::string
  strong_file()
  {
    return TestPaths::parameter_path(
      "gtests/parameters/elastodynamics_adapter_mms_strong_dirichlet.prm");
  }

  std::string
  neumann_file()
  {
    return TestPaths::parameter_path(
      "gtests/parameters/elastodynamics_adapter_mms_neumann.prm");
  }

  void
  expect_rates(const std::vector<Measurements> &results)
  {
    ASSERT_EQ(results.size(), 3u);
    for (unsigned int i = 1; i < results.size(); ++i)
      {
        const double h_ratio = results[i - 1].h / results[i].h;
        const double l2_rate = std::log(results[i - 1].l2_displacement /
                                        results[i].l2_displacement) /
                               std::log(h_ratio);
        const double h1_rate = std::log(results[i - 1].h1_displacement /
                                        results[i].h1_displacement) /
                               std::log(h_ratio);
        const double v_rate =
          std::log(results[i - 1].l2_velocity / results[i].l2_velocity) /
          std::log(h_ratio);
        EXPECT_GT(l2_rate, 1.7);
        EXPECT_GT(h1_rate, 0.8);
        EXPECT_GT(v_rate, 1.5);
      }
  }
} // namespace

TEST(ElastodynamicsAdapterMMS, StrongDirichlet)
{
  const auto result = measure_case(strong_file(), 2, true);
  EXPECT_LT(result.l2_displacement, 8.e-2);
  EXPECT_LT(result.h1_displacement, 7.e-1);
  EXPECT_LT(result.l2_velocity, 5.e-1);
}

TEST(ElastodynamicsAdapterMMS, Neumann)
{
  const auto result = measure_case(neumann_file(), 1);
  EXPECT_LT(result.l2_displacement, 3.e-1);
  EXPECT_LT(result.h1_displacement, 1.7);
  EXPECT_LT(result.l2_velocity, 8.e-1);
}

TEST(ElastodynamicsAdapterMMS, StrongDirichletSpatialConvergence)
{
  std::vector<Measurements> results;
  for (const auto refinement : {1u, 2u, 3u})
    results.push_back(measure_case(strong_file(), refinement));
  expect_rates(results);
}

TEST(ElastodynamicsAdapterMMS, NeumannSpatialConvergence)
{
  std::vector<Measurements> results;
  for (const auto refinement : {1u, 2u, 3u})
    results.push_back(measure_case(neumann_file(), refinement));
  expect_rates(results);
}
#else
TEST(ElastodynamicsAdapterMMS, RequiresSundials)
{
  GTEST_SKIP() << "deal.II was built without SUNDIALS support";
}
#endif
