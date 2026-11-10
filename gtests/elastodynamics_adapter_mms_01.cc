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
    ASSERT_EQ(results.size(), 2u);
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

TEST(ElastodynamicsAdapterMMS, BOTH_LocalAffineMovingBoundary)
{
  ParameterAcceptor::clear();
  ElastodynamicsParameters<2> parameters;
  initialize_parameters_from_string(R"(
    subsection Elastodynamics
      set Dirichlet boundary ids = 0,1,2,3
      subsection Domain
        set Initial refinement = 1
        set Grid generator = hyper_cube
        set Grid generator arguments = 0: 1: true
        set Triangulation type = distributed
      end
      subsection Material properties
        subsection default
          set Density = 1
          set Lame mu = 1
          set Lame lambda = 2
          set Rayleigh alpha = 0
          set Rayleigh beta = 0.2
          set Viscosity eta = 0.1
        end
      end
      subsection Time interval
        set Initial time = 0.1
        set Final time = 0.13
        set Output time interval = 0.01
      end
      subsection IDA
        subsection Running parameters
          set Initial step size = 0.0001
          set Maximum order of BDF = 2
        end
        subsection Error control
          set Absolute error tolerance = 1e-10
          set Relative error tolerance = 1e-9
        end
      end
      subsection Functions
        subsection Body force
          set Function expression = 2*(1+x); 0
        end
        subsection Displacement boundary
          set Function expression = t*t*(1+x); 0
        end
        subsection Velocity boundary
          set Function expression = 2*t*(1+x); 0
        end
        subsection Initial displacement
          set Function expression = 0.01*(1+x); 0
        end
        subsection Initial velocity
          set Function expression = 0.2*(1+x); 0
        end
        subsection Exact solution
          set Function expression = t*t*(1+x); 0
        end
      end
    end
  )");
  Problem problem(parameters);
  problem.make_grid();
  auto &mesh = problem.triangulation();
  mesh.set_mesh_smoothing(Triangulation<2>::limit_level_difference_at_vertices);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < .5 &&
        cell->center()[1] < .5)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  ASSERT_EQ(mesh.n_global_active_cells(), 7u);
  problem.setup_fe();
  problem.setup_system();
  problem.assemble_operators();
  problem.set_initial_conditions();
  ASSERT_GT(
    Utilities::MPI::sum(problem.hanging_node_constraints().n_constraints(),
                        MPI_COMM_WORLD),
    0u);
  Adapter    adapter(parameters.time_parameters,
                  parameters.ida_parameters,
                  MPI_COMM_WORLD);
  const auto fields    = adapter.add(problem, "local-affine");
  auto       state     = adapter.make_state();
  auto       state_dot = adapter.make_state();
  initialize_elastodynamics_adapter_state(
    adapter, fields, problem, state, state_dot);
  unsigned int nonzero_mesh_data = 0;
  for (const auto &line : problem.constraints().get_lines())
    if (!line.entries.empty() &&
        problem.locally_owned_dofs().is_element(line.index))
      {
        EXPECT_EQ(state.block(0)[line.index], line.inhomogeneity);
        nonzero_mesh_data += line.inhomogeneity != 0.;
      }
  EXPECT_GT(Utilities::MPI::sum(nonzero_mesh_data, MPI_COMM_WORLD), 0u);
  auto initial_residual = adapter.make_state();
  adapter.solver().residual(.1, state, state_dot, initial_residual);
  EXPECT_LT(initial_residual.l2_norm(), 1.e-9);
  adapter.solve(state, state_dot);
  problem.accept_state(state.block(0), state.block(1), .13, 1);
  parameters.exact_solution.set_time(.13);
  parameters.velocity_boundary.set_time(.13);
  FieldVector expected;
  expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  VectorTools::interpolate(problem.mapping(),
                           problem.dof_handler(),
                           parameters.exact_solution,
                           expected);
  expected -= problem.displacement();
  EXPECT_LT(expected.l2_norm(), 1.e-7);
  VectorTools::interpolate(problem.mapping(),
                           problem.dof_handler(),
                           parameters.velocity_boundary,
                           expected);
  expected -= problem.velocity();
  EXPECT_LT(expected.l2_norm(), 1.e-6);
}

TEST(ElastodynamicsAdapterMMS, StrongDirichletSpatialRegression)
{
  const std::vector<Measurements> results = {
    measure_case(strong_file(), 1), measure_case(strong_file(), 2, true)};
  const auto &result = results.back();
  EXPECT_LT(result.l2_displacement, 8.e-2);
  EXPECT_LT(result.h1_displacement, 7.e-1);
  EXPECT_LT(result.l2_velocity, 5.e-1);
  expect_rates(results);
}

TEST(ElastodynamicsAdapterMMS, NeumannSpatialRegression)
{
  const std::vector<Measurements> results = {measure_case(neumann_file(), 1),
                                             measure_case(neumann_file(), 2)};
  const auto                     &result  = results.front();
  EXPECT_LT(result.l2_displacement, 3.e-1);
  EXPECT_LT(result.h1_displacement, 1.7);
  EXPECT_LT(result.l2_velocity, 8.e-1);
  expect_rates(results);
}

#else
TEST(ElastodynamicsAdapterMMS, RequiresSundials)
{
  GTEST_SKIP() << "deal.II was built without SUNDIALS support";
}
#endif
