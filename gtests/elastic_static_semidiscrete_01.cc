// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static_semidiscrete.h>

#include <cmath>

using namespace dealii;
using namespace ImmersX;

#ifdef DEAL_II_WITH_SUNDIALS
TEST(ElasticStaticSemidiscrete, BOTH_AlgebraicEquilibriumAtCandidateTime)
{
  ParameterAcceptor::clear();
  ElasticStaticParameters<3> parameters;
  TimeIntervalParameters     time_parameters("/Static time/", false);
  IDAParameters              ida_parameters("/Static IDA/", &time_parameters);
  initialize_parameters_from_string(R"(
    subsection Elastic static
      set FE degree = 2
      subsection Domain
        set Grid generator = hyper_cube
        set Grid generator arguments = 0:1:false
        set Initial refinement = 1
        set Triangulation type = distributed
      end
      subsection Material properties
        subsection default
          set Lame mu = 0.5
          set Lame lambda = 0
        end
      end
      subsection Solver
        subsection Control
          set Tolerance = 1.e-13
          set Reduction = 1.e-13
        end
      end
      subsection Functions
        subsection Right hand side
          set Function expression = -2*(1+t);0;0
          set Variable names = x,y,z,t
        end
        subsection Dirichlet boundary conditions
          set Function expression = (1+t)*(1+x*x);0;0
          set Variable names = x,y,z,t
        end
      end
    end
  )");
  ElasticStaticProblem<3> problem(parameters);
  problem.setup();
  using FieldVector  = ElasticStaticProblem<3>::VectorType;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  IDAAdapter<FieldVector, GlobalVector> adapter(time_parameters,
                                                ida_parameters,
                                                MPI_COMM_WORLD);
  const auto description = elastic_static_semidiscrete(problem);
  const auto fields      = adapter.add(description);
  auto       state       = adapter.make_state();
  auto       rate        = adapter.make_state();
  ASSERT_EQ(state.n_blocks(), 1u);
  EXPECT_EQ(adapter.differential_components().n_elements(), 0u);
  const double      candidate_time = .37;
  FunctionParser<3> exact(3);
  exact.initialize("x,y,z,t", "(1+t)*(1+x*x);0;0", {}, true);
  exact.set_time(candidate_time);
  auto &displacement = adapter.field(state, fields.fields().displacement);
  VectorTools::interpolate(problem.mapping(),
                           problem.dof_handler(),
                           exact,
                           displacement);
  rate          = 19.;
  auto residual = adapter.make_state();
  adapter.solver().residual(candidate_time, state, rate, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-11);
  // Evaluating a candidate must not change the accepted physical state.
  EXPECT_EQ(problem.solution().l2_norm(), 0.);
  rate        = -7.;
  auto second = adapter.make_state();
  adapter.solver().residual(candidate_time, state, rate, second);
  second -= residual;
  EXPECT_LT(second.l2_norm(), 1.e-14);

  auto direction = adapter.make_state();
  direction      = .125;
  auto action    = adapter.make_state();
  adapter.solver().setup_jacobian(candidate_time, state, rate, 0.);
  adapter.current_jacobian().vmult(action, direction);
  auto state_action = action;
  adapter.solver().setup_jacobian(candidate_time, state, rate, 13.);
  adapter.current_jacobian().vmult(action, direction);
  action -= state_action;
  EXPECT_LT(action.l2_norm(), 1.e-13);
  const double epsilon = 1.e-5;
  state.add(epsilon, direction);
  adapter.solver().residual(candidate_time, state, rate, second);
  second -= residual;
  second *= 1. / epsilon;
  second -= state_action;
  EXPECT_LT(second.l2_norm(), 1.e-9);

  const auto first_inverse =
    adapter.local_preconditioner(fields.fields().displacement.id(), state);
  EXPECT_EQ(description.preconditioner_builds(), 1u);
  problem.update_time(.51);
  adapter.local_preconditioner(fields.fields().displacement.id(), state);
  EXPECT_EQ(description.preconditioner_builds(), 1u);
  // The retained inverse remains usable after its source snapshot is gone.
  FieldVector inverse_action;
  first_inverse.reinit_range_vector(inverse_action, false);
  first_inverse.vmult(inverse_action,
                      adapter.field(direction, fields.fields().displacement));
  EXPECT_TRUE(std::isfinite(inverse_action.l2_norm()));
  problem.update_time(candidate_time);

  // The independent polynomial solution also verifies the native static solve.
  problem.solve();
  FieldVector difference;
  difference.reinit(problem.solution());
  VectorTools::interpolate(problem.mapping(),
                           problem.dof_handler(),
                           exact,
                           difference);
  difference -= problem.solution();
  EXPECT_LT(difference.l2_norm(), 1.e-9);
  EXPECT_GT(problem.solution().l2_norm(), 0.);
}

TEST(ElasticStaticSemidiscrete, BOTH_RobinMatrixInvalidatesCachedAMG)
{
  ParameterAcceptor::clear();
  ElasticStaticParameters<3> parameters;
  TimeIntervalParameters     time_parameters("/Static time/", false);
  IDAParameters              ida_parameters("/Static IDA/", &time_parameters);
  initialize_parameters_from_string(R"(
    subsection Elastic static
      set Dirichlet boundary ids =
      set Robin boundary ids = 0
      subsection Domain
        set Grid generator = hyper_cube
        set Grid generator arguments = 0:1:false
        set Initial refinement = 1
        set Triangulation type = distributed
      end
      subsection Functions
        subsection Robin coefficient
          set Function expression = 1+t
          set Variable names = x,y,z,t
        end
        subsection Robin boundary conditions
          set Function expression = (1+t)^2;0;0
          set Variable names = x,y,z,t
        end
      end
    end
  )");
  ElasticStaticProblem<3> problem(parameters);
  problem.setup();
  using FieldVector  = ElasticStaticProblem<3>::VectorType;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  IDAAdapter<FieldVector, GlobalVector> adapter(time_parameters,
                                                ida_parameters,
                                                MPI_COMM_WORLD);
  const auto description = elastic_static_semidiscrete(problem);
  const auto fields      = adapter.add(description);
  auto       state       = adapter.make_state();
  auto       rate        = adapter.make_state();
  state                  = 0.;
  rate                   = 0.;
  adapter.solver().setup_jacobian(0., state, rate, 1.);
  EXPECT_EQ(description.preconditioner_builds(), 1u);
  adapter.solver().setup_jacobian(.5, state, rate, 1.);
  EXPECT_EQ(description.preconditioner_builds(), 2u);
  adapter.solver().setup_jacobian(.5, state, rate, 7.);
  EXPECT_EQ(description.preconditioner_builds(), 2u);

  Functions::ConstantFunction<3> exact(std::vector<double>{1.5, 0., 0.});
  VectorTools::interpolate(problem.mapping(),
                           problem.dof_handler(),
                           exact,
                           adapter.field(state, fields.fields().displacement));
  auto residual = adapter.make_state();
  adapter.solver().residual(.5, state, rate, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-11);
}

#endif
