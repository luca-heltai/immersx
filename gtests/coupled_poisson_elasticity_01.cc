// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson_residual.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "coupled_poisson_elasticity.h"
#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

  class AdapterState : public StateAccessor<FieldVector>
  {
  public:
    AdapterState(const Adapter &adapter, const GlobalVector &state)
      : adapter_(adapter)
      , state_(state)
    {}

    const FieldVector &
    field(const FieldId id, const double) const override
    {
      return adapter_.field(state_, id);
    }

  private:
    const Adapter      &adapter_;
    const GlobalVector &state_;
  };
} // namespace

TEST(CoupledPoissonElasticity, BOTH_PoissonEndpointData)
{
  ParameterAcceptor::clear();
  PoissonParameters<1, 3> parameters;
  initialize_parameters_from_string(R"(
    subsection Poisson
      set FE degree = 2
      set Dirichlet boundary ids = 0, 1
      subsection Domain
        set Initial refinement = 2
        set Grid generator = hyper_cube
        set Grid generator arguments = 0: 1: false
        set Triangulation type = fullydistributed
      end
      subsection Dirichlet boundary conditions
        set Function expression = 1+x
        set Variable names = x,y,z,t
      end
    end
  )");
  PoissonSolver<1, 3> poisson(parameters);
  poisson.setup();
  unsigned int constrained = 0;
  for (const auto dof : poisson.locally_owned_dofs())
    constrained += poisson.constraints().is_constrained(dof);
  EXPECT_EQ(Utilities::MPI::sum(constrained, MPI_COMM_WORLD), 2u);

  poisson.solve();
  const auto points =
    DoFTools::map_dofs_to_support_points(poisson.mapping(),
                                         poisson.dof_handler());
  for (const auto dof : poisson.locally_owned_dofs())
    EXPECT_NEAR(poisson.solution()[dof], 1. + points.at(dof)[0], 1.e-10);
}

void
check_pressure_traction_balance(const bool simplex)
{
  ParameterAcceptor::clear();
  PoissonParameters<1, 3>                   poisson_parameters;
  ElasticStaticParameters<3>                elastic_parameters;
  TensorProductLift<1, 2, 3>                pressure_lift("/Pressure lift/");
  CoupledPoissonElasticity::CylinderSurface surface;
  LinearSolverParameters                    solver_parameters;
  initialize_parameters(TestPaths::parameter_path(
    simplex ? "gtests/parameters/coupled_poisson_elasticity_simplex.prm" :
              "gtests/parameters/coupled_poisson_elasticity.prm"));

  PoissonSolver<1, 3>     poisson(poisson_parameters);
  ElasticStaticProblem<3> elastic(elastic_parameters);
  poisson.setup();
  elastic.setup();
  Adapter    adapter(solver_parameters, MPI_COMM_WORLD);
  const auto poisson_fields = adapter.add(poisson).fields();
  const auto elastic_fields = adapter.add(elastic).fields();
  const auto p              = poisson_fields.solution;
  const auto u              = elastic_fields.displacement;
  const auto pressure       = make_lift(2. * value(p), pressure_lift);
  const auto traction       = pressure * normal(surface);
  adapter.add(weak_term(traction, test(u)), "pressure-traction");
  auto state = adapter.make_state();
  adapter.solve(state);
  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
  EXPECT_GT(adapter.field(state, u).l2_norm(), 1.e-12);

  // For this straight line, -p'' = 1 with zero endpoint data has the
  // quadratic exact solution p(s) = s (L-s)/2, represented exactly by Q2.
  Point<3> begin, end;
  for (unsigned int d = 0; d < 3; ++d)
    {
      begin[d] = std::numeric_limits<double>::max();
      end[d]   = -std::numeric_limits<double>::max();
    }
  for (const auto &cell : poisson.dof_handler().active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto vertex : cell->vertex_indices())
        for (unsigned int d = 0; d < 3; ++d)
          {
            begin[d] = std::min(begin[d], cell->vertex(vertex)[d]);
            end[d]   = std::max(end[d], cell->vertex(vertex)[d]);
          }
  for (unsigned int d = 0; d < 3; ++d)
    {
      begin[d] = Utilities::MPI::min(begin[d], MPI_COMM_WORLD);
      end[d]   = Utilities::MPI::max(end[d], MPI_COMM_WORLD);
    }
  const auto   chord         = end - begin;
  const double length_square = chord.norm_square();
  ASSERT_GT(length_square, 0.);
  const AdapterState                   accessor(adapter, state);
  const EvaluationContext<FieldVector> context(0., accessor);
  const auto                           samples = pressure.evaluate(context);
  EXPECT_GT(Utilities::MPI::sum(samples.locally_owned_size(), MPI_COMM_WORLD),
            0u);
  const auto &points = pressure.lifted_points();
  for (std::size_t q = 0; q < points.size(); ++q)
    {
      const double t =
        ((points[q].representative_point - begin) * chord) / length_square;
      const double expected =
        length_square * t * (1. - t) * points[q].mode_values.front();
      EXPECT_NEAR(samples[pressure.point_index(q)], expected, 1.e-10);
    }

  // Retain the former application traction-balance check at the FE level.
  const auto load = ImmersX::detail::make_normal_load_operator<
    std::decay_t<decltype(pressure)>,
    CoupledPoissonElasticity::CylinderSurface,
    std::decay_t<decltype(u)>,
    FieldVector>(traction, u);
  FieldVector expected_load, defect;
  expected_load.reinit(elastic.locally_owned_dofs(), MPI_COMM_WORLD);
  load.vmult(expected_load, samples);
  defect.reinit(expected_load);
  elastic.stiffness_operator().vmult(defect, adapter.field(state, u));
  defect -= elastic.forcing();
  defect += expected_load;
  elastic.constraints().set_zero(defect);
  EXPECT_LT(defect.l2_norm(), 1.e-9);
}

TEST(CoupledPoissonElasticity, BOTH_PressureTractionBalance)
{
  check_pressure_traction_balance(false);
}
TEST(CoupledPoissonElasticity, BOTH_SimplexPressureTractionBalance)
{
  check_pressure_traction_balance(true);
}
