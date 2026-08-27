// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/representation.h>

#include <memory>

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct SamplingData
  {
    parallel::distributed::Triangulation<2>         triangulation;
    FE_Q<2>                                         finite_element;
    DoFHandler<2>                                   dof_handler;
    IndexSet                                        locally_owned;
    IndexSet                                        locally_relevant;
    AffineConstraints<double>                       constraints;
    std::unique_ptr<FiniteElementRepresentation<2>> representation;

    explicit SamplingData(const bool inhomogeneous = false)
      : triangulation(MPI_COMM_WORLD)
      , finite_element(2)
      , dof_handler(triangulation)
    {
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(2);
      dof_handler.distribute_dofs(finite_element);
      locally_owned    = dof_handler.locally_owned_dofs();
      locally_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
      constraints.reinit(locally_owned, locally_relevant);
      if (inhomogeneous)
        {
          constraints.add_line(0);
          constraints.add_entry(0, 1, 0.5);
          constraints.set_inhomogeneity(0, 2.0);
        }
      constraints.close();
      representation =
        std::make_unique<FiniteElementRepresentation<2>>(triangulation,
                                                         dof_handler,
                                                         locally_owned,
                                                         locally_relevant,
                                                         constraints);
    }
  };

  void
  fill_state(const IndexSet &owned, ImmersXLA::MPI::Vector &state)
  {
    state.reinit(owned, MPI_COMM_WORLD);
    for (const auto index : owned)
      state[index] = 0.25 + 0.5 * index + 0.125 * index * index;
  }

  double
  local_dot(const IndexSet               &owned,
            const ImmersXLA::MPI::Vector &left,
            const ImmersXLA::MPI::Vector &right)
  {
    double result = 0.;
    for (const auto index : owned)
      result += left[index] * right[index];
    return Utilities::MPI::sum(result, MPI_COMM_WORLD);
  }
} // namespace


TEST(RetainedSampling, ValueMatchesDirectFEEvaluation) // NOLINT
{
  SamplingData    data;
  const QGauss<2> quadrature(3);
  const auto      plan =
    make_retained_sampling_plan(*data.representation, quadrature);

  ImmersXLA::MPI::Vector state;
  fill_state(data.locally_owned, state);

  const auto             operator_view = plan.linearize(state);
  ImmersXLA::MPI::Vector sampled;
  operator_view.reinit_range_vector(sampled, false);
  operator_view.vmult(sampled, state);

  ImmersXLA::MPI::Vector relevant;
  relevant.reinit(data.locally_owned, data.locally_relevant, MPI_COMM_WORLD);
  relevant = state;
  relevant.update_ghost_values();

  FEValues<2>                          fe_values(data.representation->mapping(),
                        data.finite_element,
                        quadrature,
                        update_values);
  std::vector<types::global_dof_index> dof_indices(
    data.finite_element.n_dofs_per_cell());
  std::size_t local_point = 0;
  for (const auto &cell : data.dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        for (const auto q : fe_values.quadrature_point_indices())
          {
            double direct = 0.;
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
              direct += fe_values.shape_value(i, q) * relevant[dof_indices[i]];
            EXPECT_DOUBLE_EQ(sampled[plan.point_index(local_point)], direct);
            ++local_point;
          }
      }
  ASSERT_EQ(local_point, plan.points().size());
}


TEST(RetainedSampling,
     MPI_ForwardTransposeUseTheRetainedStencil) // NOLINT
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  SamplingData    data;
  const QGauss<2> quadrature(3);
  const auto      plan =
    make_retained_sampling_plan(*data.representation, quadrature);

  ImmersXLA::MPI::Vector direction;
  fill_state(data.locally_owned, direction);
  ImmersXLA::MPI::Vector weights;
  weights.reinit(plan.locally_owned_points(), MPI_COMM_WORLD);
  for (const auto index : plan.locally_owned_points())
    weights[index] = 0.75 + 0.25 * index;

  const auto             operator_view = plan.linearize(direction);
  ImmersXLA::MPI::Vector values;
  operator_view.reinit_range_vector(values, false);
  operator_view.vmult(values, direction);

  ImmersXLA::MPI::Vector transpose;
  operator_view.reinit_domain_vector(transpose, false);
  operator_view.Tvmult(transpose, weights);

  EXPECT_NEAR(local_dot(plan.locally_owned_points(), values, weights),
              local_dot(data.locally_owned, direction, transpose),
              1.e-9);
}


TEST(RetainedSampling, InhomogeneousConstraintsSeparateStateAndLinearization)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  SamplingData    data(true);
  const QGauss<2> quadrature(3);
  const auto      plan =
    make_retained_sampling_plan(*data.representation, quadrature);

  ImmersXLA::MPI::Vector state;
  fill_state(data.locally_owned, state);

  const auto state_sample  = plan.sample(state);
  const auto operator_view = plan.linearize(state);

  ImmersXLA::MPI::Vector zero;
  operator_view.reinit_domain_vector(zero, false);
  zero = 0.;
  ImmersXLA::MPI::Vector sampled_zero;
  operator_view.reinit_range_vector(sampled_zero, false);
  operator_view.vmult(sampled_zero, zero);
  EXPECT_EQ(sampled_zero.l2_norm(), 0.);

  ImmersXLA::MPI::Vector    homogeneous_state = state;
  AffineConstraints<double> homogeneous_constraints(data.constraints);
  homogeneous_constraints.set_inhomogeneity(0, 0.);
  homogeneous_constraints.distribute(homogeneous_state);

  ImmersXLA::MPI::Vector sampled;
  operator_view.reinit_range_vector(sampled, false);
  operator_view.vmult(sampled, state);

  for (std::size_t q = 0; q < plan.points().size(); ++q)
    {
      double inhomogeneous_contribution = 0.;
      for (std::size_t i = 0; i < plan.points()[q].dof_indices.size(); ++i)
        if (plan.points()[q].dof_indices[i] == 0)
          inhomogeneous_contribution += 2. * plan.points()[q].basis_values[i];
      EXPECT_NEAR(state_sample[plan.point_index(q)] -
                    sampled[plan.point_index(q)],
                  inhomogeneous_contribution,
                  1.e-12);
    }

  for (std::size_t q = 0; q < plan.points().size(); ++q)
    {
      double expected = 0.;
      for (std::size_t i = 0; i < plan.points()[q].dof_indices.size(); ++i)
        expected += plan.points()[q].basis_values[i] *
                    homogeneous_state[plan.points()[q].dof_indices[i]];
      EXPECT_NEAR(sampled[plan.point_index(q)], expected, 1.e-12);
    }

  ImmersXLA::MPI::Vector weights;
  weights.reinit(plan.locally_owned_points(), MPI_COMM_WORLD);
  for (const auto index : plan.locally_owned_points())
    weights[index] = 0.75 + 0.25 * index;

  ImmersXLA::MPI::Vector transpose;
  operator_view.reinit_domain_vector(transpose, false);
  operator_view.Tvmult(transpose, weights);

  EXPECT_NEAR(local_dot(plan.locally_owned_points(), sampled, weights),
              local_dot(data.locally_owned, state, transpose),
              1.e-9);
}
