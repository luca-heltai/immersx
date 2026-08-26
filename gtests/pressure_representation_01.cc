// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/state.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include <cmath>

#include "coupled_poisson_elasticity.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  using StateVector = ImmersX::ImmersXLA::MPI::Vector;

  /** Configure a Poisson<1,3> interval problem with the requested FE and
   *  mesh resolution. */
  void
  configure_poisson_problem(const unsigned int       fe_degree,
                            const unsigned int       initial_refinement,
                            PoissonParameters<1, 3> &parameters)
  {
    parameters.fe_degree          = fe_degree;
    parameters.initial_refinement = initial_refinement;
    parameters.name_of_grid       = "hyper_cube";
    parameters.arguments_for_grid = "0: 1: false";
    parameters.triangulation_type = "fullydistributed";
    initialize_parameters_from_string(R"(
      subsection Poisson
        subsection Right hand side
          set Function expression = 0
          set Variable names      = x,y,z,t
        end
        subsection Dirichlet boundary conditions
          set Function expression = 0
          set Variable names      = x,y,z,t
        end
        subsection Solver
          subsection Control
            set Max steps  = 10
            set Reduction  = 1.e-12
            set Tolerance  = 1.e-14
            set Log result = false
          end
        end
      end
    )");
  }

  double
  owned_dot(const StateVector &left, const StateVector &right)
  {
    double local = 0.;
    for (const auto index : left.locally_owned_elements())
      local += left[index] * right[index];
    return Utilities::MPI::sum(local, left.get_mpi_communicator());
  }

  /**
   * Verify the operator duality <A x, y> = <x, A^T y> for the pressure
   * representation's point evaluation, together with the additive variants.
   */
  void
  check_pressure_duality(const unsigned int fe_degree,
                         const unsigned int initial_refinement)
  {
    ParameterAcceptor::clear();
    PoissonParameters<1, 3>                parameters;
    CoupledPoissonElasticity::PressureLift lift("/Pressure duality lift/");
    lift.thickness                 = "0.1";
    lift.representative_n_q_points = 2;
    configure_poisson_problem(fe_degree, initial_refinement, parameters);
    PoissonSolver<1, 3> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_system();

    // The gate requires FE degree >= 2, more than one cell, more than two DoFs.
    ASSERT_GE(fe_degree, 2u);
    ASSERT_GT(problem.triangulation().n_global_active_cells(), 1u);
    ASSERT_GE(problem.n_dofs(), 3u);

    ImmersX::StateLayout     layout;
    ImmersX::FieldDescriptor descriptor;
    descriptor.name           = "pressure";
    const auto pressure_field = layout.add_field(descriptor);

    StateVector state_values;
    state_values.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    state_values = 0.;
    ImmersX::StateView<StateVector> state_view(layout, 0.);
    state_view.bind(pressure_field, state_values);
    const ImmersX::EvaluationContext<StateVector> context(0.,
                                                          state_view,
                                                          nullptr);

    const double                            factor = 2.;
    const FiniteElementRepresentation<1, 3> source_representation(
      problem.triangulation(),
      problem.dof_handler(),
      problem.locally_owned_dofs(),
      problem.locally_relevant_dofs(),
      problem.constraints());
    const auto pressure = make_fe_expression(source_representation,
                                             {value(pressure_field, "A")},
                                             "factor*A",
                                             {{"factor", factor}});

    // Sampling is deferred until the tensor-product lift supplies its
    // representative quadrature.
    const auto quantity = pressure.lift(lift);
    ASSERT_FALSE(quantity.lifted_points().empty());
    EXPECT_EQ(quantity.source_representation().quantity_space().domain(),
              (RepresentationDomain(1, 3, "retained-fe-sampling")));
    EXPECT_EQ(quantity.quantity_space().domain(),
              (RepresentationDomain(2, 3, "tensor-product-lift")));
    const auto A = quantity.linearize(context);

    // Deterministic coefficient and point-value vectors.
    StateVector x;
    x.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    for (const auto index : x.locally_owned_elements())
      x[index] = std::sin(1. + 0.7 * static_cast<double>(index)) +
                 0.5 * static_cast<double>(index);

    using QuantityVector = typename decltype(quantity)::value_type;
    QuantityVector y;
    y.reinit(quantity.locally_owned_points(),
             quantity.locally_relevant_points(),
             MPI_COMM_WORLD);
    for (const auto index : quantity.locally_owned_points())
      y[index] = 0.3 * static_cast<double>(index) + 1.;

    QuantityVector Ax;
    A.reinit_range_vector(Ax, false);
    A.vmult(Ax, x);

    StateVector ATy;
    A.reinit_domain_vector(ATy, false);
    A.Tvmult(ATy, y);

    double forward_local = 0.;
    for (const auto index : quantity.locally_owned_points())
      forward_local += Ax[index] * y[index];
    const double inner_forward =
      Utilities::MPI::sum(forward_local, MPI_COMM_WORLD);
    const double inner_transpose = owned_dot(x, ATy);
    EXPECT_NEAR(inner_forward,
                inner_transpose,
                1.e-10 * std::max(1., std::abs(inner_forward)))
      << "Forward and transpose point evaluations are not adjoint.";

    // Additive variants must agree with the non-additive ones.
    QuantityVector Ax_add;
    Ax_add.reinit(quantity.locally_owned_points(),
                  quantity.locally_relevant_points(),
                  MPI_COMM_WORLD);
    Ax_add = 0.;
    A.vmult_add(Ax_add, x);
    for (const auto index : quantity.locally_owned_points())
      EXPECT_NEAR(Ax_add[index],
                  Ax[index],
                  1.e-12 * std::max(1., std::abs(Ax[index])))
        << "vmult_add disagrees at point " << index;

    StateVector ATy_add;
    A.reinit_domain_vector(ATy_add, false);
    ATy_add = 0.;
    A.Tvmult_add(ATy_add, y);
    for (const auto index : x.locally_owned_elements())
      EXPECT_NEAR(ATy_add[index],
                  ATy[index],
                  1.e-12 * std::max(1., std::abs(ATy[index])))
        << "Tvmult_add disagrees at DoF " << index;
  }
} // namespace


TEST(DeferredPressureExpression, PointEvaluationDuality) // NOLINT
{
  check_pressure_duality(2, 2);
}


TEST(DeferredPressureExpression, MPI_PointEvaluationDuality) // NOLINT
{
  // Refined enough that every rank owns cells in a two-rank run.
  check_pressure_duality(2, 3);
}
