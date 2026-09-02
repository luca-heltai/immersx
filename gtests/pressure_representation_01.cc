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

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/state.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include <cmath>
#include <type_traits>

#include "coupled_poisson_elasticity.h"
#include "test_paths.h"

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

  class SingleFieldState : public StateAccessor<StateVector>
  {
  public:
    SingleFieldState(const FieldId field, const StateVector &values)
      : field_(field)
      , values_(&values)
    {}

    const StateVector &
    field(const FieldId field, const double) const override
    {
      AssertThrow(field == field_,
                  ExcMessage("Unknown field in pressure test state."));
      return *values_;
    }

  private:
    FieldId            field_;
    const StateVector *values_;
  };

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
    const auto pressure = make_fe_expression(
      source_representation,
      {value(state_field(source_representation, pressure_field), "A")},
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


void
check_path_pressure_with_elasticity(const unsigned int expected_processes)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
            expected_processes);
  ParameterAcceptor::clear();

  PoissonParameters<1, 3> parameters;
  configure_poisson_problem(2, 0, parameters);
  const auto path_filename =
    TestPaths::data_filename("tests/path_length_1d.vtk");
  parameters.name_of_grid       = path_filename;
  parameters.arguments_for_grid = "";

  PoissonSolver<1, 3> poisson(parameters);
  poisson.make_grid();
  poisson.set_imported_fields(
    std::make_shared<PoissonSolver<1, 3>::ImportedFields>(
      path_filename, poisson.triangulation()));
  poisson.setup_fe();
  poisson.setup_system();
  poisson.assemble_system();

  ElasticStaticParameters<3, 3> elasticity_parameters;
  elasticity_parameters.initial_refinement = 1;
  elasticity_parameters.triangulation_type = "fullydistributed";
  ElasticStaticProblem<3, 3> elasticity(elasticity_parameters);
  elasticity.setup();
  CoupledPoissonElasticity::Traction traction;
  traction.attach(elasticity);

  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = LinearAdapter<FieldVector, GlobalVector>;
  LinearSolverParameters adapter_parameters;
  Adapter                adapter(
    adapter_parameters,
    MPI_COMM_WORLD,
    [](const auto &operator_view, const auto &rhs, auto &solution) {
      dealii::SolverControl control(500, 1.e-12 * rhs.l2_norm(), false);
      using GlobalVector = std::decay_t<decltype(solution)>;
      dealii::SolverGMRES<GlobalVector> solver(control);
      dealii::PreconditionIdentity      preconditioner;
      solver.solve(operator_view, solution, rhs, preconditioner);
    });

  const auto poisson_handle = adapter.add(poisson);
  const auto elastic_handle = adapter.add(elasticity);
  const CoupledPoissonElasticity::PathPressure path_pressure{1.25,
                                                             0.75,
                                                             0.5,
                                                             "path_length"};
  const auto pressure = poisson_handle.observe(path_pressure);
  const auto lifted   = pressure.lift(CoupledPoissonElasticity::PressureLift());
  adapter.couple(lifted, elastic_handle, traction);

  const auto &expression = lifted.source_representation();
  ASSERT_EQ(expression.dependencies().size(), 1u);
  EXPECT_EQ(expression.dependencies().front(),
            poisson_handle.fields().solution);
  ASSERT_EQ(expression.n_sampling_sources(), 2u);
  EXPECT_NE(expression.sampling_plan_for_source(0).update_flags() &
              update_gradients,
            UpdateFlags());
  EXPECT_EQ(expression.sampling_plan_for_source(1).update_flags() &
              update_gradients,
            UpdateFlags());

  auto  state         = adapter.make_state();
  auto &poisson_state = adapter.field(state, poisson_handle.fields().solution);
  for (const auto index : poisson_state.locally_owned_elements())
    poisson_state[index] = std::sin(0.2 + 0.17 * static_cast<double>(index));

  SingleFieldState original_state(poisson_handle.fields().solution,
                                  poisson_state);
  const EvaluationContext<StateVector> original_context(0., original_state);
  const auto                           sampled =
    CoupledPoissonElasticity::sample_pressure(lifted, poisson_state);
  EXPECT_GT(sampled.locally_owned_elements().n_elements(), 0u);
  const auto source_values = expression.evaluate(original_context);

  const auto source_plan = expression.sampling_plan_for_source(0);
  const auto path_plan   = expression.sampling_plan_for_source(1);
  const auto gradient_operator =
    source_plan.gradient_linearize(poisson_state, 2);
  StateVector sampled_gradient;
  gradient_operator.reinit_range_vector(sampled_gradient, false);
  gradient_operator.vmult(sampled_gradient, poisson_state);
  const auto  path_operator = path_plan.linearize(poisson_state);
  StateVector sampled_path;
  path_operator.reinit_range_vector(sampled_path, false);
  path_operator.vmult(
    sampled_path,
    poisson.imported_fields()->field("path_length").coefficients());
  for (const auto index : source_plan.locally_owned_points())
    EXPECT_NEAR(source_values[index],
                path_pressure.alpha * sampled_gradient[index] +
                  path_pressure.beta *
                    std::cos(path_pressure.omega * sampled_path[index]),
                1.e-12);

  const auto jacobian =
    expression.linearize(original_context, poisson_handle.fields().solution);
  StateVector direction = poisson_state;
  StateVector jacobian_direction;
  jacobian.reinit_range_vector(jacobian_direction, false);
  jacobian.vmult(jacobian_direction, direction);
  const double epsilon         = 1.e-7;
  auto         perturbed_state = poisson_state;
  for (const auto index : perturbed_state.locally_owned_elements())
    perturbed_state[index] += epsilon * direction[index];
  SingleFieldState perturbed_state_accessor(poisson_handle.fields().solution,
                                            perturbed_state);
  const EvaluationContext<StateVector> perturbed_context(
    0., perturbed_state_accessor);
  const auto original_value  = expression.evaluate(original_context);
  const auto perturbed_value = expression.evaluate(perturbed_context);
  for (const auto index : source_plan.locally_owned_points())
    EXPECT_NEAR((perturbed_value[index] - original_value[index]) / epsilon,
                jacobian_direction[index],
                2.e-7 * std::max(1., std::abs(jacobian_direction[index])));

  StateVector dual;
  dual.reinit(source_plan.locally_owned_points(),
              source_plan.locally_relevant_points(),
              MPI_COMM_WORLD);
  for (const auto index : dual.locally_owned_elements())
    dual[index] = 0.4 + 0.03 * static_cast<double>(index);
  StateVector transpose_action;
  jacobian.reinit_domain_vector(transpose_action, false);
  jacobian.Tvmult(transpose_action, dual);
  EXPECT_NEAR(owned_dot(jacobian_direction, dual),
              owned_dot(direction, transpose_action),
              2.e-10 *
                std::max(1., std::abs(owned_dot(jacobian_direction, dual))));

  const auto path_view     = poisson.imported_fields()->field("path_length");
  const auto path_gradient = frozen_field(path_view);
  const FiniteElementRepresentation<1, 3> source_representation(
    poisson.triangulation(),
    poisson.dof_handler(),
    poisson.locally_owned_dofs(),
    poisson.locally_relevant_dofs(),
    poisson.constraints());
  const auto gradient_representation = make_expression_representation(
    source_representation,
    QGauss<1>(2),
    {gradient(path_gradient, "path_gradient", 2)},
    "path_gradient");
  StateLayout                          empty_layout;
  StateView<StateVector>               empty_view(empty_layout, 0.);
  const EvaluationContext<StateVector> empty_context(0., empty_view);
  const auto gradient_values = gradient_representation.evaluate(empty_context);
  EXPECT_EQ(gradient_representation.sampling_plan().update_flags() &
              update_gradients,
            update_gradients);
  for (const auto index :
       gradient_representation.sampling_plan().locally_owned_points())
    EXPECT_NEAR(gradient_values[index], 1., 1.e-12);

  const auto old_pressure =
    poisson_handle.observe(CoupledPoissonElasticity::Pressure{2.});
  const auto old_expression =
    old_pressure.lift(CoupledPoissonElasticity::PressureLift())
      .source_representation();
  const auto  old_values = old_expression.evaluate(original_context);
  const auto  old_plan   = old_expression.sampling_plan();
  const auto  old_active = old_plan.linearize(poisson_state);
  StateVector old_samples;
  old_active.reinit_range_vector(old_samples, false);
  old_active.vmult(old_samples, poisson_state);
  for (const auto index : old_plan.locally_owned_points())
    EXPECT_NEAR(old_values[index], 2. * old_samples[index], 1.e-12);

  adapter.solve(state);
  auto residual = adapter.make_state();
  adapter.evaluate_residual(state, residual);
  EXPECT_LT(residual.l2_norm(), 1.e-9);
  const auto &solved_poisson =
    adapter.field(state, poisson_handle.fields().solution);
  const auto &solved_elastic =
    adapter.field(state, elastic_handle.fields().displacement);
  EXPECT_LT(CoupledPoissonElasticity::traction_balance(
              lifted, traction, solved_poisson, solved_elastic),
            1.e-9);
}


TEST(DeferredPressureExpression, ImportedPathLengthWithElasticity)
{
  check_path_pressure_with_elasticity(1);
}


TEST(DeferredPressureExpression, MPI_ImportedPathLengthWithElasticity)
{
  check_path_pressure_with_elasticity(2);
}
