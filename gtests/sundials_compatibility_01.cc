// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/compatibility/dealii_9_9/ida.h>
#include <immersx/core/time_parameters.h>

// Both implementations must be usable in the same translation unit.
#include <deal.II/sundials/ida.h>
#include <deal.II/sundials/n_vector.templates.h>

#include <cmath>
#include <limits>
#include <type_traits>

#ifdef DEAL_II_WITH_SUNDIALS
namespace
{
  namespace Backport = ImmersX::Compatibility::SUNDIALS;
  using Vector       = dealii::Vector<double>;
  using IDA          = Backport::IDA<Vector>;

  template <typename VectorType>
  void
  check_vector_operations(VectorType &x)
  {
    VectorType   constraints(x), mask(x), numerator(x), denominator(x);
    const double values[]     = {-2., 0., 2., 0., -2., 2., 0., -2., 0., 2.};
    const double codes[]      = {2., 2., 2., 1., 1., -1., -1., -2., -2., 0.};
    const double violations[] = {1., 1., 0., 0., 1., 1., 0., 0., 1., 0.};
    const auto   owned        = x.locally_owned_elements();
    for (const auto i : owned)
      {
        x[i]           = values[i];
        constraints[i] = codes[i];
        numerator[i]   = 10. - i;
        denominator[i] = i == 0 ? 0. : 2.;
      }
    x.compress(dealii::VectorOperation::insert);
    constraints.compress(dealii::VectorOperation::insert);
    numerator.compress(dealii::VectorOperation::insert);
    denominator.compress(dealii::VectorOperation::insert);

    // The IDA object owns a context with the right SUNDIALS version semantics.
    Backport::IDA<VectorType> context_owner;
    const auto                view = [&](auto &v) {
      return Backport::internal::make_nvector_view(
        v
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 0, 0)
        ,
        context_owner.get_sun_context()
#  endif
      );
    };
    auto nx = view(x);
    auto nc = view(constraints);
    auto nm = view(mask);
    auto nn = view(numerator);
    auto nd = view(denominator);

    N_VCompare(2., nx, nm);
    for (const auto i : owned)
      EXPECT_DOUBLE_EQ(mask[i], std::abs(values[i]) >= 2. ? 1. : 0.);
    EXPECT_EQ(N_VConstrMask(nc, nx, nm), SUNFALSE);
    for (const auto i : owned)
      EXPECT_DOUBLE_EQ(mask[i], violations[i]);
    constraints = 0.;
    EXPECT_EQ(N_VConstrMask(nc, nx, nm), SUNTRUE);
    EXPECT_DOUBLE_EQ(mask.linfty_norm(), 0.);
    // The minimum belongs to the last rank, not rank zero.
    EXPECT_DOUBLE_EQ(N_VMinQuotient(nn, nd), 0.5);
    denominator = 0.;
    EXPECT_DOUBLE_EQ(N_VMinQuotient(nn, nd),
                     std::numeric_limits<Backport::realtype>::max());
  }
} // namespace

TEST(SundialsCompatibility, SerialVectorOperations)
{
  Vector x(10);
  check_vector_operations(x);
  dealii::BlockVector<double> blocks(
    std::vector<dealii::types::global_dof_index>{4, 6});
  check_vector_operations(blocks);
}

TEST(SundialsCompatibility, BOTH_DistributedBlockVectorOperations)
{
  const auto rank  = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  std::vector<dealii::IndexSet> owned(2, dealii::IndexSet(5));
  for (auto &indices : owned)
    indices.add_range(5 * rank / ranks, 5 * (rank + 1) / ranks);
  ImmersX::ImmersXLA::MPI::BlockVector x;
  x.reinit(owned, MPI_COMM_WORLD);
  check_vector_operations(x);
}

TEST(SundialsCompatibility, ControlsConstraintsAndExpertSetup)
{
  static_assert(!std::is_same_v<IDA, dealii::SUNDIALS::IDA<Vector>>);
  for (const double constraint : {0., 2.})
    {
      IDA::AdditionalData data;
      EXPECT_DOUBLE_EQ(data.minimum_step_size, 1.e-15);
      data.final_time                        = 0.8;
      data.output_period                     = 0.2;
      data.initial_step_size                 = 0.01;
      data.minimum_step_size                 = 0.005;
      data.maximum_step_size                 = 0.025;
      data.maximum_number_of_steps           = 100;
      data.maximum_error_test_failures       = 7;
      data.ic_type                           = IDA::AdditionalData::use_y_diff;
      data.reset_type                        = IDA::AdditionalData::none;
      data.ignore_algebraic_terms_for_errors = false;
      IDA    ida(data);
      Vector y(1), yp(1), constraints(1);
      y[0]           = 1.;
      yp[0]          = 17.; // IDACalcIC must correct a nonzero residual.
      constraints[0] = constraint;
      unsigned int setup_count = 0;
      double       alpha       = 1.;
      EXPECT_EQ(ida.get_ida_memory(), nullptr);
      ida.reinit_vector = [](Vector &v) { v.reinit(1); };
      ida.residual = [&](double, const Vector &, const Vector &dot, Vector &r) {
        EXPECT_GT(setup_count, 0u);
        r = dot;
        r[0] += 1.;
      };
      ida.setup_jacobian =
        [&](double, const Vector &, const Vector &, double a) { alpha = a; };
      ida.solve_with_jacobian = [&](const Vector &rhs, Vector &dst, double) {
        dst = rhs;
        dst /= alpha;
      };
      ida.get_constraint_vector = [&]() -> Vector & { return constraints; };
      ida.custom_setup          = [&](void *memory) {
        ++setup_count;
        EXPECT_EQ(memory, ida.get_ida_memory());
        void *user_data = nullptr;
        EXPECT_EQ(IDAGetUserData(memory, &user_data), IDA_SUCCESS);
        EXPECT_EQ(user_data, &ida);
      };
      ida.output_step = [](double        t,
                           const Vector &state,
                           const Vector &dot,
                           unsigned int  step) {
        EXPECT_NEAR(state[0], 1. - t, 1.e-8);
        EXPECT_NEAR(dot[0], -1., 1.e-8);
        EXPECT_GT(state[0], 0.);
        if (step == 0)
          {
            EXPECT_DOUBLE_EQ(t, 0.);
          }
      };
      EXPECT_GE(ida.solve_dae(y, yp), 32u);
      EXPECT_EQ(setup_count, 1u);
      EXPECT_NEAR(y[0], 0.2, 1.e-8);
      long int steps = 0;
      EXPECT_EQ(IDAGetNumSteps(ida.get_ida_memory(), &steps), IDA_SUCCESS);
      EXPECT_GE(steps, 32);
      Backport::realtype last_step = 0.;
      EXPECT_EQ(IDAGetLastStep(ida.get_ida_memory(), &last_step), IDA_SUCCESS);
      EXPECT_LE(last_step, data.maximum_step_size * (1. + 1.e-12));
#  if DEAL_II_SUNDIALS_VERSION_GTE(6, 2, 0)
      EXPECT_GE(last_step, data.minimum_step_size);
#  endif
      ida.reset(0.1, 0.01, y, yp);
      EXPECT_EQ(setup_count, 2u);
    }
}

TEST(SundialsCompatibility, ParametersAreInstanceLocal)
{
  IDA::AdditionalData first, second;
  first.ic_type  = IDA::AdditionalData::none;
  second.ic_type = IDA::AdditionalData::use_y_dot;
  dealii::ParameterHandler first_prm, second_prm;
  first.add_parameters(first_prm);
  second.add_parameters(second_prm);
  first_prm.parse_input_from_string(
    "subsection Running parameters\n set Maximum step size = 0.125\nend\n"
    "subsection Initial condition correction parameters\n"
    " set Correction type at initial time = use_y_diff\nend\n"
    "subsection Linear solver\n set Linear tolerance factor = 0.1\nend\n");
  EXPECT_DOUBLE_EQ(first.maximum_step_size, 0.125);
  EXPECT_DOUBLE_EQ(first.linear_tolerance_factor, 0.1);
  EXPECT_EQ(first.ic_type, IDA::AdditionalData::use_y_diff);
  EXPECT_EQ(second.ic_type, IDA::AdditionalData::use_y_dot);
}

TEST(SundialsCompatibility, SharedIDASectionGivesOneValuePerEntry)
{
  using dealii::ParameterAcceptor;
  ParameterAcceptor::clear();
  auto &prm = ParameterAcceptor::prm;

  // A native solver binds the section first, as MetricFlowX does.
  dealii::SUNDIALS::IDA<Vector>::AdditionalData native;
  prm.enter_subsection("Shared IDA");
  native.add_parameters(prm);
  prm.leave_subsection();

  ImmersX::TimeIntervalParameters interval("/", false);
  ImmersX::IDAParameters          ida("/Shared IDA/", &interval);
  ParameterAcceptor::initialize();
  prm.parse_input_from_string("subsection Shared IDA\n"
                              " subsection Error control\n"
                              "  set Relative error tolerance = 0.5\n"
                              " end\n"
                              "end\n");

  // Entries missing from the input hold the ImmersX defaults in both copies.
  EXPECT_DOUBLE_EQ(native.initial_step_size, 1.e-5);
  EXPECT_EQ(native.maximum_order, 1u);
  EXPECT_DOUBLE_EQ(native.absolute_tolerance, 1.e-4);
  EXPECT_DOUBLE_EQ(native.initial_step_size, ida.initial_step_size);
  EXPECT_EQ(native.maximum_order, ida.maximum_order);
  EXPECT_DOUBLE_EQ(native.absolute_tolerance, ida.absolute_tolerance);
  EXPECT_DOUBLE_EQ(native.relative_tolerance, 0.5);
  EXPECT_DOUBLE_EQ(ida.relative_tolerance, 0.5);

  // The global handler holds actions bound to the local variables above.
  ParameterAcceptor::clear();
}
#else
TEST(SundialsCompatibility, Unavailable)
{
  GTEST_SKIP() << "deal.II was configured without SUNDIALS support.";
}
#endif
