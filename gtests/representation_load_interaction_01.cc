// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/load_interaction.h>

namespace ImmersX
{
  struct LoadFields
  {
    FieldId pressure;
    FieldId force;
  };

  template <typename Builder>
  LoadFields
  contribute(Builder &builder, const dealii::IndexSet &owned)
  {
    return {builder.algebraic_field("pressure", owned),
            builder.algebraic_field("force", owned)};
  }
} // namespace ImmersX

TEST(RepresentationLoadInteraction, PressureMapsToForceWithoutField)
{
  using FieldVector  = ImmersX::ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
  using Adapter      = ImmersX::LinearAdapter<FieldVector, GlobalVector>;

  dealii::IndexSet owned(2);
  owned.add_range(0, 2);
  owned.compress();

  Adapter    adapter(MPI_COMM_WORLD,
                  [](const auto &, const auto &, auto &solution) {
                    solution = 0.;
                  });
  const auto fields = adapter.add(owned, "state");

  ImmersX::ImmersXLA::MPI::SparseMatrix matrix;
  dealii::DynamicSparsityPattern        sparsity(2, 2);
  sparsity.add(0, 0);
  sparsity.add(1, 1);
  matrix.reinit(owned, owned, sparsity, MPI_COMM_WORLD);
  matrix.set(0, 0, 2.);
  matrix.set(1, 1, 3.);
  matrix.compress(dealii::VectorOperation::insert);

  const auto pressure = adapter.observe(fields.pressure);
  const auto load     = ImmersX::payload_free(
    dealii::linear_operator<FieldVector, FieldVector>(matrix));
  adapter.add(ImmersX::PressureLoadInteraction<FieldVector>(pressure,
                                                            fields.force,
                                                            load),
              "pressure");

  auto state                               = adapter.make_state();
  adapter.field(state, fields.pressure)[0] = 4.;
  adapter.field(state, fields.pressure)[1] = 5.;

  EXPECT_EQ(state.n_blocks(), 2u);

  GlobalVector residual;
  adapter.evaluate_residual(state, residual);
  EXPECT_DOUBLE_EQ(adapter.field(residual, fields.pressure)[0], 0.);
  EXPECT_DOUBLE_EQ(adapter.field(residual, fields.pressure)[1], 0.);
  EXPECT_DOUBLE_EQ(adapter.field(residual, fields.force)[0], 8.);
  EXPECT_DOUBLE_EQ(adapter.field(residual, fields.force)[1], 15.);

  auto direction                               = adapter.make_state();
  adapter.field(direction, fields.pressure)[0] = 1.;
  adapter.field(direction, fields.pressure)[1] = 1.;
  auto jacobian_action                         = adapter.make_state();
  adapter.jacobian(state).vmult(jacobian_action, direction);
  EXPECT_DOUBLE_EQ(adapter.field(jacobian_action, fields.force)[0], 2.);
  EXPECT_DOUBLE_EQ(adapter.field(jacobian_action, fields.force)[1], 3.);
}
