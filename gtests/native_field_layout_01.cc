// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>

#include "native_field_layout.h"


namespace
{
  struct NativeTwoByTwoOperator
  {
    dealii::FullMatrix<double> uu{1, 1};
    dealii::FullMatrix<double> up{1, 1};
    dealii::FullMatrix<double> pu{1, 1};
    dealii::FullMatrix<double> pp{1, 1};

    void
    vmult(dealii::BlockVector<double>       &dst,
          const dealii::BlockVector<double> &src) const
    {
      dst.reinit(2);
      dst.block(0).reinit(1);
      dst.block(1).reinit(1);
      dst.collect_sizes();
      uu.vmult(dst.block(0), src.block(0));
      up.vmult_add(dst.block(0), src.block(1));
      pu.vmult(dst.block(1), src.block(0));
      pp.vmult_add(dst.block(1), src.block(1));
    }
  };
} // namespace


TEST(NativeFieldLayout, GathersAndScattersLocalBlockOperator) // NOLINT
{
  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor u_descriptor;
  u_descriptor.name          = "velocity";
  const auto               u = layout.add_field(u_descriptor);
  ImmersX::FieldDescriptor p_descriptor;
  p_descriptor.name      = "pressure";
  p_descriptor.time_role = ImmersX::TimeRole::algebraic;
  const auto p           = layout.add_field(p_descriptor);

  dealii::Vector<double> u_state(1);
  dealii::Vector<double> p_state(1);
  u_state[0] = 2.;
  p_state[0] = 3.;
  ImmersX::StateView<dealii::Vector<double>> state(layout, 0.);
  state.bind(u, u_state).bind(p, p_state);

  ImmersX::NativeFieldLayout native_layout(layout);
  native_layout.add_block(u);
  native_layout.add_block(p);

  dealii::BlockVector<double> native_state;
  native_layout.gather(state, 0., native_state);
  EXPECT_DOUBLE_EQ(native_state.block(0)[0], 2.);
  EXPECT_DOUBLE_EQ(native_state.block(1)[0], 3.);

  NativeTwoByTwoOperator native_operator;
  native_operator.uu(0, 0) = 2.;
  native_operator.up(0, 0) = 3.;
  native_operator.pu(0, 0) = 5.;
  native_operator.pp(0, 0) = 7.;

  dealii::BlockVector<double> native_residual;
  native_operator.vmult(native_residual, native_state);

  dealii::Vector<double> u_residual(1);
  dealii::Vector<double> p_residual(1);
  u_residual = 0.;
  p_residual = 0.;
  ImmersX::ResidualAccumulator<dealii::Vector<double>> residual(layout);
  residual.bind(u, u_residual).bind(p, p_residual);
  native_layout.scatter_add(native_residual, residual);

  EXPECT_DOUBLE_EQ(u_residual[0], 13.);
  EXPECT_DOUBLE_EQ(p_residual[0], 31.);
}
