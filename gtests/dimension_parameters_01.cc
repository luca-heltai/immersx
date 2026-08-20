// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top of
// the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>
#include <immersx/io/utils.h>

using namespace ImmersX;


TEST(DimensionParameters, ReadTopLevelValues)
{
  dealii::ParameterHandler prm;
  declare_dimension_parameters(prm);

  prm.parse_input_from_string(R"(
set dimension         = 1
set space dimension   = 3
set reduced dimension = 1
set cross section dimension = 2
)");

  const auto dimensions = get_dimension_parameters(prm);
  EXPECT_EQ(dimensions.dimension, 1u);
  EXPECT_EQ(dimensions.space_dimension, 3u);
  EXPECT_EQ(dimensions.reduced_dimension, 1u);
  EXPECT_EQ(dimensions.cross_section_dimension, 2u);
}


TEST(DimensionParameters, Defaults)
{
  dealii::ParameterHandler prm;
  declare_dimension_parameters(prm);

  const auto dimensions = get_dimension_parameters(prm);
  EXPECT_EQ(dimensions.dimension, 2u);
  EXPECT_EQ(dimensions.space_dimension, 2u);
  EXPECT_EQ(dimensions.reduced_dimension, 1u);
  EXPECT_EQ(dimensions.cross_section_dimension, 2u);
}
