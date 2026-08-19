// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <gtest/gtest.h>

#include "representation.h"


using IdentityLine       = IdentityRepresentation<1, 2>;
using CylindricalSurface = TensorProductRepresentation<1, 2, 3, 1>;

static_assert(RepresentationConcept<IdentityLine>::value,
              "The direct FE representation must satisfy the interaction "
              "contract.");
static_assert(RepresentationConcept<CylindricalSurface>::value,
              "The tensor-product representation must satisfy the "
              "interaction contract.");


TEST(RepresentationContract, IdentityAndTensorProductDimensions) // NOLINT
{
  EXPECT_EQ(IdentityLine::support_dimension, 1u);
  EXPECT_EQ(IdentityLine::representative_dimension, 1u);
  EXPECT_EQ(IdentityLine::ambient_dimension, 2u);

  EXPECT_EQ(CylindricalSurface::support_dimension, 2u);
  EXPECT_EQ(CylindricalSurface::representative_dimension, 1u);
  EXPECT_EQ(CylindricalSurface::ambient_dimension, 3u);
}
