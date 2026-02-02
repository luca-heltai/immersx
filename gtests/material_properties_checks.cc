// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mutex.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "elasticity.h"
#include "utils.h"

using namespace dealii;

TEST(MaterialParameters, Default)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;
  initialize_parameters("", "test.prm");

  std::ifstream in("test.prm");
  ASSERT_TRUE(in.good())
    << "Expected initialize_parameters() to generate test.prm";

  std::ostringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();

  EXPECT_NE(text.find("Material tags by material id"), std::string::npos)
    << "test.prm does not contain the expected parameter entry.";
}


TEST(MaterialParameters, StiffTagReadsSection)
{
  ParameterAcceptor::clear();
  static constexpr int             dim = 2;
  ElasticityProblemParameters<dim> par;

  const std::string prm_text = R"(
    subsection Immersed Problem
      subsection Material properties
        set Material tags by material id = 0: default, 1: stiff
        subsection default
          set Lame mu     = 2.0
          set Lame lambda = 3.0
        end
        subsection stiff
          set Lame mu     = 20.0
          set Lame lambda = 30.0
        end
      end
    end
  )";

  initialize_parameters_from_string(prm_text, "test.prm");

  ASSERT_EQ(par.material_properties_by_id.size(), 2);

  ASSERT_NE(par.material_properties_by_id.at(0), nullptr);
  ASSERT_NE(par.material_properties_by_id.at(1), nullptr);

  EXPECT_NEAR(par.material_properties_by_id.at(0)->Lame_mu, 2.0, 1e-14);
  EXPECT_NEAR(par.material_properties_by_id.at(0)->Lame_lambda, 3.0, 1e-14);
  EXPECT_NEAR(par.material_properties_by_id.at(1)->Lame_mu, 20.0, 1e-14);
  EXPECT_NEAR(par.material_properties_by_id.at(1)->Lame_lambda, 30.0, 1e-14);
}
