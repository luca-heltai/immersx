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
#include <numbers>
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

TEST(BoundaryConditionParameters, DirichletFallbackCopiesDefaultConfiguration)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 501
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function expression = 1 + t; 2 - x + t
        set Modulation frequency = 2.0
        set Phase shift = 0.3
      end
    end
  )");

  const auto &default_bc  = par.bc;
  const auto &override_bc = par.get_dirichlet_bc(501);

  ASSERT_EQ(par.dirichlet_bc_by_id.size(), 1);
  ASSERT_NE(par.dirichlet_bc_by_id.at(501), nullptr);
  EXPECT_NE(&override_bc, &default_bc);

  const Point<dim> p(0.25, 0.5);
  par.set_boundary_condition_times(0.125);

  EXPECT_NEAR(override_bc.value(p, 0), default_bc.value(p, 0), 1e-14);
  EXPECT_NEAR(override_bc.value(p, 1), default_bc.value(p, 1), 1e-14);
  EXPECT_NEAR(override_bc.scale(0.125), default_bc.scale(0.125), 1e-14);
}

TEST(BoundaryConditionParameters, DirichletOverrideUsesBoundarySpecificSection)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 501, 502
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function expression = 1; 2
        set Modulation frequency = 1.0
        set Phase shift = 0.0
      end
      subsection Dirichlet boundary conditions 501
        set Function expression = 10; 20
        set Modulation frequency = 3.0
        set Phase shift = 1.5707963267948966
      end
    end
  )");

  const Point<dim> p;
  const auto      &bc_501 = par.get_dirichlet_bc(501);
  const auto      &bc_502 = par.get_dirichlet_bc(502);

  par.set_boundary_condition_times(0.0);

  EXPECT_NEAR(bc_501.value(p, 0), 10.0, 1e-14);
  EXPECT_NEAR(bc_501.value(p, 1), 20.0, 1e-14);
  EXPECT_NEAR(bc_501.scale(0.0), 1.0, 1e-14);

  EXPECT_NEAR(bc_502.value(p, 0), 1.0, 1e-14);
  EXPECT_NEAR(bc_502.value(p, 1), 2.0, 1e-14);
  EXPECT_NEAR(bc_502.scale(0.25), 1.0, 1e-14);
}

TEST(BoundaryConditionParameters, NeumannOverrideUsesBoundarySpecificSection)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Neumann boundary ids = 7, 8
    end

    subsection Functions
      subsection Neumann boundary conditions
        set Function expression = x + t; y - t
        set Modulation frequency = 0.0
        set Phase shift = 0.7
      end
      subsection Neumann boundary conditions 7
        set Function expression = 3 * x; 4 * y
        set Modulation frequency = 4.0
        set Phase shift = 0.25
      end
    end
  )");

  const Point<dim> p(2.0, 3.0);
  const auto      &bc_7 = par.get_neumann_bc(7);
  const auto      &bc_8 = par.get_neumann_bc(8);

  par.set_boundary_condition_times(0.5);

  EXPECT_NEAR(bc_7.value(p, 0), 6.0, 1e-14);
  EXPECT_NEAR(bc_7.value(p, 1), 12.0, 1e-14);
  EXPECT_NEAR(bc_7.scale(0.0), std::sin(0.25), 1e-14);

  EXPECT_NEAR(bc_8.value(p, 0), 2.5, 1e-14);
  EXPECT_NEAR(bc_8.value(p, 1), 2.5, 1e-14);
  EXPECT_NEAR(bc_8.scale(0.5), 1.0, 1e-14);
}

TEST(RhsParameters, MaterialFallbackCopiesDefaultConfiguration)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Rhs material ids = 1
    end

    subsection Functions
      subsection Right hand side
        set Function expression = x + t; y - t
        set Modulation frequency = 2.0
        set Phase shift = 0.3
      end
    end
  )");

  const auto &default_rhs  = par.rhs;
  const auto &override_rhs = par.get_rhs(1);

  ASSERT_EQ(par.rhs_by_material_id.size(), 1);
  ASSERT_NE(par.rhs_by_material_id.at(1), nullptr);
  EXPECT_NE(&override_rhs, &default_rhs);

  const Point<dim> p(0.25, 0.5);
  par.set_rhs_times(0.125);

  EXPECT_NEAR(override_rhs.value(p, 0), default_rhs.value(p, 0), 1e-14);
  EXPECT_NEAR(override_rhs.value(p, 1), default_rhs.value(p, 1), 1e-14);
  EXPECT_NEAR(override_rhs.scale(0.125), default_rhs.scale(0.125), 1e-14);
}

TEST(RhsParameters, MaterialOverrideUsesMaterialSpecificSection)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Rhs material ids = 1, 2
    end

    subsection Functions
      subsection Right hand side
        set Function expression = 1; 2
        set Modulation frequency = 1.0
        set Phase shift = 0.0
      end
      subsection Right hand side 1
        set Function expression = 10; 20
        set Modulation frequency = 3.0
        set Phase shift = 1.5707963267948966
      end
    end
  )");

  const Point<dim> p;
  const auto      &rhs_1 = par.get_rhs(1);
  const auto      &rhs_2 = par.get_rhs(2);

  par.set_rhs_times(0.0);

  EXPECT_NEAR(rhs_1.value(p, 0), 10.0, 1e-14);
  EXPECT_NEAR(rhs_1.value(p, 1), 20.0, 1e-14);
  EXPECT_NEAR(rhs_1.scale(0.0), 1.0, 1e-14);

  EXPECT_NEAR(rhs_2.value(p, 0), 1.0, 1e-14);
  EXPECT_NEAR(rhs_2.value(p, 1), 2.0, 1e-14);
  EXPECT_NEAR(rhs_2.scale(0.25), 1.0, 1e-14);
}

TEST(BoundaryConditionParameters, DirichletOverrideInheritsBaseConstants)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 41, 42
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function constants = a=2.0, b=3.0
        set Function expression = a * x; b * y
        set Modulation frequency = 1.5
        set Phase shift = 0.1
      end
      subsection Dirichlet boundary conditions 41
        set Function expression = b * x; a * y
      end
    end
  )");

  const Point<dim> p(0.5, 0.25);
  const auto      &bc_41 = par.get_dirichlet_bc(41);
  const auto      &bc_42 = par.get_dirichlet_bc(42);

  par.set_boundary_condition_times(0.125);

  EXPECT_NEAR(bc_41.value(p, 0), 1.5, 1e-14);
  EXPECT_NEAR(bc_41.value(p, 1), 0.5, 1e-14);
  EXPECT_NEAR(bc_41.scale(0.125), std::sin(numbers::PI * 0.375 + 0.1), 1e-14);

  EXPECT_NEAR(bc_42.value(p, 0), 1.0, 1e-14);
  EXPECT_NEAR(bc_42.value(p, 1), 0.75, 1e-14);
  EXPECT_NEAR(bc_42.scale(0.125), std::sin(numbers::PI * 0.375 + 0.1), 1e-14);
}

// Tests with symbolic constants

TEST(BoundaryConditionParameters, DirichletWithSymbolicConstants)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 100
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function constants = a=1.0, b=2.0
        set Function expression = a * x; b * y
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(0.5, 0.75);
  const auto      &bc = par.get_dirichlet_bc(100);

  par.set_boundary_condition_times(0.0);

  EXPECT_NEAR(bc.value(p, 0), 0.5, 1e-14);
  EXPECT_NEAR(bc.value(p, 1), 1.5, 1e-14);
  EXPECT_NEAR(bc.scale(0.0), 1.0, 1e-14);
}

TEST(BoundaryConditionParameters,
     DirichletBaseWithSymbolicConstantsMultipleBoundaries)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 100, 101
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function constants = c=0.5
        set Function expression = c * x; c * y
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(2.0, 4.0);
  const auto      &bc_100 = par.get_dirichlet_bc(100);
  const auto      &bc_101 = par.get_dirichlet_bc(101);

  par.set_boundary_condition_times(0.0);

  EXPECT_NEAR(bc_100.value(p, 0), 1.0, 1e-14);
  EXPECT_NEAR(bc_100.value(p, 1), 2.0, 1e-14);
  EXPECT_NEAR(bc_101.value(p, 0), 1.0, 1e-14);
  EXPECT_NEAR(bc_101.value(p, 1), 2.0, 1e-14);
}

TEST(BoundaryConditionParameters, NeumannWithSymbolicConstants)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Neumann boundary ids = 50
    end

    subsection Functions
      subsection Neumann boundary conditions
        set Function constants = nx=1.5, ny=2.5
        set Function expression = nx * x; ny * y
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(1.0, 2.0);
  const auto      &bc = par.get_neumann_bc(50);

  par.set_boundary_condition_times(0.0);

  EXPECT_NEAR(bc.value(p, 0), 1.5, 1e-14);
  EXPECT_NEAR(bc.value(p, 1), 5.0, 1e-14);
  EXPECT_NEAR(bc.scale(0.0), 1.0, 1e-14);
}

TEST(BoundaryConditionParameters,
     NeumannBaseWithSymbolicConstantsMultipleBoundaries)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Neumann boundary ids = 50, 51
    end

    subsection Functions
      subsection Neumann boundary conditions
        set Function constants = cx=1.0, cy=1.0
        set Function expression = cx * x; cy * y
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(2.0, 3.0);
  const auto      &bc_50 = par.get_neumann_bc(50);
  const auto      &bc_51 = par.get_neumann_bc(51);

  par.set_boundary_condition_times(0.0);

  EXPECT_NEAR(bc_50.value(p, 0), 2.0, 1e-14);
  EXPECT_NEAR(bc_50.value(p, 1), 3.0, 1e-14);
  EXPECT_NEAR(bc_51.value(p, 0), 2.0, 1e-14);
  EXPECT_NEAR(bc_51.value(p, 1), 3.0, 1e-14);
}

TEST(RhsParameters, RhsWithSymbolicConstants)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Rhs material ids = 10
    end

    subsection Functions
      subsection Right hand side
        set Function constants = alpha=2.0, beta=3.0
        set Function expression = alpha * x; beta * y
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(0.5, 1.5);
  const auto      &rhs = par.get_rhs(10);

  par.set_rhs_times(0.0);

  EXPECT_NEAR(rhs.value(p, 0), 1.0, 1e-14);
  EXPECT_NEAR(rhs.value(p, 1), 4.5, 1e-14);
  EXPECT_NEAR(rhs.scale(0.0), 1.0, 1e-14);
}

TEST(RhsParameters, RhsBaseWithSymbolicConstantsMultipleMaterials)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Rhs material ids = 10, 11
    end

    subsection Functions
      subsection Right hand side
        set Function constants = gamma=0.1
        set Function expression = gamma * x; gamma * y
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(2.0, 3.0);
  const auto      &rhs_10 = par.get_rhs(10);
  const auto      &rhs_11 = par.get_rhs(11);

  par.set_rhs_times(0.0);

  EXPECT_NEAR(rhs_10.value(p, 0), 0.2, 1e-14);
  EXPECT_NEAR(rhs_10.value(p, 1), 0.3, 1e-14);
  EXPECT_NEAR(rhs_11.value(p, 0), 0.2, 1e-14);
  EXPECT_NEAR(rhs_11.value(p, 1), 0.3, 1e-14);
}

TEST(BoundaryConditionParameters, DirichletWithComplexSymbolicExpression)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Dirichlet boundary ids = 200
    end

    subsection Functions
      subsection Dirichlet boundary conditions
        set Function constants = A=1.0, B=0.5, k=1.57079632679
        set Function expression = A * sin(k*x); B * cos(k*y)
        set Modulation frequency = 1.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(0.5, 0.0);
  const auto      &bc = par.get_dirichlet_bc(200);

  par.set_boundary_condition_times(0.0);

  EXPECT_NEAR(bc.value(p, 0), std::sin(1.57079632679 * 0.5), 1e-14);
  EXPECT_NEAR(bc.value(p, 1), 0.5, 1e-14);
  EXPECT_NEAR(bc.scale(0.0), 0.0, 1e-14);
}

TEST(RhsParameters, RhsWithTimeAndSymbolicConstants)
{
  static constexpr int dim = 2;
  ParameterAcceptor::clear();
  ElasticityProblemParameters<dim> par;
  initialize_parameters_from_string(R"(
    subsection Immersed Problem
      set Rhs material ids = 20
    end

    subsection Functions
      subsection Right hand side
        set Function constants = f0=1.0, omega=2.0
        set Function expression = f0 * sin(omega*t); f0 * cos(omega*t)
        set Modulation frequency = 0.0
        set Phase shift = 0.0
      end
    end
  )");

  const Point<dim> p(1.0, 1.0);
  const auto      &rhs = par.get_rhs(20);

  par.set_rhs_times(0.0);
  EXPECT_NEAR(rhs.value(p, 0), 0.0, 1e-14);
  EXPECT_NEAR(rhs.value(p, 1), 1.0, 1e-14);

  par.set_rhs_times(numbers::PI / 4.0);
  EXPECT_NEAR(rhs.value(p, 0), std::sin(numbers::PI / 2.0), 1e-14);
  EXPECT_NEAR(rhs.value(p, 1), std::cos(numbers::PI / 2.0), 1e-14);
}
