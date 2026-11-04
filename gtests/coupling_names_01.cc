// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/physics/vessel_elasticity.h>

#include <type_traits>

using namespace ImmersX;
using namespace dealii;

static_assert(
  std::is_same_v<VesselElasticityProblem<2>, CoupledElasticityProblem<2>>);
static_assert(std::is_same_v<VesselElasticityProblem<2, 3>,
                             CoupledElasticityProblem<2, 3>>);
static_assert(
  std::is_same_v<VesselElasticityProblem<3>, CoupledElasticityProblem<3>>);

TEST(CouplingNames, PrescribedPressureParameterAliases) // NOLINT
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> parameters;
  EXPECT_FALSE(parameters.uses_prescribed_vessel_pressure());
  parameters.set_prescribed_vessel_pressure(true);
  EXPECT_TRUE(parameters.pressure_coupling);
  parameters.pressure_coupling = false;
  EXPECT_FALSE(parameters.uses_prescribed_vessel_pressure());

  // Both names bind the same boolean; normal ParameterHandler assignment order
  // applies even when an input file deliberately uses both spellings.
  ParameterAcceptor::prm.parse_input_from_string(R"(
    subsection Immersed Problem
      set Prescribed vessel pressure = true
    end
  )");
  EXPECT_TRUE(parameters.uses_prescribed_vessel_pressure());
  ParameterAcceptor::prm.parse_input_from_string(R"(
    subsection Immersed Problem
      set Pressure coupling = false
    end
  )");
  EXPECT_FALSE(parameters.uses_prescribed_vessel_pressure());
  ParameterAcceptor::prm.parse_input_from_string(R"(
    subsection Immersed Problem
      set Prescribed vessel pressure = false
      set Pressure coupling = true
    end
  )");
  EXPECT_TRUE(parameters.uses_prescribed_vessel_pressure());
  ParameterAcceptor::prm.parse_input_from_string(R"(
    subsection Immersed Problem
      set Pressure coupling = true
      set Prescribed vessel pressure = false
    end
  )");
  EXPECT_FALSE(parameters.uses_prescribed_vessel_pressure());
}

TEST(CouplingNames, MultiplierMatrixCompatibilityStorage) // NOLINT
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> parameters;
  ElasticityProblem<2>           problem(parameters);
  EXPECT_EQ(&problem.multiplier_load_matrix(), &problem.coupling_matrix);
  const auto &constant_problem = problem;
  EXPECT_EQ(&constant_problem.multiplier_load_matrix(),
            &constant_problem.coupling_matrix);
}
