#include <gtest/gtest.h>
#include <immersx/physics/elasticity_problem_parameters.h>

using namespace ImmersX;

using namespace dealii;

TEST(ElasticityCouplingParameters, ParseCouplingSelector)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;

  // Initialize global defaults used by the test harness (this mirrors how the
  // existing tests call initialize_parameters() before parsing test-specific
  // snippets).
  initialize_parameters();

  // Minimal parameter snippet selecting tensor-product coupling. Do not
  // attempt to set the full tensor-product subsections here; just verify
  // selector parsing and that ParameterAcceptor does not throw.
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Coupling type = TensorProduct
    end
  )");

  EXPECT_NO_THROW(ParameterAcceptor::parse_all_parameters());
  EXPECT_EQ(par.coupling_type, CouplingType::TensorProduct);
}
