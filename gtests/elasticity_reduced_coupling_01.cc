#include <gtest/gtest.h>

#include "elasticity_problem_parameters.h"

using namespace dealii;

TEST(ElasticityReducedCouplingParameters, ParseMinimalRepresentativeDomain)
{
  ParameterAcceptor::clear();
  ElasticityProblemParameters<2> par;

  // Initialize global defaults used by the test harness (this mirrors how the
  // existing tests call initialize_parameters() before parsing test-specific
  // snippets).
  initialize_parameters();

  // Minimal parameter snippet enabling reduced coupling. Do not attempt to set
  // the full tensor-product subsections here; just verify parsing of the new
  // flag and that ParameterAcceptor does not throw.
  ParameterAcceptor::prm.parse_input_from_string(
    R"(
    subsection Immersed Problem
      set Use reduced coupling = true
    end
  )");

  EXPECT_NO_THROW(ParameterAcceptor::parse_all_parameters());
}
