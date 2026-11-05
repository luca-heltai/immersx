// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function_lib.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/lifted_elasticity.h>

using namespace ImmersX;
using namespace dealii;

TEST(LiftedElasticity, BOTH_RobinUncoupledManufacturedTranslation)
{
  ParameterAcceptor::clear();
  LiftedElasticityParameters<2> parameters;
  initialize_parameters_from_string(R"(
subsection Lifted elasticity
 set Coupling = none
 set Dirichlet boundary ids =
 set Robin boundary ids = 0
 subsection Domain
  set Initial refinement = 1
  set Grid generator arguments = 0:1:false
 end
 subsection Functions
  subsection Robin coefficient
   set Function expression = 2+x
  end
  subsection Robin boundary conditions
   set Function expression = 2+x;2*(2+x)
  end
 end
end
)");
  LiftedElasticity<2> problem(parameters);
  problem.setup();
  problem.solve();
  const auto                 &field = problem.displacement();
  LiftedElasticity<2>::Vector expected;
  expected.reinit(field.locally_owned_dofs(), MPI_COMM_WORLD);
  Functions::ConstantFunction<2> exact(std::vector<double>{1., 2.});
  VectorTools::interpolate(field.mapping(),
                           field.dof_handler(),
                           exact,
                           expected);
  auto error = problem.solution();
  error -= expected;
  EXPECT_LT(error.linfty_norm(), 1.e-10);
  EXPECT_LT(problem.residual_norm(), 1.e-10);
}

TEST(LiftedElasticity, BOTH_RobinRejectsWeakAndNormalBoundaryOverlap)
{
  for (const bool normal : {false, true})
    {
      ParameterAcceptor::clear();
      LiftedElasticityParameters<2> parameters;
      initialize_parameters();
      parameters.coupling = "none";
      parameters.dirichlet_ids.clear();
      parameters.robin_ids = {0};
      if (normal)
        parameters.normal_flux_ids = {0};
      else
        parameters.weak_dirichlet_ids = {0};
      LiftedElasticity<2> problem(parameters);
      EXPECT_THROW(problem.setup(), dealii::ExceptionBase);
    }
}
