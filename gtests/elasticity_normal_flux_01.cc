// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/function.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity.h>

#include <cmath>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  // Plane-strain radial Lamé field in a cylinder shell with ri = 0.2, re = 2
  // and inner displacement di = 0.1.
  class RadialLame : public Function<3>
  {
  public:
    RadialLame()
      : Function<3>(3)
    {}

    double
    value(const Point<3> &p, const unsigned int component) const override
    {
      if (component == 2)
        return 0.;
      const double ri = 0.2, re = 2., di = 0.1;
      const double r2 = p[0] * p[0] + p[1] * p[1];
      return ri * di / (re * re - ri * ri) * (re * re / r2 - 1.) * p[component];
    }
  };

  // Solve on a cylinder shell refined twice by Kelly, with exact Dirichlet
  // data on the curved faces and either normal-flux or Dirichlet end faces.
  // Return the relative L2 error.
  double
  solve_cylinder_shell(const bool normal_flux_end_faces)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> par;
    par.output_directory =
      TestPaths::output_directory("elasticity-normal-flux");
    ElasticityProblem<3> problem(par);
    initialize_parameters_from_string(std::string(R"(
      subsection Functions
        subsection Dirichlet boundary conditions
          set Function constants  = ri=0.2, re=2, di=0.1
          set Function expression = ri*di/(re*re-ri*ri)*(re*re/(x*x+y*y)-1)*x; ri*di/(re*re-ri*ri)*(re*re/(x*x+y*y)-1)*y; 0
          set Variable names      = x,y,z,t
        end
      end
      subsection Immersed Problem
        set Initial refinement       = 1
        set Output name              = normal_flux
        subsection Grid generation
          set Domain type              = generate
          set Grid generator           = cylinder_shell
          set Grid generator arguments = 4: 0.2: 2: 0: 0: true
        end
        subsection Refinement and remeshing
          set Number of refinement cycles = 2
          set Strategy                    = fixed_number
          set Refinement fraction         = 0.3
        end
      )") + (normal_flux_end_faces ? R"(
        set Dirichlet boundary ids   = 0, 1
        set Normal flux boundary ids = 2, 3
      end)" :
                                     R"(
        set Dirichlet boundary ids   = 0, 1, 2, 3
        set Normal flux boundary ids =
      end)"));
    problem.run();

    const auto &u = problem.solution.block(0);
    EXPECT_TRUE(std::isfinite(u.l2_norm()));
    // More DoFs than the AMG coarse size, so the null space is used.
    if (normal_flux_end_faces)
      {
        EXPECT_GT(problem.dh.n_dofs(), 2000u);
      }

    LA::MPI::Vector ghosted(problem.dh.locally_owned_dofs(),
                            DoFTools::extract_locally_relevant_dofs(problem.dh),
                            MPI_COMM_WORLD);
    ghosted       = u;
    const auto l2 = [&](const Function<3> &reference) {
      Vector<float> cell_errors(
        problem.dh.get_triangulation().n_active_cells());
      VectorTools::integrate_difference(MappingQ1<3>(),
                                        problem.dh,
                                        ghosted,
                                        reference,
                                        cell_errors,
                                        QGauss<3>(3),
                                        VectorTools::L2_norm);
      return VectorTools::compute_global_error(problem.dh.get_triangulation(),
                                               cell_errors,
                                               VectorTools::L2_norm);
    };
    return l2(RadialLame()) / l2(Functions::ZeroFunction<3>(3));
  }
} // namespace

// Issue #458: the static AMG read a freed null space, which gave NaN in the
// first CG step. The second cycle has hanging nodes and is the first solve
// above the AMG coarse size. The exact field satisfies both kinds of end-face
// conditions. Kelly refines different cells in the two runs, so the errors
// differ; the normal-flux run must not be less accurate than the reference.
TEST(ElasticityProblem, BOTH_NormalFluxEndFacesWithHangingNodes)
{
  const double normal_flux = solve_cylinder_shell(true);
  const double dirichlet   = solve_cylinder_shell(false);
  EXPECT_LT(normal_flux, 1.05 * dirichlet);
}
