// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  void
  check_imported_materials(const std::string &extension)
  {
    ParameterAcceptor::clear();
    ElastodynamicsParameters<2> parameters;
    initialize_parameters_from_string(R"(
      subsection Elastodynamics
        set Dirichlet boundary ids =
        subsection Domain
          set Initial refinement = 0
          set Triangulation type = fullydistributed
        end
        subsection Material properties
          set Material tags by material id = 1: stiff
          subsection default
            set Density = 2
            set Lame mu = 3
            set Lame lambda = 4
            set Rayleigh alpha = 0.2
            set Rayleigh beta = 0.1
            set Viscosity eta = 0.3
          end
          subsection stiff
            set Density = 5
            set Lame mu = 7
            set Lame lambda = 11
            set Rayleigh alpha = 0.4
            set Rayleigh beta = 0.6
            set Viscosity eta = 0.8
          end
        end
        subsection Functions
          subsection Initial displacement
            set Function expression = x; 0
          end
        end
      end
    )");
    parameters.domain_parameters.name_of_grid = TestPaths::data_filename(
      "tests/elastodynamics_material_ids." + extension);
    parameters.domain_parameters.vtk_material_id_field = "material_id";
    ElastodynamicsSolver<2> problem(parameters);
    problem.make_grid();
    unsigned int stiff = 0, fallback = 0;
    for (const auto &cell : problem.triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          EXPECT_EQ(cell->material_id(), cell->center()[0] < .5 ? 1 : 99);
          stiff += cell->material_id() == 1;
          fallback += cell->material_id() == 99;
        }
    EXPECT_EQ(Utilities::MPI::sum(stiff, MPI_COMM_WORLD), 1u);
    EXPECT_EQ(Utilities::MPI::sum(fallback, MPI_COMM_WORLD), 1u);
    problem.setup_fe();
    problem.setup_system();
    problem.assemble_operators();
    ElastodynamicsSolver<2>::VectorType displacement, product;
    displacement.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
    product.reinit(displacement);
    VectorTools::interpolate(problem.mapping(),
                             problem.dof_handler(),
                             parameters.initial_displacement,
                             displacement);
    // Integrals for u=(x,0): rho*x^2, lambda+2*mu, and alpha*rho*x^2
    // + beta*(lambda+2*mu) + eta, using the imported left/right half IDs.
    problem.mass_matrix().vmult(product, displacement);
    EXPECT_NEAR(displacement * product, 19. / 24., 1.e-12);
    problem.stiffness_matrix().vmult(product, displacement);
    EXPECT_NEAR(displacement * product, 17.5, 1.e-12);
    problem.damping_matrix().vmult(product, displacement);
    EXPECT_NEAR(displacement * product, 8.75, 1.e-12);
  }
} // namespace

TEST(Elastodynamics, BOTH_UcdMaterialIdsSelectOperators)
{
  check_imported_materials("inp");
}

#ifdef DEAL_II_WITH_VTK
TEST(Elastodynamics, BOTH_VtkMaterialIdsSelectOperators)
{
  check_imported_materials("vtk");
}
#endif
