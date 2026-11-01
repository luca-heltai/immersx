// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <deal.II/base/config.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_VTK
#  include <deal.II/base/bounding_box.h>

#  include <deal.II/fe/fe_values.h>

#  include <immersx/core/symbolic_field_evaluator.h>
#  include <immersx/physics/elasticity.h>
#  include <immersx/physics/reduced_poisson.h>
#  include <vtkNew.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkXMLUnstructuredGridReader.h>

#  include <cmath>
#  include <filesystem>
#  include <vector>

#  include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

namespace
{
  unsigned int
  output_cells(const std::filesystem::path &filename)
  {
    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName(filename.string().c_str());
    reader->Update();
    return reader->GetOutput()->GetNumberOfCells();
  }

  void
  check_properties(const ReducedCoupling<1, 2, 3, 3> &coupling)
  {
    const auto &dh      = coupling.get_properties_dh();
    const auto &catalog = coupling.get_properties_catalog();
    ASSERT_EQ(catalog.size(), 2u);
    unsigned int height = numbers::invalid_unsigned_int;
    unsigned int radius = numbers::invalid_unsigned_int;
    for (const auto &field : catalog)
      {
        if (field.name == "height")
          height = field.first_fe_component;
        else if (field.name == "radius")
          radius = field.first_fe_component;
      }
    ASSERT_NE(height, numbers::invalid_unsigned_int);
    ASSERT_NE(radius, numbers::invalid_unsigned_int);
    FEValues<1, 3>              values(dh.get_fe(),
                          QGauss<1>(2),
                          update_values | update_quadrature_points);
    std::vector<Vector<double>> samples(
      2, Vector<double>(dh.get_fe().n_components()));
    for (const auto &cell : dh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          values.reinit(cell);
          values.get_function_values(coupling.get_properties(), samples);
          for (unsigned int q = 0; q < samples.size(); ++q)
            {
              const double z = values.quadrature_point(q)[2];
              EXPECT_NEAR(samples[q][height], 1. + z, 1.e-12);
              EXPECT_NEAR(samples[q][radius], z < .82 ? .01 : .02, 1.e-12);
            }
        }
  }

  void
  check_size_bounds(const Triangulation<3>    &space,
                    const Triangulation<1, 3> &embedded)
  {
    // Enumerate all pairs independently of the adjustment rtrees. Embedded
    // ownership can be empty on a rank, so collect its boxes before checking.
    std::vector<BoundingBox<3>> local_boxes;
    for (const auto &cell : embedded.active_cell_iterators())
      if (cell->is_locally_owned())
        local_boxes.push_back(cell->bounding_box());
    const auto boxes = Utilities::MPI::all_gather(MPI_COMM_WORLD, local_boxes);
    unsigned int pairs = 0;
    for (const auto &cell : space.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto   space_box = cell->bounding_box();
          const auto   corners   = space_box.get_boundary_points();
          const double h_space   = corners.first.distance(corners.second);
          for (const auto &rank_boxes : boxes)
            for (const auto &embedded_box : rank_boxes)
              {
                if (space_box.has_overlap_with(embedded_box))
                  {
                    ++pairs;
                    const auto   bounds = embedded_box.get_boundary_points();
                    const double h_embedded =
                      bounds.first.distance(bounds.second);
                    EXPECT_GT(h_embedded, .5 * h_space);
                    EXPECT_LT(h_embedded, 2. * h_space);
                  }
              }
        }
    EXPECT_GT(Utilities::MPI::sum(pairs, MPI_COMM_WORLD), 0u);
  }
} // namespace

TEST(LocalCoRefinement, BOTH_ElasticityApplication)
{
  if (!SymbolicFieldEvaluator::available())
    GTEST_SKIP() << "deal.II was built without SymEngine";

  ParameterAcceptor::clear();
  ElasticityProblemParameters<2, 3> par;
  initialize_parameters(TestPaths::parameter_path(
    "gtests/parameters/elasticity_co_refinement_both.prm"));
  par.output_directory = TestPaths::output_directory(
    "both-elasticity-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  std::filesystem::create_directories(par.output_directory);
  ASSERT_EQ(par.tensor_product_coupling_parameters.refinement_parameters
              .refinement_strategy,
            "both");
  ElasticityProblem<2, 3> problem(par);
  problem.make_grid();
  EXPECT_EQ(problem.tria->n_global_active_cells(), 1u);
  problem.setup_fe();
  problem.setup_dofs();
  ASSERT_NE(problem.tensor_product_coupling, nullptr);
  const auto &coupling = *problem.tensor_product_coupling;
  EXPECT_GT(problem.tria->n_global_active_cells(), 1u);
  EXPECT_GT(coupling.get_triangulation().n_global_active_cells(), 2u);
  check_size_bounds(*problem.tria, coupling.get_triangulation());
  check_properties(coupling);
  problem.assemble_elasticity_system();
  problem.assemble_coupling();
  EXPECT_GT(problem.system_rhs.block(1).l2_norm(), 0.);
  problem.solve_static();
  EXPECT_TRUE(std::isfinite(problem.solution.block(0).l2_norm()));
  EXPECT_GT(problem.solution.block(0).l2_norm(), 1.e-12);
}

TEST(LocalCoRefinement, BOTH_ReducedPoissonApplication)
{
  if (!SymbolicFieldEvaluator::available())
    GTEST_SKIP() << "deal.II was built without SymEngine";

  ParameterAcceptor::clear();
  ReducedPoissonParameters<3> par;
  initialize_parameters(TestPaths::parameter_path(
    "gtests/parameters/reduced_poisson_co_refinement_both.prm"));
  par.output_directory = TestPaths::output_directory(
    "both-reduced-poisson-" +
    std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  std::filesystem::create_directories(par.output_directory);
  ASSERT_EQ(
    par.reduced_coupling_parameters.refinement_parameters.refinement_strategy,
    "both");
  ReducedPoisson<3> problem(par);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
  EXPECT_GT(problem.n_reduced_dofs(), 3u);
#  ifndef MATRIX_FREE_PATH
  problem.assemble_poisson_system();
#  else
  problem.assemble_rhs();
#  endif
  problem.assemble_coupling_system();
  EXPECT_GT(problem.coupling_matrix_frobenius_norm(), 0.);
  problem.solve();
  EXPECT_TRUE(problem.bulk_solution_is_finite());
  EXPECT_TRUE(problem.multiplier_solution_is_finite());
  EXPECT_GT(problem.bulk_solution_l2_norm(), 1.e-12);
  const auto output = problem.output_solution();
  EXPECT_GT(output_cells(std::filesystem::path(par.output_directory) / output),
            1u);
}
#endif
