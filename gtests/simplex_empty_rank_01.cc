// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/function_parser.h>

#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/domain.h>
#include <immersx/core/known_term.h>
#include <immersx/coupling/particle_coupling.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>

#include "test_paths.h"

using namespace dealii;
using namespace ImmersX;

#ifdef DEAL_II_WITH_VTK
TEST(Domain, MPI_SimplexSpaceWithEmptyRanks)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  DomainParameters<3> parameters("/Tests/Empty simplex rank/");
  parameters.triangulation_type = "fullydistributed";
  parameters.name_of_grid =
    TestPaths::data_filename("tests/simplex_fields_single_tetrahedron.vtk");
  parameters.read_vtk_fields = true;
  Domain<3> domain(parameters);
  domain.make_grid();
  EXPECT_EQ(domain.triangulation().n_global_active_cells(), 1u);
  EXPECT_EQ(
    Utilities::MPI::min(domain.triangulation().n_locally_owned_active_cells(),
                        MPI_COMM_WORLD),
    0u);
  const auto reference_cell =
    detail::uniform_reference_cell(domain.triangulation());
  ASSERT_TRUE(reference_cell.has_value());
  EXPECT_TRUE(reference_cell->is_simplex());
  const auto imported = domain.field("affine");
  EXPECT_TRUE(imported.field().finite_element().reference_cell().is_simplex());
  FiniteElementSpace<3> space(domain.triangulation(),
                              std::make_unique<FE_SimplexP<3>>(1));
  const auto            V = space.view();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 1)
    {
      const auto local_view = space.view();
      EXPECT_TRUE(local_view.finite_element().reference_cell().is_simplex());
    }
  MPI_Barrier(MPI_COMM_WORLD);
  StateLayout layout;
  const auto  source = V.field(layout, "source");
  const auto  target = V.field(layout, "target");
  using VectorType   = ImmersXLA::MPI::Vector;
  using MatrixType   = ImmersXLA::MPI::SparseMatrix;
  VectorType ones, zero, result;
  ones.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
  zero.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
  result.reinit(space.locally_owned_dofs(), MPI_COMM_WORLD);
  ones = 1.;
  zero = 0.;
  StateView<VectorType> view(layout, 0.);
  view.bind(source.id(), ones);
  view.bind(target.id(), zero);
  const EvaluationContext<VectorType>         context(0., view);
  SemiDiscreteModel<VectorType, MatrixType>   model;
  SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
  weak_term(value(source), test(target)).add(builder);
  model.evaluate_row(target.id(), context, result);
  EXPECT_NEAR(ones * result, 1. / 6., 1.e-12);
  VectorType coefficients;
  coefficients.reinit(imported.field().locally_owned_dofs(),
                      imported.field().locally_relevant_dofs(),
                      MPI_COMM_WORLD);
  coefficients = imported.coefficients();
  coefficients.update_ghost_values();
  SemiDiscreteModel<VectorType, MatrixType>   frozen_model;
  SemidiscreteBuilder<VectorType, MatrixType> frozen_builder(layout,
                                                             frozen_model);
  known_term(frozen(imported.field(), coefficients), test(target))
    .add(frozen_builder);
  result = 0.;
  frozen_model.evaluate_row(target.id(), context, result);
  EXPECT_NEAR(ones * result, 5. / 12., 1.e-12);

  ParticleCouplingParameters<3> particle_parameters;
  ParticleCoupling<3>           particles(particle_parameters);
  particles.initialize_particle_handler(domain.triangulation());
  std::vector<Point<3>>            positions;
  std::vector<std::vector<double>> properties;
  if (domain.triangulation().n_locally_owned_active_cells() == 0)
    {
      positions.emplace_back(.1, .2, .1);
      properties.push_back({1.});
    }
  particles.insert_points(positions, properties);
  EXPECT_EQ(particles.get_particles().n_global_particles(), 1u);

  DomainParameters<3> geometry_parameters("/Tests/Empty geometry rank/");
  geometry_parameters.triangulation_type = "fullydistributed";
  geometry_parameters.name_of_grid =
    TestPaths::data_filename("tests/simplex_single_tetrahedron.msh");
  Domain<3> geometry(geometry_parameters);
  geometry.make_grid();
  SemiDiscreteModel<VectorType, MatrixType>   geometry_model;
  SemidiscreteBuilder<VectorType, MatrixType> geometry_builder(layout,
                                                               geometry_model);
  weak_term(value(source), test(target))
    .on(geometry.support())
    .add(geometry_builder);
  result = 0.;
  geometry_model.evaluate_row(target.id(), context, result);
  EXPECT_NEAR(ones * result, 1. / 6., 1.e-12);
}

#endif

TEST(Poisson, MPI_SimplexWithEmptyRanks)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  PoissonParameters<3> parameters;
  initialize_parameters_from_string(
    "subsection Poisson\n"
    " set FE degree = 2\n"
    " set Output directory = " +
    TestPaths::output_directory("poisson-empty-simplex-rank") +
    "\n"
    " subsection Domain\n"
    "  set Triangulation type = fullydistributed\n"
    "  set Initial refinement = 0\n"
    "  set Grid generator = " +
    TestPaths::data_filename("tests/simplex_single_tetrahedron.msh") +
    "\n"
    " end\n"
    " subsection Right hand side\n"
    "  set Function expression = -6\n"
    " end\n"
    " subsection Dirichlet boundary conditions\n"
    "  set Function expression = 1+x*x+y*y+z*z\n"
    " end\n"
    "end\n");
  PoissonSolver<3> problem(parameters);
  problem.run();
  EXPECT_EQ(
    Utilities::MPI::min(problem.triangulation().n_locally_owned_active_cells(),
                        MPI_COMM_WORLD),
    0u);
  EXPECT_TRUE(problem.dof_handler().get_fe().reference_cell().is_simplex());
  FunctionParser<3> exact;
  exact.initialize("x,y,z", "1+x*x+y*y+z*z", {});
  PoissonSolver<3>::VectorType expected;
  expected.reinit(problem.locally_owned_dofs(), MPI_COMM_WORLD);
  VectorTools::interpolate(problem.mapping(),
                           problem.dof_handler(),
                           exact,
                           expected);
  expected -= problem.solution();
  EXPECT_LT(expected.linfty_norm(), 1.e-11);
}

TEST(Domain, MPI_RejectsMixedPhysicalCellsAcrossRanks)
{
  ASSERT_EQ(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 2u);
  ParameterAcceptor::clear();
  Triangulation<2>            coarse;
  const std::vector<Point<2>> vertices = {{0., 0.},
                                          {1., 0.},
                                          {0., 1.},
                                          {1., 1.},
                                          {3., 0.},
                                          {4., 0.},
                                          {3., 1.},
                                          {4., 1.}};
  std::vector<CellData<2>>    cells    = {CellData<2>(4),
                                          CellData<2>(3),
                                          CellData<2>(3)};
  cells[0].vertices                    = {0, 1, 2, 3};
  cells[1].vertices                    = {4, 5, 6};
  cells[2].vertices                    = {5, 7, 6};
  coarse.create_triangulation(vertices, cells, SubCellData());
  parallel::fullydistributed::Triangulation<2> distributed(MPI_COMM_WORLD);
  distributed.set_partitioner(
    [](auto &tria, const unsigned int n_partitions) {
      for (const auto &cell : tria.active_cell_iterators())
        cell->set_subdomain_id(
          cell->reference_cell().is_hyper_cube() || n_partitions == 1 ? 0 : 1);
    },
    TriangulationDescription::Settings::default_setting);
  distributed.copy_triangulation(coarse);
  EXPECT_GT(Utilities::MPI::min(distributed.n_locally_owned_active_cells(),
                                MPI_COMM_WORLD),
            0u);
  DomainParameters<2> parameters("/Tests/Distributed mixed cells/");
  parameters.triangulation_type = "fullydistributed";
  EXPECT_THROW((Domain<2>(parameters, distributed)), ExceptionBase);
}
