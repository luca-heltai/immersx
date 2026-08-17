#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include <algorithm>

#include "reduced_coupling.h"
#include "reduced_poisson.h"
#include "tensor_product_space.h"
#include "utils.h"

using namespace dealii;

TEST(TensorProductSpace0D, PointsWeightsAndDofs)
{
  TensorProductSpaceParameters<0, 2, 2, 1> params;
  params.thickness                     = "0.5";
  params.section.selected_coefficients = {0};
  TensorProductSpace<0, 2, 2, 1>                      space(params);
  std::vector<ZeroDimensionalRepresentativeEntity<2>> entities(3);
  entities[0].position = Point<2>(0.0, 0.0);
  entities[1].position = Point<2>(0.25, 0.25);
  entities[2].position = Point<2>(0.5, 0.5);
  entities[0].weight   = 1.0;
  entities[1].weight   = 2.0;
  entities[2].weight   = 3.0;
  space.set_representative_entities(entities);
  space.initialize();

  EXPECT_EQ(space.n_representative_entities(), 3u);
  EXPECT_EQ(space.n_representative_dofs(), 3u);
  EXPECT_EQ(space.get_locally_owned_reduced_qpoints().size(), 3u);
  EXPECT_EQ(space.get_locally_owned_reduced_weights().size(), 3u);
  EXPECT_EQ(space.get_locally_owned_qpoints().size(),
            space.get_reference_cross_section().n_quadrature_points() * 3u);
  for (unsigned int i = 0; i < 3; ++i)
    EXPECT_EQ(space.get_representative_dof_indices(i).size(), 1u);
}

TEST(TensorProductSpace0D, UnsupportedImportedInputs)
{
  ZeroDimensionalRepresentativeEntity<2> entity;
  entity.position = Point<2>(0., 0.);

  {
    TensorProductSpaceParameters<0, 2, 2, 1> params;
    params.section.selected_coefficients = {0};
    params.reduced_grid_name             = "unsupported.vtk";
    params.representative_entities       = {entity};
    TensorProductSpace<0, 2, 2, 1> space(params);
    EXPECT_THROW(space.initialize(), ExceptionBase);
  }

  {
    TensorProductSpaceParameters<0, 2, 2, 1> params;
    params.section.selected_coefficients = {0};
    params.input_file_fields             = "radius";
    params.representative_entities       = {entity};
    TensorProductSpace<0, 2, 2, 1> space(params);
    EXPECT_THROW(space.initialize(), ExceptionBase);
  }
}

#ifdef DEAL_II_WITH_VTK
TEST(TensorProductSpace0D, ImportsRadiusThicknessFromPointCloud)
{
  for (const std::string &filename :
       {std::string(SOURCE_DIR) + "/gtests/fixtures/point_cloud_minimal.vtk",
        std::string(SOURCE_DIR) + "/gtests/fixtures/point_cloud_minimal.vtu"})
    {
      TensorProductSpaceParameters<0, 2, 2, 1> params;
      params.reduced_grid_name             = filename;
      params.input_file_fields             = "radius";
      params.thickness                     = "radius";
      params.section.selected_coefficients = {0};

      TensorProductSpace<0, 2, 2, 1> space(params);
      ASSERT_NO_THROW(space.initialize());
      ASSERT_EQ(space.n_representative_entities(), 2u);
      ASSERT_EQ(space.get_properties_bindings().size(), 1u);
      ASSERT_EQ(space.get_entity_property_values(0).size(), 1u);
      EXPECT_DOUBLE_EQ(space.get_entity_property_values(0)[0], 0.25);
      EXPECT_DOUBLE_EQ(space.get_entity_thickness(0), 0.25);
      EXPECT_DOUBLE_EQ(space.get_entity_thickness(1), 0.5);
    }
}
#endif

TEST(TensorProductSpace0D, MPI_StableParticleMapping)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(2);

  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.coupling_rhs_expressions                                      = {"1"};
  std::vector<ZeroDimensionalRepresentativeEntity<2>> entities(4);
  entities[0].position = Point<2>(-0.7, -0.7);
  entities[1].position = Point<2>(-0.7, 0.7);
  entities[2].position = Point<2>(0.7, -0.7);
  entities[3].position = Point<2>(0.7, 0.7);
  params.tensor_product_space_parameters.representative_entities = entities;

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  coupling.initialize();

  std::vector<unsigned int> local_entity_ids;
  for (const auto &particle : coupling.get_particles())
    local_entity_ids.push_back(std::get<0>(
      coupling.particle_id_to_representative_indices(particle.get_id())));
  const auto gathered =
    Utilities::MPI::all_gather(MPI_COMM_WORLD, local_entity_ids);
  std::vector<unsigned int> all_entity_ids;
  for (const auto &rank_ids : gathered)
    all_entity_ids.insert(all_entity_ids.end(),
                          rank_ids.begin(),
                          rank_ids.end());
  std::sort(all_entity_ids.begin(), all_entity_ids.end());
  ASSERT_EQ(all_entity_ids.size(),
            entities.size() *
              coupling.get_reference_cross_section().n_quadrature_points());
  for (unsigned int i = 0; i < all_entity_ids.size(); ++i)
    EXPECT_EQ(all_entity_ids[i],
              i / coupling.get_reference_cross_section().n_quadrature_points());
}

TEST(ReducedPoisson0D, TemplatePathCompiles)
{
  ReducedPoissonParameters<2, 0> params;
  params.reduced_coupling_parameters.tensor_product_space_parameters.section
    .selected_coefficients = {0};
  ZeroDimensionalRepresentativeEntity<2> entity;
  entity.position = Point<2>(0., 0.);
  params.reduced_coupling_parameters.tensor_product_space_parameters
    .representative_entities = {entity};
  ReducedPoisson<2, 2, 0> problem(params);
  (void)problem;
}

TEST(ReducedPoisson0D, MPI_OneCycleAssemblySolve)
{
  ParameterAcceptor::clear();
  ReducedPoissonParameters<2, 0> params;
  initialize_parameters_from_string(
    "subsection Reduced Poisson\n"
    "  subsection Right hand side\n"
    "    set Function expression = 0\n"
    "    set Variable names = x,y,t\n"
    "  end\n"
    "  subsection Dirichlet boundary conditions\n"
    "    set Function expression = 0\n"
    "    set Variable names = x,y,t\n"
    "  end\n"
    "end\n");
  // Use a 2x2 subdivision so the origin is an interior Q1 DoF rather
  // than a boundary DoF eliminated by the Dirichlet constraints.
  params.name_of_grid       = "subdivided_hyper_cube";
  params.arguments_for_grid = "2: -1: 1: false";
  params.solver_name        = "Schur";
  params.reduced_coupling_parameters.tensor_product_space_parameters.section
    .selected_coefficients                                    = {0};
  params.reduced_coupling_parameters.coupling_rhs_expressions = {"1"};
  ZeroDimensionalRepresentativeEntity<2> entity;
  entity.position = Point<2>(0., 0.);
  params.reduced_coupling_parameters.tensor_product_space_parameters
    .representative_entities = {entity};

  ReducedPoisson<2, 2, 0> problem(params);
  problem.make_grid();
  problem.setup_fe();
  problem.setup_dofs();
#ifndef MATRIX_FREE_PATH
  problem.assemble_poisson_system();
#else
  problem.assemble_rhs();
#endif
  problem.assemble_coupling_system();

  EXPECT_EQ(problem.n_reduced_dofs(), 1u);
  EXPECT_GT(problem.coupling_matrix_frobenius_norm(), 0.);

  problem.solve();
  EXPECT_TRUE(problem.bulk_solution_is_finite());
  EXPECT_TRUE(problem.multiplier_solution_is_finite());
  EXPECT_GT(problem.bulk_solution_l2_norm(), 0.);
  EXPECT_GT(problem.multiplier_solution_l2_norm(), 0.);
}

TEST(ReducedCoupling0D, MPI_AssemblyInterfaces)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(1);
  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.thickness                     = "0.5";
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  ZeroDimensionalRepresentativeEntity<2> entity;
  entity.position = Point<2>(0., 0.);
  params.tensor_product_space_parameters.representative_entities = {entity};
  params.coupling_rhs_expressions                                = {"1"};
  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  coupling.initialize();

  FE_Q<2>       fe(1);
  DoFHandler<2> dh(background);
  dh.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  DynamicSparsityPattern dsp(dh.n_dofs(), coupling.n_representative_dofs());
  coupling.assemble_coupling_sparsity(dsp, dh, constraints);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> matrix(sparsity);
  coupling.assemble_coupling_matrix(matrix, dh, constraints);
  DynamicSparsityPattern mdsp(coupling.n_representative_dofs(),
                              coupling.n_representative_dofs());
  mdsp.add(0, 0);
  SparsityPattern mass_sparsity;
  mass_sparsity.copy_from(mdsp);
  SparseMatrix<double> mass(mass_sparsity);
  coupling.assemble_coupling_mass_matrix(mass);
  Vector<double> rhs(coupling.n_representative_dofs());
  coupling.assemble_reduced_rhs(rhs);
  const double global_matrix_norm =
    Utilities::MPI::sum(matrix.frobenius_norm(), MPI_COMM_WORLD);
  const double global_mass_diagonal =
    Utilities::MPI::sum(mass.diag_element(0), MPI_COMM_WORLD);
  const double global_rhs_norm =
    Utilities::MPI::sum(rhs.l2_norm(), MPI_COMM_WORLD);
  EXPECT_GT(global_matrix_norm, 0.);
  EXPECT_GT(global_mass_diagonal, 0.);
  EXPECT_GT(global_rhs_norm, 0.);
}
