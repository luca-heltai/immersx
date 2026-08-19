#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

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
  TensorProductSpace<0, 2, 2, 1> space(params);
  PointCloud<2>                  cloud;
  cloud.points.resize(3);
  cloud.points[0] = Point<2>(0.0, 0.0);
  cloud.points[1] = Point<2>(0.25, 0.25);
  cloud.points[2] = Point<2>(0.5, 0.5);
  space.set_point_cloud(cloud);
  space.initialize();

  EXPECT_EQ(space.n_representative_entities(), 3u);
  EXPECT_EQ(space.n_representative_dofs(), 3u);
  EXPECT_EQ(space.get_locally_owned_reduced_qpoints().size(), 3u);
  EXPECT_EQ(space.get_locally_owned_reduced_weights().size(), 3u);
  EXPECT_EQ(space.get_locally_owned_qpoints().size(),
            space.get_reference_cross_section().n_quadrature_points() * 3u);
  for (unsigned int i = 0; i < 3; ++i)
    {
      EXPECT_EQ(space.get_representative_dof_indices(i).size(), 1u);
      EXPECT_DOUBLE_EQ(space.get_locally_owned_reduced_weights()[i][0], 1.0);
    }
  double lifted_sum = 0.;
  for (const auto &weight : space.get_locally_owned_weights())
    lifted_sum += weight[0];
  EXPECT_NEAR(lifted_sum,
              3. * space.get_reference_cross_section().measure(0.5),
              1e-12);
}

TEST(TensorProductSpace0D, VectorSelectedCoefficientsHaveContiguousDofs)
{
  TensorProductSpaceParameters<0, 2, 2, 2> params;
  params.thickness                     = "0.25";
  params.section.inclusion_degree      = 1;
  params.section.selected_coefficients = {0, 1, 2};
  params.point_cloud.points            = {Point<2>(0., 0.), Point<2>(1., 0.)};
  TensorProductSpace<0, 2, 2, 2> space(params);
  space.initialize();
  ASSERT_EQ(space.n_representative_dofs_per_entity(), 3u);
  ASSERT_EQ(space.n_representative_dofs(), 6u);
  for (unsigned int entity = 0; entity < 2; ++entity)
    {
      const auto &indices = space.get_representative_dof_indices(entity);
      ASSERT_EQ(indices.size(), 3u);
      for (unsigned int j = 0; j < indices.size(); ++j)
        EXPECT_EQ(indices[j], entity * 3 + j);
    }
}

TEST(TensorProductSpace0D, PointCloudOrientationAndDefault)
{
  TensorProductSpaceParameters<0, 2, 2, 1> default_params;
  default_params.section.selected_coefficients = {0};
  default_params.point_cloud.points            = {Point<2>(0., 0.)};
  TensorProductSpace<0, 2, 2, 1> default_space(default_params);
  default_space.initialize();
  EXPECT_NEAR(default_space.get_entity_orientation(0)[0], 0., 1e-12);
  EXPECT_NEAR(default_space.get_entity_orientation(0)[1], 1., 1e-12);

  TensorProductSpaceParameters<0, 2, 2, 1> oriented_params;
  oriented_params.section.selected_coefficients = {0};
  oriented_params.point_cloud.points            = {Point<2>(0., 0.)};
  oriented_params.point_cloud.catalog           = {
    {"orientation", FieldAssociation::point_data, 2, 0, 0}};
  oriented_params.point_cloud.properties = {{1., 0.}};
  TensorProductSpace<0, 2, 2, 1> oriented_space(oriented_params);
  oriented_space.initialize();
  EXPECT_NEAR(oriented_space.get_entity_orientation(0)[0], 1., 1e-12);
  EXPECT_NEAR(oriented_space.get_entity_orientation(0)[1], 0., 1e-12);
}

TEST(TensorProductSpace0D, RejectsTimeDependentThickness)
{
  TensorProductSpaceParameters<0, 2, 2, 1> params;
  params.thickness                     = "1+t";
  params.section.selected_coefficients = {0};
  params.point_cloud.points            = {Point<2>(0., 0.)};
  TensorProductSpace<0, 2, 2, 1> space(params);
  EXPECT_THROW(space.initialize(), ExceptionBase);
}

TEST(TensorProductSpace0D, UnsupportedImportedInputs)
{
  PointCloud<2> cloud;
  cloud.points = {Point<2>(0., 0.)};

  {
    TensorProductSpaceParameters<0, 2, 2, 1> params;
    params.section.selected_coefficients = {0};
    params.reduced_grid_name             = "unsupported.vtk";
    params.point_cloud                   = cloud;
    TensorProductSpace<0, 2, 2, 1> space(params);
    EXPECT_THROW(space.initialize(), ExceptionBase);
  }

  {
    TensorProductSpaceParameters<0, 2, 2, 1> params;
    params.section.selected_coefficients = {0};
    params.input_file_fields             = "radius";
    params.point_cloud                   = cloud;
    TensorProductSpace<0, 2, 2, 1> space(params);
    EXPECT_ANY_THROW(space.initialize());
  }
}

TEST(TensorProductSpace0D, ProgrammaticPointCloudRadiusThickness)
{
  TensorProductSpaceParameters<0, 2, 2, 1> params;
  params.input_file_fields             = "radius";
  params.thickness                     = "radius";
  params.section.selected_coefficients = {0};
  params.point_cloud.points            = {Point<2>(0., 0.), Point<2>(1., 0.)};
  params.point_cloud.catalog           = {
    {"radius", FieldAssociation::point_data, 1, 0, 0}};
  params.point_cloud.properties = {{0.25, 0.5}};

  TensorProductSpace<0, 2, 2, 1> space(params);
  ASSERT_NO_THROW(space.initialize());
  ASSERT_EQ(space.get_properties_bindings().size(), 1u);
  ASSERT_EQ(space.get_entity_property_values(0).size(), 1u);
  EXPECT_DOUBLE_EQ(space.get_entity_property_values(0)[0], 0.25);
  EXPECT_DOUBLE_EQ(space.get_entity_property_values(1)[0], 0.5);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(0), 0.25);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(1), 0.5);
}

TEST(ReducedCoupling0D, SetTimeAcceptsStaticThicknessAndUpdatesRhs)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.input_file_fields = "radius";
  params.tensor_product_space_parameters.thickness         = "radius";
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.tensor_product_space_parameters.point_cloud.points            = {
    Point<2>(0., 0.)};
  params.tensor_product_space_parameters.point_cloud.catalog = {
    {"radius", FieldAssociation::point_data, 1, 0, 0}};
  params.tensor_product_space_parameters.point_cloud.properties = {{0.25}};
  params.coupling_rhs_expressions                               = {"1+t"};

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  ASSERT_NO_THROW(coupling.initialize());
  ASSERT_NO_THROW(coupling.set_time(2.));
  EXPECT_DOUBLE_EQ(coupling.get_entity_thickness(0), 0.25);
}

TEST(ReducedCoupling0D, MPI_RankLocalProgrammaticPointCloud)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(background, 2, -1., 1.);

  const unsigned int rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int nproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  if (nproc == 1)
    return;

  Point<2> local_cell_center;
  bool     found_local_cell = false;
  for (const auto &cell : background.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        local_cell_center = cell->center();
        found_local_cell  = true;
        break;
      }
  ASSERT_TRUE(found_local_cell);
  const std::vector<double> local_center = {local_cell_center[0],
                                            local_cell_center[1]};
  const auto                all_local_centers =
    Utilities::MPI::all_gather(MPI_COMM_WORLD, local_center);
  const auto    &target_center = all_local_centers[(rank + 1) % nproc];
  const Point<2> point(target_center[0], target_center[1]);

  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.input_file_fields = "radius";
  params.tensor_product_space_parameters.thickness         = "radius";
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.tensor_product_space_parameters.point_cloud.points  = {point};
  params.tensor_product_space_parameters.point_cloud.catalog = {
    {"radius", FieldAssociation::point_data, 1, 0, 0}};
  params.tensor_product_space_parameters.point_cloud.properties = {
    {0.25 + rank}};
  params.tensor_product_space_parameters.point_cloud.distribution =
    PointCloudDistribution::rank_local;
  params.coupling_rhs_expressions = {"1"};

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  ASSERT_NO_THROW(coupling.initialize());
  EXPECT_EQ(coupling.n_representative_entities(), nproc);

  const auto locally_owned =
    coupling.get_representative_particles().locally_owned_particle_ids();
  // Representative properties are authoritative in the owning particle.
  // Inspect them only on that rank; the owner map below checks the global
  // coverage and uniqueness invariant.
  std::vector<int> local_owner(nproc, -1);
  for (unsigned int entity = 0; entity < nproc; ++entity)
    if (locally_owned.is_element(entity))
      {
        EXPECT_DOUBLE_EQ(coupling.get_entity_property_values(entity)[0],
                         0.25 + entity);
        EXPECT_DOUBLE_EQ(coupling.get_entity_thickness(entity), 0.25 + entity);
        local_owner[entity] = rank;
      }
  // The source point was deliberately placed in the next rank's background
  // cell. The lifted particle therefore crosses the source/owner partition,
  // and the receiving rank must mark the representative DoFs as relevant.
  const auto relevant = coupling.locally_relevant_representative_dofs();
  for (const auto &particle : coupling.get_particles())
    {
      const auto entity = std::get<0>(
        coupling.particle_id_to_representative_indices(particle.get_id()));
      EXPECT_TRUE(relevant.is_element(entity));
    }
  const auto owners = Utilities::MPI::all_gather(MPI_COMM_WORLD, local_owner);
  for (unsigned int entity = 0; entity < nproc; ++entity)
    {
      unsigned int n_owners = 0;
      int          owner    = -1;
      for (const auto &rank_owners : owners)
        if (rank_owners[entity] >= 0)
          {
            ++n_owners;
            owner = rank_owners[entity];
          }
      EXPECT_EQ(n_owners, 1u);
      EXPECT_NE(owner, static_cast<int>(entity));
    }
}

TEST(ReducedCoupling0D, MPI_RankLocalEmptySourceRanks)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(background, 2, -1., 1.);

  const unsigned int rank  = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int nproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  if (nproc == 1)
    return;

  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.tensor_product_space_parameters.point_cloud.distribution =
    PointCloudDistribution::rank_local;
  params.tensor_product_space_parameters.point_cloud.catalog = {
    {"radius", FieldAssociation::point_data, 1, 0, 0}};
  params.tensor_product_space_parameters.point_cloud.property_names = {
    "radius"};
  params.tensor_product_space_parameters.input_file_fields = "radius";
  params.tensor_product_space_parameters.thickness         = "radius";
  if (rank == 0)
    {
      params.tensor_product_space_parameters.point_cloud.points = {
        Point<2>(-0.5, -0.5), Point<2>(0.5, 0.5)};
      params.tensor_product_space_parameters.point_cloud.properties = {
        {0.25, 0.5}};
    }
  params.coupling_rhs_expressions = {"1"};

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  ASSERT_NO_THROW(coupling.initialize());
  ASSERT_EQ(coupling.n_representative_entities(), 2u);

  std::vector<int> local_owner(2, -1);
  for (const auto entity :
       coupling.get_representative_particles().locally_owned_particle_ids())
    local_owner[entity] = rank;
  const auto owners = Utilities::MPI::all_gather(MPI_COMM_WORLD, local_owner);
  for (unsigned int entity = 0; entity < 2; ++entity)
    {
      unsigned int n_owners = 0;
      for (const auto &rank_owners : owners)
        n_owners += rank_owners[entity] >= 0;
      EXPECT_EQ(n_owners, 1u);
    }
}

TEST(TensorProductSpace0D, ThicknessScalingAndLiftedMeasure)
{
  TensorProductSpaceParameters<0, 2, 2, 1> unit_params;
  unit_params.thickness                     = "1";
  unit_params.section.selected_coefficients = {0};
  unit_params.point_cloud.points            = {Point<2>(0., 0.)};
  TensorProductSpace<0, 2, 2, 1> unit_space(unit_params);
  unit_space.initialize();

  TensorProductSpaceParameters<0, 2, 2, 1> double_params;
  double_params.thickness                     = "2";
  double_params.section.selected_coefficients = {0};
  double_params.point_cloud.points            = {Point<2>(0., 0.)};
  TensorProductSpace<0, 2, 2, 1> double_space(double_params);
  double_space.initialize();

  const double unit_measure =
    unit_space.get_locally_owned_section_measure().front().front();
  const double double_measure =
    double_space.get_locally_owned_section_measure().front().front();
  EXPECT_NEAR(double_measure / unit_measure, 4., 1e-12);

  double lifted_sum = 0.;
  for (const auto &weight : double_space.get_locally_owned_weights())
    lifted_sum += weight.front();
  EXPECT_NEAR(lifted_sum, double_measure, 1e-12);
}

#ifdef DEAL_II_WITH_VTK
TEST(TensorProductSpace0D, ImportsReorderedCellDataFromPointCloud)
{
  TensorProductSpaceParameters<0, 2, 2, 1> params;
  params.reduced_grid_name =
    SOURCE_DIR "/gtests/fixtures/point_cloud_cell_data_reordered.vtk";
  params.input_file_fields             = "radius";
  params.thickness                     = "radius";
  params.section.selected_coefficients = {0};

  TensorProductSpace<0, 2, 2, 1> space(params);
  ASSERT_NO_THROW(space.initialize());
  ASSERT_EQ(space.get_properties_bindings().size(), 1u);
  ASSERT_EQ(space.n_representative_entities(), 3u);
  EXPECT_DOUBLE_EQ(space.get_entity_property_values(0)[0], 10.);
  EXPECT_DOUBLE_EQ(space.get_entity_property_values(1)[0], 20.);
  EXPECT_DOUBLE_EQ(space.get_entity_property_values(2)[0], 30.);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(0), 10.);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(1), 20.);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(2), 30.);
}

TEST(TensorProductSpace0D, ImportsRadiusThicknessFromPointCloud)
{
  for (const std::string &filename :
       {std::string(SOURCE_DIR) + "/gtests/fixtures/point_cloud_minimal.vtk",
        std::string(SOURCE_DIR) + "/gtests/fixtures/point_cloud_minimal.vtu",
        std::string(SOURCE_DIR) + "/gtests/fixtures/point_cloud_minimal.pvtu"})
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

TEST(ReducedCoupling0D, MPI_ImportsDistributedPVTU)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(background, 2, -1., 1.);

  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.reduced_grid_name =
    SOURCE_DIR "/gtests/fixtures/point_cloud_minimal.pvtu";
  params.tensor_product_space_parameters.input_file_fields = "radius";
  params.tensor_product_space_parameters.thickness         = "radius";
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.coupling_rhs_expressions                                      = {"1"};

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  ASSERT_NO_THROW(coupling.initialize());
  ASSERT_EQ(coupling.n_representative_entities(), 2u);
  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<int>   local_owner(2, -1);
  for (const auto entity :
       coupling.get_representative_particles().locally_owned_particle_ids())
    {
      local_owner[entity] = rank;
      EXPECT_DOUBLE_EQ(coupling.get_entity_thickness(entity),
                       entity == 0 ? 0.25 : 0.5);
    }
  const auto owners = Utilities::MPI::all_gather(MPI_COMM_WORLD, local_owner);
  for (unsigned int entity = 0; entity < 2; ++entity)
    {
      unsigned int n_owners = 0;
      for (const auto &rank_owners : owners)
        n_owners += rank_owners[entity] >= 0;
      EXPECT_EQ(n_owners, 1u);
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
  PointCloud<2> cloud;
  cloud.points.resize(4);
  cloud.points[0]                                    = Point<2>(-0.7, -0.7);
  cloud.points[1]                                    = Point<2>(-0.7, 0.7);
  cloud.points[2]                                    = Point<2>(0.7, -0.7);
  cloud.points[3]                                    = Point<2>(0.7, 0.7);
  params.tensor_product_space_parameters.point_cloud = cloud;

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
            cloud.points.size() *
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
  PointCloud<2> cloud;
  cloud.points = {Point<2>(0., 0.)};
  params.reduced_coupling_parameters.tensor_product_space_parameters
    .point_cloud = cloud;
  ReducedPoisson<2, 2, 0> problem(params);
  (void)problem;
}

TEST(ReducedPoisson0D, ThreeDimensionalSphereCrossSectionTemplatePathCompiles)
{
  ReducedPoissonParameters<3, 0, 3> params;
  params.reduced_coupling_parameters.tensor_product_space_parameters.section
    .selected_coefficients = {0};
  PointCloud<3> cloud;
  cloud.points = {Point<3>(0., 0., 0.)};
  params.reduced_coupling_parameters.tensor_product_space_parameters
    .point_cloud = cloud;
  ReducedPoisson<3, 3, 0, 3> problem(params);
  (void)problem;
}

TEST(ReducedCoupling0D, ThreeDimensionalSphereCrossSectionInitializes)
{
  parallel::distributed::Triangulation<3> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, 0., 1.);

  ReducedCouplingParameters<0, 3, 3, 1> params;
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.tensor_product_space_parameters.point_cloud.points            = {
    Point<3>(0.5, 0.5, 0.5)};

  ReducedCoupling<0, 3, 3, 1> coupling(background, params);
  ASSERT_NO_THROW(coupling.initialize());
  EXPECT_GT(coupling.get_reference_cross_section().n_quadrature_points(), 0u);
  EXPECT_GT(coupling.get_reference_cross_section().measure(1.), 0.);
}

TEST(ReducedCoupling0D, SpaceRefinementParametersRefineBulk)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);

  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.tensor_product_space_parameters.point_cloud.points            = {
    Point<2>(0., 0.)};
  params.refinement_parameters.refinement_factor            = 1000.;
  params.refinement_parameters.max_refinement_level         = 10;
  params.refinement_parameters.space_pre_refinement_cycles  = 1;
  params.refinement_parameters.space_post_refinement_cycles = 1;

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  coupling.initialize();

  EXPECT_EQ(background.n_global_active_cells(), 16u);
}

TEST(ReducedCoupling0D, PointScaleRefinementTargetsBulkCells)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);

  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  params.tensor_product_space_parameters.point_cloud.points            = {
    Point<2>(0., 0.)};
  params.refinement_parameters.refinement_factor            = 1.;
  params.refinement_parameters.max_refinement_level         = 1;
  params.refinement_parameters.space_post_refinement_cycles = 0;

  ReducedCoupling<0, 2, 2, 1> coupling(background, params);
  coupling.initialize();

  EXPECT_EQ(background.n_global_active_cells(), 4u);
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
  PointCloud<2> cloud;
  cloud.points = {Point<2>(0., 0.)};
  params.reduced_coupling_parameters.tensor_product_space_parameters
    .point_cloud = cloud;

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

TEST(ReducedCoupling0D, VectorInitializationUsesSelectedBasisCount)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  ReducedCouplingParameters<0, 2, 2, 2> params;
  params.tensor_product_space_parameters.section.inclusion_degree      = 1;
  params.tensor_product_space_parameters.section.selected_coefficients = {0,
                                                                          1,
                                                                          2};
  params.tensor_product_space_parameters.point_cloud.points            = {
    Point<2>(0., 0.)};
  params.coupling_rhs_expressions = {"1", "2", "3"};

  ReducedCoupling<0, 2, 2, 2> coupling(background, params);
  ASSERT_NO_THROW(coupling.initialize());
  EXPECT_EQ(coupling.n_representative_dofs_per_entity(), 3u);

  FESystem<2>   fe(FE_Q<2>(1), 2);
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
  EXPECT_GT(matrix.frobenius_norm(), 0.);
}

TEST(ReducedCoupling0D, MPI_AssemblyInterfaces)
{
  parallel::distributed::Triangulation<2> background(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background, -1., 1.);
  background.refine_global(1);
  ReducedCouplingParameters<0, 2, 2, 1> params;
  params.tensor_product_space_parameters.thickness                     = "0.5";
  params.tensor_product_space_parameters.section.selected_coefficients = {0};
  PointCloud<2> cloud;
  cloud.points                                       = {Point<2>(0., 0.)};
  params.tensor_product_space_parameters.point_cloud = cloud;
  params.coupling_rhs_expressions                    = {"1"};
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
