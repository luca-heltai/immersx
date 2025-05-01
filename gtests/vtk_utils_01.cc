#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "vtk_utils.h"

namespace LA = dealii::LinearAlgebra;
using namespace dealii;

#ifdef DEAL_II_WITH_VTK

TEST(VTKUtils, ReadCellData)
{
  // Provide a valid VTK file with known cell data for this test
  std::string    vtk_filename   = SOURCE_DIR "/data/tests/mstree_10.vtk";
  std::string    cell_data_name = "edge_length";
  Vector<double> output_vector;
  ASSERT_NO_THROW(
    VTKUtils::read_cell_data(vtk_filename, cell_data_name, output_vector));
  // Optionally, check expected size or values
  EXPECT_EQ(output_vector.size(), 9);
}

TEST(VTKUtils, ReadPointDataScalar)
{
  std::string    vtk_filename    = SOURCE_DIR "/data/tests/mstree_10.vtk";
  std::string    point_data_name = "path_distance";
  Vector<double> output_vector;
  ASSERT_NO_THROW(
    VTKUtils::read_point_data(vtk_filename, point_data_name, output_vector));
  // Optionally, check expected size or values
  EXPECT_EQ(output_vector.size(), 10);
}

TEST(VTKUtils, ReadVtkMesh)
{
  std::string         vtk_filename = SOURCE_DIR "/data/tests/mstree_10.vtk";
  Triangulation<1, 3> tria;
  ASSERT_NO_THROW(VTKUtils::read_vtk(vtk_filename, tria));

  EXPECT_EQ(tria.n_vertices(), 10);
  EXPECT_EQ(tria.n_active_cells(), 9);
}



TEST(VTKUtils, ReadVtkDH)
{
  std::string         vtk_filename = SOURCE_DIR "/data/tests/mstree_10.vtk";
  Triangulation<1, 3> tria;
  DoFHandler<1, 3>    dof_handler(tria);
  Vector<double>      output_vector;
  std::vector<std::string> data_names;
  ASSERT_NO_THROW(
    VTKUtils::read_vtk(vtk_filename, dof_handler, output_vector, data_names));

  EXPECT_EQ(tria.n_vertices(), 10);
  EXPECT_EQ(tria.n_active_cells(), 9);
  EXPECT_EQ(dof_handler.n_dofs(), 19);
  EXPECT_EQ(data_names.size(), 2);
  EXPECT_EQ(data_names[0], "edge_length");
  EXPECT_EQ(data_names[1], "path_distance");
  EXPECT_EQ(output_vector.size(), 19);
  std::cout << "output vector norm: " << output_vector.l2_norm() << std::endl;
}

TEST(VTKUtils, MPI_FillDistributedVectorFromSerial)
{
  const int          dim       = 2;
  const unsigned int fe_degree = 1;
  FE_Q<dim>          fe(fe_degree);
  MappingQ1<dim>     mapping;


  // Setup serial tria
  Triangulation<dim> serial_tria;
  GridGenerator::hyper_cube(serial_tria, 0, 1);
  serial_tria.refine_global(2);

  DoFHandler<dim> serial_dof_handler(serial_tria);
  serial_dof_handler.distribute_dofs(fe);

  // Build support point map (serial)
  std::map<types::global_dof_index, Point<dim>> serial_support_points;
  DoFTools::map_dofs_to_support_points(mapping,
                                       serial_dof_handler,
                                       serial_support_points);
  std::map<Point<dim>, types::global_dof_index, VTKUtils::PointComparator<dim>>
    serial_map;
  for (const auto &pair : serial_support_points)
    serial_map[pair.second] = pair.first;


  // Fill serial vector with position-dependent values
  Vector<double> serial_vec(serial_dof_handler.n_dofs());
  for (const auto &pair : serial_map)
    {
      const Point<dim>             &pt           = pair.first;
      const types::global_dof_index serial_index = pair.second;

      // Calculate value based on point coordinates
      double value             = 10.0 * pt[0] + 1. * pt[1];
      serial_vec[serial_index] = value;
    }


  // Parallel tria, it must be identical to the one I generated above

  parallel::distributed::Triangulation<dim> parallel_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(parallel_tria, 0, 1);
  parallel_tria.refine_global(2);

  DoFHandler<dim> parallel_dof_handler(parallel_tria);
  parallel_dof_handler.distribute_dofs(fe);

  // Build support point map (parallel)

  std::map<types::global_dof_index, Point<dim>> parallel_support_points;
  DoFTools::map_dofs_to_support_points(mapping,
                                       parallel_dof_handler,
                                       parallel_support_points);
  std::map<Point<dim>, types::global_dof_index, VTKUtils::PointComparator<dim>>
    parallel_map;
  for (const auto &pair : parallel_support_points)
    parallel_map[pair.second] = pair.first;

  // Fill parallel vector using the utility function
  LA::distributed::Vector<double> parallel_vec;
  VTKUtils::fill_distributed_vector_from_serial(
    parallel_dof_handler.locally_owned_dofs(),
    serial_vec,
    serial_map,
    parallel_vec,
    parallel_map,
    MPI_COMM_WORLD);


  // Check values in distributed vector
  for (const auto &p_pair : parallel_map)
    {
      const auto &pt             = p_pair.first;
      const auto &parallel_index = p_pair.second;

      if (!parallel_dof_handler.locally_owned_dofs().is_element(parallel_index))
        continue;

      // Calculate expected value based on point coordinates
      double expected_value = 10.0 * pt[0] + 1.0 * pt[1];

      // Check if vector has correct value
      ASSERT_NEAR(parallel_vec[parallel_index], expected_value, 1e-10)
        << "Mismatch at point " << pt;
    }
}

TEST(VTKUtils, MPI_TransferVTKDataToParallel)
{
  const int dim      = 1;
  const int spacedim = 3;

  std::string vtk_filename = SOURCE_DIR "/data/tests/mstree_100.vtk";

  Triangulation<dim, spacedim> serial_tria;
  DoFHandler<dim, spacedim>    serial_dof_handler(serial_tria);
  Vector<double>               serial_data;
  std::vector<std::string>     data_names;

  VTKUtils::read_vtk(vtk_filename, serial_dof_handler, serial_data, data_names);

  // Store the original L2 norm
  const double serial_norm = serial_data.l2_norm();

  MappingQ1<dim, spacedim>                           mapping;
  std::map<types::global_dof_index, Point<spacedim>> serial_support_points;
  DoFTools::map_dofs_to_support_points(mapping,
                                       serial_dof_handler,
                                       serial_support_points);

  std::map<Point<spacedim>,
           types::global_dof_index,
           VTKUtils::PointComparator<spacedim>>
    serial_map;
  for (const auto &pair : serial_support_points)
    serial_map[pair.second] = pair.first;


  // First, partition the serial triangulation
  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_tria);

  // Create description for fully distributed triangulation
  const TriangulationDescription::Description<dim, spacedim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, MPI_COMM_WORLD);

  // Create the parallel triangulation
  parallel::fullydistributed::Triangulation<dim, spacedim> parallel_tria(
    MPI_COMM_WORLD);
  parallel_tria.create_triangulation(description);


  DoFHandler<dim, spacedim> parallel_dof_handler(parallel_tria);
  parallel_dof_handler.distribute_dofs(serial_dof_handler.get_fe());

  // point-to-DoF mapping for parallel mesh
  std::map<types::global_dof_index, Point<spacedim>> parallel_support_points;
  DoFTools::map_dofs_to_support_points(mapping,
                                       parallel_dof_handler,
                                       parallel_support_points);

  std::map<Point<spacedim>,
           types::global_dof_index,
           VTKUtils::PointComparator<spacedim>>
    parallel_map;
  for (const auto &pair : parallel_support_points)
    parallel_map[pair.second] = pair.first;


  LA::distributed::Vector<double> parallel_data;
  VTKUtils::fill_distributed_vector_from_serial(
    parallel_dof_handler.locally_owned_dofs(),
    serial_data,
    serial_map,
    parallel_data,
    parallel_map,
    MPI_COMM_WORLD);

  //  compare norms
  double local_norm_sq = 0.0;
  for (const auto &pair : parallel_map)
    {
      const auto &dof_index = pair.second;

      if (parallel_dof_handler.locally_owned_dofs().is_element(dof_index))
        local_norm_sq += parallel_data[dof_index] * parallel_data[dof_index];
    }

  const double global_norm_sq =
    Utilities::MPI::sum(local_norm_sq, MPI_COMM_WORLD);
  const double parallel_norm = std::sqrt(global_norm_sq);

  // Assert that the parallel norm matches the serial norm
  ASSERT_NEAR(serial_norm, parallel_norm, 1e-10)
    << "Data transfer failed: serial norm = " << serial_norm
    << ", parallel norm = " << parallel_norm;

  // Additional verification: check sizes match properly
  EXPECT_EQ(serial_tria.n_active_cells(),
            parallel_tria.n_global_active_cells());
  EXPECT_EQ(serial_dof_handler.n_dofs(), parallel_dof_handler.n_dofs());
}

#endif // DEAL_II_WITH_VTK
