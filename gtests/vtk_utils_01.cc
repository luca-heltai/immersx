#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <mpi.h>

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
  std::string    vtk_filename     = SOURCE_DIR "/data/tests/mstree_10.vtk";
  std::string    vertex_data_name = "path_distance";
  Vector<double> output_vector;
  ASSERT_NO_THROW(
    VTKUtils::read_vertex_data(vtk_filename, vertex_data_name, output_vector));
  // Optionally, check expected size or values
  EXPECT_EQ(output_vector.size(), 10);
}

TEST(VTKUtils, ReadPointDataScalarAndIndexIt)
{
  std::string    vtk_filename     = SOURCE_DIR "/data/tests/mstree_10.vtk";
  std::string    vertex_data_name = "path_distance";
  Vector<double> output_vector;
  ASSERT_NO_THROW(
    VTKUtils::read_vertex_data(vtk_filename, vertex_data_name, output_vector));
  // Optionally, check expected size or values
  EXPECT_EQ(output_vector.size(), 10);

  Triangulation<1, 3> triangulation;
  GridIn<1, 3>        grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream f(vtk_filename);
  grid_in.read_vtk(f);

  for (const auto &cell : triangulation.active_cell_iterators())
    for (unsigned int i = 0; i < 1; ++i)
      std::cout << "Global vertex index " << cell->vertex_index(i)
                << " has data " << output_vector[cell->vertex_index(i)]
                << std::endl;
}

TEST(VTKUtils, MPI_ReadPointDataScalarAndIndexIt)
{
  std::string    vtk_filename     = SOURCE_DIR "/data/tests/mstree_10.vtk";
  std::string    vertex_data_name = "path_distance";
  Vector<double> output_vector;
  ASSERT_NO_THROW(
    VTKUtils::read_vertex_data(vtk_filename, vertex_data_name, output_vector));
  EXPECT_EQ(output_vector.size(), 10);

  Triangulation<1, 3> serial_tria;
  GridIn<1, 3>        grid_in;
  grid_in.attach_triangulation(serial_tria);
  std::ifstream f(vtk_filename);
  grid_in.read_vtk(f);

  ASSERT_GT(serial_tria.n_vertices(), 0);
  ASSERT_GT(serial_tria.n_active_cells(), 0);
  std::cout << "Serial triangulation: " << serial_tria.n_vertices()
            << " vertices, " << serial_tria.n_active_cells() << " cells"
            << std::endl;

  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_tria);

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<1, 3> dist_tria(MPI_COMM_WORLD);
  dist_tria.create_triangulation(construction_data);

  ASSERT_GT(dist_tria.n_active_cells(), 0);
  std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Distributed triangulation has "
            << dist_tria.n_locally_owned_active_cells() << " local cells"
            << std::endl;

  // Verify mapping
  auto dist_to_serial_mapping =
    VTKUtils::create_vertex_mapping(serial_tria, dist_tria);
  ASSERT_GT(dist_to_serial_mapping.size(), 0);
  std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Created mapping for " << dist_to_serial_mapping.size()
            << " vertices" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  unsigned int verified_vertices = 0;

  for (const auto &cell : dist_tria.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          {
            for (unsigned int v = 0; v < cell->n_vertices(); ++v)
              {
                const unsigned int dist_vertex_index = cell->vertex_index(v);

                //  corresponding serial index
                auto it = dist_to_serial_mapping.find(dist_vertex_index);
                if (it != dist_to_serial_mapping.end())
                  {
                    const unsigned int serial_vertex_index = it->second;
                    const double       data_value =
                      output_vector[serial_vertex_index];

                    std::cout << "Vertex " << dist_vertex_index
                              << " (serial: " << serial_vertex_index
                              << ") with location " << cell->vertex(v)
                              << " data: " << data_value << std::endl;
                    verified_vertices++;
                  }
              }
          }
        }
    }
  // Verify we accessed at least some vertices
  ASSERT_GT(verified_vertices, 0);
  std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Successfully accessed data for " << verified_vertices
            << " vertices" << std::endl;
}


TEST(VTKUtils, MPI_ReadCellDataScalarAndIndexIt)
{
  std::string    vtk_filename   = SOURCE_DIR "/data/tests/mstree_10.vtk";
  std::string    cell_data_name = "edge_length";
  Vector<double> output_vector;
  ASSERT_NO_THROW(
    VTKUtils::read_cell_data(vtk_filename, cell_data_name, output_vector));
  EXPECT_EQ(output_vector.size(), 9);

  Triangulation<1, 3> serial_tria;
  GridIn<1, 3>        grid_in;
  grid_in.attach_triangulation(serial_tria);
  std::ifstream f(vtk_filename);
  grid_in.read_vtk(f);

  ASSERT_GT(serial_tria.n_vertices(), 0);
  ASSERT_GT(serial_tria.n_active_cells(), 0);
  std::cout << "Serial triangulation: " << serial_tria.n_vertices()
            << " vertices, " << serial_tria.n_active_cells() << " cells"
            << std::endl;

  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_tria);

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, MPI_COMM_WORLD);
  parallel::fullydistributed::Triangulation<1, 3> dist_tria(MPI_COMM_WORLD);
  dist_tria.create_triangulation(construction_data);

  ASSERT_GT(dist_tria.n_active_cells(), 0);
  std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Distributed triangulation has "
            << dist_tria.n_locally_owned_active_cells() << " local cells"
            << std::endl;

  // Verify mapping
  auto dist_to_serial_mapping =
    VTKUtils::create_vertex_mapping(serial_tria, dist_tria);
  ASSERT_GT(dist_to_serial_mapping.size(), 0);
  std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Created mapping for " << dist_to_serial_mapping.size()
            << " vertices" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  unsigned int verified_cells = 0;

  for (const auto &cell : dist_tria.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          std::cout << "Process "
                    << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                    << ": Cell with index " << cell->index()
                    << " reads cell data"
                    << output_vector[cell->id().get_coarse_cell_id()]
                    << std::endl;
          ++verified_cells;
        }
    }

  // Verify we accessed at least some vertices
  ASSERT_GT(verified_cells, 0);
  std::cout << "Process " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Successfully accessed data for " << verified_cells
            << " vertices" << std::endl;
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

  // original L2 norm for comparison
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

  // Store few sample points and their values for later verification
  std::vector<Point<spacedim>> sample_points;
  std::vector<double>          sample_values;

  // Select some specific points to verify (first, middle, last)
  unsigned int count = 0;
  for (const auto &pair : serial_map)
    {
      // Take the first point, a point in the middle, and the last point
      if (count == 0 || count == serial_map.size() / 2 ||
          count == serial_map.size() - 1)
        {
          sample_points.push_back(pair.first);
          sample_values.push_back(serial_data[pair.second]);
          std::cout << "Serial point " << pair.first << " has value "
                    << serial_data[pair.second] << std::endl;
        }
      count++;
    }


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

  // compute and compare norms
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

  // point value verification: Each process checks if it owns any of the
  // sample points, then verifies values
  for (size_t i = 0; i < sample_points.size(); ++i)
    {
      const Point<spacedim> &point          = sample_points[i];
      const double           expected_value = sample_values[i];

      // Find this point in the parallel mesh
      auto it = parallel_map.find(point);
      if (it != parallel_map.end())
        {
          const types::global_dof_index dof_index = it->second;

          // Check if this process owns this DoF
          if (parallel_dof_handler.locally_owned_dofs().is_element(dof_index))
            {
              const double actual_value = parallel_data[dof_index];
              std::cout << "Process "
                        << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                        << " owns point " << point << " with value "
                        << actual_value << " (expected: " << expected_value
                        << ")" << std::endl;

              ASSERT_NEAR(expected_value, actual_value, 1e-10)
                << "Point data transfer failed at " << point;
            }
        }
    }

  // Gather all point verification results to ensure all points were checked
  std::vector<int> points_verified_local(sample_points.size(), 0);
  std::vector<int> points_verified_global(sample_points.size(), 0);

  for (size_t i = 0; i < sample_points.size(); ++i)
    {
      const Point<spacedim> &point = sample_points[i];
      auto                   it    = parallel_map.find(point);
      if (it != parallel_map.end() &&
          parallel_dof_handler.locally_owned_dofs().is_element(it->second))
        {
          points_verified_local[i] = 1;
        }
    }

  // Sum up verification status across processes
  Utilities::MPI::sum(points_verified_local,
                      MPI_COMM_WORLD,
                      points_verified_global);

  // On root process, verify all points were checked
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (size_t i = 0; i < sample_points.size(); ++i)
        {
          ASSERT_GT(points_verified_global[i], 0)
            << "Sample point " << sample_points[i]
            << " was not verified by any process";
        }
    }

  // Additional verification: check sizes match properly
  EXPECT_EQ(serial_tria.n_active_cells(),
            parallel_tria.n_global_active_cells());
  EXPECT_EQ(serial_dof_handler.n_dofs(), parallel_dof_handler.n_dofs());
}



TEST(VTKUtils, MPI_SerialToDistributed)
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

  // First, partition the serial triangulation
  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_tria);

  // Create description for fully distributed triangulation
  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, MPI_COMM_WORLD);

  // Create the parallel triangulation
  parallel::fullydistributed::Triangulation<dim> parallel_tria(MPI_COMM_WORLD);
  parallel_tria.create_triangulation(description);

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

  parallel_vec.reinit(parallel_dof_handler.locally_owned_dofs(),
                      MPI_COMM_WORLD);

  VTKUtils::serial_vector_to_distributed_vector(serial_dof_handler,
                                                parallel_dof_handler,
                                                serial_vec,
                                                parallel_vec);

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

TEST(VTKUtils, MPI_DistributedVerticesToSerialVertices)
{
  const int dim = 2;

  // Setup serial tria
  Triangulation<dim> serial_tria;
  GridGenerator::hyper_cube(serial_tria, 0, 1);
  serial_tria.refine_global(2);

  // First, partition the serial triangulation
  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), serial_tria);

  // Create description for fully distributed triangulation
  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, MPI_COMM_WORLD);

  // Create the parallel triangulation
  parallel::fullydistributed::Triangulation<dim> parallel_tria(MPI_COMM_WORLD);
  parallel_tria.create_triangulation(description);

  std::vector<types::global_vertex_index> distributed_to_serial_vertex_indices =
    VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
      serial_tria, parallel_tria);

  const std::vector<Point<dim>> &serial_vertices = serial_tria.get_vertices();
  const std::vector<Point<dim>> &parallel_vertices =
    parallel_tria.get_vertices();

  unsigned int local_vertices_checked = 0;
  for (unsigned int i = 0; i < parallel_vertices.size(); ++i)
    {
      const auto &serial_index = distributed_to_serial_vertex_indices[i];
      if (serial_index != numbers::invalid_unsigned_int)
        {
          const Point<dim> &parallel_vertex = parallel_vertices[i];
          const Point<dim> &serial_vertex   = serial_vertices[serial_index];

          ASSERT_LT(parallel_vertex.distance(serial_vertex), 1e-10)
            << "Mismatch for parallel vertex " << i << " (serial vertex "
            << serial_index << "). Coordinates: " << parallel_vertex << " vs "
            << serial_vertex;
          local_vertices_checked++;
        }
    }
}

#endif // DEAL_II_WITH_VTK
