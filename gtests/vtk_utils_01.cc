#include <deal.II/base/function_parser.h> // Added for FunctionParser
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

#include <deal.II/numerics/vector_tools.h> // Added for VectorTools

#include <mpi.h>

#include <cstdio>  // For std::remove
#include <fstream> // For writing temporary VTK file
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
    VTKUtils::distributed_to_serial_vertex_indices(serial_tria, dist_tria);
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
                const auto serial_vertex_index =
                  dist_to_serial_mapping[dist_vertex_index];
                if (serial_vertex_index != numbers::invalid_unsigned_int)
                  {
                    const double data_value =
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
    VTKUtils::distributed_to_serial_vertex_indices(serial_tria, dist_tria);
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
  EXPECT_EQ(data_names[0], "path_distance");
  EXPECT_EQ(data_names[1], "edge_length");
  EXPECT_EQ(output_vector.size(), 19);
  std::cout << "output vector norm: " << output_vector.l2_norm() << std::endl;
}


TEST(VTKUtils, ReadVtkWithData)
{
  std::string vtk_filename =
    SOURCE_DIR "/data/tests/one_cylinder_properties.vtk";
  Triangulation<1, 3>      tria;
  DoFHandler<1, 3>         dof_handler(tria);
  Vector<double>           output_vector;
  std::vector<std::string> data_names;
  ASSERT_NO_THROW(
    VTKUtils::read_vtk(vtk_filename, dof_handler, output_vector, data_names));

  EXPECT_EQ(tria.n_vertices(), 2);
  EXPECT_EQ(tria.n_active_cells(), 1);
  EXPECT_EQ(dof_handler.n_dofs(), 4);
  EXPECT_EQ(data_names.size(), 2);
  EXPECT_EQ(data_names[0], "path_distance");
  EXPECT_EQ(data_names[1], "radius");
  EXPECT_EQ(output_vector.size(), 4);
  EXPECT_GT(output_vector.l2_norm(), 0.0);
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

  // Fill serial vector with position-dependent values
  Vector<double> serial_vec(serial_dof_handler.n_dofs());

  // 7. Create ParsedFunction
  FunctionParser<dim>           function_parser;
  std::string                   parsed_function_str = "x + 10.0 * y";
  std::map<std::string, double> constants;
  function_parser.initialize(FunctionParser<dim>::default_variable_names(),
                             parsed_function_str,
                             constants,
                             false);
  VectorTools::interpolate(serial_dof_handler, function_parser, serial_vec);

  // Parallel tria, it must be identical to the one I generated above

  parallel::distributed::Triangulation<dim> parallel_tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(parallel_tria, 0, 1);
  parallel_tria.refine_global(2);

  DoFHandler<dim> parallel_dof_handler(parallel_tria);
  parallel_dof_handler.distribute_dofs(fe);

  // Fill parallel vector using the utility function
  LA::distributed::Vector<double> parallel_vec;
  LA::distributed::Vector<double> parallel_vec_expected;


  parallel_vec.reinit(parallel_dof_handler.locally_owned_dofs(),
                      MPI_COMM_WORLD);
  parallel_vec_expected.reinit(parallel_dof_handler.locally_owned_dofs(),
                               DoFTools::extract_locally_relevant_dofs(
                                 parallel_dof_handler),
                               MPI_COMM_WORLD);

  VectorTools::interpolate(parallel_dof_handler,
                           function_parser,
                           parallel_vec_expected);

  VTKUtils::serial_vector_to_distributed_vector(serial_dof_handler,
                                                parallel_dof_handler,
                                                serial_vec,
                                                parallel_vec);
  ASSERT_EQ(parallel_vec.size(), serial_vec.size());
  // Check that the norms match
  const double serial_norm   = serial_vec.l2_norm();
  const double parallel_norm = parallel_vec.l2_norm();
  ASSERT_DOUBLE_EQ(serial_norm, parallel_norm);
  parallel_vec_expected -= parallel_vec;
  const double error_norm = parallel_vec_expected.l2_norm();
  ASSERT_NEAR(error_norm, 0.0, 1e-12);
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

  LA::distributed::Vector<double> parallel_data;
  parallel_data.reinit(parallel_dof_handler.locally_owned_dofs(),
                       MPI_COMM_WORLD);
  VTKUtils::serial_vector_to_distributed_vector(serial_dof_handler,
                                                parallel_dof_handler,
                                                serial_data,
                                                parallel_data);

  ASSERT_EQ(parallel_data.size(), serial_data.size());
  // Check that the norms match
  const double serial_norm   = serial_data.l2_norm();
  const double parallel_norm = parallel_data.l2_norm();
  ASSERT_DOUBLE_EQ(serial_norm, parallel_norm);
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
    VTKUtils::distributed_to_serial_vertex_indices(serial_tria, parallel_tria);

  const std::vector<Point<dim>> &serial_vertices = serial_tria.get_vertices();
  const std::vector<Point<dim>> &parallel_vertices =
    parallel_tria.get_vertices();

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
        }
    }
}


TEST(VTKUtils, VtkToFiniteElement)
{
  std::string vtk_filename = SOURCE_DIR "/data/tests/mstree_10.vtk";
  const auto [fe, data_names] =
    VTKUtils::vtk_to_finite_element<1, 3>(vtk_filename);

  EXPECT_EQ(data_names[0], "path_distance");
  EXPECT_EQ(data_names[1], "edge_length");

  EXPECT_EQ(fe->n_blocks(), 2);
  EXPECT_EQ(fe->n_components(), 2);
  EXPECT_EQ(fe->n_dofs_per_cell(), 3);
  EXPECT_EQ(fe->base_element(0).dofs_per_vertex, 1);
  EXPECT_EQ(fe->base_element(1).dofs_per_vertex, 0);
  EXPECT_EQ(fe->base_element(0).dofs_per_line, 0);
  EXPECT_EQ(fe->base_element(1).dofs_per_line, 1);
}

TEST(VTKUtils, ReadData)
{
  const std::string vtk_filename = SOURCE_DIR "/data/tests/simple_1d_grid.vtk";
  Vector<double>    actual_data;
  VTKUtils::read_data(vtk_filename, actual_data);

  // 4. Verify the data
  // Expected order: All PointData arrays first, then all CellData arrays.
  // Within PointData: "x", then "xyz".
  // Within CellData: "center_x", then "center_xyz".
  // Sizes for simple_1d_grid.vtk:
  // PointData "x": 10 values
  // PointData "xyz": 10 points * 3 components = 30 values
  // CellData "center_x": 9 values
  // CellData "center_xyz": 9 cells * 3 components = 27 values
  // Total = 10 + 30 + 9 + 27 = 76 values
  Vector<double> expected_data(76);
  unsigned int   k = 0;

  // PointData "x" (10 points * 1 component)
  for (int i = 0; i < 10; ++i)
    expected_data[k++] = i / 9.0;

  // PointData "xyz" (10 points * 3 components)
  for (int i = 0; i < 10; ++i)
    {
      expected_data[k++] = 0 + i / 9.0; // x component
      expected_data[k++] = 1 + i / 9.0; // y component
      expected_data[k++] = 2 + i / 9.0; // z component
    }

  // CellData "center_x" (9 cells * 1 component)
  for (int i = 0; i < 9;
       ++i) // Cell centers are ( (i/9.0) + ((i+1)/9.0) ) / 2.0 = (2i+1)/18.0
    expected_data[k++] = (2.0 * i + 1.0) / 18.0;

  // CellData "center_xyz" (9 cells * 3 components)
  for (int i = 0; i < 9; ++i)
    {
      expected_data[k++] = 0.0 + (2.0 * i + 1.0) / 18.0; // x component
      expected_data[k++] = 1.0 + (2.0 * i + 1.0) / 18.0; // y component
      expected_data[k++] = 2.0 + (2.0 * i + 1.0) / 18.0; // z component
    }


  ASSERT_EQ(actual_data.size(), expected_data.size())
    << "Actual data size: " << actual_data.size()
    << ", Expected data size: " << expected_data.size();

  for (unsigned int i = 0; i < actual_data.size(); ++i)
    {
      EXPECT_NEAR(actual_data[i], expected_data[i], 1e-9)
        << "Mismatch at index " << i << "; actual value: " << actual_data[i]
        << ", expected value: " << expected_data[i];
    }
}


TEST(VTKUtils, DataToDealiiVectorAndInterpolate)
{
  const int         dim      = 1;
  const int         spacedim = 3;
  const std::string temp_vtk_filename =
    SOURCE_DIR "/data/tests/simple_1d_grid.vtk";

  Triangulation<dim, spacedim> tria;
  ASSERT_NO_THROW(
    VTKUtils::read_vtk(temp_vtk_filename, tria, /*cleanup=*/true));
  ASSERT_EQ(tria.n_vertices(), 10);
  ASSERT_EQ(tria.n_active_cells(), 9);

  // 3. Read data using VTKUtils::read_data
  Vector<double> raw_data_vector;
  ASSERT_NO_THROW(VTKUtils::read_data(temp_vtk_filename, raw_data_vector));
  // Expected size: x(10*1) + xyz(10*3) + center_x(9*1) + center_xyz(9*3) =
  // 10+30+9+27 = 76
  ASSERT_EQ(raw_data_vector.size(), 76);

  // 4. Create FiniteElement using VTKUtils::vtk_to_finite_element
  auto [fe_system_ptr, data_names_from_fe] =
    VTKUtils::vtk_to_finite_element<dim, spacedim>(temp_vtk_filename);
  ASSERT_TRUE(fe_system_ptr);
  ASSERT_EQ(data_names_from_fe.size(), 4);
  EXPECT_EQ(data_names_from_fe[0], "x");
  EXPECT_EQ(data_names_from_fe[1], "xyz");
  EXPECT_EQ(data_names_from_fe[2], "center_x");
  EXPECT_EQ(data_names_from_fe[3], "center_xyz");
  ASSERT_EQ(fe_system_ptr->n_blocks(), 4);
  // x(1) + xyz(3) + center_x(1) + center_xyz(3) = 8 components
  ASSERT_EQ(fe_system_ptr->n_components(), 8);

  std::cout << "FiniteElement: " << fe_system_ptr->get_name()
            << ", n_blocks: " << fe_system_ptr->n_blocks()
            << ", n_components: " << fe_system_ptr->n_components() << std::endl;

  const auto block_indices = VTKUtils::get_block_indices(*fe_system_ptr);
  ASSERT_EQ(block_indices.total_size(), 8);
  ASSERT_EQ(block_indices.size(), 4);
  ASSERT_EQ(block_indices.block_size(0), 1);
  ASSERT_EQ(block_indices.block_size(1), 3);
  ASSERT_EQ(block_indices.block_size(2), 1);
  ASSERT_EQ(block_indices.block_size(3), 3);

  // 5. Create DoFHandler and distribute DoFs
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe_system_ptr);
  // DoFs: Q1 for "x" (10*1), Q1 for "xyz" (10*3), DG0 for "center_x" (9*1), DG0
  // for "center_xyz" (9*3)
  // Total = 10 + 30 + 9 + 27 = 76
  ASSERT_EQ(dof_handler.n_dofs(), 76);
  ASSERT_EQ(dof_handler.n_dofs(), raw_data_vector.size());


  // 6. Call data_to_dealii_vector
  Vector<double> vector_from_data_to_dealii(dof_handler.n_dofs());
  ASSERT_NO_THROW(VTKUtils::data_to_dealii_vector(
    tria, raw_data_vector, dof_handler, vector_from_data_to_dealii));

  // 7. Create ParsedFunction
  FunctionParser<spacedim> function_parser(fe_system_ptr->n_components());
  // FE blocks order from vtk_to_finite_element: x, xyz, center_x, center_xyz
  // Corresponding expressions:
  // "x" (Q1, 1 comp) -> "x"
  // "xyz" (Q1, 3 comps) -> "x;y;z" (mesh on x-axis)
  // "center_x" (DG0, 1 comp) -> "x" (eval at cell center)
  // "center_xyz" (DG0, 3 comps) -> "x;y;z" (eval at cell center)
  std::string                   parsed_function_str = "x; x;y;z; x; x;y;z";
  std::map<std::string, double> constants;
  function_parser.initialize(FunctionParser<spacedim>::default_variable_names(),
                             parsed_function_str,
                             constants,
                             false);


  // 8. Interpolate ParsedFunction
  Vector<double>           interpolated_vector(dof_handler.n_dofs());
  MappingQ1<dim, spacedim> mapping; // Standard mapping
  ASSERT_NO_THROW(VectorTools::interpolate(
    mapping, dof_handler, function_parser, interpolated_vector));

  // 9. Compare vectors
  ASSERT_EQ(vector_from_data_to_dealii.size(), interpolated_vector.size());
  bool mismatch_found = false;
  for (unsigned int i = 0; i < vector_from_data_to_dealii.size(); ++i)
    {
      if (std::abs(vector_from_data_to_dealii(i) - interpolated_vector(i)) >
          1e-9)
        {
          std::cerr << "Mismatch at index " << i << ": data_to_dealii_vector = "
                    << vector_from_data_to_dealii(i)
                    << ", interpolated_vector = " << interpolated_vector(i)
                    << "\\n"; // Replaced std::endl
          mismatch_found = true;
        }
      // Use EXPECT_NEAR for individual checks to see all failures if any
      EXPECT_NEAR(vector_from_data_to_dealii(i), interpolated_vector(i), 1e-9);
    }
  ASSERT_FALSE(mismatch_found) << "Vectors do not match.";


  // 10. Delete temporary file - REMOVED
}

TEST(VTKUtils, DataFromSimpleVtkToDealiiVectorAndInterpolate)
{
  const int         dim          = 1;
  const int         spacedim     = 3;
  const std::string vtk_filename = SOURCE_DIR "/data/tests/simple_1d_grid.vtk";

  // 1. Read VTK mesh
  Triangulation<dim, spacedim> tria;
  ASSERT_NO_THROW(VTKUtils::read_vtk(vtk_filename, tria, /*cleanup=*/true));
  ASSERT_EQ(tria.n_vertices(), 10);
  ASSERT_EQ(tria.n_active_cells(), 9);

  // 2. Read data using VTKUtils::read_data
  Vector<double> raw_data_vector;
  ASSERT_NO_THROW(VTKUtils::read_data(vtk_filename, raw_data_vector));
  // Expected size: x(10*1) + xyz(10*3) + center_x(9*1) + center_xyz(9*3) =
  // 10+30+9+27 = 76
  ASSERT_EQ(raw_data_vector.size(), 76);

  // 3. Create FiniteElement using VTKUtils::vtk_to_finite_element
  auto [fe_system_ptr, data_names_from_fe] =
    VTKUtils::vtk_to_finite_element<dim, spacedim>(vtk_filename);
  ASSERT_TRUE(fe_system_ptr);
  ASSERT_EQ(data_names_from_fe.size(), 4);
  EXPECT_EQ(data_names_from_fe[0], "x");
  EXPECT_EQ(data_names_from_fe[1], "xyz");
  EXPECT_EQ(data_names_from_fe[2], "center_x");
  EXPECT_EQ(data_names_from_fe[3], "center_xyz");
  ASSERT_EQ(fe_system_ptr->n_blocks(), 4);
  ASSERT_EQ(fe_system_ptr->n_components(), 8);

  // 4. Create DoFHandler and distribute DoFs
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe_system_ptr);
  ASSERT_EQ(dof_handler.n_dofs(), 76);

  // 5. Call data_to_dealii_vector
  Vector<double> vector_from_data_to_dealii(dof_handler.n_dofs());
  ASSERT_NO_THROW(VTKUtils::data_to_dealii_vector(
    tria, raw_data_vector, dof_handler, vector_from_data_to_dealii));

  // 6. Create ParsedFunction
  FunctionParser<spacedim>      function_parser(fe_system_ptr->n_components());
  std::string                   parsed_function_str = "x; x;y;z; x; x;y;z";
  std::map<std::string, double> constants;
  function_parser.initialize(FunctionParser<spacedim>::default_variable_names(),
                             parsed_function_str,
                             constants,
                             false);

  // 7. Interpolate ParsedFunction
  Vector<double>           interpolated_vector(dof_handler.n_dofs());
  MappingQ1<dim, spacedim> mapping;
  ASSERT_NO_THROW(VectorTools::interpolate(
    mapping, dof_handler, function_parser, interpolated_vector));

  // 8. Compare vectors
  ASSERT_EQ(vector_from_data_to_dealii.size(), interpolated_vector.size());
  bool mismatch_found = false;
  for (unsigned int i = 0; i < vector_from_data_to_dealii.size(); ++i)
    {
      if (std::abs(vector_from_data_to_dealii(i) - interpolated_vector(i)) >
          1e-9)
        {
          std::cerr << "Mismatch at global DoF index " << i
                    << ": data_to_dealii_vector = "
                    << vector_from_data_to_dealii(i)
                    << ", interpolated_vector = " << interpolated_vector(i)
                    << "\n";
          mismatch_found = true;
        }
      EXPECT_NEAR(vector_from_data_to_dealii(i), interpolated_vector(i), 1e-9);
    }
  ASSERT_FALSE(mismatch_found) << "Vectors do not match. See cerr for details.";
}

#endif // DEAL_II_WITH_VTK
