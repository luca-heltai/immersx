#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "vtk_utils.h"

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

#endif // DEAL_II_WITH_VTK
