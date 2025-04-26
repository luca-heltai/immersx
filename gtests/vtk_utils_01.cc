#include <deal.II/lac/vector.h>

#include <string>

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

#endif // DEAL_II_WITH_VTK
