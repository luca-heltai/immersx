#include <gtest/gtest.h>

#include "vtk_utils.h"

#ifdef DEAL_II_WITH_VTK
TEST(VTKFieldCatalog, PreservesAssociationAndComponentLayout)
{
  VTKFieldCatalog catalog;
  const auto      fe = VTKUtils::vtk_to_finite_element<1, 3>(
    std::string(SOURCE_DIR) + "/data/tests/simple_1d_grid.vtk", catalog);
  ASSERT_EQ(catalog.size(), 4u);
  EXPECT_EQ(catalog[0].vtk_name, "x");
  EXPECT_EQ(catalog[0].association, VTKFieldAssociation::point_data);
  EXPECT_EQ(catalog[0].n_components, 1u);
  EXPECT_EQ(catalog[0].first_fe_component, 0u);
  EXPECT_EQ(catalog[1].vtk_name, "xyz");
  EXPECT_EQ(catalog[1].n_components, 3u);
  EXPECT_EQ(catalog[1].first_fe_component, 1u);
  EXPECT_EQ(catalog[2].association, VTKFieldAssociation::cell_data);
  EXPECT_EQ(catalog[3].n_components, 3u);
  EXPECT_EQ(fe->n_blocks(), catalog.size());
}

TEST(VTKFieldCatalog, ReadCatalogOverloadLinksAndImports)
{
  using namespace dealii;
  Triangulation<1, 3> tria;
  DoFHandler<1, 3>    dof_handler(tria);
  Vector<double>      properties;
  VTKFieldCatalog     catalog;
  VTKUtils::read_vtk(std::string(SOURCE_DIR) + "/data/tests/simple_1d_grid.vtk",
                     dof_handler,
                     properties,
                     catalog);
  ASSERT_EQ(catalog.size(), 4u);
  EXPECT_EQ(properties.size(), dof_handler.n_dofs());
}
#endif
