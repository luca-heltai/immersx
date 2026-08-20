#include <gtest/gtest.h>

#include "test_paths.h"
#include "vtk_utils.h"

#ifdef DEAL_II_WITH_VTK
TEST(FieldCatalog, PreservesAssociationAndComponentLayout)
{
  FieldCatalog catalog;
  const auto   fe = VTKUtils::vtk_to_finite_element<1, 3>(
    ImmersX::TestPaths::data_filename("tests/simple_1d_grid.vtk"), catalog);
  ASSERT_EQ(catalog.size(), 4u);
  EXPECT_EQ(catalog[0].name, "x");
  EXPECT_EQ(catalog[0].association, FieldAssociation::point_data);
  EXPECT_EQ(catalog[0].n_components, 1u);
  EXPECT_EQ(catalog[0].first_fe_component, 0u);
  EXPECT_EQ(catalog[1].name, "xyz");
  EXPECT_EQ(catalog[1].n_components, 3u);
  EXPECT_EQ(catalog[1].first_fe_component, 1u);
  EXPECT_EQ(catalog[2].association, FieldAssociation::cell_data);
  EXPECT_EQ(catalog[3].n_components, 3u);
  EXPECT_EQ(fe->n_blocks(), catalog.size());
}

TEST(FieldCatalog, ReadCatalogOverloadLinksAndImports)
{
  using namespace dealii;
  Triangulation<1, 3> tria;
  DoFHandler<1, 3>    dof_handler(tria);
  Vector<double>      properties;
  FieldCatalog        catalog;
  VTKUtils::read_vtk(ImmersX::TestPaths::data_filename(
                       "tests/simple_1d_grid.vtk"),
                     dof_handler,
                     properties,
                     catalog);
  ASSERT_EQ(catalog.size(), 4u);
  EXPECT_EQ(properties.size(), dof_handler.n_dofs());
}
#endif
