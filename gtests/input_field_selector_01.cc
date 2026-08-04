#include <gtest/gtest.h>

#include <stdexcept>

#include "input_field_selector.h"

TEST(InputFieldSelector, ResolvesAliasesAndWildcard)
{
  VTKFieldCatalog catalog = {
    {"radius", VTKFieldAssociation::point_data, 1, 0, 0},
    {"velocity", VTKFieldAssociation::point_data, 2, 1, 1},
    {"radius", VTKFieldAssociation::cell_data, 1, 3, 2}};

  const auto bindings =
    InputFieldSelector::resolve("r=point:radius, ux=velocity[0]", catalog);
  ASSERT_EQ(bindings.size(), 2u);
  EXPECT_EQ(bindings[0].symbol_name, "r");
  EXPECT_EQ(bindings[0].field_index, 0u);
  EXPECT_EQ(bindings[0].fe_component, 0u);
  EXPECT_EQ(bindings[1].symbol_name, "ux");
  EXPECT_EQ(bindings[1].vtk_component, 0u);

  VTKFieldCatalog wildcard_catalog(catalog.begin(), catalog.begin() + 2);
  const auto      all = InputFieldSelector::resolve("*", wildcard_catalog);
  ASSERT_EQ(all.size(), 3u);
  EXPECT_EQ(all[0].symbol_name, "radius");
  EXPECT_EQ(all[1].symbol_name, "velocity_0");
  EXPECT_EQ(all[2].symbol_name, "velocity_1");
}

TEST(InputFieldSelector, RejectsAmbiguousAndInvalidSelectors)
{
  VTKFieldCatalog catalog = {
    {"radius", VTKFieldAssociation::point_data, 1, 0, 0},
    {"radius", VTKFieldAssociation::cell_data, 1, 1, 1}};
  EXPECT_THROW(InputFieldSelector::resolve("radius", catalog),
               std::runtime_error);
  EXPECT_THROW(InputFieldSelector::resolve("x=point:radius", catalog),
               std::runtime_error);
  EXPECT_THROW(InputFieldSelector::resolve("r=point:radius[1]", catalog),
               std::runtime_error);
}

TEST(InputFieldSelector, RejectsInvalidWildcardAliases)
{
  VTKFieldCatalog catalog = {
    {"not-a-symbol", VTKFieldAssociation::point_data, 1, 0, 0}};
  EXPECT_THROW(InputFieldSelector::resolve("*", catalog), std::runtime_error);
}
