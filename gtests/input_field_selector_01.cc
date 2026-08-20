#include <gtest/gtest.h>
#include <immersx/core/input_field_selector.h>

#include <stdexcept>

using namespace ImmersX;

TEST(InputFieldSelector, ResolvesAliasesAndWildcard)
{
  FieldCatalog catalog = {{"radius", FieldAssociation::point_data, 1, 0, 0},
                          {"velocity", FieldAssociation::point_data, 2, 1, 1},
                          {"radius", FieldAssociation::cell_data, 1, 3, 2}};

  const auto bindings =
    InputFieldSelector::resolve("r=point:radius, ux=velocity[0]", catalog);
  ASSERT_EQ(bindings.size(), 2u);
  EXPECT_EQ(bindings[0].symbol_name, "r");
  EXPECT_EQ(bindings[0].field_index, 0u);
  EXPECT_EQ(bindings[0].fe_component, 0u);
  EXPECT_EQ(bindings[1].symbol_name, "ux");
  EXPECT_EQ(bindings[1].field_component, 0u);

  FieldCatalog wildcard_catalog(catalog.begin(), catalog.begin() + 2);
  const auto   all = InputFieldSelector::resolve("*", wildcard_catalog);
  ASSERT_EQ(all.size(), 3u);
  EXPECT_EQ(all[0].symbol_name, "radius");
  EXPECT_EQ(all[1].symbol_name, "velocity_0");
  EXPECT_EQ(all[2].symbol_name, "velocity_1");
}

TEST(InputFieldSelector, RejectsAmbiguousAndInvalidSelectors)
{
  FieldCatalog catalog = {{"radius", FieldAssociation::point_data, 1, 0, 0},
                          {"radius", FieldAssociation::cell_data, 1, 1, 1}};
  EXPECT_THROW(InputFieldSelector::resolve("radius", catalog),
               std::runtime_error);
  EXPECT_THROW(InputFieldSelector::resolve("x=point:radius", catalog),
               std::runtime_error);
  EXPECT_THROW(InputFieldSelector::resolve("r=point:radius[1]", catalog),
               std::runtime_error);
}

TEST(InputFieldSelector, RejectsInvalidWildcardAliases)
{
  FieldCatalog catalog = {
    {"not-a-symbol", FieldAssociation::point_data, 1, 0, 0}};
  EXPECT_THROW(InputFieldSelector::resolve("*", catalog), std::runtime_error);
}
