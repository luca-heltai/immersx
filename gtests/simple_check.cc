#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>

using namespace dealii;

template <class D>
class SimpleClass : public ::testing::Test
{
public:
  static constexpr int dim = D::value;

  Triangulation<dim> tria;

protected:
  SimpleClass()
  {
    GridGenerator::hyper_cube(tria);
    tria.refine_global(1);
  }
};

using Dimensions = ::testing::Types<std::integral_constant<unsigned int, 2>,
                                    std::integral_constant<unsigned int, 3>>;

TYPED_TEST_SUITE(SimpleClass, Dimensions, );


TYPED_TEST(SimpleClass, CheckSize) // NOLINT
{
  auto n_cells = std::pow(2.0, this->tria.dimension);

  ASSERT_EQ(n_cells, this->tria.n_active_cells());
}
