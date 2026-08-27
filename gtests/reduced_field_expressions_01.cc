#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <gtest/gtest.h>

#include <cmath>

#include "test_paths.h"

#ifdef DEAL_II_WITH_VTK

#  include <immersx/coupling/reduced_coupling.h>

using namespace ImmersX;

using namespace dealii;

namespace
{
  void
  check_field_dependent_reduced_rhs()
  {
    constexpr int reduced_dim  = 1;
    constexpr int dim          = 2;
    constexpr int spacedim     = 3;
    constexpr int n_components = 1;

    parallel::distributed::Triangulation<spacedim> background_tria(
      MPI_COMM_WORLD);
    GridGenerator::hyper_cube(background_tria, -0.2, 1.2);
    background_tria.refine_global(3);

    ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;
    par.tensor_product_space_parameters.reduced_grid_name =
      ImmersX::TestPaths::data_filename("tests/one_cylinder_properties.vtk");
    par.tensor_product_space_parameters.input_file_fields = "radius";
    par.tensor_product_space_parameters.thickness         = "radius";
    par.coupling_rhs_expressions                          = {"radius^2 + 5"};

    ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
      background_tria, par);
    coupling.initialize();

    LinearAlgebraTrilinos::MPI::Vector reduced_rhs;
    reduced_rhs.reinit(coupling.get_dof_handler().locally_owned_dofs(),
                       MPI_COMM_WORLD);
    coupling.assemble_reduced_rhs(reduced_rhs);

    FEValues<reduced_dim, spacedim> reduced_fe_values(
      coupling.get_dof_handler().get_fe(),
      coupling.get_quadrature(),
      update_values | update_quadrature_points | update_JxW_values);
    FEValues<reduced_dim, spacedim> properties_fe_values(
      coupling.get_properties_dh().get_fe(),
      coupling.get_quadrature(),
      update_values);
    const auto &binding = coupling.get_properties_bindings().front();
    FEValuesExtractors::Scalar radius(binding.fe_component);
    std::vector<double>        radius_values(coupling.get_quadrature().size());

    double expected_local_integral = 0.;
    for (const auto &cell : coupling.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          reduced_fe_values.reinit(cell);
          properties_fe_values.reinit(
            cell->as_dof_handler_iterator(coupling.get_properties_dh()));
          properties_fe_values[radius].get_function_values(
            coupling.get_properties(), radius_values);
          for (const auto q : reduced_fe_values.quadrature_point_indices())
            expected_local_integral +=
              (radius_values[q] * radius_values[q] + 5.) *
              reduced_fe_values.JxW(q) *
              coupling.get_reference_cross_section().measure(radius_values[q]);
        }

    const double expected_integral =
      Utilities::MPI::sum(expected_local_integral, MPI_COMM_WORLD);
    EXPECT_NEAR(reduced_rhs.l1_norm(), expected_integral, 1.e-12);
  }
} // namespace

TEST(ReducedCoupling, FieldDependentRhsUsesInterpolatedFields)
{
  ParameterAcceptor::clear();
  check_field_dependent_reduced_rhs();
}

TEST(ReducedCoupling, MPI_FieldDependentRhsUsesInterpolatedFields)
{
  ParameterAcceptor::clear();
  check_field_dependent_reduced_rhs();
}

TEST(ReducedCoupling, MPI_TensorProductDuplicateRhsPopulatesBothComponents)
{
  ParameterAcceptor::clear();

  constexpr int reduced_dim  = 1;
  constexpr int dim          = 2;
  constexpr int spacedim     = 3;
  constexpr int n_components = 3;

  parallel::distributed::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(background_tria, 0.0, 1.0);
  background_tria.refine_global(2);

  ReducedCouplingParameters<reduced_dim, dim, spacedim, n_components> par;
  par.tensor_product_space_parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder.vtk");
  par.tensor_product_space_parameters.fe_degree                     = 1;
  par.tensor_product_space_parameters.thickness                     = "0.2";
  par.tensor_product_space_parameters.section.inclusion_degree      = 1;
  par.tensor_product_space_parameters.section.refinement_level      = 1;
  par.tensor_product_space_parameters.section.selected_coefficients = {3, 7};
  par.coupling_rhs_expressions = {"0.1", "0.1"};

  ReducedCoupling<reduced_dim, dim, spacedim, n_components> coupling(
    background_tria, par);
  coupling.initialize();

  LinearAlgebraTrilinos::MPI::Vector reduced_rhs;
  reduced_rhs.reinit(coupling.get_dof_handler().locally_owned_dofs(),
                     MPI_COMM_WORLD);
  coupling.assemble_reduced_rhs(reduced_rhs);

  const auto &dof_handler = coupling.get_dof_handler();
  const auto &fe          = dof_handler.get_fe();
  ASSERT_EQ(fe.n_components(), 2u);

  std::vector<IndexSet> component_dofs(fe.n_components());
  std::vector<double>   component_norms(fe.n_components(), 0.0);
  for (unsigned int component = 0; component < fe.n_components(); ++component)
    {
      std::vector<bool> mask(fe.n_components(), false);
      mask[component] = true;
      component_dofs[component] =
        DoFTools::extract_dofs(dof_handler, ComponentMask(mask));

      double local_squared_norm = 0.0;
      for (const auto index : component_dofs[component])
        local_squared_norm += reduced_rhs[index] * reduced_rhs[index];
      component_norms[component] =
        std::sqrt(Utilities::MPI::sum(local_squared_norm, MPI_COMM_WORLD));
    }

  ASSERT_GT(component_norms[0], 0.0);
  ASSERT_GT(component_norms[1], 0.0);
  EXPECT_NEAR(component_norms[0], component_norms[1], 1.e-12);

  auto first  = component_dofs[0].begin();
  auto second = component_dofs[1].begin();
  for (; first != component_dofs[0].end(); ++first, ++second)
    EXPECT_NEAR(reduced_rhs[*first], reduced_rhs[*second], 1.e-12);
  EXPECT_EQ(second, component_dofs[1].end());
}

TEST(TensorProductSpace, ThicknessExpressionUsesAResolvedBinding)
{
  ParameterAcceptor::clear();
  TensorProductSpaceParameters<1, 2, 3, 1> parameters;
  parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder_properties.vtk");
  parameters.input_file_fields = "radius";
  parameters.thickness         = "radius";

  TensorProductSpace<1, 2, 3, 1> space(parameters);
  space.initialize();

  ASSERT_EQ(space.get_properties_bindings().size(), 1u);
  EXPECT_EQ(space.get_properties_bindings().front().symbol_name, "radius");
  EXPECT_EQ(space.get_thickness_expression(), "radius");
}

TEST(TensorProductSpace, ThicknessExpressionCanDependOnTime)
{
  ParameterAcceptor::clear();
  TensorProductSpaceParameters<1, 2, 3, 1> parameters;
  parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder_properties.vtk");
  parameters.input_file_fields = "radius";
  parameters.thickness         = "radius*sin(t)";

  TensorProductSpace<1, 2, 3, 1> space(parameters);
  space.set_time(numbers::PI / 2.0);
  ASSERT_NO_THROW(space.initialize());
  EXPECT_EQ(space.get_thickness_expression(), "radius*sin(t)");
}

TEST(TensorProductSpace, TransfersRequestedFieldsAfterRefinement)
{
  ParameterAcceptor::clear();
  TensorProductSpaceParameters<1, 2, 3, 1> parameters;
  parameters.reduced_grid_name =
    ImmersX::TestPaths::data_filename("tests/one_cylinder_properties.vtk");
  parameters.input_file_fields = "radius";

  TensorProductSpace<1, 2, 3, 1> space(parameters);
  space.preprocess_serial_triangulation = [](auto &tria) {
    tria.refine_global(1);
    tria.refine_global(1);
  };
  ASSERT_NO_THROW(space.initialize());
  EXPECT_GT(space.get_properties_dh().n_dofs(), 0u);
  EXPECT_GT(space.get_properties().l2_norm(), 0.0);
}

#endif
