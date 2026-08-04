#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <gtest/gtest.h>

#ifdef DEAL_II_WITH_VTK

#  include "reduced_coupling.h"

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
      SOURCE_DIR "/data/tests/one_cylinder_properties.vtk";
    par.tensor_product_space_parameters.input_file_fields = "radius";
    par.tensor_product_space_parameters.thickness_expression = "radius";
    par.coupling_rhs_expressions = {"radius^2 + 5"};

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
    std::vector<double> radius_values(coupling.get_quadrature().size());

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
}

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

TEST(TensorProductSpace, LegacyThicknessFieldUsesAResolvedBinding)
{
  ParameterAcceptor::clear();
  TensorProductSpaceParameters<1, 2, 3, 1> parameters;
  parameters.reduced_grid_name =
    SOURCE_DIR "/data/tests/one_cylinder_properties.vtk";
  parameters.thickness_field_name = "radius";

  TensorProductSpace<1, 2, 3, 1> space(parameters);
  space.initialize();

  ASSERT_EQ(space.get_properties_bindings().size(), 1u);
  EXPECT_EQ(space.get_properties_bindings().front().symbol_name, "radius");
  if (SymbolicFieldEvaluator::available())
    EXPECT_EQ(space.get_thickness_expression(), "radius");
  else
    EXPECT_NE(space.get_legacy_thickness_binding(),
              numbers::invalid_unsigned_int);
}

TEST(TensorProductSpace, RejectsRequestedFieldsAfterRefinement)
{
  ParameterAcceptor::clear();
  TensorProductSpaceParameters<1, 2, 3, 1> parameters;
  parameters.reduced_grid_name =
    SOURCE_DIR "/data/tests/one_cylinder_properties.vtk";
  parameters.input_file_fields = "radius";

  TensorProductSpace<1, 2, 3, 1> space(parameters);
  space.preprocess_serial_triangulation = [](auto &tria) {
    tria.refine_global(1);
  };
  EXPECT_THROW(space.initialize(), std::exception);
}

#endif
