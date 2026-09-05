// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>
#include <immersx/core/observable.h>
#include <immersx/physics/poisson.h>

#include <type_traits>

namespace
{
  using namespace dealii;

  struct ExternalFESystem
  {
    Triangulation<2>          triangulation;
    FESystem<2>               finite_element{FE_Q<2>(1), 2};
    DoFHandler<2>             dof_handler{triangulation};
    MappingQ1<2>              mapping;
    AffineConstraints<double> constraints;
    IndexSet                  relevant;

    ExternalFESystem()
    {
      GridGenerator::hyper_cube(triangulation);
      dof_handler.distribute_dofs(finite_element);
      relevant = dof_handler.locally_owned_dofs();
    }
  };
} // namespace

TEST(FESpace, IsANonOwningViewAndSupportsSubspaces)
{
  ExternalFESystem     system;
  ImmersX::StateLayout layout;
  const auto           V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);

  const auto displacement =
    V.field(layout, "displacement", FEValuesExtractors::Vector(0));
  const auto pressure =
    V.field(layout, "pressure", FEValuesExtractors::Scalar(0));
  const auto displacement_again =
    V[FEValuesExtractors::Vector(0)].field("displacement-preview");

  EXPECT_EQ(&V.dof_handler(), &system.dof_handler);
  EXPECT_EQ(&V.mapping(), &system.mapping);
  EXPECT_EQ(&V.constraints(), &system.constraints);
  EXPECT_EQ(&V.locally_relevant_dofs(), &system.relevant);
  EXPECT_EQ(displacement.field_id(), ImmersX::FieldId(0));
  EXPECT_EQ(pressure.field_id(), ImmersX::FieldId(1));
  EXPECT_NE(displacement.field_id(), pressure.field_id());
  EXPECT_FALSE(displacement_again.is_registered());
  EXPECT_EQ(&displacement.dof_handler(), &system.dof_handler);
  EXPECT_EQ(displacement.extractor().first_vector_component, 0u);
}

TEST(FESpace, ValueAndGradientExposeTypedDependencies)
{
  ExternalFESystem     system;
  ImmersX::StateLayout layout;
  const auto           V = ImmersX::fe_space(system.dof_handler,
                                   system.mapping,
                                   system.constraints,
                                   system.relevant);
  const auto           displacement =
    V.field(layout, "displacement", FEValuesExtractors::Vector(0));
  const auto pressure =
    V.field(layout, "pressure", FEValuesExtractors::Scalar(0));

  const auto displacement_value = ImmersX::value(displacement);
  const auto pressure_value     = ImmersX::value(pressure);
  const auto displacement_grad  = ImmersX::gradient(displacement);
  const auto pressure_grad      = ImmersX::gradient(pressure);

  static_assert(
    std::is_same_v<
      std::decay_t<decltype(displacement_value)>,
      ImmersX::Observable<Tensor<1, 2>, std::decay_t<decltype(displacement)>>>);
  static_assert(std::is_same_v<
                std::decay_t<decltype(pressure_value)>,
                ImmersX::Observable<double, std::decay_t<decltype(pressure)>>>);
  static_assert(
    std::is_same_v<
      std::decay_t<decltype(displacement_grad)>,
      ImmersX::Observable<Tensor<2, 2>, std::decay_t<decltype(displacement)>>>);
  static_assert(
    std::is_same_v<
      std::decay_t<decltype(pressure_grad)>,
      ImmersX::Observable<Tensor<1, 2>, std::decay_t<decltype(pressure)>>>);

  EXPECT_EQ(displacement_value.dependencies(),
            std::vector<ImmersX::FieldId>{displacement.field_id()});
  EXPECT_EQ(pressure_grad.dependencies(),
            std::vector<ImmersX::FieldId>{pressure.field_id()});
  EXPECT_EQ(displacement_value.operation(),
            ImmersX::ObservableOperation::value);
  EXPECT_EQ(pressure_grad.operation(), ImmersX::ObservableOperation::gradient);
  EXPECT_EQ(displacement_value.dimension(), 2u);
  EXPECT_EQ(displacement_value.spacedimension(), 2u);
  EXPECT_TRUE(displacement_value.is_differentiable());
  EXPECT_TRUE(displacement_value.is_linear());
  EXPECT_EQ(displacement_value.space_dimension(), 2u);
  EXPECT_EQ(displacement_value.spacedim(), 2u);
  EXPECT_EQ(displacement_value.source_field(), displacement.field_id());
}

TEST(FESpace, WrapsAnExistingProblemFromTheOutside)
{
  ParameterAcceptor::clear();
  ImmersX::PoissonParameters<2> parameters;
  parameters.initial_refinement  = 0;
  parameters.n_refinement_cycles = 1;
  ImmersX::PoissonSolver<2> problem(parameters);
  problem.make_grid();
  problem.setup_fe();

  const auto V        = ImmersX::fe_space(problem.dof_handler(),
                                   StaticMappingQ1<2>::mapping,
                                   problem.constraints(),
                                   problem.locally_relevant_dofs());
  const auto pressure = V.field("pressure");

  EXPECT_EQ(&pressure.dof_handler(), &problem.dof_handler());
  EXPECT_EQ(pressure.name(), "pressure");
  EXPECT_FALSE(pressure.is_registered());
}
