// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/core/representation.h>

#include <type_traits>

using namespace ImmersX;


using namespace dealii;


TEST(VectorRepresentation, MixedExtractorSelectsVelocityOnly) // NOLINT
{
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);

  // A small synthetic Stokes-like FE system: two velocity components followed
  // by one scalar pressure component. The representation below selects only
  // the vector extractor beginning at component zero.
  FESystem<2>   mixed_fe(FE_Q<2>(1), 2, FE_Q<2>(1), 1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(mixed_fe);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);
  AffineConstraints<double> constraints(locally_owned_dofs,
                                        locally_relevant_dofs);
  constraints.close();

  ImmersX::StateLayout     layout;
  ImmersX::FieldDescriptor velocity_descriptor;
  velocity_descriptor.name          = "velocity";
  const auto               velocity = layout.add_field(velocity_descriptor);
  ImmersX::FieldDescriptor pressure_descriptor;
  pressure_descriptor.name = "pressure";
  const auto pressure      = layout.add_field(pressure_descriptor);

  ImmersX::RepresentationMetadata metadata;
  metadata.dependencies.push_back(velocity);
  metadata.dependencies.push_back(pressure);

  using Representation = VectorFiniteElementRepresentation<2>;
  static_assert(
    std::is_same<typename Representation::QuadraturePoint::value_type,
                 Tensor<1, 2>>::value,
    "Vector representation quadrature values must remain strongly typed.");

  Representation representation(triangulation,
                                dof_handler,
                                locally_owned_dofs,
                                locally_relevant_dofs,
                                constraints,
                                StaticMappingQ1<2>::mapping,
                                FEValuesExtractors::Vector(0),
                                metadata);

  EXPECT_EQ(representation.metadata().dependencies.size(), 2u);
  EXPECT_EQ(representation.dependencies()[0], velocity);
  EXPECT_EQ(representation.dependencies()[1], pressure);
  EXPECT_EQ(representation.geometry_version(), 0u);
  representation.invalidate_geometry();
  EXPECT_EQ(representation.geometry_version(), 1u);

  const auto points =
    representation.locally_owned_quadrature_points(QGauss<2>(2));
  ASSERT_FALSE(points.empty());

  double velocity_basis_norm = 0.;
  double pressure_basis_norm = 0.;
  for (const auto &point : points)
    {
      ASSERT_EQ(point.dof_indices.size(), point.basis_values.size());
      for (unsigned int i = 0; i < point.basis_values.size(); ++i)
        {
          const unsigned int component =
            mixed_fe.system_to_component_index(i).first;
          if (component < 2)
            velocity_basis_norm += point.basis_values[i].norm();
          else
            pressure_basis_norm += point.basis_values[i].norm();
        }
    }

  EXPECT_GT(velocity_basis_norm, 1.e-12);
  EXPECT_NEAR(pressure_basis_norm, 0., 1.e-14);
}
