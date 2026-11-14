// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>
#include <immersx/io/lifted_particle_output.h>

#include <string>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace
{
  /** Two wall points of one representative point, with two selected modes. */
  std::vector<LiftedWallPoint<3>>
  make_points()
  {
    std::vector<LiftedWallPoint<3>> points(2);
    for (unsigned int q = 0; q < 2; ++q)
      {
        auto &point                = points[q];
        point.representative_point = Point<3>(0., 0., 1.);
        point.point  = Point<3>(q == 0 ? .5 : 0., q == 0 ? 0. : .5, 1.);
        point.weight = .25 + q;
        point.source_entity_id      = 7;
        point.representative_qpoint = 3;
        point.section_qpoint        = q;
        point.mode_values           = {1., q == 0 ? 1. : -1.};
      }
    return points;
  }

  const LiftedParticleColumn &
  column(const std::vector<LiftedParticleColumn> &columns,
         const std::string                       &name,
         const unsigned int                       occurrence = 0)
  {
    unsigned int seen = 0;
    for (const auto &item : columns)
      if (item.name == name && seen++ == occurrence)
        return item;
    ADD_FAILURE() << "Missing column " << name;
    return columns.front();
  }
} // namespace

TEST(LiftedParticleOutput, ParsesPropertyList) // NOLINT
{
  ParameterAcceptor::clear();
  LiftedParticleOutputParameters options("/Particles/");
  EXPECT_EQ(options.properties,
            (std::vector<std::string>{"lifted_value", "weight"}));
  ParameterAcceptor::initialize("", "");
  ParameterAcceptor::prm.parse_input_from_string(
    "subsection Particles\n set Properties = radius, bulk_field\nend\n");
  ParameterAcceptor::parse_all_parameters();
  EXPECT_EQ(options.properties,
            (std::vector<std::string>{"radius", "bulk_field"}));
  EXPECT_TRUE(options.selected("radius"));
  EXPECT_FALSE(options.selected("weight"));

  EXPECT_ANY_THROW(ParameterAcceptor::prm.parse_input_from_string(
    "subsection Particles\n set Properties = pressure\nend\n"));

  ParameterAcceptor::prm.parse_input_from_string(
    "subsection Particles\n set Properties =\nend\n");
  ParameterAcceptor::parse_all_parameters();
  EXPECT_FALSE(options.enabled());
  ParameterAcceptor::clear();
}

TEST(LiftedParticleOutput, BuildsSelectedColumns) // NOLINT
{
  ParameterAcceptor::clear();
  LiftedParticleOutputParameters options("/Particles/");
  options.properties = {"weight",
                        "mode_values",
                        "modal_coefficients",
                        "lifted_value",
                        "modal_data",
                        "representative_point",
                        "radius",
                        "indices"};
  LiftedParticleLabels labels;
  labels.value_name  = "multiplier";
  labels.mode_labels = {"0", "1"};
  const auto points  = make_points();
  // Point-major: (lambda_0, lambda_1) at each wall point.
  const std::vector<double> coefficients{2., 3., 2., 3.};
  const std::vector<double> data{4., 5., 4., 5.};

  const auto columns =
    make_lifted_particle_columns(options, labels, points, coefficients, data);
  // 1 + 2 + 2 + 1 + 2 + 3 + 1 + 3 scalar columns.
  ASSERT_EQ(columns.size(), 15u);

  EXPECT_DOUBLE_EQ(column(columns, "weight").values[1], 1.25);
  EXPECT_DOUBLE_EQ(column(columns, "mode_1").values[1], -1.);
  EXPECT_DOUBLE_EQ(column(columns, "multiplier_1").values[0], 3.);
  // Lifted value: 1 * 2 + 1 * 3 and 1 * 2 - 1 * 3.
  EXPECT_DOUBLE_EQ(column(columns, "multiplier").values[0], 5.);
  EXPECT_DOUBLE_EQ(column(columns, "multiplier").values[1], -1.);
  EXPECT_DOUBLE_EQ(column(columns, "data_0").values[0], 4.);
  EXPECT_DOUBLE_EQ(column(columns, "representative_point", 2).values[0], 1.);
  EXPECT_EQ(column(columns, "representative_point").interpretation,
            DataComponentInterpretation::component_is_part_of_vector);
  EXPECT_DOUBLE_EQ(column(columns, "radius").values[1], .5);
  EXPECT_DOUBLE_EQ(column(columns, "source_cell").values[0], 7.);
  EXPECT_DOUBLE_EQ(column(columns, "representative_qpoint").values[1], 3.);
  EXPECT_DOUBLE_EQ(column(columns, "section_qpoint").values[1], 1.);
  ParameterAcceptor::clear();
}

TEST(LiftedParticleOutput, RejectsMissingInputs) // NOLINT
{
  ParameterAcceptor::clear();
  LiftedParticleOutputParameters options("/Particles/");
  LiftedParticleLabels           labels;
  labels.mode_labels = {"0", "1"};
  const auto points  = make_points();

  options.properties = {"lifted_value"};
  EXPECT_ANY_THROW(make_lifted_particle_columns(options, labels, points));
  options.properties = {"modal_data"};
  EXPECT_ANY_THROW(make_lifted_particle_columns(options, labels, points));
  options.properties = {"not_a_property"};
  EXPECT_ANY_THROW(make_lifted_particle_columns(options, labels, points));
  ParameterAcceptor::clear();
}

TEST(LiftedParticleOutput, VectorModeShapes) // NOLINT
{
  ParameterAcceptor::clear();
  LiftedParticleOutputParameters options("/Particles/");
  options.properties = {"mode_values", "lifted_value"};
  LiftedParticleLabels labels;
  labels.value_name      = "multiplier";
  labels.mode_labels     = {"0", "1"};
  labels.mode_components = 3;
  auto points            = make_points();
  for (auto &point : points)
    point.mode_values = {1., 0., 0., 0., 1., 0.};
  const std::vector<double> coefficients{2., 3., 2., 3.};

  const auto columns =
    make_lifted_particle_columns(options, labels, points, coefficients);
  ASSERT_EQ(columns.size(), 9u);
  EXPECT_DOUBLE_EQ(column(columns, "mode_1", 1).values[0], 1.);
  EXPECT_DOUBLE_EQ(column(columns, "multiplier", 0).values[0], 2.);
  EXPECT_DOUBLE_EQ(column(columns, "multiplier", 1).values[1], 3.);
  EXPECT_DOUBLE_EQ(column(columns, "multiplier", 2).values[1], 0.);
  EXPECT_EQ(column(columns, "multiplier").interpretation,
            DataComponentInterpretation::component_is_part_of_vector);

  labels.mode_components = 1;
  EXPECT_ANY_THROW(
    make_lifted_particle_columns(options, labels, points, coefficients));
  ParameterAcceptor::clear();
}

TEST(LiftedParticleOutput,
     BOTH_ReusesLocationsAndUpdatesRemoteProperties) // NOLINT
{
  ParameterAcceptor::clear();
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);
  MappingQ1<2>       mapping;
  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  // Two ranks supply points when available; MPI4 keeps two empty source
  // ranks. Put the outside point in the middle to exercise gaps in ids.
  std::vector<LiftedWallPoint<2>> points;
  if (rank < 2)
    for (const auto &position : {Point<2>(.1 + .1 * rank, .2 + .1 * rank),
                                 Point<2>(2. + rank, 2. + rank),
                                 Point<2>(.8 - .1 * rank, .7 + .1 * rank)})
      {
        LiftedWallPoint<2> point;
        point.point = position;
        points.push_back(point);
      }
  std::vector<LiftedParticleColumn> columns(1);
  columns[0].name = "value";
  for (const auto &point : points)
    columns[0].values.push_back(point.point[0] + 2. * point.point[1] +
                                100. * rank);
  LiftedParticleOutputCache<2> cache;
  auto &handler = cache.update(tria, mapping, points, columns, 1);
  EXPECT_EQ(handler.n_global_particles(),
            2u * std::min(2u, Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)));
  std::map<types::particle_index, Point<2>> references;
  for (auto particle = handler.begin(); particle != handler.end(); ++particle)
    {
      const auto position    = particle->get_location();
      const auto source_rank = particle->get_id() / 3;
      EXPECT_NE(particle->get_id() % 3, 1u);
      EXPECT_NEAR(particle->get_properties()[0],
                  position[0] + 2. * position[1] + 100. * source_rank,
                  1.e-14);
      references[particle->get_id()] = particle->get_reference_location();
    }
  for (auto &value : columns[0].values)
    value += 7. * (rank + 1);
  auto &updated = cache.update(tria, mapping, points, columns, 1);
  EXPECT_EQ(&handler, &updated);
  for (auto particle = updated.begin(); particle != updated.end(); ++particle)
    {
      const auto position    = particle->get_location();
      const auto source_rank = particle->get_id() / 3;
      EXPECT_NEAR(particle->get_properties()[0],
                  position[0] + 2. * position[1] + 100. * source_rank +
                    7. * (source_rank + 1),
                  1.e-14);
      EXPECT_EQ(references.at(particle->get_id()),
                particle->get_reference_location());
    }
}
