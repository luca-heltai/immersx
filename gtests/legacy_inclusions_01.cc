#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numeric>

#include "legacy_inclusions.h"
#include "reference_cross_section.h"
#include "tensor_product_space.h"

using namespace dealii;

TEST(LegacyInclusions, PointCloud2DUsesCanonicalPropertiesAndThickness)
{
  TensorProductSpaceParameters<0, 1, 2, 1> params;
  params.inclusions_file =
    SOURCE_DIR "/gtests/fixtures/legacy_inclusions_2d.txt";
  params.data_file = SOURCE_DIR "/gtests/fixtures/legacy_inclusions_2d.data";
  params.legacy_n_coefficients    = 3;
  params.input_file_fields        = "coefficient_0,coefficient_1,coefficient_2";
  params.section.refinement_level = 5;

  TensorProductSpace<0, 1, 2, 1> space(params);
  ASSERT_NO_THROW(space.initialize());

  ASSERT_EQ(space.n_representative_entities(), 2u);
  ASSERT_EQ(space.n_representative_dofs_per_entity(), 3u);
  ASSERT_EQ(space.get_properties_bindings().size(), 4u);
  EXPECT_EQ(space.get_properties_catalog()[0].association,
            FieldAssociation::point_data);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(0), 0.25);
  EXPECT_DOUBLE_EQ(space.get_entity_thickness(1), 0.5);

  const auto first = space.get_entity_property_values(0);
  ASSERT_EQ(first.size(), 4u);
  EXPECT_DOUBLE_EQ(first[0], 1.0);
  EXPECT_DOUBLE_EQ(first[1], 2.0 / std::sqrt(2.));
  EXPECT_DOUBLE_EQ(first[2], 3.0 / std::sqrt(2.));
  EXPECT_DOUBLE_EQ(first[3], 0.25);

  double local_measure = 0.;
  for (const auto &weight : space.get_locally_owned_weights())
    local_measure += weight[0];
  EXPECT_NEAR(local_measure, 2. * numbers::PI * (0.25 + 0.5), 1e-3);
}

TEST(LegacyInclusions, Vector2DInstantiationUsesSelectedReferenceBasis)
{
  TensorProductSpaceParameters<0, 1, 2, 2> params;
  params.inclusions_file =
    SOURCE_DIR "/gtests/fixtures/legacy_inclusions_2d.txt";
  params.legacy_n_coefficients           = 1;
  params.legacy_reference_inclusion_data = {1., 2.};
  params.legacy_selected_coefficients    = {0, 1};
  params.input_file_fields               = "coefficient_0,coefficient_1";

  TensorProductSpace<0, 1, 2, 2> space(params);
  ASSERT_NO_THROW(space.initialize());
  EXPECT_EQ(space.n_representative_entities(), 2u);
  EXPECT_EQ(space.n_representative_dofs(), 4u);
  EXPECT_EQ(space.n_representative_dofs_per_entity(), 2u);
}

TEST(LegacyInclusions, Segment3DReconstructsCellsAndProperties)
{
  TensorProductSpaceParameters<1, 2, 3, 3> params;
  params.inclusions_file =
    SOURCE_DIR "/gtests/fixtures/legacy_inclusions_3d.txt";
  params.data_file =
    SOURCE_DIR "/gtests/fixtures/legacy_inclusions_3d_vector.data";
  params.legacy_n_coefficients = 1;
  params.input_file_fields =
    "coefficient_0,coefficient_1,coefficient_2,vessel_id";
  params.section.refinement_level      = 5;
  params.section.selected_coefficients = {0, 1, 2};

  TensorProductSpace<1, 2, 3, 3> space(params);
  ASSERT_NO_THROW(space.initialize());

  const auto &tria = space.get_triangulation();
  EXPECT_EQ(tria.n_global_active_cells(), 3u);
  EXPECT_EQ(tria.n_vertices(), 5u);
  ASSERT_EQ(space.get_properties_catalog().size(), 5u);
  EXPECT_EQ(space.get_properties_catalog()[0].name, "radius");
  EXPECT_EQ(space.get_properties_catalog()[1].name, "vessel_id");
  EXPECT_EQ(space.get_properties_catalog()[2].name, "coefficient_0");
  EXPECT_EQ(space.get_properties_bindings().size(), 5u);

  std::vector<types::global_dof_index> dof_indices(
    space.get_properties_dh().get_fe().n_dofs_per_cell());
  for (const auto &cell : space.get_properties_dh().active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell->get_dof_indices(dof_indices);
        const auto                 &values  = space.get_properties();
        const auto                  record  = cell->global_active_cell_index();
        const std::array<double, 3> radii   = {0.25, 0.5, 0.75};
        const std::array<double, 3> vessels = {0., 0., 1.};
        const std::array<double, 3> coefficient_0 = {1., 4., 7.};
        const std::array<double, 3> coefficient_1 = {2., 5., 8.};
        const std::array<double, 3> coefficient_2 = {3., 6., 9.};
        EXPECT_DOUBLE_EQ(values[dof_indices[0]], radii[record]);
        EXPECT_DOUBLE_EQ(values[dof_indices[1]], vessels[record]);
        EXPECT_DOUBLE_EQ(values[dof_indices[2]], coefficient_0[record]);
        EXPECT_DOUBLE_EQ(values[dof_indices[3]], coefficient_1[record]);
        EXPECT_DOUBLE_EQ(values[dof_indices[4]], coefficient_2[record]);
      }
}

void
check_segment_3d_measure()
{
  TensorProductSpaceParameters<1, 2, 3, 1> params;
  params.inclusions_file =
    SOURCE_DIR "/gtests/fixtures/legacy_inclusions_3d.txt";
  params.legacy_n_coefficients         = 1;
  params.section.refinement_level      = 5;
  params.section.selected_coefficients = {0};

  TensorProductSpace<1, 2, 3, 1> space(params);
  ASSERT_NO_THROW(space.initialize());

  double local_measure = 0.;
  for (const auto &weight : space.get_locally_owned_weights())
    local_measure += weight[0];
  const double global_measure =
    Utilities::MPI::sum(local_measure, MPI_COMM_WORLD);
  const double expected_measure = (0.25 + 0.50 + 0.75) * 2. * numbers::PI;
  EXPECT_NEAR(global_measure, expected_measure, 1e-2);
}

TEST(LegacyInclusions, Segment3DMeasureUsesCellJacobian)
{
  check_segment_3d_measure();
}

TEST(LegacyInclusions, MPI_Segment3DMeasureUsesCellJacobian)
{
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
    return;
  check_segment_3d_measure();
}

TEST(LegacyInclusions, FourierNormalizationIsExplicit)
{
  EXPECT_DOUBLE_EQ(LegacyInclusions::fourier_to_reference_scale(0), 1.);
  EXPECT_DOUBLE_EQ(LegacyInclusions::fourier_to_reference_scale(1),
                   1. / std::sqrt(2.));
  EXPECT_DOUBLE_EQ(LegacyInclusions::fourier_to_reference_scale(2),
                   1. / std::sqrt(2.));
}

TEST(LegacyInclusions, ReferenceBasisFollowsFourierOrdering)
{
  ReferenceCrossSectionParameters<1, 2, 1> params;
  params.inclusion_degree = 2;
  params.refinement_level = 5;
  const auto canonical_indices =
    LegacyInclusions::fourier_to_reference_indices(5, 1, 2, 2);
  ASSERT_EQ(canonical_indices, std::vector<unsigned int>({0, 1, 2, 4, 3}));
  params.selected_coefficients = canonical_indices;
  ReferenceCrossSection<1, 2, 1> reference(params);

  const auto &quadrature = reference.get_global_quadrature();
  const auto  expected   = [](const unsigned int mode, const Point<2> &point) {
    if (mode == 0)
      return 1.;
    const auto frequency = (mode + 1) / 2;
    const auto angle     = std::atan2(point[1], point[0]);
    return std::sqrt(2.) * (mode % 2 == 1 ? std::cos(frequency * angle) :
                                               std::sin(frequency * angle));
  };

  for (unsigned int basis = 0; basis < 5; ++basis)
    for (unsigned int mode = 0; mode < 5; ++mode)
      {
        double correlation = 0.;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          correlation += reference.shape_value(basis, q, 0) *
                         expected(mode, quadrature.point(q)) *
                         quadrature.weight(q);

        if (basis == mode)
          EXPECT_GT(correlation, 0.95 * reference.measure());
        else
          EXPECT_LT(std::abs(correlation), 0.15 * reference.measure());
      }
}
