// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/network.h>

#include <cmath>
#include <limits>

using namespace ImmersX::Vascularization;

namespace
{
  NetworkData
  straight_vessel()
  {
    NetworkData network;
    network.nodes[10].position = dealii::Point<3>(0., 0., 0.);
    network.nodes[20].position = dealii::Point<3>(.03, 0., 0.);
    auto &vessel               = network.vessels[70];
    vessel.nodes               = {{10, 20}};
    vessel.radii               = {{.001, .002}};
    vessel.viscosity           = .004;
    return network;
  }
} // namespace

TEST(VascularNetworkData, CylinderAndTaperAnalyticalIntegrals)
{
  auto         network = straight_vessel();
  const double pi      = dealii::numbers::PI;
  EXPECT_NEAR(volume(network, 70) / (pi * .03 * 7.e-6 / 3.), 1., 1.e-14);
  EXPECT_NEAR(resistance(network, 70) /
                (8. * .004 / pi * .03 * 7. / 24. * 1.e12),
              1.,
              1.e-14);
  network.vessels.at(70).radii[1] = .001;
  EXPECT_NEAR(volume(network, 70) / (pi * .03 * 1.e-6), 1., 1.e-14);
  EXPECT_NEAR(resistance(network, 70) / (8. * .004 * .03 / (pi * 1.e-12)),
              1.,
              1.e-14);
}

TEST(VascularNetworkData, IndependentQuadratureForTaper)
{
  const auto network = straight_vessel();
  // Composite midpoint quadrature is independent of the analytical formula.
  constexpr unsigned int samples = 20000;
  double                 v       = 0.;
  double                 r       = 0.;
  for (unsigned int i = 0; i < samples; ++i)
    {
      const double radius = .001 + .001 * (i + .5) / samples;
      v += dealii::numbers::PI * radius * radius * .03 / samples;
      r +=
        8. * .004 / dealii::numbers::PI * std::pow(radius, -4.) * .03 / samples;
    }
  EXPECT_NEAR(volume(network, 70) / v, 1., 1.e-8);
  EXPECT_NEAR(resistance(network, 70) / r, 1., 1.e-8);
}

TEST(VascularNetworkData, SubdivisionPreservesPhysicsAndTopology)
{
  const auto network = straight_vessel();
  const auto cells   = subdivide(network, 70, {0., .001, .05, .25, .9, 1.});
  double     v       = 0.;
  double     r       = 0.;
  for (const auto &cell : cells)
    {
      ASSERT_EQ(cell.vessel, 70u);
      v += volume(network, cell.vessel, cell.begin, cell.end);
      r += resistance(network, cell.vessel, cell.begin, cell.end);
    }
  EXPECT_NEAR(v / volume(network, 70), 1., 1.e-14);
  EXPECT_NEAR(r / resistance(network, 70), 1., 1.e-14);
  EXPECT_EQ(network.vessels.size(), 1u);
  EXPECT_EQ(network.nodes.size(), 2u);
  EXPECT_EQ(network.vessels.at(70).nodes, (std::array<Id, 2>{{10, 20}}));
  EXPECT_EQ(network.vessels.at(70).provenance, Provenance::imported);
  EXPECT_DOUBLE_EQ(network.vessels.at(70).radii[0], .001);
  EXPECT_DOUBLE_EQ(network.vessels.at(70).radii[1], .002);
}

TEST(VascularNetworkData, GeneralizedMurrayAnalyticalBifurcations)
{
  EXPECT_NEAR(murray_radius({3., 4., 5.}, 3.), 6., 1.e-14);
  EXPECT_NEAR(murray_radius({3., 4.}, 2.), 5., 1.e-14);
  EXPECT_NEAR(murray_radius({.001, .001}, 3.) / .001, std::cbrt(2.), 1.e-14);
  EXPECT_THROW(murray_radius({}, 3.), dealii::ExceptionBase);
  EXPECT_THROW(murray_radius({0., 1.}, 3.), dealii::ExceptionBase);
  EXPECT_THROW(murray_radius({1., 1.}, 0.), dealii::ExceptionBase);
}

TEST(VascularNetworkData, GhostDataNeedsOnlyRelevantTopology)
{
  auto network                 = straight_vessel();
  network.vessels.at(70).owner = 1;
  network.nodes.at(20).owner   = 1;
  EXPECT_TRUE(validate_data(network, 2).valid());
  EXPECT_FALSE(validate_data(network, 1).valid());
  network.nodes.erase(20);
  const auto report = validate_data(network, 2);
  ASSERT_EQ(report.issues.size(), 1u);
  EXPECT_EQ(report.issues.front().entity, "vessel");
  EXPECT_EQ(report.issues.front().id, 70u);
}

TEST(VascularNetworkData, RejectsInvalidPhysicalDataAndIntervals)
{
  auto network = straight_vessel();
  EXPECT_TRUE(validate_data(network, 1).valid());
  network.vessels.at(70).viscosity = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(validate_data(network, 1).valid());
  EXPECT_THROW(resistance(network, 70), dealii::ExceptionBase);
  network.vessels.at(70).viscosity = .004;
  network.vessels.at(70).radii[0]  = -.001;
  EXPECT_FALSE(validate_data(network, 1).valid());
  EXPECT_THROW(volume(network, 70), dealii::ExceptionBase);
  network.vessels.at(70).radii[0] = .001;
  EXPECT_THROW(volume(network, 70, -.01, 1.), dealii::ExceptionBase);
  EXPECT_THROW(resistance(network, 70, .8, .2), dealii::ExceptionBase);
  EXPECT_THROW(subdivide(network, 70, {0., .5, .5, 1.}), dealii::ExceptionBase);
  EXPECT_THROW(subdivide(network, 70, {0., .5}), dealii::ExceptionBase);
  network.nodes.at(20).position = network.nodes.at(10).position;
  EXPECT_FALSE(validate_data(network, 1).valid());
}

TEST(VascularNetworkData, ExplicitSystemAndPortRoles)
{
  auto network                    = straight_vessel();
  network.vessels.at(70).system   = System::venous;
  network.nodes.at(10).port       = Port::venous_outlet;
  network.terminals[90].node      = 20;
  network.terminals.at(90).system = System::venous;
  EXPECT_TRUE(validate_data(network, 1).valid());
  network.terminals.at(90).node = 10;
  EXPECT_FALSE(validate_data(network, 1).valid());
}
