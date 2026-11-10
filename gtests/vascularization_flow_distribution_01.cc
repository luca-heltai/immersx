// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/io.h>

#include <array>
#include <chrono>
#include <cmath>
#include <sstream>

#include "../source/vascularization/generation_pairs.h"
#include "test_paths.h"

using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  constexpr double q0 = 1.e-9, mu = .004, radius = .001, base_length = .01;
  double
  R()
  {
    return 8. * mu * base_length / (numbers::PI * std::pow(radius, 4));
  }

  void
  diagnostic(const std::string &stage, const GenerationResult &r)
  {
#ifdef DEAL_II_WITH_VTK
    static const std::string token = [] {
      std::string value;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        value = std::to_string(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
      return Utilities::MPI::broadcast(MPI_COMM_WORLD, value, 0);
    }();
#  ifdef DEBUG
    const std::string configuration = "debug";
#  else
    const std::string configuration = "release";
#  endif
    const auto test = ::testing::UnitTest::GetInstance()->current_test_info();
    write_diagnostic_vtk(
      ImmersX::TestPaths::output_directory(
        "reference-flow-distribution/" + configuration + "-mpi" +
        std::to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) +
        "/run-" + token + "/" + test->name() + "/" + stage),
      r.network,
      r.validation,
      MPI_COMM_WORLD);
#else
    (void)stage;
    (void)r;
#endif
  }

  std::string
  issues(const GenerationResult &r)
  {
    std::ostringstream text;
    for (const auto &issue : r.validation.issues)
      text << issue.entity << ": " << issue.message << '\n';
    return text.str();
  }

  void
  add_role(GenerationResult     &r,
           System                system,
           Port                  port,
           Id                    node,
           Id                    vessel,
           Id                    terminal,
           std::array<double, 2> ratios,
           double                total)
  {
    auto        &n           = r.network;
    const double y           = static_cast<unsigned int>(system);
    const double root_radius = std::cbrt(2.) * radius;
    const double trunk_length =
      base_length * std::pow(root_radius / radius, 4) / 4.;
    n.nodes[node]     = {Point<3>(0., y, 0.), 0, port, Provenance::generated};
    n.nodes[node + 1] = {Point<3>(trunk_length, y, 0.),
                         0,
                         Port::none,
                         Provenance::generated};
    n.nodes[node + 2] = {Point<3>(trunk_length + ratios[0] * base_length,
                                  y,
                                  0.),
                         0,
                         Port::none,
                         Provenance::generated};
    n.nodes[node + 3] = {Point<3>(trunk_length,
                                  y + ratios[1] * base_length,
                                  0.),
                         0,
                         Port::none,
                         Provenance::generated};
    Vessel v;
    v.nodes           = {{node, node + 1}};
    v.radii           = {{root_radius, root_radius}};
    v.viscosity       = mu;
    v.system          = system;
    v.provenance      = Provenance::generated;
    n.vessels[vessel] = v;
    for (unsigned int k = 0; k < 2; ++k)
      {
        v.nodes                   = {{node + 1, node + 2 + k}};
        v.radii                   = {{radius, radius}};
        n.vessels[vessel + 1 + k] = v;
        n.terminals[terminal + k] = {node + 2 + k, 1000 + k, system, 0, 1.};
        r.target_terminal_flows[terminal + k] = total / 2.;
      }
    r.target_attachment_flows[vessel] = total;
  }

  GenerationResult
  fixture(bool three, bool retained)
  {
    GenerationResult r;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return r;
    add_role(r,
             System::arterial,
             Port::arterial_inlet,
             0,
             10,
             100,
             {{.5, 1.5}},
             3. * q0);
    add_role(r,
             System::venous,
             Port::venous_outlet,
             10,
             20,
             200,
             three ? std::array<double, 2>{{.5, .5}} :
                     std::array<double, 2>{{.5, 1.5}},
             (three ? 6. : 3.) * q0);
    if (three)
      add_role(r,
               System::portal,
               Port::portal_inlet,
               20,
               30,
               300,
               {{1.5, .5}},
               3. * q0);
    for (unsigned int k = 0; k < 2; ++k)
      r.territories.push_back(
        {1000 + k, r.network.nodes.at(2 + k).position, 0, 1.});
    if (retained)
      {
        for (unsigned int k = 0; k < 2; ++k)
          {
            const Id n             = 500 + 10 * k;
            r.network.nodes[n]     = {Point<3>(0., 4. + k, 0.),
                                      0,
                                  k == 0 ? Port::arterial_inlet :
                                               Port::venous_outlet};
            r.network.nodes[n + 1] = {Point<3>(base_length, 4. + k, 0.), 0};
            Vessel v;
            v.nodes              = {{n, n + 1}};
            v.radii              = {{radius, radius}};
            v.viscosity          = mu;
            v.system             = k == 0 ? System::arterial : System::venous;
            r.network.vessels[n] = v;
            r.network.terminals[400 + k] = {n + 1, 9000, v.system, 0, 999.};
          }
      }
    return r;
  }

  std::vector<TerminalPair>
  retained_beds(bool retained)
  {
    if (retained && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      return {{9000, 400, 401, R()}};
    return {};
  }

  void
  controls(Parameters &p, bool three, bool retained)
  {
    p.venous.enabled       = true;
    p.portal.enabled       = three;
    p.reference_flow       = (three ? 6. : 3.) * q0;
    p.portal_flow_fraction = .5;
    if (three)
      p.compartment_resistances = {{R() / 2., R() / 2., R() / 2.}};
    else
      p.venous.microvascular_resistance = R();
    // Exercise boundary data supplied by a rank with no network entities.
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) + 1 ==
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      {
        p.reference_conditions.inflows[0]    = 3. * q0;
        p.reference_conditions.pressures[10] = 600.;
        if (three)
          p.reference_conditions.inflows[20] = 3. * q0;
        if (retained)
          {
            p.reference_conditions.pressures[500] = 620.;
            p.reference_conditions.pressures[510] = 600.;
          }
      }
  }

  double
  owned_value(const std::map<Id, double> &values, Id id)
  {
    const auto found = values.find(id);
    const bool present =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
      found != values.end();
    EXPECT_EQ(Utilities::MPI::sum(present ? 1u : 0u, MPI_COMM_WORLD), 1u);
    const double local = present ? found->second : 0.;
    return Utilities::MPI::sum(local, MPI_COMM_WORLD);
  }

  void
  expect_reference(const GenerationResult &r)
  {
    EXPECT_TRUE(r.reference.validation.valid());
    EXPECT_LT(r.reference.relative_flow_defect, 1.e-9);
    EXPECT_NEAR(r.reference.boundary_flow_balance / q0, 0., 1.e-8);
    for (const auto &[id, v] : r.reference.vessels)
      {
        EXPECT_TRUE(std::isfinite(v.resistance));
        EXPECT_GT(v.resistance, 0.);
      }
  }
} // namespace

TEST(VascularReferenceFlow, BOTH_ResistanceDistributionThreeRolesKeepsNewTotals)
{
  Parameters p("/Resistance distribution three roles/");
  controls(p, true, true);
  p.reference_flow_distribution = "network resistance";
  auto       r                  = fixture(true, true);
  const auto terminal_design    = r.target_terminal_flows;
  const auto attachment_design  = r.target_attachment_flows;
  diagnostic("input", r);
  ImmersX::Vascularization::internal::pair_territories(r,
                                                       p,
                                                       MPI_COMM_WORLD,
                                                       retained_beds(true));
  diagnostic("resistance-reference", r);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  expect_reference(r);
  EXPECT_EQ(r.target_terminal_flows, terminal_design);
  EXPECT_EQ(r.target_attachment_flows, attachment_design);
  const std::array<double, 2> a{{2., 1.}}, pv{{1., 2.}};
  for (unsigned int k = 0; k < 2; ++k)
    {
      const double qa =
        owned_value(r.reference.terminal_outflows, 100 + k) / q0;
      const double qp =
        owned_value(r.reference.terminal_outflows, 300 + k) / q0;
      const double qv =
        owned_value(r.reference.terminal_outflows, 200 + k) / q0;
      EXPECT_NEAR(qa, a[k], 1.e-8);
      EXPECT_NEAR(qp, pv[k], 1.e-8);
      EXPECT_NEAR(qv, -3., 1.e-8);
      EXPECT_NEAR(qa + qp + qv, 0., 1.e-8);
      EXPECT_NEAR(qp / (qa + qp), (k == 0 ? 1. : 2.) / 3., 1.e-8);
    }
  EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 0) / q0, 3., 1.e-8);
  EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 20) / q0, 3., 1.e-8);
  EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 10) / q0, -6., 1.e-8);
  EXPECT_NEAR(owned_value(r.reference.pressures, 0),
              600. + 7.25 * R() * q0,
              1.e-8);
  EXPECT_NEAR(owned_value(r.reference.pressures, 20),
              600. + 7.25 * R() * q0,
              1.e-8);
  EXPECT_NEAR(owned_value(r.reference.terminal_outflows, 400),
              20. / (3. * R()),
              1.e-16);
  EXPECT_NEAR(owned_value(r.reference.terminal_outflows, 401),
              -20. / (3. * R()),
              1.e-16);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  r.terminal_pairs.size()),
                                MPI_COMM_WORLD),
            1u);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  r.terminal_triplets.size()),
                                MPI_COMM_WORLD),
            2u);
  for (const auto &bed : r.terminal_triplets)
    for (double arm : bed.resistances)
      EXPECT_DOUBLE_EQ(arm, R() / 2.);
  std::map<Id, double> pc;
  for (const auto &[id, bed] : r.reference.territories)
    pc[id] = bed.compartment_pressure;
  for (Id id : {1000, 1001})
    EXPECT_NEAR(owned_value(pc, id), 600. + 4.5 * R() * q0, 1.e-8);
}

TEST(VascularReferenceFlow, BOTH_ResistanceDistributionArteriovenousNetwork)
{
  Parameters p("/Resistance distribution two roles/");
  controls(p, false, false);
  p.reference_flow_distribution = "network resistance";
  auto       r                  = fixture(false, false);
  const auto design             = r.target_terminal_flows;
  diagnostic("input", r);
  ImmersX::Vascularization::internal::pair_territories(r, p, MPI_COMM_WORLD);
  diagnostic("resistance-reference", r);
  ASSERT_TRUE(r.validation.valid()) << issues(r);
  expect_reference(r);
  EXPECT_EQ(r.target_terminal_flows, design);
  for (unsigned int k = 0; k < 2; ++k)
    {
      const double qa =
        owned_value(r.reference.terminal_outflows, 100 + k) / q0;
      const double qv =
        owned_value(r.reference.terminal_outflows, 200 + k) / q0;
      EXPECT_NEAR(qa, k == 0 ? 2. : 1., 1.e-8);
      EXPECT_NEAR(qa + qv, 0., 1.e-8);
    }
  EXPECT_NEAR(owned_value(r.reference.pressures, 0),
              600. + 5.5 * R() * q0,
              1.e-8);
  EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 0) / q0, 3., 1.e-8);
  EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 10) / q0, -3., 1.e-8);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(
                                  r.terminal_pairs.size()),
                                MPI_COMM_WORLD),
            2u);
  for (const auto &bed : r.terminal_pairs)
    EXPECT_DOUBLE_EQ(bed.resistance, R());
}

TEST(VascularReferenceFlow,
     BOTH_ResistanceDistributionKeepsLegacyAndTotalRejections)
{
  Parameters p("/Resistance distribution rejections/");
  controls(p, true, false);
  EXPECT_EQ(p.reference_flow_distribution, "tissue demand");
  auto legacy = fixture(true, false);
  diagnostic("legacy-input", legacy);
  ImmersX::Vascularization::internal::pair_territories(legacy,
                                                       p,
                                                       MPI_COMM_WORLD);
  diagnostic("legacy-rejection", legacy);
  EXPECT_TRUE(legacy.reference.validation.valid());
  EXPECT_FALSE(legacy.validation.valid());
  EXPECT_EQ(Utilities::MPI::max(static_cast<unsigned int>(
                                  issues(legacy).find(
                                    "per-system demand flow") !=
                                  std::string::npos),
                                MPI_COMM_WORLD),
            1u);
  p.reference_flow_distribution = "network resistance";
  p.reference_flow *= 2.;
  auto mismatch = fixture(true, false);
  diagnostic("global-total-input", mismatch);
  ImmersX::Vascularization::internal::pair_territories(mismatch,
                                                       p,
                                                       MPI_COMM_WORLD);
  diagnostic("global-total-rejection", mismatch);
  EXPECT_TRUE(mismatch.reference.validation.valid());
  EXPECT_FALSE(mismatch.validation.valid());
  EXPECT_NE(issues(mismatch).find("per-system total"), std::string::npos);
}

TEST(VascularReferenceFlow, BOTH_ReferenceFlowDistributionParameters)
{
  Parameters p("/Resistance distribution validation/");
  controls(p, true, false);
  p.reference_flow_distribution = "network resistance";
  EXPECT_NO_THROW(p.validate(false));
  p.compartment_resistances = {{0., 0., 0.}};
  EXPECT_THROW(p.validate(false), ExceptionBase);
  p.reference_flow_distribution = "tissue demand";
  EXPECT_NO_THROW(p.validate(false));
  p.reference_flow_distribution = "not a distribution";
  EXPECT_THROW(p.validate(false), ExceptionBase);
  p.reference_flow_distribution = "network resistance";
  p.compartment_resistances     = {{R() / 2., R() / 2., R() / 2.}};
  p.reference_flow              = 0.;
  EXPECT_THROW(p.validate(false), ExceptionBase);
  p.reference_flow                  = 3. * q0;
  p.portal.enabled                  = false;
  p.venous.enabled                  = false;
  p.venous.microvascular_resistance = R();
  EXPECT_THROW(p.validate(false), ExceptionBase);
}

TEST(VascularReferenceFlow, MPI_ReferenceFlowDistributionControlsMustAgree)
{
  ASSERT_GT(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  Parameters p("/Resistance distribution collective validation/");
  controls(p, true, false);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    p.reference_flow_distribution = "network resistance";
  auto r = fixture(true, false);
  diagnostic("asymmetric-controls-input", r);
  ImmersX::Vascularization::internal::pair_territories(r, p, MPI_COMM_WORLD);
  diagnostic("asymmetric-controls-rejection", r);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.reference.vessels.empty());
}

namespace
{
  GenerationResult
  unequal_fixture()
  {
    auto r = fixture(true, true);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return r;
    const std::array<Id, 3> branches{{11, 21, 31}}, terminals{{100, 200, 300}};
    const std::array<std::array<double, 2>, 3> ratios{
      {{{.25, 1.625}}, {{.25, .625}}, {{1.25, .625}}}};
    for (unsigned int role = 0; role < 3; ++role)
      for (unsigned int k = 0; k < 2; ++k)
        {
          const double demand       = k + 1.;
          const double child_radius = radius * std::cbrt(2. * demand / 3.);
          auto        &v            = r.network.vessels.at(branches[role] + k);
          v.radii                   = {{child_radius, child_radius}};
          auto tip                  = r.network.nodes.at(v.nodes[0]).position;
          tip[k] +=
            base_length * ratios[role][k] * std::pow(child_radius / radius, 4);
          r.network.nodes.at(v.nodes[1]).position            = tip;
          r.network.terminals.at(terminals[role] + k).demand = demand;
          r.target_terminal_flows.at(terminals[role] + k) =
            (role == 1 ? 6. : 3.) * q0 * demand / 3.;
        }
    for (unsigned int k = 0; k < 2; ++k)
      {
        r.territories[k].position = r.network.nodes.at(2 + k).position;
        r.territories[k].demand   = k + 1.;
      }
    return r;
  }
} // namespace

TEST(VascularReferenceFlow, BOTH_CompartmentDemandScalingUsesOnlyNewTerritories)
{
  Parameters p("/Resistance distribution demand scaling/");
  controls(p, true, true);
  p.reference_flow_distribution = "network resistance";
  EXPECT_FALSE(p.scale_compartment_resistances_by_demand);
  for (const bool scaled : {false, true})
    {
      p.scale_compartment_resistances_by_demand = scaled;
      auto              r                       = unequal_fixture();
      const auto        design                  = r.target_terminal_flows;
      const auto        attachments             = r.target_attachment_flows;
      const std::string stage = scaled ? "scaled-arms" : "uniform-arms";
      diagnostic(stage + "-input", r);
      ImmersX::Vascularization::internal::pair_territories(r,
                                                           p,
                                                           MPI_COMM_WORLD,
                                                           retained_beds(true));
      diagnostic(stage + "-reference", r);
      ASSERT_TRUE(r.validation.valid()) << issues(r);
      expect_reference(r);
      EXPECT_EQ(r.target_terminal_flows, design);
      EXPECT_EQ(r.target_attachment_flows, attachments);
      // New demands 1 and 2 have mean 1.5; retained demand 999 is excluded.
      // Scaled branch+arm resistances are (R,2R,R) and (2R,R,R).
      // Uniform-arm values solve the four rational node-balance equations.
      const std::array<std::array<double, 3>, 2> expected =
        scaled ? std::array<std::array<double, 3>, 2>{{{{2., 1., 3.}},
                                                       {{1., 2., 3.}}}} :
                 std::array<std::array<double, 3>, 2>{
                   {{{2775. / 1219., 1503. / 1219., 186. / 53.}},
                    {{882. / 1219., 2154. / 1219., 132. / 53.}}}};
      for (unsigned int k = 0; k < 2; ++k)
        {
          const std::array<double, 3> actual{
            {owned_value(r.reference.terminal_outflows, 100 + k) / q0,
             owned_value(r.reference.terminal_outflows, 300 + k) / q0,
             -owned_value(r.reference.terminal_outflows, 200 + k) / q0}};
          for (unsigned int arm = 0; arm < 3; ++arm)
            EXPECT_NEAR(actual[arm], expected[k][arm], 1.e-8);
          EXPECT_NEAR(actual[0] + actual[1] - actual[2], 0., 1.e-8);
          EXPECT_NEAR(owned_value(r.target_terminal_flows, 100 + k) / q0,
                      k + 1.,
                      1.e-14);
        }
      EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 0) / q0, 3., 1.e-8);
      EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 20) / q0,
                  3.,
                  1.e-8);
      EXPECT_NEAR(owned_value(r.reference.boundary_inflows, 10) / q0,
                  -6.,
                  1.e-8);
      for (const auto &bed : r.terminal_triplets)
        {
          const double factor =
            scaled ? (bed.territory == 1000 ? .75 : .375) : .5;
          for (double arm : bed.resistances)
            EXPECT_NEAR(arm / R(), factor, 1.e-13);
        }
      for (const auto &bed : r.terminal_pairs)
        EXPECT_DOUBLE_EQ(bed.resistance, R());
      EXPECT_NEAR(owned_value(r.reference.terminal_outflows, 400),
                  20. / (3. * R()),
                  1.e-16);
    }
}

TEST(VascularReferenceFlow, BOTH_CompartmentDemandScalingRequiresResistanceMode)
{
  Parameters p("/Resistance distribution scaling validation/");
  controls(p, true, false);
  p.reference_flow_distribution             = "network resistance";
  p.scale_compartment_resistances_by_demand = true;
  EXPECT_NO_THROW(p.validate(false));
  p.reference_flow_distribution = "tissue demand";
  EXPECT_THROW(p.validate(false), ExceptionBase);
  p.reference_flow_distribution     = "network resistance";
  p.portal.enabled                  = false;
  p.venous.microvascular_resistance = R();
  EXPECT_THROW(p.validate(false), ExceptionBase);
}

TEST(VascularReferenceFlow, MPI_CompartmentDemandScalingControlsMustAgree)
{
  ASSERT_GT(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1u);
  Parameters p("/Resistance distribution collective scaling/");
  controls(p, true, false);
  p.reference_flow_distribution = "network resistance";
  p.scale_compartment_resistances_by_demand =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  auto r = fixture(true, false);
  diagnostic("asymmetric-scaling-input", r);
  ImmersX::Vascularization::internal::pair_territories(r, p, MPI_COMM_WORLD);
  diagnostic("asymmetric-scaling-rejection", r);
  EXPECT_FALSE(r.validation.valid());
  EXPECT_TRUE(r.reference.vessels.empty());
}
