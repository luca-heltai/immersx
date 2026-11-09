// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>

#include <gtest/gtest.h>
#include <immersx/vascularization/global_geometry.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../source/vascularization/global_geometry_internal.h"
#include "test_paths.h"

using namespace dealii;
using namespace ImmersX::Vascularization;

namespace
{
  std::filesystem::path
  case_directory(MPI_Comm comm)
  {
#ifdef DEBUG
    const std::string configuration = "debug";
#else
    const std::string configuration = "release";
#endif
    return ImmersX::TestPaths::output_path(
      "global-forest-geometry/" + configuration + "-mpi" +
      std::to_string(Utilities::MPI::n_mpi_processes(comm)) + "/" +
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
  }
  void
  export_background(const Triangulation<3> &bg, MPI_Comm comm)
  {
    const auto directory = case_directory(comm);
    std::filesystem::create_directories(directory);
    GridOut().write_mesh_per_processor_as_vtu(
      bg, (directory / "background").string());
  }
  void
  export_network(const NetworkData          &network,
                 const std::map<Id, double> &flows,
                 const std::string          &stage,
                 MPI_Comm                    comm)
  {
    const auto me        = Utilities::MPI::this_mpi_process(comm),
               processes = Utilities::MPI::n_mpi_processes(comm);
    const auto directory = case_directory(comm) / stage;
    std::filesystem::create_directories(directory);
    const std::string part = "-rank" + std::to_string(me);
    std::ofstream     nodes(directory / ("nodes" + part + ".csv")),
      vessels(directory / ("vessels" + part + ".csv")),
      terminals(directory / ("terminals" + part + ".csv"));
    nodes << std::setprecision(17) << "id,x,y,z,port,provenance,owner\n";
    for (const auto &[id, n] : network.nodes)
      if (n.owner == me)
        nodes << id << "," << n.position[0] << "," << n.position[1] << ","
              << n.position[2] << "," << static_cast<unsigned int>(n.port)
              << "," << static_cast<unsigned int>(n.provenance) << ","
              << n.owner << "\n";
    vessels
      << std::setprecision(17)
      << "id,node0,node1,radius0,radius1,viscosity,system,provenance,owner\n";
    for (const auto &[id, v] : network.vessels)
      if (v.owner == me)
        vessels << id << "," << v.nodes[0] << "," << v.nodes[1] << ","
                << v.radii[0] << "," << v.radii[1] << "," << v.viscosity << ","
                << static_cast<unsigned int>(v.system) << ","
                << static_cast<unsigned int>(v.provenance) << "," << v.owner
                << "\n";
    terminals << std::setprecision(17)
              << "id,node,territory,system,demand,owner\n";
    for (const auto &[id, t] : network.terminals)
      if (t.owner == me)
        terminals << id << "," << t.node << "," << t.territory << ","
                  << static_cast<unsigned int>(t.system) << "," << t.demand
                  << "," << t.owner << "\n";
    nodes.close();
    vessels.close();
    terminals.close();
    bool valid = nodes.good() && vessels.good() && terminals.good();
    for (const auto system : {System::arterial, System::portal, System::venous})
      {
        const std::string role = system == System::arterial ? "arterial" :
                                 system == System::portal   ? "portal" :
                                                              "venous";
        std::vector<Id>   edges;
        for (const auto &[id, v] : network.vessels)
          if (v.owner == me && v.system == system)
            edges.push_back(id);
        std::ofstream out(directory / (role + part + ".vtk"));
        out << std::setprecision(17)
            << "# vtk DataFile Version 3.0\nGlobal forest geometry " << stage
            << " " << role
            << (edges.empty() ? " (no owned vessels of this role)" : "")
            << "\nASCII\nDATASET POLYDATA\nPOINTS " << 2 * edges.size()
            << " double\n";
        for (auto id : edges)
          for (auto node : network.vessels.at(id).nodes)
            {
              const auto &p = network.nodes.at(node).position;
              out << p[0] << " " << p[1] << " " << p[2] << "\n";
            }
        out << "LINES " << edges.size() << " " << 3 * edges.size() << "\n";
        for (std::size_t i = 0; i < edges.size(); ++i)
          out << "2 " << 2 * i << " " << 2 * i + 1 << "\n";
        out << "POINT_DATA " << 2 * edges.size()
            << "\nSCALARS radius_m double 1\nLOOKUP_TABLE default\n";
        for (auto id : edges)
          for (auto r : network.vessels.at(id).radii)
            out << r << "\n";
        out << "CELL_DATA " << edges.size()
            << "\nSCALARS vessel_id unsigned_long 1\nLOOKUP_TABLE default\n";
        for (auto id : edges)
          out << id << "\n";
        out << "SCALARS growth_flow_m3_s double 1\nLOOKUP_TABLE default\n";
        for (auto id : edges)
          out << (flows.count(id) ? flows.at(id) : 0.) << "\n";
        out.close();
        valid = valid && out.good();
      }
    const bool all_valid =
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1;
    EXPECT_TRUE(all_valid);
    if (me == 0 && all_valid)
      {
        const auto    temporary = directory / "stage.json.tmp";
        std::ofstream out(temporary);
        out
          << "{\n  \"stage\": \"" << stage << "\",\n  \"geometry_status\": \""
          << (stage == "after-certified" ? "certified" :
              stage == "before-draft"    ? "draft" :
                                           "invalid")
          << "\",\n  \"adoptable\": "
          << (stage == "after-certified" ? "true" : "false")
          << ",\n  \"ranks\": " << processes
          << ",\n  \"units\": {\"coordinates\": \"m\", \"radius\": \"m\", \"viscosity\": \"Pa s\", \"flow\": \"m^3/s\"}\n}\n";
        out.close();
        std::filesystem::rename(temporary, directory / "stage.json");
      }
    ASSERT_EQ(MPI_Barrier(comm), MPI_SUCCESS);
  }
  void
  record_result(const GlobalGeometryResult &result, MPI_Comm comm)
  {
    if (Utilities::MPI::this_mpi_process(comm) != 0)
      return;
    const auto   &r = result.report;
    std::ofstream out(case_directory(comm) / "case.json");
    out << std::setprecision(17)
        << "{\n  \"certified\": " << (r.certified ? "true" : "false")
        << ",\n  \"validation_valid\": "
        << (result.validation.valid() ? "true" : "false")
        << ",\n  \"backend_reason\": " << r.backend_reason
        << ",\n  \"outer_iterations\": " << r.outer_iterations
        << ",\n  \"inner_iterations\": " << r.inner_iterations
        << ",\n  \"inserted_nodes\": " << r.inserted_nodes
        << ",\n  \"collision_constraints\": " << r.collision_constraints
        << ",\n  \"boundary_constraints\": " << r.boundary_constraints
        << ",\n  \"objective_W\": " << r.objective_W
        << ",\n  \"maximum_constraint_violation\": "
        << r.maximum_constraint_violation << "\n}\n";
    std::cout << "GlobalGeometryResult certified=" << r.certified
              << " reason=" << r.backend_reason
              << " outer=" << r.outer_iterations
              << " inner=" << r.inner_iterations
              << " objective_W=" << r.objective_W
              << " max_constraint=" << r.maximum_constraint_violation
              << " output=" << case_directory(comm) << std::endl;
  }
} // namespace

TEST(VascularForest, BOTH_GlobalGeometryConstraintGradient)
{
  // Finite differences evaluate geometric squared distances directly rather
  // than calling the constraint kernel that supplies the analytic gradient.
  const std::array<double, 7> x = {{.2, -.3, .7, .8, .4, -.1, 1.4}};
  const auto                  derivative =
    ImmersX::Vascularization::internal::global_length_constraint(x).second;
  const auto residual = [](const std::array<double, 7> &z) {
    const Point<3> a(z[0], z[1], z[2]), b(z[3], z[4], z[5]);
    return z[6] * z[6] - a.distance_square(b);
  };
  for (unsigned int i = 0; i < x.size(); ++i)
    {
      auto plus = x, minus = x;
      plus[i] += 1.e-6;
      minus[i] -= 1.e-6;
      EXPECT_NEAR((residual(plus) - residual(minus)) / 2.e-6,
                  derivative[i],
                  4.e-10);
    }
}

TEST(VascularForest, BOTH_GlobalGeometryAlreadyValidAndEmptyOwners)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2, 0., 1.);
  MappingQ1<3>         mapping;
  NetworkData          original, candidate;
  std::map<Id, double> flows;
  const auto           owner = Utilities::MPI::n_mpi_processes(comm) - 1;
  if (Utilities::MPI::this_mpi_process(comm) == owner)
    {
      Node a, b;
      a.position = Point<3>(.2, .3, .5);
      b.position = Point<3>(.8, .3, .5);
      a.owner = b.owner  = owner;
      original.nodes[10] = a;
      candidate.nodes    = {{10, a}, {11, b}};
      Vessel v;
      v.nodes                 = {{10, 11}};
      v.radii                 = {{.005, .005}};
      v.viscosity             = .004;
      v.owner                 = owner;
      v.provenance            = Provenance::generated;
      candidate.vessels[20]   = v;
      candidate.terminals[30] = {11, 40, System::arterial, owner, 1.};
      flows[20]               = 2.e-9;
    }
  GlobalGeometryParameters p;
  export_background(bg, comm);
  export_network(candidate, flows, "before-draft", comm);
  const auto result =
    optimize_forest_geometry(bg, mapping, original, candidate, flows, p, comm);
  export_network(result.network,
                 result.design_flows,
                 result.report.certified ? "after-certified" : "after-invalid",
                 comm);
  record_result(result, comm);
#ifdef DEAL_II_WITH_PETSC
  for (const auto &issue : result.validation.issues)
    ADD_FAILURE() << issue.message;
  EXPECT_TRUE(result.report.certified);
  EXPECT_GT(result.report.backend_reason, 0);
  if (result.network.nodes.count(10))
    {
      EXPECT_EQ(result.network.nodes.at(10).position, Point<3>(.2, .3, .5));
    }
  if (result.network.nodes.count(11))
    {
      EXPECT_EQ(result.network.nodes.at(11).position, Point<3>(.8, .3, .5));
    }
  double local_volume = 0.;
  for (const auto &[id, v] : result.network.vessels)
    if (v.owner == Utilities::MPI::this_mpi_process(comm))
      local_volume += volume(result.network, id);
  EXPECT_NEAR(Utilities::MPI::sum(local_volume, comm),
              .6 * numbers::PI * .005 * .005,
              1.e-13);
#else
  EXPECT_FALSE(result.validation.valid());
  EXPECT_FALSE(result.report.backend_available);
#endif
}

TEST(VascularForest, BOTH_GlobalGeometryImmutableCollisionRejected)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2, 0., 1.);
  MappingQ1<3> mapping;
  NetworkData  n;
  const auto   owner = Utilities::MPI::n_mpi_processes(comm) - 1;
  if (Utilities::MPI::this_mpi_process(comm) == owner)
    {
      const std::array<Point<3>, 4> points = {
        {{.2, .5, .5}, {.8, .5, .5}, {.5, .2, .5}, {.5, .8, .5}}};
      for (unsigned int i = 0; i < 4; ++i)
        {
          n.nodes[i].position = points[i];
          n.nodes[i].owner    = owner;
        }
      for (unsigned int i = 0; i < 2; ++i)
        {
          Vessel v;
          v.nodes           = {{2 * i, 2 * i + 1}};
          v.radii           = {{.005, .005}};
          v.viscosity       = .004;
          v.owner           = owner;
          v.system          = i == 0 ? System::arterial : System::venous;
          n.vessels[20 + i] = v;
        }
    }
  export_background(bg, comm);
  export_network(n, {}, "before-draft", comm);
  const auto result = optimize_forest_geometry(
    bg, mapping, n, n, {}, GlobalGeometryParameters{}, comm);
  export_network(result.network,
                 result.design_flows,
                 result.report.certified ? "after-certified" : "after-invalid",
                 comm);
  record_result(result, comm);
  EXPECT_FALSE(result.report.certified);
  EXPECT_FALSE(result.validation.valid());
}

namespace
{
  double
  segment_square_distance(const Point<3> &a,
                          const Point<3> &b,
                          const Point<3> &c,
                          const Point<3> &d)
  {
    // Independent convex two-variable quadratic: interior critical point and
    // the minima on each of the four edges of the parameter square.
    const auto        u = b - a, v = d - c, w = a - c;
    const long double A = u * u, B = u * v, C = v * v, D = u * w, E = v * w;
    double            best     = std::numeric_limits<double>::infinity();
    auto              evaluate = [&](double s, double t) {
      if (s >= 0. && s <= 1. && t >= 0. && t <= 1.)
        best = std::min(best, (a + s * u - c - t * v).norm_square());
    };
    if (A * C - B * B > 0.)
      evaluate((B * E - C * D) / (A * C - B * B),
               (A * E - B * D) / (A * C - B * B));
    for (double s : {0., 1.})
      evaluate(s, std::clamp(double((E + B * s) / C), 0., 1.));
    for (double t : {0., 1.})
      evaluate(std::clamp(double((B * t - D) / A), 0., 1.), t);
    return best;
  }
} // namespace

TEST(VascularForest, BOTH_GlobalGeometryThreeCrossingTrees)
{
#ifndef DEAL_II_WITH_PETSC
  GTEST_SKIP() << "PETSc TAO is unavailable";
#else
  const auto comm  = MPI_COMM_WORLD;
  const auto me    = Utilities::MPI::this_mpi_process(comm),
             owner = Utilities::MPI::n_mpi_processes(comm) - 1;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2, -2., 2.);
  MappingQ1<3>         mapping;
  NetworkData          original, candidate;
  std::map<Id, double> flows;
  if (me == owner)
    {
      const std::array<Point<3>, 6> points = {{{-1., 0., 0.},
                                               {1., 0., 0.},
                                               {0., -1., 0.},
                                               {0., 1., 0.},
                                               {-1., -1., 0.},
                                               {1., 1., 0.}}};
      for (unsigned int i = 0; i < 6; ++i)
        {
          Node n;
          n.position              = points[i];
          n.owner                 = owner;
          n.provenance            = Provenance::generated;
          candidate.nodes[10 + i] = n;
          if (i % 2 == 0)
            original.nodes[10 + i] = n;
        }
      for (unsigned int tree = 0; tree < 3; ++tree)
        {
          Vessel v;
          v.nodes                        = {{10 + 2 * tree, 11 + 2 * tree}};
          v.radii                        = {{.05, .05}};
          v.viscosity                    = .004;
          v.owner                        = owner;
          v.provenance                   = Provenance::generated;
          v.system                       = tree == 0 ? System::arterial :
                                           tree == 1 ? System::portal :
                                                       System::venous;
          candidate.vessels[20 + tree]   = v;
          candidate.terminals[30 + tree] = {
            11 + 2 * tree, 40 + tree, v.system, owner, 1.};
          flows[20 + tree] = 2.e-9;
        }
    }
  GlobalGeometryParameters controls;
  controls.clearance_m              = .02;
  controls.maximum_outer_iterations = 12;
  export_background(bg, comm);
  export_network(candidate, flows, "before-draft", comm);
  const auto result = optimize_forest_geometry(
    bg, mapping, original, candidate, flows, controls, comm);
  export_network(result.network,
                 result.design_flows,
                 result.report.certified ? "after-certified" : "after-invalid",
                 comm);
  record_result(result, comm);
  for (const auto &issue : result.validation.issues)
    ADD_FAILURE() << issue.message;
  ASSERT_TRUE(result.report.certified);
  EXPECT_GT(result.report.inserted_nodes, 0);
  std::map<Id, std::pair<std::array<Id, 2>, std::array<Point<3>, 2>>> local;
  for (const auto &[id, v] : result.network.vessels)
    if (v.owner == me)
      local[id] = {v.nodes,
                   {{result.network.nodes.at(v.nodes[0]).position,
                     result.network.nodes.at(v.nodes[1]).position}}};
  // This tiny fixture gathers only its oracle records, never in production.
  std::map<Id, decltype(local)::mapped_type> all;
  for (const auto &part : Utilities::MPI::all_gather(comm, local))
    all.insert(part.begin(), part.end());
  for (auto a = all.begin(); a != all.end(); ++a)
    for (auto b = std::next(a); b != all.end(); ++b)
      {
        const auto &an = a->second.first, &bn = b->second.first;
        if (an[0] == bn[0] || an[0] == bn[1] || an[1] == bn[0] ||
            an[1] == bn[1])
          continue;
        const auto &ap = a->second.second, &bp = b->second.second;
        EXPECT_GE(
          std::sqrt(segment_square_distance(ap[0], ap[1], bp[0], bp[1])), .12);
      }
  double local_flow = 0.;
  for (const auto &[id, q] : result.design_flows)
    {
      EXPECT_NEAR(q, 2.e-9, 1.e-24);
      local_flow += q;
    }
  EXPECT_GT(Utilities::MPI::sum(local_flow, comm), 3. * 2.e-9);
#endif
}

TEST(VascularForest, BOTH_GlobalGeometryNonconvexNotchThreeTrees)
{
#ifndef DEAL_II_WITH_PETSC
  GTEST_SKIP() << "PETSc TAO is unavailable";
#else
  const auto comm  = MPI_COMM_WORLD;
  const auto me    = Utilities::MPI::this_mpi_process(comm),
             owner = Utilities::MPI::n_mpi_processes(comm) - 1;
  Triangulation<3> box, notch;
  GridGenerator::subdivided_hyper_rectangle(box,
                                            {8, 8, 2},
                                            Point<3>(-2., -2., -.5),
                                            Point<3>(2., 2., .5));
  std::set<Triangulation<3>::active_cell_iterator> removed;
  for (const auto &cell : box.active_cell_iterators())
    if (std::abs(cell->center()[0]) < .5 && cell->center()[1] > 0.)
      removed.insert(cell);
  GridGenerator::create_triangulation_with_removed_cells(box, removed, notch);
  parallel::distributed::Triangulation<3> bg(comm);
  bg.copy_triangulation(notch);
  MappingQ1<3>         mapping;
  NetworkData          original, candidate;
  std::map<Id, double> flows;
  if (me == owner)
    for (unsigned int tree = 0; tree < 3; ++tree)
      {
        const double z = tree == 0 ? 0. : tree == 1 ? .2 : -.2;
        Node         root, leaf;
        root.owner = leaf.owner        = owner;
        root.position                  = Point<3>(-1., 1., z);
        leaf.position                  = Point<3>(1., 1., z);
        candidate.nodes[10 + 2 * tree] = root;
        candidate.nodes[11 + 2 * tree] = leaf;
        original.nodes[10 + 2 * tree]  = root;
        Vessel v;
        v.nodes                        = {{10 + 2 * tree, 11 + 2 * tree}};
        v.radii                        = {{.05, .05}};
        v.viscosity                    = .004;
        v.owner                        = owner;
        v.provenance                   = Provenance::generated;
        v.system                       = tree == 0 ? System::arterial :
                                         tree == 1 ? System::portal :
                                                     System::venous;
        candidate.vessels[20 + tree]   = v;
        candidate.terminals[30 + tree] = {
          11 + 2 * tree, 40 + tree, v.system, owner, 1.};
        flows[20 + tree] = 2.e-9;
      }
  GlobalGeometryParameters controls;
  controls.clearance_m              = .02;
  controls.maximum_outer_iterations = 16;
  export_background(bg, comm);
  export_network(candidate, flows, "before-draft", comm);
  const auto result = optimize_forest_geometry(
    bg, mapping, original, candidate, flows, controls, comm);
  export_network(result.network,
                 result.design_flows,
                 result.report.certified ? "after-certified" : "after-invalid",
                 comm);
  record_result(result, comm);
  for (const auto &issue : result.validation.issues)
    ADD_FAILURE() << issue.message;
  EXPECT_TRUE(result.report.certified);
  EXPECT_GT(result.report.boundary_constraints, 0);
  // Independent axis-aligned notch oracle: an accepted edge must neither
  // enter the missing prism nor leave the exterior box. The native verifier
  // separately checks the full finite-radius envelope.
  for (const auto &[id, v] : result.network.vessels)
    if (v.owner == me)
      {
        const auto a = result.network.nodes.at(v.nodes[0]).position,
                   b = result.network.nodes.at(v.nodes[1]).position;
        for (const auto &p : {a, b})
          {
            EXPECT_GE(p[0], -2. + .07);
            EXPECT_LE(p[0], 2. - .07);
            EXPECT_GE(p[1], -2. + .07);
            EXPECT_LE(p[1], 2. - .07);
            EXPECT_GE(p[2], -.5 + .07);
            EXPECT_LE(p[2], .5 - .07);
          }
        double                      low = 0., high = 1.;
        bool                        intersects = true;
        const std::array<double, 3> lo = {{-.5, 0., -.5}}, hi = {{.5, 2., .5}};
        for (unsigned int d = 0; d < 3; ++d)
          if (a[d] == b[d])
            intersects = intersects && a[d] >= lo[d] && a[d] <= hi[d];
          else
            {
              const double t0 = (lo[d] - a[d]) / (b[d] - a[d]),
                           t1 = (hi[d] - a[d]) / (b[d] - a[d]);
              low             = std::max(low, std::min(t0, t1));
              high            = std::min(high, std::max(t0, t1));
            }
        EXPECT_FALSE(intersects && low <= high);
      }
#endif
}

TEST(VascularForest, BOTH_GlobalGeometryMissingOriginalRejectedCollectively)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  GridGenerator::subdivided_hyper_cube(bg, 2, 0., 1.);
  MappingQ1<3> mapping;
  NetworkData  original;
  const auto   owner = Utilities::MPI::n_mpi_processes(comm) - 1;
  if (Utilities::MPI::this_mpi_process(comm) == owner)
    {
      Node n;
      n.owner    = owner;
      n.position = Point<3>(.2, .3, .4);
      original.nodes.emplace(47, n);
    }
  // No vessel network exists in this metadata-rejection unit test. Only one
  // rank detects a lost original node; it must still join every lookup.
  const auto result = optimize_forest_geometry(
    bg, mapping, original, {}, {}, GlobalGeometryParameters{}, comm);
  EXPECT_FALSE(result.validation.valid());
  EXPECT_FALSE(result.report.certified);
}
