// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/numbers.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_description.h>

#include <gtest/gtest.h>
#include <immersx/vascularization/distribution.h>
#include <immersx/vascularization/io.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>

#include "test_paths.h"
using namespace dealii;
using namespace ImmersX::Vascularization;
namespace
{
  constexpr Id base = Id{1} << 60;
  std::filesystem::path
  path(const std::string &name, MPI_Comm comm)
  {
    std::string token;
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      token = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    token = Utilities::MPI::broadcast(comm, token, 0);
    return ImmersX::TestPaths::output_path("vascular-io") /
           (name + "-" + token);
  }
  NetworkData
  physical_fixture()
  {
    NetworkData n;
    for (unsigned int i = 0; i < 2; ++i)
      {
        const Id     b = base + 1000 * i;
        const double y = .25 + .5 * i;
        n.nodes[b]     = {Point<3>(i == 0 ? 0. : -.1, y, .5),
                          0,
                          Port::arterial_inlet,
                          Provenance::imported,
                      i == 0 ? Point<3>(2., 0., 0.) : Point<3>()};
        n.nodes[b + 1] = {Point<3>(.6, y, .5),
                          0,
                          Port::none,
                          Provenance::generated};
        n.nodes[b + 2] = {Point<3>(i == 0 ? 1. : 1.1, y, .5),
                          0,
                          Port::venous_outlet,
                          Provenance::generated,
                          i == 0 ? Point<3>(-3., 0., 0.) : Point<3>()};
        n.nodes[b + 3] = {Point<3>(.7, y, .5),
                          0,
                          Port::none,
                          Provenance::imported};
        Vessel v;
        v.nodes      = {{b, b + 1}};
        v.radii      = {{.0031415926535897933, .0017312345678901234}};
        v.viscosity  = .004123456789012345;
        v.provenance = i == 0 ? Provenance::imported : Provenance::generated;
        v.physical_properties = {{"Young modulus", 123456.78901234567},
                                 {"coefficient\nwith \"quotes\"",
                                  -1.2345678901234567e-19}};
        n.vessels[b]          = v;
        v.nodes               = {{b + 2, b + 3}};
        v.system              = System::venous;
        v.radii               = {{.0043212345678901234, .0030123456789012345}};
        v.provenance = i == 0 ? Provenance::generated : Provenance::imported;
        n.vessels[b + 1]    = v;
        n.terminals[b + 10] = {b + 1, b + 20, System::arterial, 0, 2. + i};
        n.terminals[b + 11] = {b + 3, b + 20, System::venous, 0, 2. + i};
      }
    n.nodes[base + 9000] = {Point<3>(-.5, .5, .5),
                            0,
                            Port::none,
                            Provenance::generated};
    return n;
  }
  NetworkArchive
  make_archive(const Triangulation<3> &bg, MPI_Comm comm)
  {
    NetworkArchive a;
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      a.network = physical_fixture();
    a.network       = redistribute(bg, MappingQ1<3>(), a.network, 0., comm);
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    for (const auto &[id, t] : a.network.terminals)
      if (t.owner == rank && t.system == System::arterial)
        {
          a.territories.push_back(
            {t.territory, a.network.nodes.at(t.node).position, rank, t.demand});
          const unsigned int i = (t.territory - base - 20) / 1000;
          a.terminal_pairs.push_back(
            {t.territory, id, id + 1, 4.e12 + i * 1.e12});
        }
    return a;
  }
  void
  check(const NetworkArchive &a, MPI_Comm comm)
  {
    const auto expected = physical_fixture();
    const auto rank     = Utilities::MPI::this_mpi_process(comm);
    ASSERT_TRUE(
      validate_data(a.network, Utilities::MPI::n_mpi_processes(comm)).valid());
    unsigned int nn = 0, nv = 0, nt = 0;
    for (const auto &[id, n] : a.network.nodes)
      {
        ASSERT_TRUE(expected.nodes.count(id));
        const auto &e = expected.nodes.at(id);
        EXPECT_EQ(n.position, e.position);
        EXPECT_EQ(n.port, e.port);
        EXPECT_EQ(n.provenance, e.provenance);
        EXPECT_EQ(n.inward_port_normal, e.inward_port_normal);
        nn += n.owner == rank;
      }
    for (const auto &[id, v] : a.network.vessels)
      {
        ASSERT_TRUE(expected.vessels.count(id));
        const auto &e = expected.vessels.at(id);
        EXPECT_EQ(v.nodes, e.nodes);
        EXPECT_EQ(v.radii, e.radii);
        EXPECT_DOUBLE_EQ(v.viscosity, e.viscosity);
        EXPECT_EQ(v.system, e.system);
        EXPECT_EQ(v.provenance, e.provenance);
        EXPECT_EQ(v.physical_properties, e.physical_properties);
        nv += v.owner == rank;
      }
    for (const auto &[id, t] : a.network.terminals)
      {
        ASSERT_TRUE(expected.terminals.count(id));
        const auto &e = expected.terminals.at(id);
        EXPECT_EQ(t.node, e.node);
        EXPECT_EQ(t.territory, e.territory);
        EXPECT_EQ(t.system, e.system);
        EXPECT_DOUBLE_EQ(t.demand, e.demand);
        EXPECT_EQ(t.owner, a.network.nodes.at(t.node).owner);
        nt += t.owner == rank;
      }
    EXPECT_EQ(Utilities::MPI::sum(nn, comm), 9u);
    EXPECT_EQ(Utilities::MPI::sum(nv, comm), 4u);
    EXPECT_EQ(Utilities::MPI::sum(nt, comm), 4u);
    EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(a.territories.size()), comm),
              2u);
    EXPECT_EQ(Utilities::MPI::sum(static_cast<Id>(a.terminal_pairs.size()),
                                  comm),
              2u);
    for (const auto &t : a.territories)
      {
        const Id node = t.id - 19;
        EXPECT_EQ(t.owner, rank);
        ASSERT_TRUE(a.network.nodes.count(node));
        EXPECT_EQ(a.network.nodes.at(node).owner, rank);
        EXPECT_EQ(t.position, expected.nodes.at(node).position);
        EXPECT_DOUBLE_EQ(t.demand, 2. + (t.id - base - 20) / 1000);
      }
    for (const auto &p : a.terminal_pairs)
      {
        EXPECT_EQ(p.arterial, p.territory - 10);
        EXPECT_EQ(p.venous, p.territory - 9);
        EXPECT_DOUBLE_EQ(p.resistance,
                         4.e12 + ((p.territory - base - 20) / 1000) * 1.e12);
      }
    // The unrelated exterior point cloud is preserved by I/O; exclude it from
    // the independent resistive problem, which intentionally rejects isolates.
    auto active = a.network;
    active.nodes.erase(base + 9000);
    ReferenceConditions bc;
    for (unsigned int i = 0; i < 2; ++i)
      {
        bc.pressures[base + 1000 * i]     = 12000.;
        bc.pressures[base + 1000 * i + 2] = 4000.;
      }
    const auto reference =
      solve_reference(active, a.terminal_pairs, bc, {}, comm);
    ASSERT_TRUE(reference.validation.valid());
    const auto r = [&](Id id) {
      const auto  &v = expected.vessels.at(id);
      const double l =
        expected.nodes.at(v.nodes[0])
          .position.distance(expected.nodes.at(v.nodes[1]).position);
      const double r0 = v.radii[0], r1 = v.radii[1];
      return 8. * v.viscosity * l / numbers::PI *
             (r0 * r0 + r0 * r1 + r1 * r1) / (3. * std::pow(r0 * r1, 3));
    };
    for (const auto &p : a.terminal_pairs)
      {
        const Id     b = p.territory - 20;
        const double q = 8000. / (r(b) + r(b + 1) + p.resistance);
        EXPECT_NEAR(reference.territories.at(p.territory).flow / q, 1., 1.e-11);
      }
  }
  void
  mesh(parallel::distributed::Triangulation<3> &bg)
  {
    GridGenerator::subdivided_hyper_rectangle(bg,
                                              {8, 1, 1},
                                              Point<3>(),
                                              Point<3>(1., 1., 1.));
  }
} // namespace
TEST(VascularIO, BOTH_RoundTripRetainsStablePhysicalDataAndPositiveBeds)
{
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<3> bg(comm);
  mesh(bg);
  const auto original  = make_archive(bg, comm);
  const auto directory = path("roundtrip", comm);
  save_network(directory,
               original.network,
               original.territories,
               original.terminal_pairs,
               comm);
  const auto restored = load_network(directory, bg, MappingQ1<3>(), 0., comm);
  check(restored, comm);
  check(original, comm);
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      EXPECT_TRUE(std::filesystem::is_regular_file(directory / "manifest"));
      for (unsigned int r = 0; r < Utilities::MPI::n_mpi_processes(comm); ++r)
        EXPECT_TRUE(std::filesystem::is_regular_file(
          directory / ("part-" + std::to_string(r) + "-0.txt")));
    }
  EXPECT_THROW(save_network(directory,
                            original.network,
                            original.territories,
                            original.terminal_pairs,
                            comm),
               ExceptionBase);
  check(load_network(directory, bg, MappingQ1<3>(), 0., comm), comm);
}
TEST(VascularIO, BOTH_EmptyWritersAndEmptyNetworkRemainCollective)
{
  const auto       comm = MPI_COMM_WORLD;
  Triangulation<3> serial;
  GridGenerator::hyper_cube(serial);
  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial, comm);
  parallel::fullydistributed::Triangulation<3> bg(comm);
  bg.create_triangulation(description);
  const auto a         = make_archive(bg, comm);
  const auto directory = path("empty-writers", comm);
  save_network(directory, a.network, a.territories, a.terminal_pairs, comm);
  check(load_network(directory, bg, MappingQ1<3>(), 0., comm), comm);
  const auto empty = path("empty-network", comm);
  save_network(empty, {}, {}, {}, comm);
  const auto restored = load_network(empty, bg, MappingQ1<3>(), 0., comm);
  EXPECT_TRUE(restored.network.nodes.empty());
  EXPECT_TRUE(restored.network.vessels.empty());
  EXPECT_TRUE(restored.network.terminals.empty());
  EXPECT_TRUE(restored.territories.empty());
  EXPECT_TRUE(restored.terminal_pairs.empty());
}
TEST(VascularIO, BOTH_MissingCorruptAndIncompleteFilesFailCollectively)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  mesh(bg);
  const auto a       = make_archive(bg, comm);
  const auto missing = path("missing", comm);
  EXPECT_THROW(load_network(missing, bg, MappingQ1<3>(), 0., comm),
               ExceptionBase);
  for (unsigned int mode = 0; mode < 3; ++mode)
    {
      const auto directory = path("broken", comm);
      save_network(directory, a.network, a.territories, a.terminal_pairs, comm);
      if (rank + 1 == Utilities::MPI::n_mpi_processes(comm))
        {
          const auto part =
            directory / ("part-" + std::to_string(rank) + "-0.txt");
          if (mode == 0)
            std::filesystem::remove(part);
          else if (mode == 1)
            {
              std::ofstream out(part);
              out << "truncated\n";
            }
          else
            std::filesystem::remove(directory / "manifest");
        }
      MPI_Barrier(comm);
      EXPECT_THROW(load_network(directory, bg, MappingQ1<3>(), 0., comm),
                   ExceptionBase);
    }
  const auto blocked = path("blocked", comm);
  if (rank == 0)
    {
      std::ofstream out(blocked);
      out << "existing file\n";
    }
  MPI_Barrier(comm);
  EXPECT_THROW(
    save_network(blocked, a.network, a.territories, a.terminal_pairs, comm),
    ExceptionBase);
}
TEST(VascularIO, BOTH_InvalidPhysicalOrTerritoryRecordsCannotPublishAManifest)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  mesh(bg);
  for (unsigned int mode = 0; mode < 5; ++mode)
    {
      auto a = make_archive(bg, comm);
      if (mode == 0 && rank == 0)
        a.network.nodes.at(base).position[0] =
          std::numeric_limits<double>::quiet_NaN();
      else if (mode == 1 && !a.terminal_pairs.empty())
        a.terminal_pairs[0].resistance = -1.;
      else if (mode == 2 && !a.territories.empty())
        a.territories[0].demand += 1.;
      else if (mode == 3 && !a.terminal_pairs.empty())
        a.terminal_pairs.push_back(a.terminal_pairs.front());
      else if (mode == 4 && a.network.nodes.count(base + 1) &&
               a.network.nodes.at(base + 1).owner == rank)
        {
          a.network.nodes[base + 4]    = {Point<3>(.65, .35, .5), rank};
          auto v                       = physical_fixture().vessels.at(base);
          v.nodes                      = {{base + 1, base + 4}};
          v.owner                      = rank;
          a.network.vessels[base + 99] = v;
        }
      const auto directory = path("invalid", comm);
      EXPECT_THROW(save_network(directory,
                                a.network,
                                a.territories,
                                a.terminal_pairs,
                                comm),
                   ExceptionBase);
      if (rank == 0)
        {
          EXPECT_FALSE(std::filesystem::exists(directory));
        }
    }
}
TEST(VascularIO, MPI_ReloadWithFewerOrMoreReadersThanWriters)
{
  const auto comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm),
             size = Utilities::MPI::n_mpi_processes(comm);
  ASSERT_GT(size, 1u);
  parallel::distributed::Triangulation<3> bg(comm);
  mesh(bg);
  const auto a         = make_archive(bg, comm);
  const auto directory = path("fewer-readers", comm);
  save_network(directory, a.network, a.territories, a.terminal_pairs, comm);
  MPI_Comm reader;
  MPI_Comm_split(comm, rank % 2, rank, &reader);
  {
    parallel::distributed::Triangulation<3> smaller(reader);
    mesh(smaller);
    check(load_network(directory, smaller, MappingQ1<3>(), 0., reader), reader);
  }
  MPI_Comm_free(&reader);
  MPI_Barrier(comm);
  const auto one_writer = path("more-readers", comm);
  if (rank + 1 == size)
    {
      parallel::distributed::Triangulation<3> serial(MPI_COMM_SELF);
      mesh(serial);
      const auto single = make_archive(serial, MPI_COMM_SELF);
      save_network(one_writer,
                   single.network,
                   single.territories,
                   single.terminal_pairs,
                   MPI_COMM_SELF);
    }
  MPI_Barrier(comm);
  check(load_network(one_writer, bg, MappingQ1<3>(), 0., comm), comm);
}
TEST(VascularIO, MPI_DuplicateAuthorityAndStaleGhostsAreRejected)
{
  const auto comm = MPI_COMM_WORLD;
  ASSERT_GT(Utilities::MPI::n_mpi_processes(comm), 1u);
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  mesh(bg);
  auto         a       = make_archive(bg, comm);
  unsigned int changed = 0;
  for (auto &[id, node] : a.network.nodes)
    if (node.owner != rank)
      {
        node.position[1] += .01;
        changed = 1;
        break;
      }
  ASSERT_GT(Utilities::MPI::sum(changed, comm), 0u);
  EXPECT_THROW(save_network(path("stale", comm), a.network, {}, {}, comm),
               ExceptionBase);
  a                    = make_archive(bg, comm);
  const Id replacement = base + 99999;
  changed              = 0;
  for (const auto &[id, v] : a.network.vessels)
    if (v.owner == rank)
      {
        const auto value = v;
        a.network.vessels.erase(id);
        a.network.vessels.emplace(replacement, value);
        changed = 1;
        break;
      }
  ASSERT_GT(Utilities::MPI::sum(changed, comm), 1u);
  EXPECT_THROW(save_network(path("duplicate", comm), a.network, {}, {}, comm),
               ExceptionBase);
}

TEST(VascularIO, BOTH_BoundedPartsRedistributeSerialAuthorityAcrossReaders)
{
  const auto             comm       = MPI_COMM_WORLD;
  const auto             rank       = Utilities::MPI::this_mpi_process(comm);
  const auto             size       = Utilities::MPI::n_mpi_processes(comm);
  constexpr unsigned int components = 1100;
  const auto             directory  = path("bounded", comm);
  if (rank + 1 == size)
    {
      NetworkArchive a;
      for (unsigned int i = 0; i < components; ++i)
        {
          const Id     b     = base + 10 * i;
          const double y     = .25 + .5 * i / components;
          a.network.nodes[b] = {Point<3>(0., y, .5), 0, Port::arterial_inlet};
          a.network.nodes[b + 1] = {Point<3>(.6, y, .5), 0};
          a.network.nodes[b + 2] = {Point<3>(1., y, .5),
                                    0,
                                    Port::venous_outlet};
          a.network.nodes[b + 3] = {Point<3>(.7, y, .5), 0};
          Vessel v;
          v.nodes                            = {{b, b + 1}};
          v.radii                            = {{.002, .001}};
          v.viscosity                        = .004;
          v.physical_properties["component"] = i;
          a.network.vessels[b]               = v;
          v.nodes                            = {{b + 2, b + 3}};
          v.system                           = System::venous;
          a.network.vessels[b + 1]           = v;
          const double demand = 1. + static_cast<double>(i) / components;
          a.network.terminals[base + 2 * i] = {
            b + 1, base + i, System::arterial, 0, demand};
          a.network.terminals[base + 2 * i + 1] = {
            b + 3, base + i, System::venous, 0, demand};
          a.territories.push_back(
            {base + i, a.network.nodes.at(b + 1).position, 0, demand});
          a.terminal_pairs.push_back(
            {base + i, base + 2 * i, base + 2 * i + 1, 4.e12 + i * 1.e9});
        }
      save_network(
        directory, a.network, a.territories, a.terminal_pairs, MPI_COMM_SELF);
      EXPECT_TRUE(std::filesystem::exists(directory / "part-0-4.txt"));
      EXPECT_FALSE(std::filesystem::exists(directory / "part-0-5.txt"));
    }
  MPI_Barrier(comm);
  parallel::distributed::Triangulation<3> bg(comm);
  mesh(bg);
  const auto   a  = load_network(directory, bg, MappingQ1<3>(), 0., comm);
  unsigned int nn = 0, nv = 0, nt = 0;
  for (const auto &[id, node] : a.network.nodes)
    {
      EXPECT_LT(id, base + 10 * components);
      nn += node.owner == rank;
    }
  for (const auto &[id, v] : a.network.vessels)
    {
      const unsigned int i = (id - base) / 10;
      EXPECT_DOUBLE_EQ(v.physical_properties.at("component"), i);
      EXPECT_EQ(v.radii, (std::array<double, 2>{{.002, .001}}));
      nv += v.owner == rank;
    }
  for (const auto &[id, t] : a.network.terminals)
    {
      EXPECT_EQ(t.owner, a.network.nodes.at(t.node).owner);
      EXPECT_DOUBLE_EQ(t.demand,
                       1. +
                         static_cast<double>(t.territory - base) / components);
      nt += t.owner == rank;
    }
  for (const auto &t : a.territories)
    {
      EXPECT_EQ(t.owner, rank);
      const Id node = base + 10 * (t.id - base) + 1;
      ASSERT_TRUE(a.network.nodes.count(node));
      EXPECT_EQ(a.network.nodes.at(node).owner, rank);
      EXPECT_EQ(t.position, a.network.nodes.at(node).position);
    }
  for (const auto &p : a.terminal_pairs)
    {
      const Id i = p.territory - base;
      EXPECT_EQ(p.arterial, base + 2 * i);
      EXPECT_EQ(p.venous, base + 2 * i + 1);
      EXPECT_DOUBLE_EQ(p.resistance, 4.e12 + i * 1.e9);
    }
  EXPECT_EQ(Utilities::MPI::sum(nn, comm), 4 * components);
  EXPECT_EQ(Utilities::MPI::sum(nv, comm), 2 * components);
  EXPECT_EQ(Utilities::MPI::sum(nt, comm), 2 * components);
  EXPECT_EQ(Utilities::MPI::sum(static_cast<unsigned int>(a.territories.size()),
                                comm),
            components);
  EXPECT_EQ(Utilities::MPI::sum(
              static_cast<unsigned int>(a.terminal_pairs.size()), comm),
            components);
}
