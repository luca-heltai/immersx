// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <gtest/gtest.h>
#include <immersx/vascularization/io.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

#include "test_paths.h"
#ifdef DEAL_II_WITH_VTK
#  include <vtkCell.h>
#  include <vtkCellData.h>
#  include <vtkDataArray.h>
#  include <vtkNew.h>
#  include <vtkPointData.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkVariant.h>
#  include <vtkXMLPUnstructuredGridReader.h>
#endif
using namespace dealii;
using namespace ImmersX::Vascularization;

TEST(VascularIO, BOTH_InvalidDiagnosticVTK)
{
#ifndef DEAL_II_WITH_VTK
  GTEST_SKIP() << "VTK is unavailable.";
#else
  const auto  comm  = MPI_COMM_WORLD;
  const auto  me    = Utilities::MPI::this_mpi_process(comm);
  const auto  owner = Utilities::MPI::n_mpi_processes(comm) - 1;
  std::string token;
  if (me == 0)
    token = std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  token               = Utilities::MPI::broadcast(comm, token, 0);
  const auto   output = ImmersX::TestPaths::output_path("invalid-vtk") / token;
  constexpr Id base   = Id{1} << 60;
  NetworkData  n;
  ValidationReport              report;
  const std::array<Point<3>, 6> x = {{{-1., 0., 0.},
                                      {1., 0., 0.},
                                      {0., -1., 0.},
                                      {0., 1., 0.},
                                      {-1., -1., 0.},
                                      {1., 1., 0.}}};
  if (me == owner)
    {
      // Three mutually crossing lines, a collapsed line, and two edges whose
      // endpoints cannot be represented. This input must not be repaired.
      for (unsigned int i = 0; i < 6; ++i)
        n.nodes[base + i] = {x[i], owner};
      for (unsigned int i = 0; i < 3; ++i)
        {
          Vessel v;
          v.nodes                   = {{base + 2 * i, base + 2 * i + 1}};
          v.owner                   = owner;
          v.system                  = i == 0 ? System::arterial :
                                      i == 1 ? System::portal :
                                               System::venous;
          v.radii                   = {{.012, .010}};
          v.viscosity               = .004;
          n.vessels[base + 100 + i] = v;
        }
      Vessel collapsed      = n.vessels.at(base + 100);
      collapsed.nodes       = {{base, base}};
      collapsed.radii       = {{-.001, .009}};
      n.vessels[base + 103] = collapsed;
      Vessel missing        = collapsed;
      missing.nodes[1]      = base + 999;
      n.vessels[base + 104] = missing;
      n.nodes[base + 10] = {{std::numeric_limits<double>::quiet_NaN(), 0., 0.},
                            owner};
      missing.nodes[1]   = base + 10;
      n.vessels[base + 105] = missing;
      Vessel unknown        = collapsed;
      unknown.system        = static_cast<System>(99);
      n.vessels[base + 106] = unknown;
      report.issues.push_back(
        {"network", base, "Intentional invalid diagnostic fixture."});
    }
  write_diagnostic_vtk(output, n, report, comm);
  EXPECT_EQ(n.vessels.size(), me == owner ? 7u : 0u);
  if (me == 0)
    {
      const std::array<std::pair<const char *, unsigned int>, 3> roles = {
        {{"arterial", 2}, {"portal", 1}, {"venous", 1}}};
      for (const auto &[name, count] : roles)
        {
          vtkNew<vtkXMLPUnstructuredGridReader> reader;
          reader->SetFileName((output / (std::string(name) + ".pvtu")).c_str());
          reader->Update();
          auto *grid = reader->GetOutput();
          EXPECT_EQ(grid->GetNumberOfCells(), count);
          EXPECT_EQ(grid->GetNumberOfPoints(), 2 * count);
          auto *ids      = grid->GetCellData()->GetArray("VesselId");
          auto *node_ids = grid->GetPointData()->GetArray("PhysicalNodeId");
          auto *radius   = grid->GetPointData()->GetArray("Radius");
          EXPECT_NE(ids, nullptr);
          EXPECT_NE(node_ids, nullptr);
          EXPECT_NE(radius, nullptr);
          // XML readers may return native unsigned arrays instead of the
          // vtkTypeUInt64Array subclass used by the writer.
          // Read integer variants to preserve IDs above double's exact range.
          for (auto *array : {ids, node_ids})
            if (array)
              {
                EXPECT_TRUE(array->GetDataType() == VTK_UNSIGNED_LONG ||
                            array->GetDataType() == VTK_UNSIGNED_LONG_LONG);
                EXPECT_EQ(array->GetDataTypeSize(), 8);
                EXPECT_EQ(array->GetNumberOfComponents(), 1);
              }
          if (ids && node_ids && radius)
            for (vtkIdType c = 0; c < grid->GetNumberOfCells(); ++c)
              {
                const auto id = ids->GetVariantValue(c).ToUnsignedLongLong();
                EXPECT_GE(id, base + 100);
                EXPECT_LE(id, base + 103);
                auto      *cell = grid->GetCell(c);
                const auto p0 = cell->GetPointId(0), p1 = cell->GetPointId(1);
                for (const auto point : {p0, p1})
                  {
                    const auto physical =
                      node_ids->GetVariantValue(point).ToUnsignedLongLong();
                    EXPECT_GE(physical, base);
                    EXPECT_LT(physical, base + 6);
                    if (physical >= base && physical < base + 6)
                      {
                        for (unsigned int d = 0; d < 3; ++d)
                          EXPECT_DOUBLE_EQ(grid->GetPoint(point)[d],
                                           x[physical - base][d]);
                      }
                  }
                auto *mu = grid->GetCellData()->GetArray("DynamicViscosity");
                EXPECT_NE(mu, nullptr);
                if (mu)
                  {
                    EXPECT_DOUBLE_EQ(mu->GetTuple1(c), .004);
                  }
                if (id == base + 103)
                  {
                    EXPECT_EQ(
                      node_ids->GetVariantValue(p0).ToUnsignedLongLong(), base);
                    EXPECT_EQ(
                      node_ids->GetVariantValue(p1).ToUnsignedLongLong(), base);
                    EXPECT_DOUBLE_EQ(radius->GetTuple1(p0), -.001);
                    EXPECT_DOUBLE_EQ(radius->GetTuple1(p1), .009);
                  }
                else
                  {
                    const auto i = id - (base + 100);
                    EXPECT_EQ(
                      node_ids->GetVariantValue(p0).ToUnsignedLongLong(),
                      base + 2 * i);
                    EXPECT_EQ(
                      node_ids->GetVariantValue(p1).ToUnsignedLongLong(),
                      base + 2 * i + 1);
                    EXPECT_DOUBLE_EQ(radius->GetTuple1(p0), .012);
                    EXPECT_DOUBLE_EQ(radius->GetTuple1(p1), .010);
                  }
                EXPECT_DOUBLE_EQ(
                  grid->GetCellData()->GetArray("DiagnosticOnly")->GetTuple1(c),
                  1.);
                EXPECT_DOUBLE_EQ(grid->GetCellData()
                                   ->GetArray("ValidationFailed")
                                   ->GetTuple1(c),
                                 1.);
              }
        }
      std::ifstream     metadata(output / "diagnostic.json");
      const std::string text((std::istreambuf_iterator<char>(metadata)), {});
      EXPECT_NE(text.find("\"validation_valid\": false"), std::string::npos);
      EXPECT_NE(text.find("\"archive_adoptable\": false"), std::string::npos);
      EXPECT_NE(text.find("\"omitted_vessels\": 3"), std::string::npos);
      std::cout << "Invalid diagnostic VTK: " << output << std::endl;
    }
  EXPECT_THROW(write_diagnostic_vtk(output, n, report, comm), ExceptionBase);
#endif
}

TEST(VascularIO, BOTH_EmptyDiagnosticVTK)
{
#ifndef DEAL_II_WITH_VTK
  GTEST_SKIP() << "VTK is unavailable.";
#else
  const auto  comm = MPI_COMM_WORLD;
  std::string token;
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    token = std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  token = Utilities::MPI::broadcast(comm, token, 0);
  const auto output =
    ImmersX::TestPaths::output_path("invalid-vtk") / ("empty-" + token);
  write_diagnostic_vtk(output, {}, {}, comm);
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    for (const auto *name : {"arterial", "portal", "venous"})
      {
        vtkNew<vtkXMLPUnstructuredGridReader> reader;
        reader->SetFileName((output / (std::string(name) + ".pvtu")).c_str());
        reader->Update();
        EXPECT_EQ(reader->GetOutput()->GetNumberOfCells(), 0);
      }
#endif
}
