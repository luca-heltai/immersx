// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/utilities.h>

#include <immersx/vascularization/io.h>

#ifdef DEAL_II_WITH_VTK
#  include <vtkCellData.h>
#  include <vtkDataArray.h>
#  include <vtkDoubleArray.h>
#  include <vtkNew.h>
#  include <vtkPointData.h>
#  include <vtkPoints.h>
#  include <vtkTypeUInt64Array.h>
#  include <vtkUnsignedIntArray.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkXMLUnstructuredGridWriter.h>
#endif

#include <cmath>
#include <fstream>
#include <iostream>

namespace ImmersX::Vascularization
{
  void
  write_diagnostic_vtk(const std::filesystem::path &directory,
                       const NetworkData           &network,
                       const ValidationReport      &validation,
                       MPI_Comm                     comm)
  {
#ifndef DEAL_II_WITH_VTK
    (void)directory;
    (void)network;
    (void)validation;
    (void)comm;
    AssertThrow(
      false, dealii::ExcMessage("Diagnostic VTK output requires VTK support."));
#else
    using namespace dealii;
    const auto rank = Utilities::MPI::this_mpi_process(comm);
    const auto size = Utilities::MPI::n_mpi_processes(comm);
    const bool valid =
      Utilities::MPI::min(validation.valid() ? 1u : 0u, comm) == 1;
    bool ready = !directory.empty();
    for (const auto &other :
         Utilities::MPI::all_gather(comm, directory.string()))
      ready = ready && other == directory.string();
    AssertThrow(
      Utilities::MPI::min(ready ? 1u : 0u, comm) == 1,
      ExcMessage(
        "Diagnostic output requires the same nonempty directory on every rank."));
    if (rank == 0)
      try
        {
          AssertThrow(!std::filesystem::exists(directory),
                      ExcMessage(
                        "Diagnostic output directory already exists."));
          std::filesystem::create_directories(directory);
        }
      catch (const std::exception &e)
        {
          std::cerr << e.what() << '\n';
          ready = false;
        }
    AssertThrow(Utilities::MPI::min(ready ? 1u : 0u, comm) == 1,
                ExcMessage("Cannot prepare diagnostic VTK output."));
    const std::array<std::pair<System, std::string>, 3> roles = {
      {{System::arterial, "arterial"},
       {System::portal, "portal"},
       {System::venous, "venous"}}};
    std::uint64_t omitted = 0;
    try
      {
        std::ofstream skipped(directory /
                              ("omitted-" + std::to_string(rank) + ".csv"));
        skipped.exceptions(std::ios::failbit | std::ios::badbit);
        skipped << "vessel_id,reason\n";
        for (const auto &[id, vessel] : network.vessels)
          if (vessel.owner == rank && vessel.system != System::arterial &&
              vessel.system != System::portal &&
              vessel.system != System::venous)
            {
              ++omitted;
              skipped << id << ",unknown_system\n";
            }
        for (const auto &[system, name] : roles)
          {
            vtkNew<vtkPoints> points;
            points->SetDataTypeToDouble();
            vtkNew<vtkUnstructuredGrid> grid;
            grid->SetPoints(points);
            vtkNew<vtkDoubleArray> radii, viscosity;
            radii->SetName("Radius");
            viscosity->SetName("DynamicViscosity");
            vtkNew<vtkTypeUInt64Array> nodes, vessels;
            nodes->SetName("PhysicalNodeId");
            vessels->SetName("VesselId");
            vtkNew<vtkUnsignedIntArray> systems, generated, diagnostic, failed;
            systems->SetName("System");
            generated->SetName("Generated");
            diagnostic->SetName("DiagnosticOnly");
            failed->SetName("ValidationFailed");
            for (const auto &[id, v] : network.vessels)
              if (v.owner == rank && v.system == system)
                {
                  const auto a = network.nodes.find(v.nodes[0]);
                  const auto b = network.nodes.find(v.nodes[1]);
                  bool       representable =
                    a != network.nodes.end() && b != network.nodes.end();
                  if (representable)
                    for (unsigned int d = 0; d < 3; ++d)
                      representable = representable &&
                                      std::isfinite(a->second.position[d]) &&
                                      std::isfinite(b->second.position[d]);
                  if (!representable)
                    {
                      ++omitted;
                      skipped << id << ",missing_or_nonfinite_endpoint\n";
                      continue;
                    }
                  vtkIdType endpoints[2];
                  for (unsigned int end = 0; end < 2; ++end)
                    {
                      const auto &x = (end == 0 ? a : b)->second.position;
                      endpoints[end] =
                        points->InsertNextPoint(x[0], x[1], x[2]);
                      radii->InsertNextValue(v.radii[end]);
                      nodes->InsertNextValue(v.nodes[end]);
                    }
                  // Separate endpoints preserve discontinuous radius traces at
                  // a physical junction, including coincident or collapsed
                  // edges.
                  grid->InsertNextCell(VTK_LINE, 2, endpoints);
                  vessels->InsertNextValue(id);
                  systems->InsertNextValue(static_cast<unsigned int>(system));
                  generated->InsertNextValue(
                    static_cast<unsigned int>(v.provenance));
                  diagnostic->InsertNextValue(1);
                  failed->InsertNextValue(valid ? 0 : 1);
                  viscosity->InsertNextValue(v.viscosity);
                }
            grid->GetPointData()->AddArray(radii);
            grid->GetPointData()->AddArray(nodes);
            for (auto *array :
                 {static_cast<vtkDataArray *>(vessels.GetPointer()),
                  static_cast<vtkDataArray *>(systems.GetPointer()),
                  static_cast<vtkDataArray *>(generated.GetPointer()),
                  static_cast<vtkDataArray *>(diagnostic.GetPointer()),
                  static_cast<vtkDataArray *>(failed.GetPointer()),
                  static_cast<vtkDataArray *>(viscosity.GetPointer())})
              grid->GetCellData()->AddArray(array);
            vtkNew<vtkXMLUnstructuredGridWriter> writer;
            writer->SetFileName(
              (directory / (name + "-" + std::to_string(rank) + ".vtu"))
                .c_str());
            writer->SetInputData(grid);
            writer->SetDataModeToAscii();
            AssertThrow(writer->Write() == 1,
                        ExcMessage("Cannot write diagnostic VTK piece."));
          }
        skipped.close();
      }
    catch (const std::exception &e)
      {
        std::cerr << e.what() << '\n';
        ready = false;
      }
    AssertThrow(Utilities::MPI::min(ready ? 1u : 0u, comm) == 1,
                ExcMessage("Cannot write diagnostic VTK pieces."));
    const auto omissions = Utilities::MPI::sum(omitted, comm);
    if (rank == 0)
      try
        {
          for (const auto &[system, name] : roles)
            {
              (void)system;
              std::ofstream out(directory / (name + ".pvtu"));
              out.exceptions(std::ios::failbit | std::ios::badbit);
              out
                << "<?xml version=\"1.0\"?>\n"
                   "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                   "<PUnstructuredGrid GhostLevel=\"0\">\n<PPointData>\n"
                   "<PDataArray type=\"Float64\" Name=\"Radius\"/>\n"
                   "<PDataArray type=\"UInt64\" Name=\"PhysicalNodeId\"/>\n"
                   "</PPointData>\n<PCellData>\n"
                   "<PDataArray type=\"UInt64\" Name=\"VesselId\"/>\n"
                   "<PDataArray type=\"UInt32\" Name=\"System\"/>\n"
                   "<PDataArray type=\"UInt32\" Name=\"Generated\"/>\n"
                   "<PDataArray type=\"UInt32\" Name=\"DiagnosticOnly\"/>\n"
                   "<PDataArray type=\"UInt32\" Name=\"ValidationFailed\"/>\n"
                   "<PDataArray type=\"Float64\" Name=\"DynamicViscosity\"/>\n"
                   "</PCellData>\n<PPoints><PDataArray type=\"Float64\" NumberOfComponents=\"3\"/></PPoints>\n";
              for (unsigned int piece = 0; piece < size; ++piece)
                out << "<Piece Source=\"" << name << '-' << piece
                    << ".vtu\"/>\n";
              out << "</PUnstructuredGrid>\n</VTKFile>\n";
              out.close();
            }
          std::ofstream marker(directory / "diagnostic.json.tmp");
          marker.exceptions(std::ios::failbit | std::ios::badbit);
          marker << "{\n  \"diagnostic_only\": true,\n"
                    "  \"validation_valid\": "
                 << (valid ? "true" : "false")
                 << ",\n  \"generation_complete\": false,\n"
                    "  \"archive_adoptable\": false,\n"
                    "  \"omitted_vessels\": "
                 << omissions << "\n}\n";
          marker.close();
          std::filesystem::rename(directory / "diagnostic.json.tmp",
                                  directory / "diagnostic.json");
        }
      catch (const std::exception &e)
        {
          std::cerr << e.what() << '\n';
          ready = false;
        }
    AssertThrow(Utilities::MPI::min(ready ? 1u : 0u, comm) == 1,
                ExcMessage("Cannot publish diagnostic VTK metadata."));
#endif
  }
} // namespace ImmersX::Vascularization
