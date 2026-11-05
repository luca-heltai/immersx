// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/core/domain.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/io.h>

#ifdef DEAL_II_WITH_VTK
#  include <vtkCell.h>
#  include <vtkDataArray.h>
#  include <vtkDataSet.h>
#  include <vtkGenericDataObjectReader.h>
#  include <vtkPointData.h>
#  include <vtkSmartPointer.h>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>

namespace
{
  namespace V = ImmersX::Vascularization;
  using namespace dealii;

  struct Settings : ParameterAcceptor
  {
    Settings()
      : ParameterAcceptor("/Liver/")
      , domain("/Liver/Domain/")
      , growth("/Liver/Vascularization/")
    {
      add_parameter("Arterial grid", arterial_grid);
      add_parameter("Venous grid", venous_grid);
      add_parameter(
        "Portal grid",
        portal_grid,
        "Optional rooted 1D VTK tree; required when Portal network/Enabled is true.");
      add_parameter("Output directory", output_directory);
      domain.triangulation_type               = "fullydistributed";
      domain.surface_mesh_size                = .012;
      domain.surface_feature_angle            = 60.;
      growth.length_scale                     = "meter";
      growth.terminal_count_multiplier_factor = "none";
      growth.minimum_length                   = 0.;
      growth.density_beta                     = 0.;
      growth.coverage.service_radius          = 0.;
      growth.viscosity                        = .004;
      growth.seed_length                      = .003;
      growth.terminal_count                   = 1000;
      growth.candidate_count                  = 16;
      growth.venous.enabled                   = true;
      growth.venous.terminal_separation       = .003;
    }
    std::string arterial_grid, venous_grid, portal_grid, output_directory;
    ImmersX::DomainParameters<3> domain;
    V::Parameters                growth;
  };

#ifdef DEAL_II_WITH_VTK
  void
  read_tree(const std::string  &filename,
            const V::System     system,
            const double        viscosity,
            V::NetworkData     &network,
            std::vector<V::Id> &selected)
  {
    auto reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(filename.c_str());
    reader->ReadAllScalarsOn();
    reader->ReadAllFieldsOn();
    reader->Update();
    auto *grid = vtkDataSet::SafeDownCast(reader->GetOutput());
    AssertThrow(grid != nullptr && grid->GetNumberOfPoints() > 1,
                ExcMessage("Expected a legacy VTK 1D tree: " + filename));
    auto *radius     = grid->GetPointData()->GetArray("Radius");
    auto *root_flags = grid->GetPointData()->GetArray("Root");
    auto *grow_flags = grid->GetPointData()->GetArray("Grow");
    AssertThrow(radius && root_flags && grow_flags,
                ExcMessage(
                  "Line grids require Radius, Root and Grow point arrays."));
    const auto count = static_cast<std::size_t>(grid->GetNumberOfPoints());
    for (auto *array : {radius, root_flags, grow_flags})
      AssertThrow(
        array->GetNumberOfComponents() == 1 &&
          static_cast<std::size_t>(array->GetNumberOfTuples()) == count,
        ExcMessage("Tree point arrays must contain one scalar per point."));
    const V::Id                        offset = network.nodes.size();
    std::vector<std::set<std::size_t>> adjacency(count);
    std::size_t                        edge_count = 0;
    for (vtkIdType c = 0; c < grid->GetNumberOfCells(); ++c)
      {
        auto *cell = grid->GetCell(c);
        AssertThrow(cell->GetCellDimension() == 1,
                    ExcMessage("The input tree must contain only line cells."));
        for (vtkIdType j = 1; j < cell->GetNumberOfPoints(); ++j)
          {
            const auto a = static_cast<std::size_t>(cell->GetPointId(j - 1));
            const auto b = static_cast<std::size_t>(cell->GetPointId(j));
            AssertThrow(a != b && adjacency[a].insert(b).second,
                        ExcMessage(
                          "Tree contains a repeated or collapsed edge."));
            adjacency[b].insert(a);
            ++edge_count;
          }
      }
    std::size_t root = count;
    for (std::size_t i = 0; i < count; ++i)
      {
        const double r = radius->GetTuple1(i);
        AssertThrow(std::isfinite(r) && r > 0.,
                    ExcMessage("Tree radii must be positive metres."));
        const auto *x             = grid->GetPoint(i);
        network.nodes[offset + i] = {Point<3>(x[0], x[1], x[2])};
        if (root_flags->GetTuple1(i) != 0.)
          {
            AssertThrow(root == count,
                        ExcMessage("A tree must have exactly one Root point."));
            root = i;
          }
      }
    AssertThrow(root < count && adjacency[root].size() == 1 &&
                  edge_count == count - 1,
                ExcMessage("Expected a tree rooted at one leaf."));
    if (system == V::System::arterial)
      network.nodes.at(offset + root).port = V::Port::arterial_inlet;
    else if (system == V::System::venous)
      network.nodes.at(offset + root).port = V::Port::venous_outlet;
    else if (system == V::System::portal)
      network.nodes.at(offset + root).port = V::Port::portal_inlet;
    std::set<std::size_t>   seen{root};
    std::queue<std::size_t> pending;
    pending.push(root);
    while (!pending.empty())
      {
        const auto a = pending.front();
        pending.pop();
        for (const auto b : adjacency[a])
          if (seen.insert(b).second)
            {
              pending.push(b);
              V::Vessel vessel;
              vessel.nodes     = {{offset + a, offset + b}};
              vessel.radii     = {{radius->GetTuple1(a), radius->GetTuple1(b)}};
              vessel.viscosity = viscosity;
              vessel.system    = system;
              network.vessels.emplace(network.vessels.size(), vessel);
            }
      }
    AssertThrow(seen.size() == count,
                ExcMessage("Input tree is disconnected."));
    for (std::size_t i = 0; i < count; ++i)
      {
        const bool leaf = i != root && adjacency[i].size() == 1;
        if (leaf)
          network.terminals[offset + i] = {
            offset + i, V::invalid_id, system, 0, 1.};
        if (grow_flags->GetTuple1(i) != 0.)
          {
            AssertThrow(leaf,
                        ExcMessage("Grow must mark non-root leaf points."));
            selected.push_back(offset + i);
          }
      }
    AssertThrow(!selected.empty(),
                ExcMessage("Each tree needs at least one Grow leaf."));
  }
#endif
} // namespace

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize init(argc, argv, 1);
      const auto                       comm = MPI_COMM_WORLD;
      const auto rank = Utilities::MPI::this_mpi_process(comm);
      AssertThrow(argc == 2, ExcMessage("Usage: liver_vascular FILE.prm"));
      Settings      settings;
      const auto    input = std::filesystem::absolute(argv[1]);
      std::ifstream prm(input);
      AssertThrow(prm.good(), ExcMessage("Cannot open liver parameter file."));
      ParameterAcceptor::initialize(prm);
      for (auto *path : {&settings.arterial_grid,
                         &settings.venous_grid,
                         &settings.output_directory,
                         &settings.domain.name_of_grid})
        *path = (input.parent_path() / *path).lexically_normal().string();
      if (!settings.portal_grid.empty())
        settings.portal_grid = (input.parent_path() / settings.portal_grid)
                                 .lexically_normal()
                                 .string();
      AssertThrow(
        settings.growth.portal.enabled == !settings.portal_grid.empty(),
        ExcMessage("Supply Portal grid and enable Portal network together."));
      bool               valid = true;
      V::NetworkData     initial;
      std::vector<V::Id> arterial, venous, portal;
      if (rank == 0)
        try
          {
#ifdef DEAL_II_WITH_VTK
            read_tree(settings.arterial_grid,
                      V::System::arterial,
                      settings.growth.viscosity,
                      initial,
                      arterial);
            read_tree(settings.venous_grid,
                      V::System::venous,
                      settings.growth.viscosity,
                      initial,
                      venous);
            if (settings.growth.portal.enabled)
              read_tree(settings.portal_grid,
                        V::System::portal,
                        settings.growth.viscosity,
                        initial,
                        portal);
#else
            AssertThrow(false,
                        ExcMessage("The liver tutorial requires VTK support."));
#endif
            AssertThrow(!std::filesystem::exists(settings.output_directory),
                        ExcMessage("Output directory must be new."));
            std::filesystem::create_directories(settings.output_directory);
          }
        catch (const std::exception &error)
          {
            std::cerr << error.what() << '\n';
            valid = false;
          }
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage("Cannot prepare liver input and output files."));
      settings.growth.selected_terminals =
        Utilities::MPI::broadcast(comm, arterial, 0);
      settings.growth.venous.selected_terminals =
        Utilities::MPI::broadcast(comm, venous, 0);
      settings.growth.portal.selected_terminals =
        Utilities::MPI::broadcast(comm, portal, 0);
      ImmersX::Domain<3> domain(settings.domain, comm);
      domain.make_grid();
      const auto reference_cell =
        ImmersX::detail::uniform_reference_cell(domain.triangulation());
      AssertThrow(reference_cell.has_value(),
                  ExcMessage("The liver mesh is empty."));
      std::unique_ptr<Mapping<3>> mapping;
      if (reference_cell->is_simplex())
        mapping = std::make_unique<MappingFE<3>>(FE_SimplexP<3>(1));
      else if (reference_cell->is_hyper_cube())
        mapping = std::make_unique<MappingQ1<3>>();
      const auto result = V::generate(
        domain.triangulation(), *mapping, initial, settings.growth, comm);
      for (const auto &issue : result.validation.issues)
        std::cerr << "Rank " << rank << ": " << issue.entity << ' ' << issue.id
                  << ": " << issue.message << '\n';
      AssertThrow(result.validation.valid(),
                  ExcMessage("Liver vascular growth failed."));
      V::save_network(std::filesystem::path(settings.output_directory) /
                        "network",
                      result.network,
                      result.territories,
                      result.terminal_pairs,
                      result.terminal_triplets,
                      comm);
      parallel::fullydistributed::Triangulation<1, 3> lines(comm);
      const auto                                      references =
        V::create_triangulation(result.network, result.discretization, lines);
      Vector<float> systems(lines.n_active_cells()),
        generated(lines.n_active_cells()), radii(lines.n_active_cells());
      for (const auto &cell : lines.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto reference =
              V::physical_reference(cell->id(), references);
            const auto &vessel = result.network.vessels.at(reference.vessel);
            const auto  index  = cell->active_cell_index();
            systems[index]     = static_cast<unsigned int>(vessel.system);
            generated[index]   = static_cast<unsigned int>(vessel.provenance);
            const double midpoint = .5 * (reference.begin + reference.end);
            radii[index] =
              (1. - midpoint) * vessel.radii[0] + midpoint * vessel.radii[1];
          }
      DataOut<1, 3> line_output;
      line_output.attach_triangulation(lines);
      line_output.add_data_vector(systems, "System");
      line_output.add_data_vector(generated, "Generated");
      line_output.add_data_vector(radii, "Radius");
      line_output.build_patches();
      line_output.write_vtu_with_pvtu_record(settings.output_directory + "/",
                                             "vessels",
                                             0,
                                             comm);
      DataOut<3> background_output;
      background_output.attach_triangulation(domain.triangulation());
      background_output.build_patches(*mapping);
      background_output.write_vtu_with_pvtu_record(
        settings.output_directory + "/", "liver", 0, comm);
      const auto table = [&](const std::string &name) {
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.open(std::filesystem::path(settings.output_directory) /
                 (name + "-" + std::to_string(rank) + ".csv"));
        out << std::setprecision(17);
        return out;
      };
      {
        auto out = table("nodes");
        out << "id,x,y,z,port,provenance,pressure_Pa,boundary_inflow_m3_s\n";
        for (const auto &[id, node] : result.network.nodes)
          if (node.owner == rank)
            {
              out << id << ',' << node.position[0] << ',' << node.position[1]
                  << ',' << node.position[2] << ','
                  << static_cast<unsigned int>(node.port) << ','
                  << static_cast<unsigned int>(node.provenance) << ',';
              if (const auto pressure = result.reference.pressures.find(id);
                  pressure != result.reference.pressures.end())
                out << pressure->second;
              out << ',';
              const auto inflow = result.reference.boundary_inflows.find(id);
              out << (inflow == result.reference.boundary_inflows.end() ?
                        0. :
                        inflow->second)
                  << '\n';
            }
      }
      {
        auto out = table("vessels");
        out
          << "id,node0,node1,radius0,radius1,viscosity,system,provenance,resistance_Pa_s_per_m3,flow_m3_s\n";
        for (const auto &[id, vessel] : result.network.vessels)
          if (vessel.owner == rank)
            {
              out << id << ',' << vessel.nodes[0] << ',' << vessel.nodes[1]
                  << ',' << vessel.radii[0] << ',' << vessel.radii[1] << ','
                  << vessel.viscosity << ','
                  << static_cast<unsigned int>(vessel.system) << ','
                  << static_cast<unsigned int>(vessel.provenance) << ',';
              if (const auto reference = result.reference.vessels.find(id);
                  reference != result.reference.vessels.end())
                out << reference->second.resistance << ','
                    << reference->second.flow;
              else
                out << ',';
              out << '\n';
            }
      }
      {
        auto out = table("terminals");
        out << "id,node,territory,system,demand\n";
        for (const auto &[id, terminal] : result.network.terminals)
          if (terminal.owner == rank)
            out << id << ',' << terminal.node << ',' << terminal.territory
                << ',' << static_cast<unsigned int>(terminal.system) << ','
                << terminal.demand << '\n';
      }
      {
        auto out = table("pairs");
        out << "territory,arterial,venous,resistance\n";
        for (const auto &pair : result.terminal_pairs)
          out << pair.territory << ',' << pair.arterial << ',' << pair.venous
              << ',' << pair.resistance << '\n';
      }
      {
        auto out = table("triplets");
        out
          << "territory,arterial,portal,venous,arterial_resistance,portal_resistance,venous_resistance,arterial_flow,portal_flow,venous_flow,compartment_pressure\n";
        for (const auto &t : result.terminal_triplets)
          {
            const auto &reference =
              result.reference.territories.at(t.territory);
            out << t.territory << ',' << t.arterial << ',' << t.portal << ','
                << t.venous;
            for (double r : t.resistances)
              out << ',' << r;
            out << ',' << reference.arterial_flow << ','
                << reference.portal_flow << ',' << reference.venous_flow << ','
                << reference.compartment_pressure << '\n';
          }
      }
      const auto pairs =
        Utilities::MPI::sum(result.terminal_pairs.size(), comm);
      const auto triplets =
        Utilities::MPI::sum(result.terminal_triplets.size(), comm);
      const auto vessels =
        Utilities::MPI::sum(static_cast<V::Id>(
                              std::count_if(result.network.vessels.begin(),
                                            result.network.vessels.end(),
                                            [rank](const auto &entry) {
                                              return entry.second.owner == rank;
                                            })),
                            comm);
      if (rank == 0)
        {
          std::ofstream summary(
            std::filesystem::path(settings.output_directory) / "summary.csv");
          summary
            << "background_cells,vessels,new_pairs,volume_fraction,new_triplets\n"
            << domain.triangulation().n_global_active_cells() << ',' << vessels
            << ',' << pairs << ',' << result.tissue.volume_fraction << ','
            << triplets << '\n';
          std::cout << "Liver cells: "
                    << domain.triangulation().n_global_active_cells()
                    << ", physical vessels: " << vessels
                    << ", tissue pairs: " << pairs;
          if (settings.growth.portal.enabled)
            std::cout << ", tissue triplets: " << triplets;
          std::cout << '\n';
        }
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
}
