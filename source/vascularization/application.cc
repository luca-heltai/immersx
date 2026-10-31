// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/application.h>
#include <immersx/vascularization/io.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace ImmersX::Vascularization
{
  ApplicationParameters::ApplicationParameters(const std::string &section)
    : ParameterAcceptor(section)
    , generation(section)
  {
    enter_subsection("Input and ports");
    add_parameter(
      "Mode",
      mode,
      "generate, complete or load; load only rediscretizes the physical archive.");
    add_parameter("Input archive",
                  input_archive,
                  "Archive directory required for complete and load.");
    add_parameter(
      "Evaluate reference",
      evaluate_reference,
      "Evaluate arterial/loaded networks; generated AV networks always validate their paired reference.");
    leave_subsection();
    enter_subsection("Domain");
    add_parameter("Shape", shape, "cube or sphere.");
    add_parameter("Cube origin", cube_origin);
    add_parameter("Cube side length", cube_side, "Metres.");
    add_parameter("Sphere center", sphere_center);
    add_parameter("Sphere radius", sphere_radius, "Metres.");
    add_parameter("Uniform refinements", uniform_refinements);
    add_parameter("Refinement focus",
                  refinement_focus,
                  "none, edge for a cube, or center for a sphere.");
    add_parameter("Edge axis", edge_axis, "0=x, 1=y, 2=z.");
    add_parameter(
      "Edge sides",
      edge_sides,
      "Lower(0)/upper(1) faces for the two transverse axes, in coordinate order.");
    add_parameter(
      "Refinement fractions",
      refinement_fractions,
      "Successive local refinement distances, relative to cube side or sphere radius.");
    leave_subsection();
    enter_subsection("Physiology");
    add_parameter(
      "Loaded tissue reference flow",
      loaded_reference_flow,
      "m^3/s through all terminals in arterial load mode; zero uses explicit boundary data only.");
    add_parameter("Reference iteration budget",
                  generation.reference_parameters.maximum_iterations);
    add_parameter("Reference linear tolerance",
                  generation.reference_parameters.linear_relative_tolerance);
    add_parameter(
      "Reference conservation tolerance",
      generation.reference_parameters.conservation_relative_tolerance);
    add_parameter("Reference absolute flow tolerance",
                  generation.reference_parameters.absolute_flow_tolerance);
    leave_subsection();
    enter_subsection("Output");
    add_parameter("Directory",
                  output_directory,
                  "New output directory; must be supplied.");
    add_parameter("Save physical archive", save_archive);
    add_parameter("Write tables", write_tables);
    add_parameter("Write meshes", write_meshes);
    leave_subsection();
  }
  namespace
  {
    using namespace dealii;
    namespace fs = std::filesystem;
    void
    require_all(bool valid, MPI_Comm comm, const std::string &message)
    {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage(message));
    }
    bool
    valid_all(ValidationReport &report, MPI_Comm comm)
    {
      const bool valid =
        Utilities::MPI::min(static_cast<unsigned int>(report.valid()), comm) ==
        1;
      if (!valid && report.valid())
        report.issues.push_back({"application",
                                 invalid_id,
                                 "Vascular run failed on another process."});
      return valid;
    }
    std::ofstream
    output(const fs::path &file)
    {
      std::ofstream stream;
      stream.exceptions(std::ios::failbit | std::ios::badbit);
      stream.open(file);
      stream << std::setprecision(std::numeric_limits<double>::max_digits10);
      return stream;
    }
    const char *
    system_name(System s)
    {
      if (s == System::arterial)
        return "arterial";
      else if (s == System::venous)
        return "venous";
      AssertThrow(false, ExcMessage("Unknown vascular system."));
      return "";
    }
    const char *
    calibration_name(CalibrationStatus s)
    {
      if (s == CalibrationStatus::not_requested)
        return "not_requested";
      else if (s == CalibrationStatus::converged)
        return "converged";
      else if (s == CalibrationStatus::incompatible)
        return "incompatible";
      else if (s == CalibrationStatus::exhausted)
        return "exhausted";
      AssertThrow(false, ExcMessage("Unknown calibration status."));
      return "";
    }
    void
    point(std::ostream &out, const Point<3> &p)
    {
      out << p[0] << ',' << p[1] << ',' << p[2];
    }

    void
    tables(const fs::path         &dir,
           const GenerationResult &r,
           const Triangulation<3> &bg,
           const Mapping<3>       &mapping,
           unsigned int            rank)
    {
      const auto name = [&](const char *kind) {
        return dir / (std::string(kind) + "-" + std::to_string(rank) + ".csv");
      };
      {
        auto out = output(name("nodes"));
        out << "id,x,y,z,owner,port,provenance,pressure\n";
        for (const auto &[id, n] : r.network.nodes)
          if (n.owner == rank)
            {
              out << id << ',';
              point(out, n.position);
              out << ',' << n.owner << ',' << static_cast<unsigned int>(n.port)
                  << ',' << static_cast<unsigned int>(n.provenance) << ',';
              if (r.reference.pressures.count(id))
                out << r.reference.pressures.at(id);
              out << '\n';
            }
      }
      {
        auto out = output(name("vessels"));
        out
          << "id,node0,node1,system,x0,y0,z0,x1,y1,z1,radius0,radius1,viscosity,length,full_volume,resistance,flow,shear0,shear1,normalized_length,minimum_scale,maximum_scale,minimum_normalized_radius,maximum_normalized_radius,normalization_complete,has_normalized_part,inside_length,exterior_length\n";
        for (const auto &[id, v] : r.network.vessels)
          if (v.owner == rank)
            {
              out << id << ',' << v.nodes[0] << ',' << v.nodes[1] << ','
                  << system_name(v.system) << ',';
              point(out, r.network.nodes.at(v.nodes[0]).position);
              out << ',';
              point(out, r.network.nodes.at(v.nodes[1]).position);
              out << ',' << v.radii[0] << ',' << v.radii[1] << ','
                  << v.viscosity << ',' << length(r.network, id) << ','
                  << volume(r.network, id) << ',' << resistance(r.network, id)
                  << ',';
              const auto it = r.reference.vessels.find(id);
              if (it != r.reference.vessels.end())
                out << it->second.flow << ',' << it->second.wall_shear[0] << ','
                    << it->second.wall_shear[1];
              else
                out << ",,";
              const auto &measure = r.normalized_geometry.at(id);
              out << ',' << measure.normalized_length << ','
                  << measure.minimum_scale << ',' << measure.maximum_scale
                  << ',' << measure.minimum_normalized_radius << ','
                  << measure.maximum_normalized_radius << ','
                  << measure.complete << ',' << measure.has_normalized_part
                  << ',' << measure.inside_length << ','
                  << measure.exterior_length;
              out << '\n';
            }
      }
      {
        auto out = output(name("terminals"));
        out
          << "id,node,territory,system,x,y,z,demand,target_flow,actual_outflow\n";
        for (const auto &[id, t] : r.network.terminals)
          if (t.owner == rank)
            {
              out << id << ',' << t.node << ',' << t.territory << ','
                  << system_name(t.system) << ',';
              point(out, r.network.nodes.at(t.node).position);
              out << ',' << t.demand << ',';
              if (r.target_terminal_flows.count(id))
                out << r.target_terminal_flows.at(id);
              out << ',';
              if (r.reference.terminal_outflows.count(id))
                out << r.reference.terminal_outflows.at(id);
              out << '\n';
            }
      }
      {
        auto out = output(name("territories"));
        out << "id,x,y,z,demand,flow,flow_per_demand\n";
        for (const auto &t : r.territories)
          {
            out << t.id << ',';
            point(out, t.position);
            out << ',' << t.demand << ',';
            if (r.reference.territories.count(t.id))
              {
                const auto &v = r.reference.territories.at(t.id);
                out << v.flow << ',' << v.flow_per_demand;
              }
            else
              out << ',';
            out << '\n';
          }
      }
      {
        auto out = output(name("pairs"));
        out << "territory,arterial_terminal,venous_terminal,resistance,flow\n";
        for (const auto &p : r.terminal_pairs)
          {
            out << p.territory << ',' << p.arterial << ',' << p.venous << ','
                << p.resistance << ',';
            if (r.reference.territories.count(p.territory))
              out << r.reference.territories.at(p.territory).flow;
            out << '\n';
          }
      }
      {
        auto out = output(name("intervals"));
        out
          << "vessel,begin,end,x0,y0,z0,x1,y1,z1,length,h_min,h_max,length_over_h,inside,short_cell,execution_owner\n";
        for (const auto &c : r.discretization.cells)
          {
            const auto &v    = r.network.vessels.at(c.reference.vessel);
            const auto &a    = r.network.nodes.at(v.nodes[0]).position,
                       &b    = r.network.nodes.at(v.nodes[1]).position;
            const Point<3> p = a + c.reference.begin * (b - a),
                           q = a + c.reference.end * (b - a);
            out << c.reference.vessel << ',' << c.reference.begin << ','
                << c.reference.end << ',';
            point(out, p);
            out << ',';
            point(out, q);
            out << ',' << p.distance(q) << ',' << c.h_min << ',' << c.h_max
                << ',';
            if (c.inside && c.h_min > 0.)
              out << p.distance(q) / c.h_min;
            out << ',' << c.inside << ',' << c.short_cell << ',' << c.owner
                << '\n';
          }
      }
      {
        auto out = output(name("background"));
        out << "cell,x,y,z,h,volume";
        for (unsigned int v = 0; v < 8; ++v)
          out << ",x" << v << ",y" << v << ",z" << v;
        out << '\n';
        auto quadrature = output(name("quadrature"));
        quadrature << "cell,x,y,z,weight\n";
        FE_Nothing<3> fe;
        FEValues<3>   values(mapping,
                           fe,
                           QGauss<3>(2),
                           update_JxW_values | update_quadrature_points);
        for (const auto &cell : bg.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              values.reinit(cell);
              double cell_volume = 0.;
              for (double jxw : values.get_JxW_values())
                cell_volume += jxw;
              out << std::quoted(cell->id().to_string()) << ',';
              point(out, cell->center());
              out << ',' << cell->diameter() << ',' << cell_volume;
              for (unsigned int v = 0; v < 8; ++v)
                {
                  out << ',';
                  point(out, cell->vertex(v));
                }
              out << '\n';
              for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
                {
                  quadrature << std::quoted(cell->id().to_string()) << ',';
                  point(quadrature, values.quadrature_point(q));
                  quadrature << ',' << values.JxW(q) << '\n';
                }
            }
      }
    }
  } // namespace

  GenerationResult
  run(const ApplicationParameters &p, MPI_Comm comm)
  {
    using namespace dealii;
    const auto rank = Utilities::MPI::this_mpi_process(comm),
               size = Utilities::MPI::n_mpi_processes(comm);
    std::vector<std::string> strings{
      p.mode, p.input_archive, p.output_directory, p.shape, p.refinement_focus};
    std::vector<double> controls{p.cube_side,
                                 p.sphere_radius,
                                 static_cast<double>(p.uniform_refinements),
                                 static_cast<double>(p.edge_axis),
                                 static_cast<double>(p.evaluate_reference),
                                 p.loaded_reference_flow,
                                 static_cast<double>(p.save_archive),
                                 static_cast<double>(p.write_tables),
                                 static_cast<double>(p.write_meshes)};
    for (const auto &x : {p.cube_origin, p.sphere_center})
      for (unsigned int d = 0; d < 3; ++d)
        controls.push_back(x[d]);
    for (auto x : p.edge_sides)
      controls.push_back(x);
    // Vector boundaries are part of the controls, not just their concatenation.
    controls.push_back(p.edge_sides.size());
    controls.insert(controls.end(),
                    p.refinement_fractions.begin(),
                    p.refinement_fractions.end());
    bool       valid  = true;
    const auto config = std::make_pair(strings, controls);
    for (const auto &other : Utilities::MPI::all_gather(comm, config))
      valid = other == config && valid;
    require_all(
      valid,
      comm,
      "All processes must use the same vascular application configuration.");
    valid =
      (p.mode == "generate" || p.mode == "complete" || p.mode == "load") &&
      (p.shape == "cube" || p.shape == "sphere") &&
      (p.refinement_focus == "none" ||
       (p.shape == "cube" && p.refinement_focus == "edge") ||
       (p.shape == "sphere" && p.refinement_focus == "center")) &&
      std::isfinite(p.cube_side) && p.cube_side > 0. &&
      std::isfinite(p.sphere_radius) && p.sphere_radius > 0. &&
      std::isfinite(p.loaded_reference_flow) && p.loaded_reference_flow >= 0. &&
      p.edge_axis < 3 && p.edge_sides.size() == 2 &&
      !p.output_directory.empty() &&
      fs::path(p.output_directory).is_absolute() &&
      ((p.mode == "generate" && p.input_archive.empty()) ||
       (p.mode != "generate" && !p.input_archive.empty() &&
        fs::path(p.input_archive).is_absolute()));
    for (auto side : p.edge_sides)
      valid = side <= 1 && valid;
    for (double f : p.refinement_fractions)
      valid = std::isfinite(f) && f > 0. && f <= 1. && valid;
    for (unsigned int d = 0; d < 3; ++d)
      valid = std::isfinite(p.cube_origin[d]) &&
              std::isfinite(p.sphere_center[d]) && valid;
    require_all(
      valid,
      comm,
      "Invalid vascular mode, domain, refinement distances or absolute input/output paths.");
    Timer                                   timer;
    parallel::distributed::Triangulation<3> bg(comm);
    MappingQ1<3>                            mapping;
    if (p.shape == "cube")
      {
        GridGenerator::hyper_cube(bg, 0., p.cube_side);
        GridTools::shift(p.cube_origin, bg);
      }
    else if (p.shape == "sphere")
      GridGenerator::hyper_ball(bg, p.sphere_center, p.sphere_radius);
    bg.refine_global(p.uniform_refinements);
    if (p.refinement_focus != "none")
      for (double f : p.refinement_fractions)
        {
          for (const auto &cell : bg.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                double distance = 0.;
                if (p.refinement_focus == "center")
                  distance = cell->center().distance(p.sphere_center);
                else if (p.refinement_focus == "edge")
                  {
                    unsigned int side = 0;
                    for (unsigned int d = 0; d < 3; ++d)
                      if (d != p.edge_axis)
                        {
                          const double edge =
                            p.cube_origin[d] +
                            p.edge_sides[side++] * p.cube_side;
                          distance += std::pow(cell->center()[d] - edge, 2);
                        }
                    distance = std::sqrt(distance);
                  }
                if (distance <
                    f * (p.shape == "cube" ? p.cube_side : p.sphere_radius))
                  cell->set_refine_flag();
              }
          bg.execute_coarsening_and_refinement();
        }
    const double background_seconds = timer.wall_time();
    const auto   scale = mesh_length_scale(bg, p.generation.length_scale, comm);
    const double halo =
      scale.physical_length(p.generation.clearance, scale.maximum());
    NetworkArchive initial;
    if (p.mode != "generate")
      initial = load_network(p.input_archive, bg, mapping, halo, comm);
    const double     input_seconds = timer.wall_time() - background_seconds;
    GenerationResult result;
    if (p.mode == "generate")
      result = generate(bg, mapping, {}, p.generation, comm);
    else if (p.mode == "complete")
      result = generate(bg,
                        mapping,
                        initial.network,
                        p.generation,
                        comm,
                        initial.territories,
                        initial.terminal_pairs);
    else if (p.mode == "load")
      {
        result.length_scale   = scale;
        result.network        = std::move(initial.network);
        result.territories    = std::move(initial.territories);
        result.terminal_pairs = std::move(initial.terminal_pairs);
        result.discretization = discretize(
          bg, mapping, result.network, p.generation.discretization, comm);
        result.validation = result.discretization.validation;
        if (valid_all(result.validation, comm))
          result.tissue = measure_tissue(bg,
                                         mapping,
                                         result.network,
                                         result.discretization,
                                         p.generation.coverage,
                                         comm,
                                         scale);
        if (valid_all(result.validation, comm))
          result.normalized_geometry =
            measure_scaled_vessels(bg, mapping, result.network, scale, comm);
        result.effective_terminal_count = Utilities::MPI::sum(
          static_cast<unsigned int>(result.territories.size()), comm);
      }
    const double generation_seconds =
      timer.wall_time() - background_seconds - input_seconds;
    bool reference_evaluated = p.mode != "load" &&
                               p.generation.venous.enabled &&
                               result.validation.valid();
    Id local_nodes = 0;
    for (const auto &[id, node] : result.network.nodes)
      local_nodes += node.owner == rank;
    const Id global_nodes = Utilities::MPI::sum(local_nodes, comm);
    if (valid_all(result.validation, comm) && p.evaluate_reference &&
        !reference_evaluated && global_nodes > 0)
      {
        auto bc          = p.generation.reference_conditions;
        bool valid_loads = true;
        if (p.mode == "load" && p.loaded_reference_flow > 0.)
          {
            require_all(
              Utilities::MPI::sum(static_cast<Id>(result.terminal_pairs.size()),
                                  comm) == 0,
              comm,
              "Loaded demand-flow loads apply to arterial networks without paired beds.");
            const auto loads = demand_flow_conditions(
              result.network, {}, p.loaded_reference_flow, {}, comm);
            for (const auto &[id, flow] : loads.inflows)
              valid_loads = bc.inflows.emplace(id, flow).second && valid_loads;
          }
        else if (p.mode != "load")
          for (const auto &[id, flow] : result.target_terminal_flows)
            {
              const auto &t           = result.network.terminals.at(id);
              double      signed_flow = 0.;
              if (t.system == System::arterial)
                signed_flow = -flow;
              else if (t.system == System::venous)
                signed_flow = flow;
              valid_loads =
                bc.inflows.emplace(t.node, signed_flow).second && valid_loads;
            }
        require_all(
          valid_loads,
          comm,
          "A requested tissue-flow load cannot also have an independent external inflow.");
        result.reference = solve_reference(result.network,
                                           result.terminal_pairs,
                                           bc,
                                           p.generation.reference_parameters,
                                           comm);
        result.validation.issues.insert(
          result.validation.issues.end(),
          result.reference.validation.issues.begin(),
          result.reference.validation.issues.end());
        reference_evaluated = true;
      }
    valid_all(result.validation, comm);
    const double reference_seconds = timer.wall_time() - background_seconds -
                                     input_seconds - generation_seconds;
    const Id numerical_cells =
      Utilities::MPI::sum(static_cast<Id>(result.discretization.cells.size()),
                          comm);
    parallel::fullydistributed::Triangulation<1, 3> line(comm);
    if (result.validation.valid() && p.write_meshes && numerical_cells > 0)
      create_triangulation(result.network, result.discretization, line);
    const double mesh_seconds = timer.wall_time() - background_seconds -
                                input_seconds - generation_seconds -
                                reference_seconds;
    const fs::path directory(p.output_directory);
    valid = true;
    try
      {
        valid = !fs::exists(directory);
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    require_all(valid,
                comm,
                "Vascular output directory must not already exist.");
    if (rank == 0)
      try
        {
          fs::create_directories(directory.parent_path());
          valid = fs::create_directory(directory);
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(valid, comm, "Cannot create vascular output directory.");
    if (result.validation.valid() && p.save_archive)
      save_network(directory / "network",
                   result.network,
                   result.territories,
                   result.terminal_pairs,
                   comm);
    try
      {
        {
          auto stats =
            output(directory / ("growth-" + std::to_string(rank) + ".csv"));
          stats
            << "system,requested_terminals,first_terminals,last_terminals,batches,sampled_points,proposed_insertions,accepted_insertions,committed_batches,rejected_batches,active_writer_batches,owned_vessels_first,owned_vessels_last,owned_vessels_maximum,initialization_seconds,sampling_seconds,search_seconds,optimization_seconds,selection_seconds,commit_seconds,redistribution_seconds,physiology_seconds,certification_seconds,total_seconds\n";
          for (const auto &s : result.growth_statistics)
            stats << system_name(s.system) << ',' << s.requested_terminals
                  << ',' << s.first_terminals << ',' << s.last_terminals << ','
                  << s.batches << ',' << s.sampled_points << ','
                  << s.proposed_insertions << ',' << s.accepted_insertions
                  << ',' << s.committed_batches << ',' << s.rejected_batches
                  << ',' << s.active_writer_batches << ','
                  << s.owned_vessels_first << ',' << s.owned_vessels_last << ','
                  << s.owned_vessels_maximum << ',' << s.initialization_seconds
                  << ',' << s.sampling_seconds << ',' << s.search_seconds << ','
                  << s.optimization_seconds << ',' << s.selection_seconds << ','
                  << s.commit_seconds << ',' << s.redistribution_seconds << ','
                  << s.physiology_seconds << ',' << s.certification_seconds
                  << ',' << s.total_seconds << '\n';
        }
        if (result.validation.valid() && p.write_tables)
          tables(directory, result, bg, mapping, rank);
        if (result.validation.valid() && p.write_meshes)
          {
            GridOut out;
            out.write_mesh_per_processor_as_vtu(
              bg, (directory / "background").string());
            if (numerical_cells > 0)
              out.write_mesh_per_processor_as_vtu(
                line, (directory / "line").string());
          }
        auto errors =
          output(directory / ("errors-" + std::to_string(rank) + ".txt"));
        for (const auto &issue : result.validation.issues)
          errors << issue.entity << ' ' << issue.id << ": " << issue.message
                 << '\n';
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    require_all(valid,
                comm,
                "Cannot write vascular tables, meshes or diagnostics.");
    Utilities::System::MemoryStats memory{};
    Utilities::System::get_memory_stats(memory);
    std::vector<Id> counts(12);
    counts[0] = bg.n_locally_owned_active_cells();
    counts[1] = local_nodes;
    for (const auto &[id, v] : result.network.vessels)
      counts[2] += v.owner == rank;
    for (const auto &[id, t] : result.network.terminals)
      if (t.owner == rank)
        {
          if (t.system == System::arterial)
            ++counts[3];
          else if (t.system == System::venous)
            ++counts[4];
        }
    counts[5] = result.territories.size();
    counts[6] = result.terminal_pairs.size();
    counts[7] = result.discretization.cells.size();
    counts[8] =
      result.validation.valid() && p.write_meshes && numerical_cells > 0 ?
        line.n_locally_owned_active_cells() :
        0;
    counts[9]                   = result.network.nodes.size() - counts[1];
    counts[10]                  = memory.VmRSS;
    counts[11]                  = memory.VmHWM;
    const double output_seconds = timer.wall_time() - background_seconds -
                                  input_seconds - generation_seconds -
                                  reference_seconds - mesh_seconds;
    std::vector<double> times{background_seconds,
                              input_seconds,
                              generation_seconds,
                              reference_seconds,
                              mesh_seconds,
                              output_seconds,
                              timer.wall_time()};
    const auto          rows =
      Utilities::MPI::all_gather(comm, std::make_pair(counts, times));
    if (rank == 0)
      try
        {
          std::vector<Id>     total(12);
          std::vector<double> maximum(7);
          for (const auto &row : rows)
            {
              for (unsigned int i = 0; i < 12; ++i)
                total[i] += row.first[i];
              for (unsigned int i = 0; i < 7; ++i)
                maximum[i] = std::max(maximum[i], row.second[i]);
            }
          auto out = output(directory / "summary.csv");
          out
            << "valid,mode,processes,background_cells,nodes,vessels,arterial_terminals,venous_terminals,territories,pairs,numerical_cells,background_volume,inside_volume,volume_fraction,target_volume_fraction,arterial_coverage,venous_coverage,dual_coverage,coverage_evaluated,minimum_coverage,reference_evaluated,reference_flow,reference_power,maximum_flow_defect,calibration_status,calibration_attempts,sampled_points,accepted_insertions,background_seconds,input_seconds,generation_seconds,additional_reference_seconds,mesh_seconds,output_seconds,total_seconds,length_scale,mesh_h_min,mesh_h_max,base_terminal_count,initial_terminal_count,requested_terminal_count,effective_terminal_count\n";
          out << result.validation.valid() << ',' << p.mode << ',' << size;
          for (unsigned int i = 0; i < 8; ++i)
            out << ',' << total[i];
          out << ',' << result.tissue.background_volume << ','
              << result.tissue.inside_vessel_volume << ','
              << result.tissue.volume_fraction << ','
              << p.generation.calibration.volume_fraction << ','
              << result.tissue.arterial_coverage << ','
              << result.tissue.venous_coverage << ','
              << result.tissue.dual_coverage << ','
              << result.tissue.coverage_evaluated << ','
              << p.generation.calibration.minimum_coverage << ','
              << reference_evaluated << ','
              << (p.mode == "load" ? p.loaded_reference_flow :
                                     p.generation.reference_flow)
              << ',' << result.reference.dissipated_power << ','
              << result.reference.maximum_flow_defect << ','
              << calibration_name(result.calibration.status) << ','
              << result.calibration.attempts.size() << ','
              << result.sampled_points << ',' << result.accepted_insertions;
          for (double t : maximum)
            out << ',' << t;
          out << ',' << result.length_scale.name() << ','
              << result.length_scale.mesh_minimum << ','
              << result.length_scale.mesh_maximum << ','
              << result.base_terminal_count << ','
              << result.initial_terminal_count << ','
              << result.requested_terminal_count << ','
              << result.effective_terminal_count;
          out << '\n';
          auto per_rank = output(directory / "ranks.csv");
          per_rank
            << "rank,background_cells,nodes,vessels,arterial_terminals,venous_terminals,territories,pairs,physical_owner_intervals,line_owned_cells,ghost_nodes,rss_kb,peak_rss_kb,background_seconds,input_seconds,generation_seconds,additional_reference_seconds,mesh_seconds,output_seconds,total_seconds\n";
          for (unsigned int r = 0; r < size; ++r)
            {
              per_rank << r;
              for (auto n : rows[r].first)
                per_rank << ',' << n;
              for (auto t : rows[r].second)
                per_rank << ',' << t;
              per_rank << '\n';
            }
          auto attempts = output(directory / "calibration.csv");
          attempts
            << "terminals,valid,volume_fraction,arterial_coverage,dual_coverage,sampled_points\n";
          for (const auto &a : result.calibration.attempts)
            attempts << a.terminal_count << ','
                     << a.generation_validation.valid() << ','
                     << a.tissue.volume_fraction << ','
                     << a.tissue.arterial_coverage << ','
                     << a.tissue.dual_coverage << ',' << a.sampled_points
                     << '\n';
        }
      catch (const std::exception &)
        {
          valid = false;
        }
    require_all(valid, comm, "Cannot write vascular global reports.");
    return result;
  }
} // namespace ImmersX::Vascularization
