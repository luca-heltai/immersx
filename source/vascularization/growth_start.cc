// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_start.h"

#include <deal.II/base/function_parser.h>

#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include <immersx/vascularization/distribution.h>

#include "growth_batch.h"
namespace ImmersX::Vascularization::internal
{
  std::vector<dealii::Point<3>>
  initial_root_directions(
    const dealii::Point<3>                &position,
    const dealii::Point<3>                &cell_center,
    const std::vector<dealii::Point<3>>   &cell_vertices,
    const dealii::Point<3>                &explicit_direction,
    const std::optional<dealii::Point<3>> &incoming_direction,
    const Territory                       *shared,
    double                                 separation)
  {
    using namespace dealii;
    std::vector<Point<3>> directions;
    if (shared)
      for (unsigned int i = 0; i < 6; ++i)
        {
          auto               end  = shared->position;
          const unsigned int axis = (i + shared->id) % 6;
          end[axis / 2] += (axis % 2 ? 1. : -1.) * separation;
          directions.emplace_back(end - position);
        }
    else if (explicit_direction.norm() > 0.)
      directions.emplace_back(explicit_direction);
    else
      {
        if (incoming_direction)
          directions.emplace_back(*incoming_direction);
        directions.emplace_back(cell_center - position);
        for (const auto &vertex : cell_vertices)
          directions.emplace_back(.5 * (vertex + cell_center) - position);
        for (unsigned int d = 0; d < 3; ++d)
          for (const double sign : {-1., 1.})
            {
              Point<3> direction;
              direction[d] = sign;
              directions.emplace_back(direction);
            }
      }
    return directions;
  }

  namespace
  {
    struct SeedCell
    {
      dealii::CellId                id;
      double                        diameter = 0.;
      dealii::Point<3>              center;
      std::vector<dealii::Point<3>> vertices;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &diameter &center &vertices;
      }
    };
  } // namespace

  GrowthCommit
  start_seed(const dealii::Triangulation<3>  &bg,
             const dealii::Mapping<3>        &mapping,
             const Parameters                &p,
             const dealii::FunctionParser<3> &demand,
             MPI_Comm                         comm)
  {
    using namespace dealii;
    const auto rank        = Utilities::MPI::this_mpi_process(comm);
    const auto size        = Utilities::MPI::n_mpi_processes(comm);
    const auto require_all = [&](bool valid, const std::string &message) {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage(message));
    };
    std::vector<double> controls{p.seed_radius,
                                 p.seed_length,
                                 p.viscosity,
                                 p.minimum_radius,
                                 p.minimum_length,
                                 p.minimum_angle,
                                 p.clearance,
                                 static_cast<double>(p.geometry_depth),
                                 static_cast<double>(p.geometry_work),
                                 p.arterial_murray_exponent,
                                 p.reference_flow,
                                 p.dissipation_weight};
    for (const auto &point :
         {p.seed_position, p.seed_direction, p.seed_inward_port_normal})
      for (unsigned int d = 0; d < 3; ++d)
        controls.push_back(point[d]);
    const auto local_controls = std::make_pair(controls, p.tissue_demand);
    bool       valid          = true;
    for (const auto &other : Utilities::MPI::all_gather(comm, local_controls))
      valid = other == local_controls && valid;
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(bg.get_mpi_communicator(), comm, &comparison);
    const auto *q  = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *fe = dynamic_cast<const MappingFE<3> *>(&mapping);
    valid = valid && (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
            ((q && q->get_degree() == 1) || (fe && fe->get_degree() == 1));
    for (const auto &cell : bg.active_cell_iterators())
      if (cell->is_locally_owned())
        valid = (cell->reference_cell().is_simplex() ||
                 cell->reference_cell().is_hyper_cube()) &&
                mapping.is_compatible_with(cell->reference_cell()) && valid;
    require_all(
      valid,
      "Seed writers require matching controls, a first-order background mapping and congruent communicators.");
    GrowthCommit result;
    const auto   fail = [&](const std::string &message) {
      result.validation.issues.push_back({"seed", 0, message});
    };
    const auto normal = normalized_port_normal(p.seed_inward_port_normal);
    BackgroundGeometry    geometry(bg, mapping);
    std::vector<SeedCell> local_cells;
    if (const auto cell =
          geometry.containing(p.seed_position,
                              normal.norm_square() > 0. ? 1.e-12 : 0.))
      {
        SeedCell hint;
        hint.id       = (*cell)->id();
        hint.diameter = (*cell)->diameter();
        hint.center   = (*cell)->center();
        for (const auto v : (*cell)->vertex_indices())
          hint.vertices.push_back((*cell)->vertex(v));
        local_cells.push_back(std::move(hint));
      }
    const auto cells = Utilities::MPI::all_gather(comm, local_cells);
    std::optional<SeedCell> best;
    unsigned int            source = size;
    for (unsigned int r = 0; r < size; ++r)
      for (const auto &hint : cells[r])
        {
          source = std::min(source, r);
          if (!best || hint.diameter < best->diameter ||
              (hint.diameter == best->diameter && hint.id < best->id))
            best = hint;
        }
    if (!best)
      {
        fail("Seed position is outside the background.");
        return result;
      }
    if (normal.norm_square() > 0. &&
        Utilities::MPI::max(static_cast<unsigned int>(
                              geometry.planar_section(p.seed_position, normal)),
                            comm) == 0)
      {
        fail(
          "The declared planar port does not match a planar background boundary section.");
        return result;
      }
    // Only a real containing process submits the parameter-defined port.
    // Native ParticleHandler ownership then determines its physical writer.
    NetworkData port;
    if (rank == source)
      port.nodes[0] = {p.seed_position,
                       rank,
                       Port::arterial_inlet,
                       Provenance::generated,
                       normal};
    port                = redistribute(bg, mapping, port, 0., comm);
    const bool producer = port.nodes.count(0) && port.nodes.at(0).owner == rank;
    std::vector<Point<3>> directions;
    if (producer)
      directions = initial_root_directions(p.seed_position,
                                           best->center,
                                           best->vertices,
                                           p.seed_direction,
                                           {},
                                           nullptr,
                                           0.);
    const auto count =
      Utilities::MPI::max(static_cast<unsigned int>(directions.size()), comm);
    for (unsigned int i = 0; i < count; ++i)
      {
        std::vector<BatchProposal> proposals;
        Point<3>                   end;
        if (producer)
          {
            const double norm = directions[i].norm();
            if (norm > 0. && std::isfinite(norm))
              {
                end = p.seed_position + p.seed_length / norm * directions[i];
                if (end.distance(p.seed_position) > p.minimum_length)
                  {
                    BatchProposal proposal;
                    proposal.id = 0;
                    proposal.changed.push_back(
                      {0, {{0, 1}}, {{p.seed_position, end}}, p.seed_radius});
                    if (normal.norm_square() > 0.)
                      {
                        proposal.normals[0]     = normal;
                        proposal.port_owners[0] = rank;
                      }
                    proposals.push_back(std::move(proposal));
                  }
              }
          }
        const auto selected =
          select_growth_batch(geometry, port, proposals, p, comm, [](Id) {
            return 0.;
          });
        const bool accepted = selected.local.count(0) && selected.local.at(0);
        if (Utilities::MPI::max(static_cast<unsigned int>(accepted), comm) == 0)
          continue;
        double value        = 0.;
        bool   valid_demand = true;
        if (producer)
          {
            try
              {
                value = demand.value(end);
              }
            catch (const std::exception &)
              {
                valid_demand = false;
              }
            valid_demand = valid_demand && std::isfinite(value) && value > 0.;
          }
        require_all(
          valid_demand,
          "Tissue demand must be positive and finite at every initial terminal.");
        auto                 trial = port;
        std::set<Id>         mutable_owned;
        std::map<Id, double> roots;
        if (producer)
          {
            trial.nodes[1] = {end, rank, Port::none, Provenance::generated};
            Vessel vessel;
            vessel.nodes       = {{0, 1}};
            vessel.radii       = {{p.seed_radius, p.seed_radius}};
            vessel.viscosity   = p.viscosity;
            vessel.provenance  = Provenance::generated;
            vessel.owner       = rank;
            trial.vessels[0]   = vessel;
            trial.terminals[0] = {1, 0, System::arterial, rank, value};
            mutable_owned.insert(0);
            roots[0] = p.seed_radius;
          }
        auto committed = commit_growth_splits(
          bg, mapping, trial, mutable_owned, roots, {}, p, comm);
        if (committed.committed)
          return committed;
      }
    fail(
      "No admissible initial subtree direction was found within the deterministic geometry budget.");
    return result;
  }
} // namespace ImmersX::Vascularization::internal
