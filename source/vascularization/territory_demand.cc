// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "territory_demand.h"

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>

#include <deal.II/numerics/rtree.h>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_int/serialize.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/serialization/vector.hpp>

#include <cmath>
#include <limits>
#include <set>

#include "demand_mass.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using namespace dealii;
    namespace bgi    = boost::geometry::index;
    using Mass       = boost::multiprecision::cpp_int;
    using CentreLeaf = std::pair<Point<3>, Id>;
    using SampleLeaf = std::pair<BoundingBox<3>, Id>;

    bool
    valid_point(const Point<3> &point)
    {
      for (unsigned int d = 0; d < 3; ++d)
        if (!std::isfinite(point[d]))
          return false;
      return std::isfinite(point.norm_square());
    }
    bool
    prefer(const Point<3> &point, const TissueCentre &a, const TissueCentre &b)
    {
      const double da = point.distance_square(a.position),
                   db = point.distance_square(b.position);
      return da < db || (da == db && a.territory < b.territory);
    }
    void
    require_all(bool valid, MPI_Comm comm, const std::string &message)
    {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage(message));
    }
    struct Sample
    {
      TissueDemandSample point;
      TissueCentre       centre;
    };
    SampleLeaf
    influence(Id id, const Sample &sample)
    {
      const auto  &point = sample.point.position;
      const double radius =
        std::nextafter(point.distance(sample.centre.position),
                       std::numeric_limits<double>::infinity());
      const double guard =
        64. * std::numeric_limits<double>::epsilon() *
        std::max({point.norm(), sample.centre.position.norm(), radius});
      Point<3> lo = point, hi = point;
      for (unsigned int d = 0; d < 3; ++d)
        {
          lo[d] -= radius + guard;
          hi[d] += radius + guard;
        }
      return {BoundingBox<3>(std::make_pair(lo, hi)), id};
    }
    struct Demand
    {
      TissueCentre centre;
      Mass         mass;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &centre &mass;
      }
    };
    using Demands = std::map<Id, Demand>;
    void
    add_mass(Demands &values, const TissueCentre &centre, const Mass &mass)
    {
      const auto found = values.find(centre.terminal);
      if (found == values.end())
        values.emplace(centre.terminal, Demand{centre, mass});
      else
        found->second.mass += mass;
    }
    Mass
    global_mass(const Mass &local, MPI_Comm comm)
    {
      Mass total = 0;
      for (const auto &value : Utilities::MPI::all_gather(comm, local))
        total += value;
      return total;
    }
  } // namespace

  struct TerritoryDemand::Journal
  {
    const State              *origin = nullptr;
    bool                      valid  = false;
    std::uint64_t             epoch = 0, quadrature_revision = 0;
    std::map<Id, Sample>      before, after;
    Demands                   demands_before, demands_after;
    std::vector<TissueCentre> new_centres;
    Mass                      old_total, new_total;
  };

  struct TerritoryDemand::State
  {
    MPI_Comm                   comm;
    unsigned int               rank, size;
    std::uint64_t              epoch = 0, quadrature_revision = 0;
    std::map<Id, Sample>       samples;
    std::map<Id, TissueCentre> centres;
    RTree<CentreLeaf>          centre_index;
    RTree<SampleLeaf>          influence_index;
    Demands                    demands;
    Mass                       total;
    std::shared_ptr<Journal>   staged;

    State(MPI_Comm communicator)
      : comm(communicator)
      , rank(Utilities::MPI::this_mpi_process(comm))
      , size(Utilities::MPI::n_mpi_processes(comm))
    {}

    std::optional<TissueCentre>
    nearest(const Point<3> &point, TerritoryDemandWork &work) const
    {
      ++work.nearest_queries;
      std::optional<TissueCentre> best;
      if (centre_index.empty())
        return best;
      double cutoff = 0.;
      for (auto it =
             centre_index.qbegin(bgi::nearest(point, centre_index.size()));
           it != centre_index.qend();
           ++it)
        {
          ++work.nearest_hits;
          const auto  &centre   = centres.at(it->second);
          const double distance = point.distance_square(centre.position);
          const double guard =
            128. * std::numeric_limits<double>::epsilon() *
            std::max(
              {point.norm_square(), centre.position.norm_square(), distance});
          if (best && distance > cutoff + guard)
            break;
          if (!best || prefer(point, centre, *best))
            best = centre;
          if (best)
            cutoff = point.distance_square(best->position);
        }
      return best;
    }

    std::map<Id, TissueCentre>
    locate(const std::map<Id, Point<3>> &points,
           TerritoryDemandWork          &work) const
    {
      std::vector<BoundingBox<3>> local_boxes;
      if (!centres.empty())
        {
          // Native index bounds are maintained as centres are inserted.
          const auto bounds = centre_index.bounds();
          Point<3>   lo, hi;
          for (unsigned int d = 0; d < 3; ++d)
            {
              if (d == 0)
                {
                  lo[d] = boost::geometry::get<boost::geometry::min_corner, 0>(
                    bounds);
                  hi[d] = boost::geometry::get<boost::geometry::max_corner, 0>(
                    bounds);
                }
              else if (d == 1)
                {
                  lo[d] = boost::geometry::get<boost::geometry::min_corner, 1>(
                    bounds);
                  hi[d] = boost::geometry::get<boost::geometry::max_corner, 1>(
                    bounds);
                }
              else if (d == 2)
                {
                  lo[d] = boost::geometry::get<boost::geometry::min_corner, 2>(
                    bounds);
                  hi[d] = boost::geometry::get<boost::geometry::max_corner, 2>(
                    bounds);
                }
            }
          local_boxes.emplace_back(std::make_pair(lo, hi));
        }
      const auto boxes = Utilities::MPI::all_gather(comm, local_boxes);
      std::map<Id, TissueCentre>                     result;
      std::map<unsigned int, std::map<Id, Point<3>>> send;
      for (const auto &[id, point] : points)
        {
          const auto local = nearest(point, work);
          if (local)
            result[id] = *local;
          const double reach = local ? point.distance(local->position) :
                                       std::numeric_limits<double>::infinity();
          for (unsigned int owner = 0; owner < size; ++owner)
            if (owner != rank)
              for (const auto &box : boxes[owner])
                {
                  const double guard =
                    128. * std::numeric_limits<double>::epsilon() *
                    std::max({point.norm(),
                              box.get_boundary_points().first.norm(),
                              box.get_boundary_points().second.norm(),
                              std::isfinite(reach) ? reach : 0.});
                  if (box.signed_distance(point) <= reach + guard)
                    send[owner].emplace(id, point);
                }
        }
      std::map<unsigned int, std::map<Id, TissueCentre>> replies;
      for (const auto &[source, queries] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[id, point] : queries)
          if (const auto best = nearest(point, work))
            replies[source].emplace(id, *best);
      for (const auto &[source, values] :
           Utilities::MPI::some_to_some(comm, replies))
        for (const auto &[id, centre] : values)
          if (!result.count(id) || prefer(points.at(id), centre, result.at(id)))
            result[id] = centre;
      bool finite = result.size() == points.size();
      for (const auto &[id, centre] : result)
        finite = finite &&
                 std::isfinite(points.at(id).distance_square(centre.position));
      require_all(
        finite,
        comm,
        "Every tissue sample requires a nearest common territory centre at a finite representable distance.");
      return result;
    }

    Demands
    route(const Demands &local) const
    {
      std::map<unsigned int, Demands> send;
      for (const auto &[terminal, demand] : local)
        send[terminal % size].emplace(terminal, demand);
      Demands result;
      for (const auto &[source, values] :
           Utilities::MPI::some_to_some(comm, send))
        for (const auto &[terminal, demand] : values)
          add_mass(result, demand.centre, demand.mass);
      return result;
    }
  };

  TerritoryDemand::TerritoryDemand(
    const std::map<Id, TissueDemandSample> &owned_samples,
    const std::vector<TissueCentre>        &owned_centres,
    MPI_Comm                                communicator)
    : state(std::make_unique<State>(communicator))
  {
    auto   &s     = *state;
    bool    valid = true;
    Demands registration;
    for (const auto &centre : owned_centres)
      {
        valid = valid && centre.owner == s.rank &&
                centre.territory != invalid_id &&
                centre.terminal != invalid_id && valid_point(centre.position) &&
                !s.centres.count(centre.territory);
        s.centres.emplace(centre.territory, centre);
        add_mass(registration, centre, Mass(0));
      }
    std::map<Id, Point<3>> points;
    for (const auto &[id, point] : owned_samples)
      {
        valid = valid && valid_point(point.position) &&
                std::isfinite(point.mass) && point.mass > 0.;
        points.emplace(id, point.position);
      }
    const auto global_centres =
      Utilities::MPI::sum(static_cast<std::uint64_t>(owned_centres.size()),
                          s.comm);
    require_all(
      valid && global_centres > 0,
      s.comm,
      "Tissue assignment requires unique owned centres and finite positive quadrature masses.");
    for (const bool by_territory : {true, false})
      {
        std::map<unsigned int, std::vector<Id>> send;
        for (const auto &centre : owned_centres)
          {
            const Id id = by_territory ? centre.territory : centre.terminal;
            send[id % s.size].push_back(id);
          }
        std::set<Id> seen;
        for (const auto &[source, ids] :
             Utilities::MPI::some_to_some(s.comm, send))
          for (Id id : ids)
            valid = seen.insert(id).second && valid;
      }
    require_all(
      valid,
      s.comm,
      "Initial territory and terminal identifiers must be globally unique.");
    std::vector<CentreLeaf> leaves;
    for (const auto &[id, centre] : s.centres)
      leaves.emplace_back(centre.position, id);
    s.centre_index = pack_rtree(leaves);
    TerritoryDemandWork     work;
    const auto              located = s.locate(points, work);
    std::vector<SampleLeaf> influences;
    Mass                    local_total = 0;
    for (const auto &[id, point] : owned_samples)
      {
        const Sample sample{point, located.at(id)};
        s.samples.emplace(id, sample);
        influences.push_back(influence(id, sample));
        const auto mass = demand_mass_units(point.mass);
        add_mass(registration, sample.centre, mass);
        local_total += mass;
      }
    s.influence_index = pack_rtree(influences);
    s.demands         = s.route(registration);
    s.total           = global_mass(local_total, s.comm);
    require_all(
      std::isfinite(physical_demand_mass(s.total)) &&
        physical_demand_mass(s.total) > 0.,
      s.comm,
      "Total integrated demand must be positive, finite and representable.");
  }
  TerritoryDemand::~TerritoryDemand()                           = default;
  TerritoryDemand::TerritoryDemand(TerritoryDemand &&) noexcept = default;
  TerritoryDemand &
  TerritoryDemand::operator=(TerritoryDemand &&) noexcept = default;

  TerritoryDemand::Preview
  TerritoryDemand::preview(const std::vector<TissueCentre> &new_owned,
                           std::uint64_t                    snapshot_revision,
                           const TissueQuadratureChanges   &quadrature) const
  {
    auto &s = *state;
    require_all(
      !s.staged,
      s.comm,
      "A staged tissue update must be adopted or discarded before another preview.");
    Preview result;
    auto    journal                 = std::make_shared<Journal>();
    journal->origin                 = &s;
    journal->epoch                  = s.epoch;
    journal->old_total              = s.total;
    journal->new_total              = s.total;
    journal->new_centres            = new_owned;
    result.journal                  = journal;
    auto &update                    = result.demand.update;
    update.snapshot_revision        = snapshot_revision;
    update.base_quadrature_revision = s.quadrature_revision;
    bool         valid              = true;
    std::set<Id> removed;
    for (Id id : quadrature.removed)
      valid = removed.insert(id).second && s.samples.count(id) && valid;
    for (const auto &[id, point] : quadrature.added)
      valid = valid && (!s.samples.count(id) || removed.count(id)) &&
              valid_point(point.position) && std::isfinite(point.mass) &&
              point.mass > 0.;
    for (const auto &centre : new_owned)
      valid = valid && centre.owner == s.rank &&
              centre.territory != invalid_id && centre.terminal != invalid_id &&
              valid_point(centre.position);
    require_all(
      valid,
      s.comm,
      "Malformed tissue preview, quadrature replacement or centre authority.");
    const std::array<std::uint64_t, 3> versions{
      {s.epoch, s.quadrature_revision, snapshot_revision}};
    const auto batches =
      Utilities::MPI::all_gather(s.comm, std::make_pair(versions, new_owned));
    std::vector<TissueCentre> proposed;
    std::set<Id>              proposed_ids, proposed_terminals;
    for (const auto &[other_versions, batch] : batches)
      {
        valid = valid && other_versions == versions;
        for (const auto &centre : batch)
          {
            valid = proposed_ids.insert(centre.territory).second &&
                    proposed_terminals.insert(centre.terminal).second &&
                    !s.centres.count(centre.territory) && valid;
            proposed.push_back(centre);
          }
      }
    require_all(
      valid,
      s.comm,
      "New territory and terminal identities must be globally unique and absent from the accepted centre set.");
    for (const auto &centre : proposed)
      {
        ++result.work.influence_queries;
        for (const auto &[box, id] :
             s.influence_index |
               bgi::adaptors::queried(bgi::intersects(centre.position)))
          {
            ++result.work.influence_hits;
            if (removed.count(id))
              continue;
            const auto &sample   = s.samples.at(id);
            const auto  changed  = journal->after.find(id);
            const auto &previous = changed == journal->after.end() ?
                                     sample.centre :
                                     changed->second.centre;
            if (prefer(sample.point.position, centre, previous))
              {
                journal->before.emplace(id, sample);
                journal->after[id] = {sample.point, centre};
              }
          }
      }
    const bool changed_quadrature =
      Utilities::MPI::max(static_cast<unsigned int>(
                            !quadrature.removed.empty() ||
                            !quadrature.added.empty()),
                          s.comm) > 0;
    std::map<Id, Point<3>> added_points;
    for (const auto &[id, point] : quadrature.added)
      added_points.emplace(id, point.position);
    const auto located = changed_quadrature ?
                           s.locate(added_points, result.work) :
                           std::map<Id, TissueCentre>{};
    for (Id id : removed)
      journal->before[id] = s.samples.at(id);
    for (const auto &[id, point] : quadrature.added)
      {
        auto best = located.at(id);
        for (const auto &centre : proposed)
          if (prefer(point.position, centre, best))
            best = centre;
        journal->after[id] = {point, best};
      }
    result.work.reassigned_samples = journal->before.size() - removed.size();
    result.work.removed_samples    = removed.size();
    result.work.added_samples      = quadrature.added.size();
    Demands changes;
    for (const auto &centre : new_owned)
      add_mass(changes, centre, Mass(0));
    Mass local_delta = 0;
    for (const auto &[id, sample] : journal->before)
      {
        const auto mass = demand_mass_units(sample.point.mass);
        add_mass(changes, sample.centre, -mass);
        if (removed.count(id))
          local_delta -= mass;
      }
    for (const auto &[id, sample] : journal->after)
      {
        const auto mass = demand_mass_units(sample.point.mass);
        add_mass(changes, sample.centre, mass);
        if (quadrature.added.count(id))
          local_delta += mass;
      }
    journal->quadrature_revision = s.quadrature_revision + changed_quadrature;
    update.quadrature_revision   = journal->quadrature_revision;
    if (changed_quadrature)
      journal->new_total += global_mass(local_delta, s.comm);
    Demands absolute = s.route(changes);
    for (auto &[terminal, demand] : absolute)
      {
        const auto old = s.demands.find(terminal);
        if (old != s.demands.end())
          {
            valid = valid && !proposed_terminals.count(terminal);
            journal->demands_before.emplace(terminal, old->second);
            demand.mass += old->second.mass;
          }
        else
          valid = valid && proposed_terminals.count(terminal);
        valid = valid && demand.mass >= 0;
        journal->demands_after.emplace(terminal, demand);
      }
    require_all(
      valid,
      s.comm,
      "Tissue demand changes must preserve registered terminal identities and nonnegative exact mass.");
    // Bootstrap may already contain unresolved centres. Ordinary previews need
    // only inspect changed demands; this scan occurs only until initialization
    // has been resolved by quadrature refinement and adopted once.
    if (s.epoch == 0)
      for (const auto &[terminal, demand] : s.demands)
        if (!absolute.count(terminal))
          absolute.emplace(terminal, demand);
    std::map<unsigned int, std::vector<TerminalDemandUpdate>> replies;
    for (const auto &[terminal, demand] : absolute)
      {
        const double mass = physical_demand_mass(demand.mass);
        valid             = valid && std::isfinite(mass);
        if (!(mass > 0.))
          result.unresolved.push_back(demand.centre);
        else if (proposed_terminals.count(terminal))
          replies[demand.centre.owner].push_back({terminal, mass});
        else if (s.epoch == 0 || demand.mass != s.demands.at(terminal).mass)
          update.terminals.push_back({terminal, mass});
      }
    const auto new_demands = Utilities::MPI::some_to_some(s.comm, replies);
    for (const auto &[source, values] : new_demands)
      result.demand.new_terminals.insert(result.demand.new_terminals.end(),
                                         values.begin(),
                                         values.end());
    const double total = physical_demand_mass(journal->new_total);
    valid              = valid && std::isfinite(total) && total > 0.;
    require_all(valid,
                s.comm,
                "Integrated demands must remain finite and representable.");
    update.canonical_total_demand = std::array<double, 2>{{total, 0.}};
    if (Utilities::MPI::sum(
          static_cast<std::uint64_t>(result.unresolved.size()), s.comm) > 0)
      result.validation.issues.push_back(
        {"tissue quadrature",
         invalid_id,
         "A territory has no quadrature mass; refine integration or report an exhausted integration budget."});
    journal->valid = result.validation.valid();
    return result;
  }

  void
  TerritoryDemand::prepare_adopt(const Preview &preview)
  {
    auto       &s = *state;
    const auto &p = preview.journal;
    require_all(
      !s.staged && preview.validation.valid() && p && p->valid &&
        p->origin == &s && p->epoch == s.epoch,
      s.comm,
      "Tissue staging requires a valid preview of the current accepted assignment.");
    s.staged = p;
    for (const auto &[id, sample] : p->before)
      {
        s.influence_index.remove(influence(id, sample));
        s.samples.erase(id);
      }
    for (const auto &[id, sample] : p->after)
      {
        s.samples.emplace(id, sample);
        s.influence_index.insert(influence(id, sample));
      }
    for (const auto &centre : p->new_centres)
      {
        s.centres.emplace(centre.territory, centre);
        s.centre_index.insert({centre.position, centre.territory});
      }
    for (const auto &[terminal, demand] : p->demands_after)
      s.demands[terminal] = demand;
    s.total = p->new_total;
  }
  void
  TerritoryDemand::adopt()
  {
    auto &s = *state;
    AssertThrow(s.staged,
                dealii::ExcMessage("No prepared tissue update to adopt."));
    s.quadrature_revision = s.staged->quadrature_revision;
    ++s.epoch;
    s.staged.reset();
  }
  void
  TerritoryDemand::discard()
  {
    auto &s = *state;
    if (!s.staged)
      return;
    const auto &p = s.staged;
    for (const auto &[id, sample] : p->after)
      {
        s.influence_index.remove(influence(id, sample));
        s.samples.erase(id);
      }
    for (const auto &[id, sample] : p->before)
      {
        s.samples.emplace(id, sample);
        s.influence_index.insert(influence(id, sample));
      }
    for (const auto &centre : p->new_centres)
      {
        s.centre_index.remove({centre.position, centre.territory});
        s.centres.erase(centre.territory);
      }
    for (const auto &[terminal, demand] : p->demands_after)
      {
        const auto old = p->demands_before.find(terminal);
        if (old == p->demands_before.end())
          s.demands.erase(terminal);
        else
          s.demands.at(terminal) = old->second;
      }
    s.total = p->old_total;
    s.staged.reset();
  }
  std::vector<TerminalDemandUpdate>
  TerritoryDemand::owned_demands() const
  {
    AssertThrow(!state->staged,
                dealii::ExcMessage(
                  "A staged tissue assignment is not an accepted state."));
    std::vector<TerminalDemandUpdate> result;
    for (const auto &[terminal, demand] : state->demands)
      result.push_back({terminal, physical_demand_mass(demand.mass)});
    return result;
  }
  std::map<Id, Id>
  TerritoryDemand::owned_assignments() const
  {
    AssertThrow(!state->staged,
                dealii::ExcMessage(
                  "A staged tissue assignment is not an accepted state."));
    std::map<Id, Id> result;
    for (const auto &[id, sample] : state->samples)
      result.emplace(id, sample.centre.territory);
    return result;
  }
  double
  TerritoryDemand::total_demand() const
  {
    AssertThrow(!state->staged,
                dealii::ExcMessage(
                  "A staged tissue assignment is not an accepted state."));
    return physical_demand_mass(state->total);
  }
  std::uint64_t
  TerritoryDemand::quadrature_revision() const
  {
    return state->quadrature_revision;
  }
} // namespace ImmersX::Vascularization::internal
