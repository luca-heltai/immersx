// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "tissue_integration.h"

#include <deal.II/base/function_parser.h>

#include <deal.II/fe/fe_nothing.h>

#include <boost/serialization/vector.hpp>

#include <immersx/compatibility/reference_cell.h>

#include <limits>
#include <set>

#include "growth_geometry.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    using namespace dealii;
    struct CellSamples
    {
      Triangulation<3>::active_cell_iterator cell;
      unsigned int                           copies = 1;
      Id                                     first  = 0;
      std::uint64_t                          count  = 0;
    };
    void
    require_all(bool valid, MPI_Comm comm, const std::string &message)
    {
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(valid), comm) ==
                    1,
                  ExcMessage(message));
    }
    void
    accumulate(TerritoryDemandWork &a, const TerritoryDemandWork &b)
    {
      a.nearest_queries += b.nearest_queries;
      a.nearest_hits += b.nearest_hits;
      a.influence_queries += b.influence_queries;
      a.influence_hits += b.influence_hits;
      a.reassigned_samples += b.reassigned_samples;
      a.added_samples += b.added_samples;
      a.removed_samples += b.removed_samples;
    }
  } // namespace
  struct TissueDemandIntegration::Journal
  {
    const State                  *origin = nullptr;
    std::uint64_t                 epoch = 0, local_count = 0, global_count = 0;
    Id                            next  = 0;
    bool                          valid = false;
    std::map<CellId, CellSamples> before, after;
  };
  struct TissueDemandIntegration::State
  {
    BackgroundGeometry           &geometry;
    TissueIntegrationParameters   parameters;
    MPI_Comm                      comm;
    FunctionParser<3>             density;
    std::map<CellId, CellSamples> cells;
    using NativeRule =
      std::pair<std::unique_ptr<FE_Nothing<3>>, std::unique_ptr<FEValues<3>>>;
    std::map<std::pair<bool, unsigned int>, NativeRule> rules;
    std::map<bool, std::uint64_t>                       base_points;
    std::unique_ptr<TerritoryDemand>                    territories;
    ValidationReport                                    validation;
    bool                                                exhausted = false;
    TissueIntegrationWork                               initial_work;
    std::uint64_t            local_count = 0, global_count = 0, epoch = 0;
    Id                       next = 0;
    std::shared_ptr<Journal> staged;
    Id                       previous_next = 0;
    std::uint64_t previous_local_count = 0, previous_global_count = 0;

    State(BackgroundGeometry                &g,
          const TissueIntegrationParameters &p,
          MPI_Comm                           c)
      : geometry(g)
      , parameters(p)
      , comm(c)
    {}

    std::uint64_t
    count(const CellSamples &record) const
    {
      std::uint64_t value =
        base_points.at(record.cell->reference_cell().is_simplex());
      for (unsigned int d = 0; d < 3; ++d)
        {
          if (value > parameters.maximum_points / record.copies)
            return static_cast<std::uint64_t>(parameters.maximum_points) + 1;
          value *= record.copies;
        }
      return value;
    }
    std::vector<TissueDemandSample>
    generate(const CellSamples &record)
    {
      const auto reference = record.cell->reference_cell();
      const auto key   = std::make_pair(reference.is_simplex(), record.copies);
      auto       found = rules.find(key);
      if (found == rules.end())
        {
          auto element = std::make_unique<FE_Nothing<3>>(reference);
          auto values  = std::make_unique<FEValues<3>>(
            geometry.mapping,
            *element,
            tissue_demand_quadrature(reference,
                                     parameters.quadrature_order,
                                     record.copies,
                                     parameters.maximum_points),
            update_quadrature_points | update_JxW_values);
          found =
            rules
              .emplace(key, NativeRule{std::move(element), std::move(values)})
              .first;
        }
      return sample_tissue_demand(record.cell, *found->second.second, density);
    }
  };
  TissueDemandIntegration::TissueDemandIntegration(
    BackgroundGeometry                &geometry,
    const std::string                 &density,
    const TissueIntegrationParameters &parameters,
    unsigned int                       requested_terminals,
    const std::vector<TissueCentre>   &owned_centres,
    MPI_Comm                           communicator,
    bool                               require_interior_centres)
    : state(std::make_unique<State>(geometry, parameters, communicator))
  {
    auto      &s        = *state;
    const auto controls = std::make_pair(
      std::array<unsigned int, 6>{
        {parameters.quadrature_order,
         parameters.samples_per_terminal,
         parameters.maximum_points,
         parameters.refinement_budget,
         requested_terminals,
         static_cast<unsigned int>(require_interior_centres)}},
      density);
    bool valid = true;
    for (const auto &other : Utilities::MPI::all_gather(communicator, controls))
      valid = valid && other == controls;
    try
      {
        parameters.validate();
        s.density.initialize("x,y,z", density, {});
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    int         comparison = MPI_UNEQUAL;
    const auto &bg         = geometry.cache.get_triangulation();
    MPI_Comm_compare(bg.get_mpi_communicator(), communicator, &comparison);
    valid = valid && requested_terminals > 0 && bg.n_levels() > 0 &&
            (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
    require_all(
      valid,
      communicator,
      "Tissue integration requires matching controls, a finite density expression and a prepared congruent background.");
    if (require_interior_centres)
      {
        std::map<Id, Point<3>> centre_queries;
        for (const auto &centre : owned_centres)
          centre_queries.emplace(centre.territory, centre.position);
        const auto centre_cells =
          geometry.locate_points(centre_queries, communicator);
        require_all(
          centre_cells.size() == owned_centres.size(),
          communicator,
          "Tissue territory centres must lie inside the prepared background.");
      }
    std::uint64_t local_base = 0;
    try
      {
        for (const auto &cell : bg.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto reference = cell->reference_cell();
              valid =
                valid && (reference.is_hyper_cube() || reference.is_simplex());
              if (!valid)
                continue;
              const bool simplex = reference.is_simplex();
              if (!s.base_points.count(simplex))
                s.base_points[simplex] =
                  compatibility::gauss_quadrature<3>(
                    reference, parameters.quadrature_order)
                    .size();
              local_base += s.base_points.at(simplex);
              s.cells.emplace(cell->id(), CellSamples{cell, 1, 0, 0});
            }
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    require_all(valid,
                communicator,
                "Tissue integration supports owned hexahedra and tetrahedra.");
    const auto base = Utilities::MPI::sum(local_base, communicator);
    require_all(base > 0,
                communicator,
                "Tissue integration requires positive background volume.");
    const std::uint64_t target =
      static_cast<std::uint64_t>(requested_terminals) *
      parameters.samples_per_terminal;
    std::uint64_t predicted = base;
    unsigned int  copies    = 1;
    while (predicted < target && predicted <= parameters.maximum_points / 8)
      {
        predicted *= 8;
        copies *= 2;
      }
    if (predicted < target || predicted > parameters.maximum_points)
      {
        s.exhausted = true;
        s.validation.issues.push_back(
          {"tissue integration budget",
           invalid_id,
           "The initial quadrature target exceeds the global integration point budget; the requested terminal count was not reduced."});
        return;
      }
    std::map<Id, TissueDemandSample> samples;
    try
      {
        for (auto &[id, record] : s.cells)
          {
            record.copies = copies;
            record.first  = s.next;
            record.count  = s.count(record);
            AssertThrow(record.count < invalid_id - s.next,
                        ExcMessage(
                          "Tissue quadrature identifier space exhausted."));
            const auto generated = s.generate(record);
            for (const auto &point : generated)
              samples.emplace(s.next++, point);
            s.local_count += generated.size();
          }
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    require_all(
      valid,
      communicator,
      "Tissue density and physical quadrature masses must be positive, finite and representable.");
    s.global_count = Utilities::MPI::sum(s.local_count, communicator);
    require_all(
      s.global_count == predicted,
      communicator,
      "Native tissue quadrature cardinality differs from the checked allocation budget.");
    s.initial_work.generated_points = s.local_count;
    s.initial_work.peak_points      = s.global_count;
    s.territories =
      std::make_unique<TerritoryDemand>(samples, owned_centres, communicator);
    s.initial_work.assignments = s.territories->initialization_work();
  }
  TissueDemandIntegration::~TissueDemandIntegration() = default;
  const ValidationReport &
  TissueDemandIntegration::validation() const
  {
    return state->validation;
  }
  bool
  TissueDemandIntegration::integration_budget_exhausted() const
  {
    return state->exhausted;
  }
  const TissueIntegrationWork &
  TissueDemandIntegration::initialization_work() const
  {
    return state->initial_work;
  }

  TissueDemandIntegration::Preview
  TissueDemandIntegration::preview(const std::vector<TissueCentre> &new_owned,
                                   std::uint64_t snapshot_revision)
  {
    auto &s = *state;
    require_all(
      !s.staged,
      s.comm,
      "A staged tissue integration update must be adopted or discarded before preview.");
    Preview result;
    result.validation                   = s.validation;
    result.integration_budget_exhausted = s.exhausted;
    const auto finish                   = [&]() {
      result.territory.demand.validation = result.validation;
      result.territory.demand.integration_budget_exhausted =
        result.integration_budget_exhausted;
      return std::move(result);
    };
    if (!s.validation.valid())
      return finish();
    auto journal          = std::make_shared<Journal>();
    journal->origin       = &s;
    journal->epoch        = s.epoch;
    journal->local_count  = s.local_count;
    journal->global_count = s.global_count;
    journal->next         = s.next;
    result.journal        = journal;
    TissueQuadratureChanges changes;
    const auto              exhaust = [&](const std::string &message) {
      result.integration_budget_exhausted = true;
      result.validation.issues.push_back(
        {"tissue integration budget", invalid_id, message});
    };
    for (unsigned int round = 0;; ++round)
      {
        result.territory =
          s.territories->preview(new_owned, snapshot_revision, changes);
        accumulate(result.work.assignments, result.territory.work);
        result.work.peak_points =
          std::max(result.work.peak_points, journal->global_count);
        result.validation = result.territory.validation;
        if (result.validation.valid())
          {
            journal->valid = true;
            return finish();
          }
        if (round == s.parameters.refinement_budget)
          {
            exhaust(
              "The territorial quadrature refinement budget was exhausted; unresolved territories retain zero mass and cannot be adopted.");
            return finish();
          }
        std::map<Id, Point<3>> queries;
        for (const auto &centre : result.territory.unresolved)
          queries.emplace(centre.territory, centre.position);
        const auto located = s.geometry.locate_points(queries, s.comm);
        bool       valid   = located.size() == queries.size();
        std::map<unsigned int, std::vector<CellId>> send;
        for (const auto &[id, cell] : located)
          send[cell.cell_owner].push_back(cell.id);
        std::set<CellId> refine;
        for (const auto &[source, ids] :
             Utilities::MPI::some_to_some(s.comm, send))
          for (const auto &id : ids)
            {
              valid = valid && s.cells.count(id);
              refine.insert(id);
            }
        require_all(
          valid,
          s.comm,
          "Unresolved territories must lie in an owned background cell.");
        std::uint64_t proposed_local = journal->local_count;
        Id            proposed_next  = journal->next;
        for (const auto &id : refine)
          {
            auto record =
              journal->after.count(id) ? journal->after.at(id) : s.cells.at(id);
            const auto previous = record.count;
            if (record.copies > std::numeric_limits<unsigned int>::max() / 2)
              {
                valid = false;
                continue;
              }
            record.copies *= 2;
            const auto next_count = s.count(record);
            valid = valid && next_count < invalid_id - proposed_next;
            if (valid)
              proposed_next += next_count;
            proposed_local = proposed_local - previous + next_count;
          }
        const auto proposed_global =
          Utilities::MPI::sum(proposed_local, s.comm);
        valid = valid && proposed_global <= s.parameters.maximum_points;
        if (Utilities::MPI::min(static_cast<unsigned int>(valid), s.comm) == 0)
          {
            exhaust(
              "Refining unresolved territories would exceed the global quadrature point or identifier budget; no samples were allocated and the terminal count was not reduced.");
            return finish();
          }
        try
          {
            for (const auto &id : refine)
              {
                const auto  accepted = s.cells.at(id);
                auto        found    = journal->after.find(id);
                CellSamples record =
                  found == journal->after.end() ? accepted : found->second;
                if (found == journal->after.end())
                  {
                    journal->before.emplace(id, accepted);
                    for (Id i = 0; i < accepted.count; ++i)
                      changes.removed.push_back(accepted.first + i);
                  }
                else
                  for (Id i = 0; i < record.count; ++i)
                    changes.added.erase(record.first + i);
                record.copies *= 2;
                record.first         = journal->next;
                record.count         = s.count(record);
                const auto generated = s.generate(record);
                for (const auto &point : generated)
                  changes.added.emplace(journal->next++, point);
                journal->after[id] = record;
                ++result.work.refined_cells;
                result.work.generated_points += generated.size();
              }
          }
        catch (const std::exception &)
          {
            valid = false;
          }
        if (Utilities::MPI::min(static_cast<unsigned int>(valid), s.comm) == 0)
          {
            result.validation.issues.push_back(
              {"tissue integration",
               invalid_id,
               "Refined tissue density and physical quadrature masses must be positive, finite and representable."});
            return finish();
          }
        journal->local_count  = proposed_local;
        journal->global_count = proposed_global;
        ++result.work.refinement_rounds;
      }
  }
  void
  TissueDemandIntegration::prepare_adopt(const Preview &preview)
  {
    auto       &s = *state;
    const auto &p = preview.journal;
    require_all(!s.staged && preview.validation.valid() && p &&
                  p->origin == &s && p->epoch == s.epoch && p->valid,
                s.comm,
                "Tissue integration staging requires a valid current preview.");
    s.territories->prepare_adopt(preview.territory);
    s.staged                = p;
    s.previous_next         = s.next;
    s.previous_local_count  = s.local_count;
    s.previous_global_count = s.global_count;
    for (const auto &[id, record] : p->after)
      s.cells.at(id) = record;
    s.next         = p->next;
    s.local_count  = p->local_count;
    s.global_count = p->global_count;
  }
  void
  TissueDemandIntegration::adopt()
  {
    auto &s = *state;
    AssertThrow(s.staged,
                ExcMessage("No tissue integration update is prepared."));
    s.territories->adopt();
    ++s.epoch;
    s.staged.reset();
  }
  void
  TissueDemandIntegration::discard()
  {
    auto &s = *state;
    if (!s.staged)
      return;
    s.territories->discard();
    for (const auto &[id, record] : s.staged->before)
      s.cells.at(id) = record;
    s.next         = s.previous_next;
    s.local_count  = s.previous_local_count;
    s.global_count = s.previous_global_count;
    s.staged.reset();
  }
  std::uint64_t
  TissueDemandIntegration::point_count() const
  {
    AssertThrow(!state->staged,
                ExcMessage("A staged integration is not an accepted state."));
    return state->global_count;
  }
  double
  TissueDemandIntegration::total_demand() const
  {
    AssertThrow(state->territories,
                ExcMessage("No valid tissue quadrature was constructed."));
    return state->territories->total_demand();
  }
  std::vector<TerminalDemandUpdate>
  TissueDemandIntegration::owned_demands() const
  {
    AssertThrow(state->territories,
                ExcMessage("No valid tissue quadrature was constructed."));
    return state->territories->owned_demands();
  }
} // namespace ImmersX::Vascularization::internal
