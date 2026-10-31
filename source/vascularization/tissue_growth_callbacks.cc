// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "tissue_growth_callbacks.h"

#include "tissue_integration.h"
namespace ImmersX::Vascularization::internal
{
  namespace
  {
    struct TissueGrowthState
    {
      BackgroundGeometry                      &geometry;
      const std::string                        density;
      const TissueIntegrationParameters        parameters;
      const unsigned int                       requested_terminals;
      const MPI_Comm                           comm;
      std::unique_ptr<TissueDemandIntegration> integration;
      struct Descriptor
      {
        std::vector<TissueCentre> local;
        bool                      valid = false;
      };
      std::map<GrowthProposalKey, Descriptor>         proposals;
      std::optional<TissueDemandIntegration::Preview> pending;
      std::optional<
        std::pair<GrowthProposalKey, TissueDemandIntegration::Preview>>
                    isolated;
      std::uint64_t snapshot = 0, next_snapshot = 0;
      bool          staged = false, ready = false;

      TissueGrowthState(BackgroundGeometry &background,
                        const Parameters   &configuration,
                        unsigned int        requested,
                        MPI_Comm            communicator)
        : geometry(background)
        , density(configuration.tissue_demand)
        , parameters(configuration.tissue_integration)
        , requested_terminals(requested)
        , comm(communicator)
      {}

      void
      require(bool local, const std::string &message) const
      {
        AssertThrow(dealii::Utilities::MPI::min(
                      static_cast<unsigned int>(local), comm) == 1,
                    dealii::ExcMessage(message));
      }
      GrowthDemandJointUpdate
      bootstrap(const NetworkData  &network,
                const std::set<Id> &leaves,
                std::uint64_t       revision)
      {
        require(!integration, "Tissue bootstrap may run only once.");
        const auto rank  = dealii::Utilities::MPI::this_mpi_process(comm);
        bool       valid = true;
        std::vector<TissueCentre> centres;
        for (Id id : leaves)
          {
            const auto terminal = network.terminals.find(id);
            if (terminal == network.terminals.end() ||
                terminal->second.owner != rank)
              {
                valid = false;
                continue;
              }
            const auto node = network.nodes.find(terminal->second.node);
            if (node == network.nodes.end() ||
                terminal->second.system != System::arterial)
              {
                valid = false;
                continue;
              }
            centres.push_back(
              {terminal->second.territory, id, node->second.position, rank});
          }
        require(valid,
                "Tissue bootstrap requires native arterial leaf terminals.");
        snapshot    = revision;
        integration = std::make_unique<TissueDemandIntegration>(
          geometry, density, parameters, requested_terminals, centres, comm);
        if (!integration->validation().valid())
          {
            GrowthDemandJointUpdate failure;
            failure.validation = integration->validation();
            failure.integration_budget_exhausted =
              integration->integration_budget_exhausted();
            failure.update.snapshot_revision = snapshot;
            return failure;
          }
        pending = integration->preview({}, snapshot);
        return pending->territory.demand;
      }
      GrowthDemandProposal
      preview(const GrowthProposalKey &key,
              Id                       terminal,
              Id                       territory,
              const dealii::Point<3>  &point,
              std::uint64_t            revision)
      {
        require(
          integration && ready && !staged && !pending && revision == snapshot,
          "Tissue preview requires an accepted, current physical snapshot.");
        const auto rank = dealii::Utilities::MPI::this_mpi_process(comm);
        const auto size = dealii::Utilities::MPI::n_mpi_processes(comm);
        require(
          key.first < size && (rank != key.first || key.second == terminal) &&
            !proposals.count(key),
          "Tissue proposal keys must identify a fresh allocated terminal.");
        terminal = key.second;
        Descriptor descriptor;
        if (rank == key.first)
          descriptor.local.push_back({territory, terminal, point, rank});
        // Retain at most one journal, including while computing the next trial.
        isolated.reset();
        auto trial = integration->preview(descriptor.local, snapshot);
        GrowthDemandProposal result;
        result.proposal_key    = key;
        result.new_terminal_id = terminal;
        result.update          = trial.territory.demand.update;
        result.validation      = trial.validation;
        result.integration_budget_exhausted =
          trial.integration_budget_exhausted;
        for (const auto &value : trial.territory.demand.new_terminals)
          if (value.terminal == terminal)
            result.new_demand = value.demand;
        descriptor.valid = trial.validation.valid();
        proposals.emplace(key, std::move(descriptor));
        if (trial.validation.valid())
          isolated.emplace(key, std::move(trial));
        return result;
      }
      GrowthDemandJointUpdate
      joint(const std::vector<GrowthProposalKey> &selected,
            std::uint64_t                         revision)
      {
        require(integration && ready && !staged && !pending &&
                  revision == snapshot,
                "Joint tissue preview requires the current physical snapshot.");
        // The physical selector may return local or replicated selected keys.
        // Only this small batch is replicated, never the tissue or forest.
        const auto keys = dealii::Utilities::MPI::all_gather(comm, selected);
        std::set<GrowthProposalKey> unique;
        for (const auto &part : keys)
          unique.insert(part.begin(), part.end());
        bool                      valid = true;
        std::vector<TissueCentre> centres;
        for (const auto &key : unique)
          {
            const auto found = proposals.find(key);
            if (found == proposals.end() || !found->second.valid)
              {
                valid = false;
                continue;
              }
            centres.insert(centres.end(),
                           found->second.local.begin(),
                           found->second.local.end());
          }
        require(valid,
                "Joint tissue preview must select valid isolated proposals.");
        // A singleton selects the same partition as its isolated preview.
        // A joint batch must recompute every affected old and new territory.
        const bool reuse =
          unique.size() == 1 && isolated && isolated->first == *unique.begin();
        if (dealii::Utilities::MPI::min(static_cast<unsigned int>(reuse),
                                        comm) == 1)
          pending = std::move(isolated->second);
        isolated.reset();
        if (!pending)
          pending = integration->preview(centres, snapshot);
        return pending->territory.demand;
      }
      void
      prepare(std::uint64_t quadrature, std::uint64_t revision)
      {
        require(
          pending && pending->validation.valid() && !staged &&
            pending->territory.demand.update.quadrature_revision ==
              quadrature &&
            revision == snapshot + 1,
          "Tissue adoption requires the joint quadrature and next physical revision.");
        integration->prepare_adopt(*pending);
        next_snapshot = revision;
        staged        = true;
      }
      void
      adopt()
      {
        Assert(staged, dealii::ExcInternalError());
        integration->adopt();
        snapshot = next_snapshot;
        ready    = true;
        staged   = false;
        pending.reset();
        isolated.reset();
        proposals.clear();
      }
      void
      discard()
      {
        if (staged)
          integration->discard();
        staged = false;
        pending.reset();
        isolated.reset();
        proposals.clear();
      }
    };
  } // namespace
  GrowthDemandCallbacks
  tissue_growth_callbacks(BackgroundGeometry &geometry,
                          const Parameters   &parameters,
                          unsigned int        requested_terminals,
                          MPI_Comm            communicator)
  {
    auto                  state = std::make_shared<TissueGrowthState>(geometry,
                                                     parameters,
                                                     requested_terminals,
                                                     communicator);
    GrowthDemandCallbacks callbacks;
    callbacks.bootstrap = [state](const NetworkData  &network,
                                  const std::set<Id> &leaves,
                                  std::uint64_t       revision) {
      return state->bootstrap(network, leaves, revision);
    };
    callbacks.preview = [state](const GrowthProposalKey &key,
                                Id                       terminal,
                                Id                       territory,
                                const dealii::Point<3>  &point,
                                std::uint64_t            revision) {
      return state->preview(key, terminal, territory, point, revision);
    };
    callbacks.joint = [state](const std::vector<GrowthProposalKey> &selected,
                              std::uint64_t                         revision) {
      return state->joint(selected, revision);
    };
    callbacks.prepare_adopt = [state](std::uint64_t quadrature,
                                      std::uint64_t revision) {
      state->prepare(quadrature, revision);
    };
    callbacks.adopt   = [state]() { state->adopt(); };
    callbacks.discard = [state]() { state->discard(); };
    return callbacks;
  }
} // namespace ImmersX::Vascularization::internal
