// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_batch_h
#define immersx_internal_growth_batch_h
#include <deal.II/base/mpi.h>

#include <functional>

#include "growth_geometry.h"
namespace ImmersX::Vascularization
{
  struct Parameters;
  namespace internal
  {
    using ProposalKey = std::pair<unsigned int, Id>;
    struct BatchProposal
    {
      Id     id = invalid_id; // Unique on the submitting process in this batch.
      double cost = 0.;
      std::vector<Edge>               changed;
      std::map<Id, unsigned int>      owners, port_owners;
      std::map<Id, std::array<Id, 2>> old_nodes;
      std::map<Id, dealii::Point<3>>  normals;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &id &cost &changed &owners &port_owners &old_nodes &normals;
      }
    };
    struct BatchSelection
    {
      std::map<Id, bool>          local;
      std::map<ProposalKey, bool> relevant;
    };
    // Validate a common snapshot on the background's communicator. Newly
    // allocated physical ids must be globally unique. Geometry and write
    // conflicts use one deterministic independent round; this deliberately does
    // not promise a maximal independent set. No accepted network/history is
    // changed by a proposal or a vote.
    BatchSelection
    select_growth_batch(BackgroundGeometry               &geometry,
                        const NetworkData                &network,
                        const std::vector<BatchProposal> &local,
                        const Parameters                 &parameters,
                        MPI_Comm                          communicator,
                        const std::function<double(Id)>  &radius);
  } // namespace internal
} // namespace ImmersX::Vascularization
#endif
