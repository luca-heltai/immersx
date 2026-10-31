// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_forest_state_h
#define immersx_internal_forest_state_h
#include <deal.II/base/mpi.h>

#include <immersx/vascularization/mesh_scale.h>
#include <immersx/vascularization/network.h>

#include <limits>
#include <set>
namespace ImmersX::Vascularization
{
  struct Parameters;
  namespace internal
  {
    struct ForestBranch
    {
      Id parent = invalid_id, root = invalid_id, terminal = invalid_id;
      unsigned int parent_owner   = dealii::numbers::invalid_unsigned_int;
      unsigned int terminal_owner = dealii::numbers::invalid_unsigned_int;
      std::map<Id, unsigned int> children;
      std::uint64_t              terminals = 0;
      double demand = 0., root_demand = 0., root_radius = 0., radius = 0.;
      double volume_moment = 0., resistance_moment = 0.;
      double root_volume_moment = 0., root_resistance_moment = 0.;
      double flow = 0., resistance = 0.;
      double minimum_demand      = std::numeric_limits<double>::infinity();
      double root_minimum_demand = 0.;
      template <class Archive>
      void
      serialize(Archive &ar, unsigned int)
      {
        ar &parent &root &terminal &parent_owner &terminal_owner &children
          &terminals &demand &root_demand &root_radius &radius &volume_moment
            &resistance_moment &root_volume_moment &root_resistance_moment &flow
              &resistance &minimum_demand &root_minimum_demand;
      }
    };
    struct ForestState
    {
      ValidationReport validation;
      /** Authoritative branch metadata only, not a replicated topology. */
      std::map<Id, ForestBranch> owned;
      /** Desired terminal flows on the terminal/node owner. */
      std::map<Id, double> terminal_flows;
      /** Mutable leaf terminal identities on their node owner, also at Q=0. */
      std::set<Id>          owned_leaf_terminals;
      double                demand_scale = 1.;
      std::array<double, 2> total_demand{{0., 0.}}, volume{{0., 0.}},
        power{{0., 0.}}, power_numerator{{0., 0.}};
      unsigned int upward_levels = 0, downward_levels = 0;
    };
    // The mutable set and fixed-root controls contain OWNED ids only. Frozen
    // vessels, including input entries marked generated, are never normalized.
    // Node/terminal authority must be colocated, as in redistribute(). All
    // ranks participate in sparse graph sweeps, including empty ranks.
    // The scale supplies a necessary global minimum-radius bound; exact
    // along-vessel constraints belong to joint geometry certification. No
    // physical data is changed: a caller adopts radii/flows only after a valid
    // result and the geometric checks required by the batch transaction.
    ForestState
    evaluate_forest(const NetworkData          &network,
                    const std::set<Id>         &mutable_owned,
                    const std::map<Id, double> &fixed_owned_roots,
                    const Parameters           &parameters,
                    MPI_Comm                    communicator,
                    const MeshLengthScale      &scale = {});
  } // namespace internal
} // namespace ImmersX::Vascularization
#endif
