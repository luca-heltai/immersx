// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_commit_h
#define immersx_internal_growth_commit_h
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include "forest_state.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthSplit
  {
    Id                vessel = invalid_id;
    unsigned int      owner  = dealii::numbers::invalid_unsigned_int;
    std::array<Id, 2> old_nodes{{invalid_id, invalid_id}};
    std::array<dealii::Point<3>, 2> old_points;
    Id                              junction = invalid_id, tip = invalid_id;
    dealii::Point<3>                junction_position, tip_position;
    Id     continuation = invalid_id, leaf = invalid_id;
    Id     terminal = invalid_id, territory = invalid_id;
    double demand = 0.;
    template <class Archive>
    void
    serialize(Archive &ar, unsigned int)
    {
      ar &vessel &owner &old_nodes &old_points &junction &tip &junction_position
        &tip_position &continuation &leaf &terminal &territory &demand;
    }
  };
  // Shared topology edit for serial CCO and distributed batch staging. The
  // caller controls mutability and supplies globally fresh physical ids.
  bool
  apply_growth_split(NetworkData &network, const GrowthSplit &split);
  struct GrowthCommit
  {
    bool             committed = false;
    ValidationReport validation;
    // Outputs are populated only after the entire batch passes final checks.
    NetworkData          network;
    std::set<Id>         mutable_owned;
    std::map<Id, double> fixed_owned_roots;
    ForestState          physiology;
  };
  // Input remains immutable. Splits are sent to their writing owners. After
  // joint normalization, ALL generated geometry is certified on one snapshot.
  // One failed check rolls back the whole batch, including changes on ranks
  // with no local proposal. The caller may resample within its finite budget.
  GrowthCommit
  commit_growth_splits(const dealii::Triangulation<3> &background,
                       const dealii::Mapping<3>       &mapping,
                       const NetworkData              &snapshot,
                       const std::set<Id>             &mutable_owned,
                       const std::map<Id, double>     &fixed_owned_roots,
                       const std::vector<GrowthSplit> &local,
                       const Parameters               &parameters,
                       MPI_Comm                        communicator);
} // namespace ImmersX::Vascularization::internal
#endif
