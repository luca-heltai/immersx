// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_indices_h
#define immersx_internal_growth_indices_h
#include <set>

#include "growth_geometry.h"
namespace ImmersX::Vascularization::internal
{
  struct GrowthGeometryChanges
  {
    std::set<Id> changed_owned, released_owned;
  };

  // Centerline indices have no radius dependency. A fixed maximum over the
  // real root controls and frozen input radii bounds every Murray radius.
  struct GrowthIndices
  {
    using Leaf = std::pair<Segment<3>, Id>;
    using Tree = RTree<Leaf>;
    struct Record
    {
      Leaf   leaf;
      bool   mutable_branch;
      System system;
    };
    GrowthIndices(const NetworkData          &network,
                  const std::set<Id>         &mutable_owned,
                  const std::map<Id, double> &roots,
                  MPI_Comm                    comm,
                  GrowthWork                 *work)
      : rank(Utilities::MPI::this_mpi_process(comm))
      , work(work)
    {
      std::vector<Leaf>                all;
      std::array<std::vector<Leaf>, 2> candidates;
      for (const auto &[id, v] : network.vessels)
        if (v.owner == rank)
          {
            Record record{{{network.nodes.at(v.nodes[0]).position,
                            network.nodes.at(v.nodes[1]).position},
                           id},
                          mutable_owned.count(id) != 0,
                          v.system};
            records.emplace(id, record);
            all.push_back(record.leaf);
            if (record.mutable_branch)
              candidates[index(v.system)].push_back(record.leaf);
            radius_bound = std::max({radius_bound, v.radii[0], v.radii[1]});
          }
      for (const auto &[id, radius] : roots)
        radius_bound = std::max(radius_bound, radius);
      radius_bound = Utilities::MPI::max(radius_bound, comm);
      owned        = pack_rtree(all);
      count_work(work, GrowthOperation::network_index_builds);
      count_work(work, GrowthOperation::network_index_insertions, all.size());
      for (unsigned int system = 0; system < 2; ++system)
        {
          mutable_trees[system] = pack_rtree(candidates[system]);
          count_work(work, GrowthOperation::network_index_builds);
          count_work(work,
                     GrowthOperation::network_index_insertions,
                     candidates[system].size());
        }
    }
    static unsigned int
    index(System system)
    {
      if (system == System::arterial)
        return 0;
      else if (system == System::venous)
        return 1;
      AssertThrow(false,
                  ExcMessage("Invalid vascular system in centerline index."));
      return 0;
    }
    static std::vector<BoundingBox<3>>
    covering(const Tree &tree)
    {
      if (tree.empty())
        return {};
      BoundingBox<3> box;
      boost::geometry::convert(tree.bounds(), box);
      return {box};
    }
    const Tree &
    mutable_tree(System system) const
    {
      return mutable_trees[index(system)];
    }
    void
    erase(Id id)
    {
      const auto found = records.find(id);
      if (found == records.end())
        return;
      const auto &record = found->second;
      AssertThrow(owned.remove(record.leaf) == 1, ExcInternalError());
      count_work(work, GrowthOperation::network_index_removals);
      if (record.mutable_branch)
        {
          AssertThrow(mutable_trees[index(record.system)].remove(record.leaf) ==
                        1,
                      ExcInternalError());
          count_work(work, GrowthOperation::network_index_removals);
        }
      records.erase(found);
    }
    void
    update(const NetworkData           &network,
           const std::set<Id>          &mutable_owned,
           const GrowthGeometryChanges &changes)
    {
      for (Id id : changes.released_owned)
        erase(id);
      for (Id id : changes.changed_owned)
        {
          erase(id);
          const auto &v = network.vessels.at(id);
          AssertThrow(v.owner == rank, ExcInternalError());
          Record record{{{network.nodes.at(v.nodes[0]).position,
                          network.nodes.at(v.nodes[1]).position},
                         id},
                        mutable_owned.count(id) != 0,
                        v.system};
          records.emplace(id, record);
          owned.insert(record.leaf);
          count_work(work, GrowthOperation::network_index_insertions);
          if (record.mutable_branch)
            {
              mutable_trees[index(v.system)].insert(record.leaf);
              count_work(work, GrowthOperation::network_index_insertions);
            }
        }
    }
    Tree                 owned;
    std::array<Tree, 2>  mutable_trees;
    std::map<Id, Record> records;
    double               radius_bound = 0.;
    unsigned int         rank;
    GrowthWork          *work;
  };
} // namespace ImmersX::Vascularization::internal
#endif
