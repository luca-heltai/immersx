// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_territory_search.h"

#include <boost/geometry/algorithms/comparable_distance.hpp>
namespace ImmersX::Vascularization::internal
{
  VenousTerritorySearch::Cursor::Cursor(const dealii::Point<3>    &point,
                                        Tree::const_query_iterator iterator)
    : position(point)
    , next(std::move(iterator))
  {}
  VenousTerritorySearch::VenousTerritorySearch(
    const std::vector<Territory> &points,
    unsigned int                  owner,
    GrowthWork                   *counts)
    : work(counts)
  {
    std::vector<Tree::value_type> entries;
    for (const auto &point : points)
      {
        count_work(work, GrowthOperation::validation_record_visits);
        bool valid = point.id != invalid_id && point.owner == owner &&
                     std::isfinite(point.demand) && point.demand > 0.;
        for (unsigned int d = 0; d < 3; ++d)
          valid = valid && std::isfinite(point.position[d]);
        if (!valid || !available.emplace(point.id, point).second)
          {
            report.issues.push_back(
              {"venous territories",
               point.id,
               "Venous startup needs unique native territories with finite positions and positive demand."});
            continue;
          }
        entries.emplace_back(point.position, point.id);
      }
    tree = dealii::pack_rtree(entries);
    count_work(work, GrowthOperation::network_index_builds);
    count_work(work, GrowthOperation::network_index_insertions, entries.size());
  }
  const ValidationReport &
  VenousTerritorySearch::validation() const
  {
    return report;
  }
  std::size_t
  VenousTerritorySearch::size() const
  {
    return available.size();
  }
  std::size_t
  VenousTerritorySearch::active_queries() const
  {
    return queries.size();
  }
  std::optional<dealii::BoundingBox<3>>
  VenousTerritorySearch::cover() const
  {
    if (tree.empty())
      return {};
    dealii::BoundingBox<3> bounds;
    boost::geometry::convert(tree.bounds(), bounds);
    return bounds;
  }
  std::optional<Territory>
  VenousTerritorySearch::query(Id                      root,
                               const dealii::Point<3> &position,
                               Id                      rejected)
  {
    AssertThrow(
      report.valid() && root != invalid_id,
      dealii::ExcMessage(
        "Venous search needs validated points and a stable root id."));
    for (unsigned int d = 0; d < 3; ++d)
      AssertThrow(std::isfinite(position[d]),
                  dealii::ExcMessage("A root position must be finite."));
    auto found = queries.find(root);
    if (found == queries.end())
      {
        found = queries
                  .try_emplace(root,
                               position,
                               tree.qbegin(boost::geometry::index::nearest(
                                 position, tree.size())))
                  .first;
        count_work(work, GrowthOperation::network_index_queries);
      }
    auto &cursor = found->second;
    AssertThrow(cursor.position == position,
                dealii::ExcMessage(
                  "A retained startup root must keep its position."));
    if (cursor.current != rejected && available.count(cursor.current))
      return available.at(cursor.current);
    cursor.current = invalid_id;
    for (;;)
      {
        while (!cursor.pending.empty() &&
               !available.count(cursor.pending.top().second))
          cursor.pending.pop();
        if (cursor.next != tree.qend())
          {
            const double distance =
              boost::geometry::comparable_distance(position,
                                                   cursor.next->first);
            // The native cursor orders comparable distance. Prefetch a
            // conservative floating-point band before choosing exact deal.II
            // distance and physical id; this also resolves cutoff ties.
            const double lower =
              std::max(0.,
                       std::nextafter(
                         std::min(distance,
                                  std::numeric_limits<double>::max()) *
                           (1. - 64. * std::numeric_limits<double>::epsilon()),
                         -std::numeric_limits<double>::infinity()) -
                         64. * std::numeric_limits<double>::denorm_min());
            if (cursor.pending.empty() || !(cursor.pending.top().first < lower))
              {
                const auto &point = *cursor.next;
                if (available.count(point.second))
                  cursor.pending.emplace(point.first.distance_square(position),
                                         point.second);
                ++cursor.next;
                count_work(work, GrowthOperation::network_index_hits);
                continue;
              }
          }
        if (cursor.pending.empty())
          return {};
        cursor.current = cursor.pending.top().second;
        cursor.pending.pop();
        return available.at(cursor.current);
      }
  }
  bool
  VenousTerritorySearch::claim(Id id)
  {
    const bool erased = available.erase(id) == 1;
    count_work(work, GrowthOperation::transaction_records, erased);
    return erased;
  }
  void
  VenousTerritorySearch::forget(Id root)
  {
    queries.erase(root);
  }
} // namespace ImmersX::Vascularization::internal
