// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_territory_search_h
#define immersx_internal_growth_territory_search_h
#include <immersx/vascularization/generation.h>

#include <queue>

#include "growth_geometry.h"
namespace ImmersX::Vascularization::internal
{
  // Startup points are fixed. Availability changes separately so native
  // nearest iterators stay valid while other outlets claim territories.
  class VenousTerritorySearch
  {
  public:
    VenousTerritorySearch(const std::vector<Territory> &,
                          unsigned int owner,
                          GrowthWork * = nullptr);
    VenousTerritorySearch(const VenousTerritorySearch &) = delete;
    VenousTerritorySearch &
    operator=(const VenousTerritorySearch &) = delete;
    const ValidationReport &
    validation() const;
    std::size_t
    size() const;
    std::size_t
    active_queries() const;
    std::optional<dealii::BoundingBox<3>>
    cover() const;
    std::optional<Territory>
         query(Id root, const dealii::Point<3> &, Id rejected = invalid_id);
    bool claim(Id);
    void forget(Id);

  private:
    using Tree = dealii::RTree<std::pair<dealii::Point<3>, Id>>;
    struct Cursor
    {
      Cursor(const dealii::Point<3> &, Tree::const_query_iterator);
      dealii::Point<3>           position;
      Tree::const_query_iterator next;
      std::priority_queue<std::pair<double, Id>,
                          std::vector<std::pair<double, Id>>,
                          std::greater<std::pair<double, Id>>>
         pending;
      Id current = invalid_id;
    };
    Tree                    tree;
    std::map<Id, Territory> available;
    std::map<Id, Cursor>    queries;
    GrowthWork             *work;
    ValidationReport        report;
  };
} // namespace ImmersX::Vascularization::internal
#endif
