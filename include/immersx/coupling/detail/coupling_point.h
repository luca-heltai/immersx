// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coupling_detail_coupling_point_h
#define immersx_coupling_detail_coupling_point_h

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace ImmersX::detail
{
  /** Data retained by a coupling backend for one physical source point. */
  template <int spacedim, typename ValueType = double>
  struct CouplingPoint
  {
    using value_type = ValueType;

    dealii::Point<spacedim>          point;
    dealii::Point<spacedim>          representative_point;
    double                           weight = 0.;
    dealii::types::global_cell_index source_entity_id =
      dealii::numbers::invalid_unsigned_int;
    unsigned int  representative_qpoint = dealii::numbers::invalid_unsigned_int;
    unsigned int  section_qpoint        = dealii::numbers::invalid_unsigned_int;
    std::uint64_t stable_id = std::numeric_limits<std::uint64_t>::max();
    std::vector<dealii::types::global_dof_index> dof_indices;
    std::vector<ValueType>                       basis_values;
  };
} // namespace ImmersX::detail

#endif // immersx_coupling_detail_coupling_point_h
