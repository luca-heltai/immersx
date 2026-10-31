// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_tissue_demand_h
#define immersx_internal_tissue_demand_h
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <cstdint>
namespace ImmersX::Vascularization::internal
{
  struct TissueDemandSample
  {
    dealii::Point<3> position;
    double           mass = 0.;
  };

  // Refine integration on a fixed physical background cell. Check the point
  // budget before constructing a native iterated rule, including simplex rules
  // that internally build a refined reference triangulation.
  dealii::Quadrature<3>
  tissue_demand_quadrature(
    decltype(dealii::ReferenceCells::Hexahedron) reference_cell,
    unsigned int                                 order,
    unsigned int                                 copies,
    std::uint64_t                                point_budget);

  // Samples stay with their owned background cell; they are FE quadrature,
  // rather than a separately migrating point cloud. The caller may reuse its
  // native FEValues across all cells with the same rule and reference cell.
  std::vector<TissueDemandSample>
  sample_tissue_demand(
    const dealii::Triangulation<3>::active_cell_iterator &cell,
    dealii::FEValues<3>                                  &values,
    const dealii::Function<3>                            &density);
} // namespace ImmersX::Vascularization::internal
#endif
