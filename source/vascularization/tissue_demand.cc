// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "tissue_demand.h"

#include <deal.II/base/quadrature_lib.h>

#include <immersx/compatibility/reference_cell.h>

#include <cmath>
namespace ImmersX::Vascularization::internal
{
  dealii::Quadrature<3>
  tissue_demand_quadrature(
    decltype(dealii::ReferenceCells::Hexahedron) reference_cell,
    unsigned int                                 order,
    unsigned int                                 copies,
    std::uint64_t                                point_budget)
  {
    using namespace dealii;
    AssertThrow(
      order > 0 && order <= 8 && copies > 0 && (copies & (copies - 1)) == 0 &&
        (reference_cell.is_hyper_cube() || reference_cell.is_simplex()),
      ExcMessage(
        "Tissue demand requires order 1 through 8 and a positive power-of-two integration subdivision on hexahedra or tetrahedra."));
    const auto base = compatibility::gauss_quadrature<3>(reference_cell, order);
    std::uint64_t count = base.size();
    for (unsigned int d = 0; d < 3; ++d)
      {
        AssertThrow(
          count <= point_budget / copies,
          ExcMessage(
            "Tissue demand integration point budget exhausted before quadrature allocation."));
        count *= copies;
      }
    if (reference_cell.is_hyper_cube())
      return QIterated<3>(QGauss<1>(order), copies);
    else if (reference_cell.is_simplex())
      return QIteratedSimplex<3>(base, copies);
    AssertThrow(false, ExcMessage("Unsupported tissue reference cell."));
    return {};
  }

  std::vector<TissueDemandSample>
  sample_tissue_demand(
    const dealii::Triangulation<3>::active_cell_iterator &cell,
    dealii::FEValues<3>                                  &values,
    const dealii::Function<3>                            &density)
  {
    using namespace dealii;
    AssertThrow(
      cell->is_locally_owned(),
      ExcMessage(
        "Tissue quadrature must be evaluated by its background cell owner."));
    values.reinit(cell);
    std::vector<TissueDemandSample> result;
    result.reserve(values.n_quadrature_points);
    for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
      {
        const auto  &position = values.quadrature_point(q);
        const double value    = density.value(position);
        const double weight   = values.JxW(q);
        const double mass     = value * weight;
        AssertThrow(
          std::isfinite(value) && value > 0. && std::isfinite(weight) &&
            weight > 0. && std::isfinite(mass) && mass > 0.,
          ExcMessage(
            "Integrated tissue demand and physical JxW must be positive, finite and representable."));
        result.push_back({position, mass});
      }
    return result;
  }
} // namespace ImmersX::Vascularization::internal
