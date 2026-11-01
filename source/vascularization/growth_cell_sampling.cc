// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "growth_cell_sampling.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace ImmersX::Vascularization::internal
{
  BackgroundCellSampler::BackgroundCellSampler(
    const dealii::Triangulation<3> &background,
    const dealii::Mapping<3>       &mapping,
    double                          beta,
    MPI_Comm                        comm,
    GrowthWork                     *work)
    : mapping(mapping)
    , beta(beta)
  {
    using namespace dealii;
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(), comm, &comparison);
    bool valid = std::isfinite(beta) &&
                 (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
    for (double other : Utilities::MPI::all_gather(comm, beta))
      valid = valid && other == beta;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      ExcMessage(
        "Cell sampling needs matching finite density exponents and congruent communicators."));
    const FE_Nothing<3>          hex(ReferenceCells::Hexahedron);
    const FE_Nothing<3>          tet(ReferenceCells::Tetrahedron);
    std::unique_ptr<FEValues<3>> hex_values, tet_values;
    std::vector<long double>     logarithms;
    long double largest = -std::numeric_limits<long double>::infinity();
    if (background.n_levels() > 0)
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            count_work(work, GrowthOperation::background_cell_visits);
            try
              {
                FEValues<3> *values = nullptr;
                if (cell->reference_cell().is_hyper_cube())
                  {
                    if (!hex_values)
                      hex_values = std::make_unique<FEValues<3>>(
                        mapping, hex, QGauss<3>(2), update_JxW_values);
                    values = hex_values.get();
                  }
                else if (cell->reference_cell().is_simplex())
                  {
                    if (!tet_values)
                      tet_values = std::make_unique<FEValues<3>>(
                        mapping, tet, QGaussSimplex<3>(2), update_JxW_values);
                    values = tet_values.get();
                  }
                if (!values)
                  {
                    valid = false;
                    continue;
                  }
                values->reinit(cell);
                long double volume = 0.;
                for (double weight : values->get_JxW_values())
                  {
                    valid = valid && std::isfinite(weight) && weight > 0.;
                    volume += weight;
                  }
                if (!std::isfinite(volume) || !(volume > 0.))
                  {
                    valid = false;
                    continue;
                  }
                const long double log_mass =
                  (1.L - static_cast<long double>(beta) / 3.L) *
                  std::log(volume);
                valid = valid && std::isfinite(volume) && volume > 0. &&
                        std::isfinite(log_mass);
                largest = std::max(largest, log_mass);
                cells.push_back({cell, 0.});
                logarithms.push_back(log_mass);
              }
            catch (const std::exception &)
              {
                valid = false;
              }
          }
    largest = Utilities::MPI::max(largest, comm);
    valid   = valid && std::isfinite(largest);
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      ExcMessage(
        "Cell sampling needs matching finite density controls and positive mapped cell volumes."));
    for (std::size_t i = 0; i < cells.size(); ++i)
      {
        const long double mass = std::exp(logarithms[i] - largest);
        const long double next = total + mass;
        valid                  = valid && std::isfinite(mass) && mass > 0. &&
                std::isfinite(next) && next > total;
        total          = next;
        cells[i].upper = total;
      }
    maximum_rank_total = Utilities::MPI::max(total, comm);
    valid =
      valid && std::isfinite(maximum_rank_total) && maximum_rank_total > 0.;
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), comm) == 1,
      ExcMessage(
        "The cell sampling mass range is not representable; no cell weight was discarded or floored."));
  }

  double
  BackgroundCellSampler::exponent() const
  {
    return beta;
  }

  std::optional<dealii::Point<3>>
  BackgroundCellSampler::sample(std::mt19937_64 &selection,
                                std::mt19937    &physical) const
  {
    const auto uniform = [&]() {
      return static_cast<long double>(selection() >> 11) * 0x1.0p-53L;
    };
    // Equal proposal opportunities per rank need this mass correction. The
    // busiest owner always draws; empty owners participate without a point.
    if (cells.empty() || uniform() >= total / maximum_rank_total)
      return {};
    const long double value = uniform() * total;
    const auto        selected =
      std::upper_bound(cells.begin(),
                       cells.end(),
                       value,
                       [](long double value, const Entry &entry) {
                         return value < entry.upper;
                       });
    AssertThrow(selected != cells.end(), dealii::ExcInternalError());
    // deal.II 9.8's particle generator silently falls back to reference-
    // uniform points after 100 misses and only supports hypercubes in Debug.
    // Use its native inverse mapping for both supported reference cells, but
    // report exhaustion instead of changing the physical sampling density.
    const auto bounds =
      mapping.get_bounding_box(selected->cell).get_boundary_points();
    std::uniform_real_distribution<double> coordinate(0., 1.);
    for (unsigned int attempt = 0; attempt < 4096; ++attempt)
      {
        dealii::Point<3> point;
        for (unsigned int d = 0; d < 3; ++d)
          point[d] = bounds.first[d] + coordinate(physical) *
                                         (bounds.second[d] - bounds.first[d]);
        try
          {
            const auto reference =
              mapping.transform_real_to_unit_cell(selected->cell, point);
            if (selected->cell->reference_cell().contains_point(reference))
              return point;
          }
        catch (const dealii::Mapping<3>::ExcTransformationFailed &)
          {}
      }
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Physical cell sampling exhausted its inverse-map budget; no reference-uniform fallback was used."));
    return {};
  }
} // namespace ImmersX::Vascularization::internal
