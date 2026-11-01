// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_internal_growth_cell_sampling_h
#define immersx_internal_growth_cell_sampling_h
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <immersx/vascularization/growth_work.h>

#include <optional>
#include <random>
#include <vector>

namespace ImmersX::Vascularization::internal
{
  // Immutable cell masses for physical density |K|^(-beta/3). The random
  // engines belong to a growth attempt, independently of this cached table.
  class BackgroundCellSampler
  {
  public:
    BackgroundCellSampler(const dealii::Triangulation<3> &,
                          const dealii::Mapping<3> &,
                          double beta,
                          MPI_Comm,
                          GrowthWork * = nullptr);
    double
    exponent() const;
    std::optional<dealii::Point<3>>
    sample(std::mt19937_64 &selection, std::mt19937 &physical) const;

  private:
    struct Entry
    {
      dealii::Triangulation<3>::active_cell_iterator cell;
      long double                                    upper;
    };
    const dealii::Mapping<3> &mapping;
    double                    beta;
    long double               total = 0., maximum_rank_total = 0.;
    std::vector<Entry>        cells;
  };
} // namespace ImmersX::Vascularization::internal
#endif
