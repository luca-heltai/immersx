// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_overlap_classification_h
#define immersx_overlap_classification_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <immersx/core/integration_support.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ImmersX
{
  /** Cell labels assigned by set_overlap_material_ids().
   *
   * Use the same partial and inside id to select every cell with at least one
   * located sample through Domain::support(id). Existing material ids on the
   * source mesh are overwritten.
   */
  struct OverlapMaterialIds
  {
    dealii::types::material_id outside = 0;
    dealii::types::material_id partial = 1;
    dealii::types::material_id inside  = 2;
  };

  /** Assign material ids from physical samples located in a background mesh.
   *
   * The point generator is called once per locally owned active source cell
   * and must return a nonempty range of Point<spacedim>. It can supply lifted
   * wall points instead of centreline points. A cell is outside if no sample
   * is found, inside if every sample is found, and partial otherwise. This is
   * a sampled classification, not an exact geometric cell intersection.
   *
   * This operation is collective on the background communicator. Source and
   * background communicators must be identical or congruent. Source cells
   * may be owned by different ranks than the background cells containing the
   * samples; ranks without local source cells still participate. Material ids
   * are copied from source owners to ghost cells using deal.II's cell-data
   * exchange. Serial sources are also supported. Parallel source meshes must
   * support GridTools::exchange_cell_data_to_ghosts().
   *
   * Cell labels do not remove exterior samples from partially covered cells.
   * Reclassify after changing either mesh, its mapping, or the sample geometry.
   */
  template <int dim, int spacedim, typename PointGenerator>
  void
  set_overlap_material_ids(
    dealii::Triangulation<dim, spacedim>               &source,
    const dealii::GridTools::Cache<spacedim, spacedim> &background,
    PointGenerator                                    &&points_on_cell,
    const OverlapMaterialIds                           &ids       = {},
    const double                                        tolerance = 1.e-10)
  {
    const auto communicator =
      background.get_triangulation().get_mpi_communicator();
    int        comparison = MPI_UNEQUAL;
    const auto error      = MPI_Comm_compare(source.get_mpi_communicator(),
                                        communicator,
                                        &comparison);
    AssertThrow(error == MPI_SUCCESS,
                dealii::ExcMessage("Could not compare overlap communicators."));
    AssertThrow(
      comparison == MPI_IDENT || comparison == MPI_CONGRUENT,
      dealii::ExcMessage(
        "Source and background overlap communicators must be identical or congruent."));
    const bool valid_parameters =
      std::isfinite(tolerance) && tolerance >= 0. &&
      ids.outside != dealii::numbers::invalid_material_id &&
      ids.partial != dealii::numbers::invalid_material_id &&
      ids.inside != dealii::numbers::invalid_material_id &&
      ids.outside != ids.partial && ids.outside != ids.inside;
    AssertThrow(
      dealii::Utilities::MPI::min(valid_parameters ? 1u : 0u, communicator) !=
        0,
      dealii::ExcMessage(
        "Overlap classification needs a finite nonnegative tolerance and valid material ids with outside distinct from inside and partial."));

    using Cell =
      typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;
    std::vector<Cell>                    cells;
    std::vector<std::size_t>             offsets{0};
    std::vector<dealii::Point<spacedim>> positions;
    bool                                 valid_samples = true;
    for (const auto &cell : source.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto &points = points_on_cell(cell);
          valid_samples      = valid_samples && !points.empty();
          for (const auto &point : points)
            {
              for (unsigned int d = 0; d < spacedim; ++d)
                valid_samples = valid_samples && std::isfinite(point[d]);
              positions.push_back(point);
            }
          cells.push_back(cell);
          offsets.push_back(positions.size());
        }
    valid_samples = valid_samples && positions.size() <=
                                       std::numeric_limits<unsigned int>::max();
    AssertThrow(
      dealii::Utilities::MPI::min(valid_samples ? 1u : 0u, communicator) != 0,
      dealii::ExcMessage(
        "Every source cell needs at least one finite overlap sample."));

    using Search =
      dealii::Utilities::MPI::RemotePointEvaluation<spacedim, spacedim>;
    // Missing points are expected. Unique mapping would reject them in Debug.
    Search     search(typename Search::AdditionalData(tolerance, false));
    const bool has_background =
      background.get_triangulation().n_global_active_cells() > 0;
    if (has_background)
      search.reinit(background, positions);
    for (std::size_t c = 0; c < cells.size(); ++c)
      {
        std::size_t found = 0;
        for (std::size_t q = offsets[c]; q < offsets[c + 1]; ++q)
          found +=
            has_background && search.point_found(static_cast<unsigned int>(q));
        if (found == 0)
          cells[c]->set_material_id(ids.outside);
        else if (found == offsets[c + 1] - offsets[c])
          cells[c]->set_material_id(ids.inside);
        else if (found < offsets[c + 1] - offsets[c])
          cells[c]->set_material_id(ids.partial);
      }

    if (dynamic_cast<const dealii::parallel::TriangulationBase<dim, spacedim>
                       *>(&source) != nullptr)
      dealii::GridTools::exchange_cell_data_to_ghosts<
        dealii::types::material_id>(
        source,
        [](const Cell &cell) { return cell->material_id(); },
        [](const Cell &cell, const dealii::types::material_id id) {
          cell->set_material_id(id);
        });
  }

  /** Classify source cells using the mapped reference-cell quadrature points.
   *
   * The source must have a uniform reference-cell type and the quadrature
   * must match that type. No source finite-element space is needed.
   */
  template <int dim, int spacedim>
  void
  set_overlap_material_ids(
    dealii::Triangulation<dim, spacedim>               &source,
    const dealii::Mapping<dim, spacedim>               &mapping,
    const dealii::Quadrature<dim>                      &quadrature,
    const dealii::GridTools::Cache<spacedim, spacedim> &background,
    const OverlapMaterialIds                           &ids       = {},
    const double                                        tolerance = 1.e-10)
  {
    AssertThrow(dealii::Utilities::MPI::min(quadrature.size() > 0 ? 1u : 0u,
                                            source.get_mpi_communicator()) != 0,
                dealii::ExcMessage("Overlap quadrature must not be empty."));
    const auto reference_cell = detail::uniform_reference_cell(source);
    if (!reference_cell)
      return;
    const dealii::FE_Nothing<dim, spacedim> geometry_element(*reference_cell);
    dealii::FEValues<dim, spacedim>         values(mapping,
                                           geometry_element,
                                           quadrature,
                                           dealii::update_quadrature_points);
    set_overlap_material_ids(
      source,
      background,
      [&](const auto &cell) -> const auto & {
        values.reinit(cell);
        return values.get_quadrature_points();
      },
      ids,
      tolerance);
  }
} // namespace ImmersX

#endif
