// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_overlap_integration_support_h
#define immersx_overlap_integration_support_h

#include <immersx/grid/overlap_classification.h>

namespace ImmersX
{
  /** Complementary fixed quadrature supports. Physical material ids are kept.
   */
  template <int dim, int spacedim = dim>
  struct OverlapIntegrationSupports
  {
    IntegrationSupport<dim, spacedim> overlap;
    IntegrationSupport<dim, spacedim> outside;
  };

  /** Partition source-owned physical samples on the reference geometry.
   *
   * Records expose point, weight, source_entity_id, representative_qpoint and
   * section_qpoint. Weights are physical integration weights. The
   * representative support sums the selected section weights at each quadrature
   * point; lifted assembly retains each original physical weight. Missing
   * samples contribute only to outside. Rebuild after changing the mesh or the
   * sample geometry.
   */
  template <int dim, int spacedim, typename PointRange>
  OverlapIntegrationSupports<dim, spacedim>
  make_overlap_supports(
    const IntegrationSupport<dim, spacedim>            &source,
    const dealii::Mapping<dim, spacedim>               &mapping,
    const dealii::Quadrature<dim>                      &quadrature,
    const PointRange                                   &samples,
    const dealii::GridTools::Cache<spacedim, spacedim> &background,
    const double                                        tolerance = 1.e-10)
  {
    const auto &tria = source.triangulation();
    const auto  communicator =
      background.get_triangulation().get_mpi_communicator();
    int       comparison = MPI_UNEQUAL;
    const int error =
      MPI_Comm_compare(tria.get_mpi_communicator(), communicator, &comparison);
    AssertThrow(error == MPI_SUCCESS &&
                  (comparison == MPI_IDENT || comparison == MPI_CONGRUENT),
                dealii::ExcMessage(
                  "Overlap supports require congruent communicators."));
    bool valid =
      std::isfinite(tolerance) && tolerance >= 0. && quadrature.size() > 0;
    auto partition = std::make_shared<detail::OverlapQuadraturePartition>();
    const auto reference_cell = detail::uniform_reference_cell(tria);
    if (!reference_cell)
      return {source.with_quadrature_partition(quadrature, partition, false),
              source.with_quadrature_partition(quadrature, partition, true)};
    const dealii::FE_Nothing<dim, spacedim> element(*reference_cell);
    dealii::FEValues<dim, spacedim>         values(mapping,
                                           element,
                                           quadrature,
                                           dealii::update_JxW_values);
    std::map<dealii::types::global_cell_index, std::vector<double>> jxw;
    std::set<dealii::types::global_cell_index> selected_cells;
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          auto &entry = (*partition)[cell->global_active_cell_index()];
          entry.inside.resize(quadrature.size());
          entry.overlap_weights.assign(quadrature.size(), 0.);
          entry.total_weights.assign(quadrature.size(), 0.);
          if (source.includes_cell(cell))
            selected_cells.insert(cell->global_active_cell_index());
          values.reinit(cell);
          jxw[cell->global_active_cell_index()] = values.get_JxW_values();
        }
    std::vector<dealii::Point<spacedim>> positions;
    std::vector<std::size_t>             indices;
    for (std::size_t i = 0; i < samples.size(); ++i)
      {
        const auto &point = samples[i];
        if (!selected_cells.count(point.source_entity_id))
          continue;
        const auto entry = partition->find(point.source_entity_id);
        valid            = valid && entry != partition->end() &&
                point.representative_qpoint < quadrature.size();
        valid = valid && std::isfinite(point.weight) && point.weight > 0.;
        for (unsigned int d = 0; d < spacedim; ++d)
          valid = valid && std::isfinite(point.point[d]);
        if (entry != partition->end() &&
            point.representative_qpoint < quadrature.size())
          {
            auto &flags = entry->second.inside[point.representative_qpoint];
            flags.resize(std::max<std::size_t>(flags.size(),
                                               point.section_qpoint + 1),
                         0);
            positions.push_back(point.point);
            indices.push_back(i);
          }
      }
    AssertThrow(
      dealii::Utilities::MPI::min(valid ? 1u : 0u, communicator) != 0,
      dealii::ExcMessage(
        "Overlap supports require finite positive source-owned quadrature samples."));
    using Search =
      dealii::Utilities::MPI::RemotePointEvaluation<spacedim, spacedim>;
    Search     search(typename Search::AdditionalData(tolerance, false));
    const bool has_background =
      background.get_triangulation().n_global_active_cells() > 0;
    if (has_background)
      search.reinit(background, positions);
    for (std::size_t i = 0; i < indices.size(); ++i)
      {
        const auto &point  = samples[indices[i]];
        auto       &entry  = partition->at(point.source_entity_id);
        const auto  q      = point.representative_qpoint;
        const bool  inside = has_background && search.point_found(i);
        entry.inside[q][point.section_qpoint] = inside;
        const double weight =
          point.weight / jxw.at(point.source_entity_id).at(q);
        entry.total_weights[q] += weight;
        if (inside)
          entry.overlap_weights[q] += weight;
      }
    using Cell =
      typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;
    if (dynamic_cast<
          const dealii::parallel::TriangulationBase<dim, spacedim> *>(&tria))
      dealii::GridTools::exchange_cell_data_to_ghosts<
        detail::OverlapCellQuadrature>(
        tria,
        [&](const Cell &cell) {
          return partition->at(cell->global_active_cell_index());
        },
        [&](const Cell &cell, const detail::OverlapCellQuadrature &data) {
          (*partition)[cell->global_active_cell_index()] = data;
        });
    return {source.with_quadrature_partition(quadrature, partition, false),
            source.with_quadrature_partition(quadrature, partition, true)};
  }

  /** Partition an ordinary mapped cell quadrature into located and missing
   * samples. */
  template <int dim, int spacedim>
  OverlapIntegrationSupports<dim, spacedim>
  make_overlap_supports(
    const IntegrationSupport<dim, spacedim>            &source,
    const dealii::Mapping<dim, spacedim>               &mapping,
    const dealii::Quadrature<dim>                      &quadrature,
    const dealii::GridTools::Cache<spacedim, spacedim> &background,
    const double                                        tolerance = 1.e-10)
  {
    struct Sample
    {
      dealii::Point<spacedim>          point;
      double                           weight;
      dealii::types::global_cell_index source_entity_id;
      unsigned int                     representative_qpoint;
      unsigned int                     section_qpoint = 0;
    };
    std::vector<Sample> samples;
    const auto          reference_cell =
      detail::uniform_reference_cell(source.triangulation());
    if (reference_cell)
      {
        const dealii::FE_Nothing<dim, spacedim> element(*reference_cell);
        dealii::FEValues<dim, spacedim>         values(
          mapping,
          element,
          quadrature,
          dealii::update_quadrature_points | dealii::update_JxW_values);
        for (const auto &cell : source.triangulation().active_cell_iterators())
          if (cell->is_locally_owned() && source.includes_cell(cell))
            {
              values.reinit(cell);
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                samples.push_back({values.quadrature_point(q),
                                   values.JxW(q),
                                   cell->global_active_cell_index(),
                                   q,
                                   0});
            }
      }
    return make_overlap_supports(
      source, mapping, quadrature, samples, background, tolerance);
  }
} // namespace ImmersX
#endif
