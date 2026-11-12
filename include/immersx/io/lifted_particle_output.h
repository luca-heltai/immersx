// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_lifted_particle_output_h
#define immersx_lifted_particle_output_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include <immersx/coupling/detail/coupling_point.h>
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** A physical sample of a lifted wall: one quadrature point of the
   * representative-curve quadrature times one cross-section quadrature point.
   */
  template <int spacedim>
  using LiftedWallPoint = detail::CouplingPoint<spacedim, double>;

  /** Parameters selecting which quantities each lifted-wall particle carries.
   *
   * The wall particles are the physical quadrature points of the lift. The
   * "Properties" list chooses what is stored on them:
   *
   * - weight: quadrature weight of the wall point;
   * - mode_values: selected cross-section mode shapes at the point;
   * - modal_coefficients: coefficients of the modal FE field at the
   *   representative point, one per selected mode;
   * - lifted_value: the modal field lifted to the wall, the sum over the modes
   *   of mode shape times modal coefficient;
   * - modal_data: prescribed modal data at the representative point, one per
   *   selected mode;
   * - representative_point: the point of the curve the wall point is lifted
   *   from;
   * - radius: distance between the wall point and its representative point;
   * - indices: source cell, representative quadrature point and cross-section
   *   quadrature point of the wall point;
   * - bulk_field: the background field interpolated at the wall point.
   *
   * An empty list disables the output.
   */
  struct LiftedParticleOutputParameters : public dealii::ParameterAcceptor
  {
    static const std::vector<std::string> &
    available_properties()
    {
      static const std::vector<std::string> names{"weight",
                                                  "mode_values",
                                                  "modal_coefficients",
                                                  "lifted_value",
                                                  "modal_data",
                                                  "representative_point",
                                                  "radius",
                                                  "indices",
                                                  "bulk_field"};
      return names;
    }

    explicit LiftedParticleOutputParameters(
      const std::string       &subsection,
      std::vector<std::string> default_properties = {"lifted_value", "weight"})
      : dealii::ParameterAcceptor(subsection)
      , properties(std::move(default_properties))
    {
      std::string choices;
      for (const auto &name : available_properties())
        choices += (choices.empty() ? "" : "|") + name;
      add_parameter(
        "Properties",
        properties,
        "Quantities stored on the lifted-wall particles, as a "
        "comma-separated list of: " +
          choices + ". An empty list disables the particle output.",
        this->prm,
        dealii::Patterns::List(dealii::Patterns::Selection(choices),
                               0,
                               dealii::Patterns::List::max_int_value,
                               ","));
    }

    bool
    enabled() const
    {
      return !properties.empty();
    }

    bool
    selected(const std::string &name) const
    {
      return std::find(properties.begin(), properties.end(), name) !=
             properties.end();
    }

    /** Reject names outside available_properties(). The parameter pattern
     * already does so for values read from a file; this covers values set from
     * code.
     */
    void
    validate() const
    {
      for (const auto &name : properties)
        AssertThrow(std::find(available_properties().begin(),
                              available_properties().end(),
                              name) != available_properties().end(),
                    dealii::ExcMessage("Unknown lifted particle property '" +
                                       name + "'."));
    }

    std::vector<std::string> properties;
  };

  /** Names of the problem-specific quantities in the particle file. */
  struct LiftedParticleLabels
  {
    /** Name of the lifted value; the modal coefficients are named
     * value_name + "_" + mode label.
     */
    std::string value_name = "lambda";

    /** One label per selected mode. */
    std::vector<std::string> mode_labels;

    /** Values per mode shape: 1 for scalar modes, spacedim for vector modes.
     * The lifted value has the same number of components.
     */
    unsigned int mode_components = 1;
  };

  /** One scalar column of the particle file. A vector quantity is stored as
   * several columns sharing a name, all but the interpretation being scalar.
   */
  struct LiftedParticleColumn
  {
    std::string name;
    dealii::DataComponentInterpretation::DataComponentInterpretation
      interpretation = dealii::DataComponentInterpretation::component_is_scalar;
    std::vector<double> values; ///< One entry per wall point.
  };

  /** Build the columns that follow from the wall points alone.
   *
   * Each wall point stores mode_components values per selected mode in
   * mode_values. modal_coefficients and modal_data are point-major with one
   * entry per selected mode and point; either may be empty if the matching
   * property is not selected. The bulk field is not a column: it depends on
   * where each particle ends up and is sampled by write_lifted_particles().
   */
  template <int spacedim>
  std::vector<LiftedParticleColumn>
  make_lifted_particle_columns(
    const LiftedParticleOutputParameters         &options,
    const LiftedParticleLabels                   &labels,
    const std::vector<LiftedWallPoint<spacedim>> &points,
    const std::vector<double>                    &modal_coefficients = {},
    const std::vector<double>                    &modal_data         = {})
  {
    options.validate();
    const std::size_t  n_points = points.size();
    const std::size_t  n_modes  = labels.mode_labels.size();
    const unsigned int width    = labels.mode_components;
    AssertThrow(width == 1 || width == spacedim,
                dealii::ExcMessage(
                  "Mode shapes must be scalar or have spacedim components."));
    for (const auto &point : points)
      AssertThrow(!(options.selected("mode_values") ||
                    options.selected("lifted_value")) ||
                    point.mode_values.size() == n_modes * width,
                  dealii::ExcMessage(
                    "A wall point needs mode_components values per selected "
                    "mode."));
    const auto scalar_column = [&](const std::string &name) {
      LiftedParticleColumn column;
      column.name = name;
      column.values.reserve(n_points);
      return column;
    };
    const auto vector_columns = [&](const std::string &name,
                                    const unsigned int n_components) {
      if (n_components == 1)
        return std::vector<LiftedParticleColumn>{scalar_column(name)};
      std::vector<LiftedParticleColumn> columns(n_components);
      for (auto &column : columns)
        {
          column.name = name;
          column.interpretation =
            dealii::DataComponentInterpretation::component_is_part_of_vector;
          column.values.reserve(n_points);
        }
      return columns;
    };

    std::vector<LiftedParticleColumn> columns;

    if (options.selected("weight"))
      {
        auto column = scalar_column("weight");
        for (const auto &point : points)
          column.values.push_back(point.weight);
        columns.push_back(std::move(column));
      }

    if (options.selected("mode_values"))
      for (std::size_t mode = 0; mode < n_modes; ++mode)
        {
          auto shape =
            vector_columns("mode_" + labels.mode_labels[mode], width);
          for (const auto &point : points)
            for (unsigned int c = 0; c < width; ++c)
              shape[c].values.push_back(point.mode_values[mode * width + c]);
          for (auto &column : shape)
            columns.push_back(std::move(column));
        }

    if (options.selected("modal_coefficients") ||
        options.selected("lifted_value"))
      AssertThrow(modal_coefficients.size() == n_points * n_modes,
                  dealii::ExcMessage(
                    "The modal coefficients are needed for the selected "
                    "particle properties, one per mode and wall point."));

    if (options.selected("modal_coefficients"))
      for (std::size_t mode = 0; mode < n_modes; ++mode)
        {
          auto column =
            scalar_column(labels.value_name + "_" + labels.mode_labels[mode]);
          for (std::size_t q = 0; q < n_points; ++q)
            column.values.push_back(modal_coefficients[q * n_modes + mode]);
          columns.push_back(std::move(column));
        }

    if (options.selected("lifted_value"))
      {
        auto lifted = vector_columns(labels.value_name, width);
        for (std::size_t q = 0; q < n_points; ++q)
          for (unsigned int c = 0; c < width; ++c)
            {
              double value = 0.;
              for (std::size_t mode = 0; mode < n_modes; ++mode)
                value += points[q].mode_values[mode * width + c] *
                         modal_coefficients[q * n_modes + mode];
              lifted[c].values.push_back(value);
            }
        for (auto &column : lifted)
          columns.push_back(std::move(column));
      }

    if (options.selected("modal_data"))
      {
        AssertThrow(modal_data.size() == n_points * n_modes,
                    dealii::ExcMessage(
                      "The modal data are needed for the selected particle "
                      "properties, one per mode and wall point."));
        for (std::size_t mode = 0; mode < n_modes; ++mode)
          {
            auto column = scalar_column("data_" + labels.mode_labels[mode]);
            for (std::size_t q = 0; q < n_points; ++q)
              column.values.push_back(modal_data[q * n_modes + mode]);
            columns.push_back(std::move(column));
          }
      }

    if (options.selected("representative_point"))
      {
        auto vector = vector_columns("representative_point", spacedim);
        for (const auto &point : points)
          for (unsigned int d = 0; d < spacedim; ++d)
            vector[d].values.push_back(point.representative_point[d]);
        for (auto &column : vector)
          columns.push_back(std::move(column));
      }

    if (options.selected("radius"))
      {
        auto column = scalar_column("radius");
        for (const auto &point : points)
          column.values.push_back(
            point.point.distance(point.representative_point));
        columns.push_back(std::move(column));
      }

    if (options.selected("indices"))
      {
        auto cell      = scalar_column("source_cell");
        auto from_axis = scalar_column("representative_qpoint");
        auto around    = scalar_column("section_qpoint");
        for (const auto &point : points)
          {
            cell.values.push_back(static_cast<double>(point.source_entity_id));
            from_axis.values.push_back(
              static_cast<double>(point.representative_qpoint));
            around.values.push_back(static_cast<double>(point.section_qpoint));
          }
        columns.push_back(std::move(cell));
        columns.push_back(std::move(from_axis));
        columns.push_back(std::move(around));
      }

    return columns;
  }

  /** Sample a finite element function on a curve at the representative points
   * of the wall points. The result is point-major with one entry per FE
   * component.
   *
   * The wall points of one representative quadrature point share the same
   * value, so each representative point is evaluated once. The coefficients
   * must be a ghosted vector with constraints already distributed.
   */
  template <int dim, int spacedim, typename VectorType>
  std::vector<double>
  sample_at_representative_points(
    const dealii::DoFHandler<dim, spacedim>      &dof_handler,
    const dealii::Mapping<dim, spacedim>         &mapping,
    const VectorType                             &coefficients,
    const std::vector<LiftedWallPoint<spacedim>> &points)
  {
    const unsigned int  n_components = dof_handler.get_fe().n_components();
    std::vector<double> result;
    result.reserve(points.size() * n_components);
    std::map<std::pair<dealii::CellId, unsigned int>, std::vector<double>>
      cache;
    for (const auto &point : points)
      {
        const auto key =
          std::make_pair(point.source_cell_id, point.representative_qpoint);
        auto found = cache.find(key);
        if (found == cache.end())
          {
            const auto cell = dof_handler.get_triangulation()
                                .create_cell_iterator(point.source_cell_id)
                                ->as_dof_handler_iterator(dof_handler);
            const auto reference =
              mapping.transform_real_to_unit_cell(cell,
                                                  point.representative_point);
            dealii::FEValues<dim, spacedim> values(
              mapping,
              cell->get_fe(),
              dealii::Quadrature<dim>(
                std::vector<dealii::Point<dim>>{reference}),
              dealii::update_values);
            values.reinit(cell);
            std::vector<dealii::Vector<double>> samples(
              1, dealii::Vector<double>(n_components));
            values.get_function_values(coefficients, samples);
            found = cache
                      .emplace(key,
                               std::vector<double>(samples[0].begin(),
                                                   samples[0].end()))
                      .first;
          }
        result.insert(result.end(), found->second.begin(), found->second.end());
      }
    return result;
  }

  /** Background field sampled at the wall particles. The name labels the
   * output; n_components consecutive FE components starting at
   * first_component are interpolated and written as a scalar or a vector.
   * values may be null if the bulk_field property is not selected.
   */
  template <int spacedim, typename VectorType>
  struct LiftedBulkField
  {
    std::string                         name;
    const dealii::DoFHandler<spacedim> &dof_handler;
    const dealii::Mapping<spacedim>    &mapping;
    const VectorType                   *values          = nullptr;
    unsigned int                        first_component = 0;
    unsigned int                        n_components    = 1;
  };

  namespace detail
  {
    /** Particle coupling whose handler can be written to. */
    template <int spacedim>
    class LiftedParticleSearch : public ParticleCoupling<spacedim>
    {
    public:
      using ParticleCoupling<spacedim>::ParticleCoupling;

      dealii::Particles::ParticleHandler<spacedim> &
      handler()
      {
        return this->particles;
      }
    };
  } // namespace detail

  /** Write the lifted wall as a particle file.
   *
   * The wall points are located in the background mesh, so each particle is
   * owned by the rank holding the background cell it lies in. Points outside
   * the background are dropped. The call is collective; every rank must call
   * it, also with no wall points.
   *
   * @param columns Columns from make_lifted_particle_columns().
   * @param points The wall points the columns were built from.
   * @param bulk The background mesh and, if the bulk_field property is
   * selected, the background field to interpolate.
   */
  template <int spacedim, typename VectorType>
  void
  write_lifted_particles(const LiftedParticleOutputParameters         &options,
                         const std::vector<LiftedParticleColumn>      &columns,
                         const std::vector<LiftedWallPoint<spacedim>> &points,
                         const LiftedBulkField<spacedim, VectorType>  &bulk,
                         const std::string                            &filename)
  {
    options.validate();
    const bool sample_bulk = options.selected("bulk_field");
    AssertThrow(!sample_bulk ||
                  (bulk.values != nullptr && bulk.n_components > 0 &&
                   bulk.first_component + bulk.n_components <=
                     bulk.dof_handler.get_fe().n_components()),
                dealii::ExcMessage(
                  "The bulk_field property needs the background field values "
                  "and a valid component range."));
    for (const auto &column : columns)
      AssertThrow(column.values.size() == points.size(),
                  dealii::ExcMessage(
                    "Each particle column needs one value per wall point."));

    const auto        &tria         = bulk.dof_handler.get_triangulation();
    const MPI_Comm     communicator = tria.get_mpi_communicator();
    const unsigned int n_source     = columns.size();
    const unsigned int n_bulk       = sample_bulk ? bulk.n_components : 0;
    const unsigned int n_total      = n_source + n_bulk;
    AssertThrow(n_total > 0,
                dealii::ExcMessage(
                  "No particle property is selected for the lifted wall."));

    std::vector<dealii::Point<spacedim>> positions;
    std::vector<std::vector<double>>     properties;
    positions.reserve(points.size());
    properties.reserve(points.size());
    for (std::size_t q = 0; q < points.size(); ++q)
      {
        positions.push_back(points[q].point);
        std::vector<double> row(n_total, 0.);
        for (unsigned int c = 0; c < n_source; ++c)
          row[c] = columns[c].values[q];
        properties.push_back(std::move(row));
      }

    ParticleCouplingParameters<spacedim> search_parameters(
      "/Lifted particle search/");
    detail::LiftedParticleSearch<spacedim> search(search_parameters);
    search.initialize_particle_handler(tria, bulk.mapping, n_total);
    search.insert_points(positions, properties);

    auto &handler = search.handler();
    if (sample_bulk)
      {
        const auto &fe       = bulk.dof_handler.get_fe();
        auto        particle = handler.begin();
        while (particle != handler.end())
          {
            const auto &cell = particle->get_surrounding_cell();
            const auto  dh_cell =
              typename dealii::DoFHandler<spacedim>::cell_iterator(
                *cell, &bulk.dof_handler);
            const auto range = handler.particles_in_cell(cell);
            std::vector<dealii::Point<spacedim>> reference_points;
            for (const auto &item : range)
              reference_points.push_back(item.get_reference_location());
            dealii::FEValues<spacedim> values(bulk.mapping,
                                              fe,
                                              dealii::Quadrature<spacedim>(
                                                reference_points),
                                              dealii::update_values);
            values.reinit(dh_cell);
            std::vector<dealii::Vector<double>> samples(reference_points.size(),
                                                        dealii::Vector<double>(
                                                          fe.n_components()));
            values.get_function_values(*bulk.values, samples);
            unsigned int q = 0;
            for (auto item = range.begin(); item != range.end(); ++item, ++q)
              {
                auto data = item->get_properties();
                for (unsigned int c = 0; c < n_bulk; ++c)
                  data[n_source + c] = samples[q][bulk.first_component + c];
              }
            particle = range.end();
          }
      }

    std::vector<std::string> names;
    std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      interpretations;
    for (const auto &column : columns)
      {
        names.push_back(column.name);
        interpretations.push_back(column.interpretation);
      }
    for (unsigned int c = 0; c < n_bulk; ++c)
      {
        names.push_back(bulk.name);
        interpretations.push_back(
          n_bulk == 1 ?
            dealii::DataComponentInterpretation::component_is_scalar :
            dealii::DataComponentInterpretation::component_is_part_of_vector);
      }

    dealii::Particles::DataOut<spacedim> output;
    output.build_patches(handler, names, interpretations);

    const auto directory = std::filesystem::path(filename).parent_path();
    if (!directory.empty() &&
        dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      std::filesystem::create_directories(directory);
    MPI_Barrier(communicator);
    output.write_vtu_in_parallel(filename, communicator);
  }
} // namespace ImmersX

#endif // immersx_lifted_particle_output_h
