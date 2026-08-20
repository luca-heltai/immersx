// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/polynomials_p.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_tools_topology.h>

#include <immersx/core/reduced_field_utils.h>
#include <immersx/coupling/legacy_inclusions.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>

namespace ImmersX
{
  namespace
  {
    template <int spacedim>
    std::vector<unsigned int>
    make_fourier_to_reference_indices(const unsigned int n_coefficients,
                                      const unsigned int n_components,
                                      const unsigned int polynomial_degree)
    {
      const dealii::PolynomialsP<spacedim> polynomials(polynomial_degree);
      const auto invalid = std::numeric_limits<unsigned int>::max();
      std::vector<unsigned int> canonical_indices(polynomials.n(), invalid);
      unsigned int              n_in_plane = 0;
      for (unsigned int polynomial = 0; polynomial < polynomials.n();
           ++polynomial)
        {
          const auto exponents = polynomials.directional_degrees(polynomial);
          bool       in_plane  = true;
          for (unsigned int d = 2; d < spacedim; ++d)
            in_plane = in_plane && exponents[d] == 0;
          if (in_plane)
            canonical_indices[polynomial] = n_in_plane++;
        }

      std::vector<unsigned int> result;
      result.reserve(n_coefficients * n_components);
      for (unsigned int mode = 0; mode < n_coefficients; ++mode)
        {
          std::array<unsigned int, spacedim> target{};
          if (mode != 0)
            {
              if (mode % 2 == 1)
                target[0] = (mode + 1) / 2;
              else
                {
                  target[0] = mode / 2 - 1;
                  target[1] = 1;
                }
            }

          unsigned int canonical_scalar_index = invalid;
          for (unsigned int polynomial = 0; polynomial < polynomials.n();
               ++polynomial)
            if (canonical_indices[polynomial] != invalid &&
                polynomials.directional_degrees(polynomial) == target)
              {
                canonical_scalar_index = canonical_indices[polynomial];
                break;
              }

          AssertThrow(canonical_scalar_index != invalid,
                      dealii::ExcMessage(
                        "The requested legacy Fourier mode is not present in "
                        "the canonical polynomial basis."));
          for (unsigned int component = 0; component < n_components;
               ++component)
            result.push_back(canonical_scalar_index * n_components + component);
        }

      return result;
    }

    struct LegacyRecord
    {
      dealii::Point<3>     center;
      dealii::Tensor<1, 3> direction;
      double               radius    = 0.;
      double               vessel_id = 0.;
    };

    std::vector<std::vector<double>>
    read_data(const std::string         &filename,
              const unsigned int         n_records,
              const unsigned int         n_values,
              const std::vector<double> &reference_data)
    {
      std::vector<std::vector<double>> data;
      data.reserve(n_records);

      if (filename.empty())
        {
          std::vector<double> defaults(n_values, 0.);
          for (unsigned int i = 0;
               i < std::min<unsigned int>(n_values, reference_data.size());
               ++i)
            defaults[i] = reference_data[i];
          data.assign(n_records, defaults);
          return data;
        }

      std::ifstream input(filename);
      AssertThrow(input, dealii::ExcIO());
      std::string line;
      while (std::getline(input, line))
        {
          std::istringstream  stream(line);
          std::vector<double> row;
          double              value = 0.;
          while (stream >> value)
            row.push_back(value);
          if (!row.empty())
            data.push_back(std::move(row));
        }

      AssertThrow(data.size() == n_records,
                  dealii::ExcDimensionMismatch(data.size(), n_records));
      for (const auto &row : data)
        AssertThrow(row.size() >= n_values,
                    dealii::ExcDimensionMismatch(row.size(), n_values));
      for (auto &row : data)
        row.resize(n_values);
      return data;
    }

    std::vector<LegacyRecord>
    read_2d_records(const std::string &filename)
    {
      std::ifstream input(filename);
      AssertThrow(input, dealii::ExcIO());

      std::vector<LegacyRecord> records;
      double                    value = 0.;
      while (input >> value)
        {
          LegacyRecord record;
          record.center[0] = value;
          AssertThrow(input >> record.center[1] >> record.radius,
                      dealii::ExcIO());
          AssertThrow(std::isfinite(record.radius) && record.radius > 0.,
                      dealii::ExcMessage(
                        "Legacy inclusion radii must be finite and positive."));
          records.push_back(record);
        }
      return records;
    }

    std::vector<LegacyRecord>
    read_3d_records(const std::string &filename)
    {
      std::ifstream input(filename);
      AssertThrow(input, dealii::ExcIO());

      std::vector<LegacyRecord> records;
      double                    value = 0.;
      while (input >> value)
        {
          LegacyRecord record;
          record.center[0] = value;
          AssertThrow(input >> record.center[1] >> record.center[2] >>
                        record.direction[0] >> record.direction[1] >>
                        record.direction[2] >> record.radius >>
                        record.vessel_id,
                      dealii::ExcIO());
          AssertThrow(std::isfinite(record.radius) && record.radius > 0.,
                      dealii::ExcMessage(
                        "Legacy inclusion radii must be finite and positive."));
          AssertThrow(std::isfinite(record.vessel_id), dealii::ExcIO());
          AssertThrow(record.direction.norm() > 0.,
                      dealii::ExcMessage(
                        "Legacy inclusion directions must be non-zero."));
          records.push_back(record);
        }
      return records;
    }

    void
    append_scalar_catalog_field(
      const std::string     &name,
      FieldCatalog          &catalog,
      const FieldAssociation association = FieldAssociation::cell_data)
    {
      ReducedFieldDescriptor descriptor;
      descriptor.name         = name;
      descriptor.association  = association;
      descriptor.n_components = 1;
      descriptor.first_fe_component =
        catalog.empty() ?
          0 :
          catalog.back().first_fe_component + catalog.back().n_components;
      descriptor.block_index = catalog.size();
      catalog.push_back(descriptor);
    }

    dealii::Vector<double>
    make_field_major_data(const std::vector<double>              &radius,
                          const std::vector<double>              &vessel_id,
                          const std::vector<std::vector<double>> &coefficients,
                          const bool include_vessel)
    {
      const unsigned int n_records = radius.size();
      unsigned int       n_fields  = 1 + coefficients.front().size();
      if (include_vessel)
        ++n_fields;

      dealii::Vector<double> result(n_fields * n_records);
      unsigned int           offset = 0;
      for (const auto value : radius)
        result[offset++] = value;
      if (include_vessel)
        for (const auto value : vessel_id)
          result[offset++] = value;
      for (unsigned int coefficient = 0;
           coefficient < coefficients.front().size();
           ++coefficient)
        for (const auto &row : coefficients)
          result[offset++] = row[coefficient];
      return result;
    }

    std::vector<std::vector<double>>
    normalized_coefficients(const std::vector<std::vector<double>> &data,
                            const unsigned int n_components)
    {
      std::vector<std::vector<double>> result = data;
      for (auto &row : result)
        for (unsigned int i = 0; i < row.size(); ++i)
          row[i] *=
            LegacyInclusions::fourier_to_reference_scale(i / n_components);
      return result;
    }
  } // namespace

  namespace LegacyInclusions
  {
    double
    fourier_to_reference_scale(const unsigned int mode)
    {
      return mode == 0 ? 1. : 1. / std::sqrt(2.);
    }

    std::vector<unsigned int>
    fourier_to_reference_indices(const unsigned int n_coefficients,
                                 const unsigned int n_components,
                                 const unsigned int polynomial_degree,
                                 const unsigned int spacedim)
    {
      AssertThrow(n_components > 0,
                  dealii::ExcMessage(
                    "The number of components must be positive."));
      AssertThrow(spacedim == 2 || spacedim == 3, dealii::ExcNotImplemented());
      if (spacedim == 2)
        return make_fourier_to_reference_indices<2>(n_coefficients,
                                                    n_components,
                                                    polynomial_degree);
      return make_fourier_to_reference_indices<3>(n_coefficients,
                                                  n_components,
                                                  polynomial_degree);
    }

    void
    read_2d(const std::string         &inclusions_file,
            const std::string         &data_file,
            const unsigned int         n_coefficients,
            const unsigned int         n_components,
            const std::vector<double> &reference_data,
            PointCloud<2>             &point_cloud)
    {
      AssertThrow(n_coefficients > 0,
                  dealii::ExcMessage(
                    "The legacy number of coefficients must be positive."));
      AssertThrow(n_components > 0,
                  dealii::ExcMessage(
                    "The number of components must be positive."));

      const auto records = read_2d_records(inclusions_file);
      AssertThrow(
        !records.empty(),
        dealii::ExcMessage(
          "The legacy inclusion file must contain at least one record."));
      const auto data         = read_data(data_file,
                                  records.size(),
                                  n_coefficients * n_components,
                                  reference_data);
      const auto coefficients = normalized_coefficients(data, n_components);

      point_cloud = PointCloud<2>();
      point_cloud.points.reserve(records.size());
      for (const auto &record : records)
        point_cloud.points.emplace_back(record.center[0], record.center[1]);
      point_cloud.distribution = PointCloudDistribution::replicated;

      append_scalar_catalog_field("radius",
                                  point_cloud.catalog,
                                  FieldAssociation::point_data);
      for (unsigned int coefficient = 0;
           coefficient < n_coefficients * n_components;
           ++coefficient)
        append_scalar_catalog_field("coefficient_" +
                                      std::to_string(coefficient),
                                    point_cloud.catalog,
                                    FieldAssociation::point_data);

      point_cloud.property_names.reserve(point_cloud.catalog.size());
      for (const auto &field : point_cloud.catalog)
        point_cloud.property_names.push_back(field.name);

      point_cloud.properties.resize(point_cloud.catalog.size());
      point_cloud.properties[0].reserve(records.size());
      for (const auto &record : records)
        point_cloud.properties[0].push_back(record.radius);
      for (unsigned int coefficient = 0;
           coefficient < n_coefficients * n_components;
           ++coefficient)
        {
          auto &values = point_cloud.properties[coefficient + 1];
          values.reserve(records.size());
          for (const auto &row : coefficients)
            values.push_back(row[coefficient]);
        }
    }

    void
    read_3d(const std::string           &inclusions_file,
            const std::string           &data_file,
            const unsigned int           n_coefficients,
            const unsigned int           n_components,
            const std::vector<double>   &reference_data,
            dealii::Triangulation<1, 3> &tria,
            dealii::DoFHandler<1, 3>    &properties_dh,
            dealii::Vector<double>      &properties,
            FieldCatalog                &catalog)
    {
      AssertThrow(n_coefficients > 0,
                  dealii::ExcMessage(
                    "The legacy number of coefficients must be positive."));
      AssertThrow(n_components > 0,
                  dealii::ExcMessage(
                    "The number of components must be positive."));

      const auto records = read_3d_records(inclusions_file);
      AssertThrow(
        !records.empty(),
        dealii::ExcMessage(
          "The legacy inclusion file must contain at least one record."));
      const auto data         = read_data(data_file,
                                  records.size(),
                                  n_coefficients * n_components,
                                  reference_data);
      const auto coefficients = normalized_coefficients(data, n_components);

      std::vector<dealii::Point<3>>    vertices;
      std::vector<dealii::CellData<1>> cells;
      vertices.reserve(2 * records.size());
      cells.reserve(records.size());
      double scale = 1.;
      for (const auto &record : records)
        {
          scale = std::max(scale, record.center.norm());
          scale =
            std::max(scale, (record.center - record.direction / 2.).norm());
          scale =
            std::max(scale, (record.center + record.direction / 2.).norm());
          dealii::CellData<1> cell;
          cell.vertices[0] = vertices.size();
          vertices.push_back(record.center - record.direction / 2.);
          cell.vertices[1] = vertices.size();
          vertices.push_back(record.center + record.direction / 2.);
          cell.material_id = 0;
          cells.push_back(cell);
        }

      dealii::SubCellData       subcell_data;
      std::vector<unsigned int> considered_vertices;
      dealii::GridTools::delete_duplicated_vertices(
        vertices, cells, subcell_data, considered_vertices, 1e-12 * scale);
      tria.clear();
      tria.create_triangulation(vertices, cells, subcell_data);

      catalog.clear();
      append_scalar_catalog_field("radius", catalog);
      append_scalar_catalog_field("vessel_id", catalog);
      for (unsigned int coefficient = 0;
           coefficient < n_coefficients * n_components;
           ++coefficient)
        append_scalar_catalog_field("coefficient_" +
                                      std::to_string(coefficient),
                                    catalog);

      std::vector<double> radius;
      std::vector<double> vessel_id;
      radius.reserve(records.size());
      vessel_id.reserve(records.size());
      for (const auto &record : records)
        {
          radius.push_back(record.radius);
          vessel_id.push_back(record.vessel_id);
        }
      const auto raw_data =
        make_field_major_data(radius, vessel_id, coefficients, true);

      auto fe =
        ReducedFieldUtils::field_catalog_to_finite_element<1, 3>(catalog);
      properties_dh.clear();
      properties_dh.distribute_dofs(*fe);
      properties.reinit(properties_dh.n_dofs());
      ReducedFieldUtils::data_to_dealii_vector(tria,
                                               raw_data,
                                               properties_dh,
                                               properties);
    }
  } // namespace LegacyInclusions
} // namespace ImmersX
