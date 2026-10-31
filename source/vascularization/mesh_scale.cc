// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <immersx/vascularization/mesh_scale.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "background_intersections.h"

namespace ImmersX::Vascularization
{
  void
  MeshLengthScale::validate() const
  {
    AssertThrow(std::isfinite(mesh_minimum) && mesh_minimum > 0. &&
                  std::isfinite(mesh_maximum) && mesh_maximum >= mesh_minimum &&
                  (policy == LengthScalePolicy::meter ||
                   policy == LengthScalePolicy::local_mesh_size ||
                   policy == LengthScalePolicy::min_mesh_size ||
                   policy == LengthScalePolicy::max_mesh_size),
                dealii::ExcMessage("Invalid mesh length scale."));
  }

  double
  MeshLengthScale::value(double local_diameter) const
  {
    validate();
    if (policy == LengthScalePolicy::meter)
      return 1.;
    else if (policy == LengthScalePolicy::local_mesh_size)
      {
        AssertThrow(std::isfinite(local_diameter) && local_diameter > 0.,
                    dealii::ExcMessage(
                      "Local mesh size is undefined outside the background."));
        return local_diameter;
      }
    else if (policy == LengthScalePolicy::min_mesh_size)
      return mesh_minimum;
    else if (policy == LengthScalePolicy::max_mesh_size)
      return mesh_maximum;
    AssertThrow(false, dealii::ExcInternalError());
    return 0.;
  }

  double
  MeshLengthScale::minimum() const
  {
    return value(mesh_minimum);
  }
  double
  MeshLengthScale::maximum() const
  {
    return value(mesh_maximum);
  }

  double
  MeshLengthScale::physical_length(double coefficient,
                                   double local_diameter) const
  {
    validate();
    AssertThrow(std::isfinite(coefficient) && coefficient >= 0.,
                dealii::ExcMessage(
                  "Length coefficients must be finite and nonnegative."));
    if (coefficient == 0.)
      return 0.;
    const double result = coefficient * value(local_diameter);
    AssertThrow(std::isfinite(result) && result > 0.,
                dealii::ExcMessage(
                  "Scaled length is not finite and representable."));
    return result;
  }

  std::string
  MeshLengthScale::name() const
  {
    if (policy == LengthScalePolicy::meter)
      return "meter";
    else if (policy == LengthScalePolicy::local_mesh_size)
      return "local mesh size";
    else if (policy == LengthScalePolicy::min_mesh_size)
      return "min mesh size";
    else if (policy == LengthScalePolicy::max_mesh_size)
      return "max mesh size";
    AssertThrow(false, dealii::ExcInternalError());
    return {};
  }

  MeshLengthScale
  mesh_length_scale(const dealii::Triangulation<3> &background,
                    const std::string              &policy,
                    MPI_Comm                        communicator)
  {
    using namespace dealii;
    for (const auto &other : Utilities::MPI::all_gather(communicator, policy))
      AssertThrow(other == policy,
                  ExcMessage("Length scale policies must agree across ranks."));
    MeshLengthScale result;
    if (policy == "meter")
      result.policy = LengthScalePolicy::meter;
    else if (policy == "local mesh size")
      result.policy = LengthScalePolicy::local_mesh_size;
    else if (policy == "min mesh size")
      result.policy = LengthScalePolicy::min_mesh_size;
    else if (policy == "max mesh size")
      result.policy = LengthScalePolicy::max_mesh_size;
    else
      AssertThrow(false, ExcMessage("Unknown length scale policy."));
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(comparison == MPI_IDENT ||
                                                    comparison ==
                                                      MPI_CONGRUENT),
                          communicator) == 1,
      ExcMessage(
        "Background and length-scale communicators must be congruent."));
    result.mesh_minimum = std::numeric_limits<double>::infinity();
    result.mesh_maximum = 0.;
    bool valid          = true;
    if (background.n_levels() > 0)
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const double h      = cell->diameter();
            valid               = valid && std::isfinite(h) && h > 0.;
            result.mesh_minimum = std::min(result.mesh_minimum, h);
            result.mesh_maximum = std::max(result.mesh_maximum, h);
          }
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), communicator) == 1,
      ExcMessage("Background diameters must be finite and positive."));
    result.mesh_minimum =
      Utilities::MPI::min(result.mesh_minimum, communicator);
    result.mesh_maximum =
      Utilities::MPI::max(result.mesh_maximum, communicator);
    result.validate();
    return result;
  }

  ScaledVesselMeasure
  internal::measure_background_profile(
    double                                 physical_length,
    const std::array<double, 2>           &radii,
    const std::vector<BackgroundInterval> &intervals,
    const MeshLengthScale                 &scale)
  {
    ScaledVesselMeasure measure;
    for (const auto &interval : intervals)
      {
        const double piece = physical_length * (interval.end - interval.begin);
        if (interval.h > 0.)
          measure.inside_length += piece;
        else
          measure.exterior_length += piece;
        if (interval.h == 0. &&
            scale.policy == LengthScalePolicy::local_mesh_size)
          {
            measure.complete = false;
            continue;
          }
        const double h = scale.value(interval.h);
        const double a = radii[0] + interval.begin * (radii[1] - radii[0]);
        const double b = radii[0] + interval.end * (radii[1] - radii[0]);
        measure.normalized_length += piece / h;
        const double lo = std::min(a, b) / h, hi = std::max(a, b) / h;
        if (!measure.has_normalized_part)
          {
            measure.minimum_scale = measure.maximum_scale = h;
            measure.minimum_normalized_radius             = lo;
            measure.maximum_normalized_radius             = hi;
            measure.has_normalized_part                   = true;
          }
        else
          {
            measure.minimum_scale = std::min(measure.minimum_scale, h);
            measure.maximum_scale = std::max(measure.maximum_scale, h);
            measure.minimum_normalized_radius =
              std::min(measure.minimum_normalized_radius, lo);
            measure.maximum_normalized_radius =
              std::max(measure.maximum_normalized_radius, hi);
          }
      }
    return measure;
  }

  std::map<Id, ScaledVesselMeasure>
  measure_scaled_vessels(const dealii::Triangulation<3> &background,
                         const dealii::Mapping<3>       &mapping,
                         const NetworkData              &network,
                         const MeshLengthScale          &scale,
                         MPI_Comm                        communicator)
  {
    using namespace dealii;
    const auto rank  = Utilities::MPI::this_mpi_process(communicator);
    const auto size  = Utilities::MPI::n_mpi_processes(communicator);
    bool       valid = true;
    try
      {
        scale.validate();
      }
    catch (const std::exception &)
      {
        valid = false;
      }
    const auto *q          = dynamic_cast<const MappingQ<3> *>(&mapping);
    const auto *fe         = dynamic_cast<const MappingFE<3> *>(&mapping);
    int         comparison = MPI_UNEQUAL;
    MPI_Comm_compare(background.get_mpi_communicator(),
                     communicator,
                     &comparison);
    valid = valid && (comparison == MPI_IDENT || comparison == MPI_CONGRUENT) &&
            ((q && q->get_degree() == 1) || (fe && fe->get_degree() == 1)) &&
            validate_data(network, size).valid();
    valid = valid && background.n_levels() > 0;
    if (background.n_levels() > 0)
      for (const auto &cell : background.active_cell_iterators())
        if (cell->is_locally_owned())
          valid = valid && mapping.is_compatible_with(cell->reference_cell()) &&
                  (cell->reference_cell().is_simplex() ||
                   cell->reference_cell().is_hyper_cube());
    AssertThrow(
      Utilities::MPI::min(static_cast<unsigned int>(valid), communicator) == 1,
      ExcMessage(
        "Scaled vessel measurement requires valid data, first-order mappings and congruent communicators."));
    const std::array<double, 3> controls{{static_cast<double>(scale.policy),
                                          scale.mesh_minimum,
                                          scale.mesh_maximum}};
    for (const auto &other : Utilities::MPI::all_gather(communicator, controls))
      AssertThrow(
        other == controls,
        ExcMessage(
          "Scaled vessel measurement controls must agree across ranks."));
    std::map<Id, Segment<3>> segments;
    for (const auto &[id, vessel] : network.vessels)
      if (vessel.owner == rank)
        segments.emplace(
          id,
          Segment<3>(network.nodes.at(vessel.nodes[0]).position,
                     network.nodes.at(vessel.nodes[1]).position));
    GridTools::Cache<3> cache(background, mapping);
    const auto          profiles =
      internal::intersect_background(cache, mapping, segments, communicator);
    std::map<Id, ScaledVesselMeasure> result;
    for (const auto &[id, intervals] : profiles)
      {
        auto        &measure = result[id];
        const auto  &vessel  = network.vessels.at(id);
        const double physical =
          segments.at(id).first.distance(segments.at(id).second);
        measure = internal::measure_background_profile(physical,
                                                       vessel.radii,
                                                       intervals,
                                                       scale);
      }
    bool finite_measures = true;
    for (const auto &[id, measure] : result)
      finite_measures = finite_measures &&
                        std::isfinite(measure.normalized_length) &&
                        std::isfinite(measure.inside_length) &&
                        std::isfinite(measure.exterior_length) &&
                        std::isfinite(measure.minimum_normalized_radius) &&
                        std::isfinite(measure.maximum_normalized_radius);
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(finite_measures),
                                    communicator) == 1,
                ExcMessage("Scaled vessel measures are not representable."));
    return result;
  }
} // namespace ImmersX::Vascularization
