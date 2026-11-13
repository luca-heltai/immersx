// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_detail_harmonic_robin_boundary_h
#define immersx_detail_harmonic_robin_boundary_h

#include <immersx/core/robin_boundary.h>
#include <immersx/physics/detail/harmonic_reinforcement.h>

#include <variant>

namespace ImmersX::harmonic_detail
{
  /** Physical representative boundary extruded through the fiber section. */
  template <int dim, int spacedim, typename Lift>
  auto
  make_reinforcement_boundary_supports(
    const ElasticityFreqProblem<dim, spacedim> &problem,
    const Lift                                 *lift,
    const dealii::GridTools::Cache<spacedim>   &background)
  {
    using Support    = ReinforcementSupport<spacedim>;
    using BoundaryId = dealii::types::boundary_id;
    std::map<BoundaryId, Support> result;
    const auto                   &parameters = problem.parameters();
    if (parameters.robin_ids.empty())
      return result;
    const auto reference =
      detail::uniform_reference_cell(problem.triangulation());
    AssertThrow(reference.has_value(),
                dealii::ExcMessage("The reinforcement mesh has no cells."));
    const auto quadrature = compatibility::gauss_quadrature<dim - 1>(
      reference->face_reference_cell(0), parameters.fe_degree + 2);
    const auto mapping = [&]() {
      if constexpr (dim < spacedim)
        return compatibility::default_mapping<dim, spacedim>(*reference, 2);
      else
        return compatibility::linear_mapping<dim, spacedim>(*reference).clone();
    }();
    const dealii::FE_Nothing<dim, spacedim> fe(*reference);
    dealii::FEFaceValues<dim, spacedim>     values(
      *mapping,
      fe,
      quadrature,
      dealii::update_quadrature_points | dealii::update_JxW_values |
        dealii::update_jacobians);
    const auto rank =
      dealii::Utilities::MPI::this_mpi_process(problem.communicator());
    const auto section = [&]() {
      if constexpr (dim < spacedim)
        {
          AssertThrow(lift != nullptr, dealii::ExcNotInitialized());
          return std::make_unique<
            TensorProductLiftSupport<dim, spacedim, spacedim>>(
            lift->parameters());
        }
      else
        return std::monostate{};
    }();
    for (const auto id : parameters.robin_ids)
      {
        std::vector<typename Support::Point> points;
        for (const auto &cell : problem.triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary() &&
                  cell->face(face)->boundary_id() == id)
                {
                  values.reinit(cell, face);
                  for (const auto q : values.quadrature_point_indices())
                    {
                      const auto base_id =
                        (static_cast<std::uint64_t>(
                           cell->global_active_cell_index()) *
                           cell->n_faces() +
                         face) *
                          quadrature.size() +
                        q;
                      if constexpr (dim < spacedim)
                        {
                          const auto &origin = values.quadrature_point(q);
                          const auto  lifted =
                            section->transform(origin,
                                               values.jacobian(q),
                                               values.JxW(q),
                                               section->thickness(origin, 0.),
                                               q);
                          const auto count = section->reference_cross_section()
                                               .n_quadrature_points();
                          for (const auto &point : lifted)
                            points.push_back(
                              {point.point,
                               point.weight,
                               base_id * count + point.section_qpoint,
                               rank,
                               0});
                        }
                      else
                        points.push_back({values.quadrature_point(q),
                                          values.JxW(q),
                                          base_id,
                                          rank,
                                          0});
                    }
                }
        result.emplace(id,
                       Support(std::move(points), problem.communicator())
                         .inside(background));
      }
    return result;
  }

  /** Add fiber boundary springs and loads to the shared matrix displacement. */
  template <typename Builder,
            typename Fields,
            typename Supports,
            typename Parameters>
  void
  add_shared_robin(Builder          &builder,
                   const Fields     &fields,
                   const Supports   &supports,
                   const Parameters &parameters)
  {
    constexpr int spacedim =
      std::decay_t<decltype(fields.real)>::spacedimension();
    using Support = ReinforcementSupport<spacedim>;
    using Value   = dealii::Tensor<1, spacedim>;
    if (supports.empty())
      return;
    const auto communicator = supports.begin()->second.mpi_communicator();
    std::vector<typename Support::Point> weighted;
    // Plain locals instead of structured bindings: the forcing lambda below
    // captures id, and capturing a structured binding requires C++20.
    for (const auto &support : supports)
      {
        const auto   id          = support.first;
        const auto  &geometry    = support.second;
        unsigned int invalid     = 0;
        const auto  &coefficient = parameters.get_robin_coefficient(id);
        const auto  &load        = parameters.get_robin_bc(id);
        for (const auto &point : geometry.locally_owned_quadrature_points())
          {
            try
              {
                const double beta  = coefficient.value(point.point);
                bool         valid = std::isfinite(beta) && beta >= 0. &&
                             std::isfinite(beta * point.weight);
                for (unsigned int c = 0; c < spacedim; ++c)
                  valid = valid && std::isfinite(load.value(point.point, c));
                if (!valid)
                  invalid = 1;
                else if (beta * point.weight > 0.)
                  {
                    auto entry = point;
                    entry.weight *= beta;
                    weighted.push_back(entry);
                  }
              }
            catch (const std::exception &)
              {
                invalid = 1;
              }
          }
        for (const bool imaginary : {false, true})
          if (!std::isfinite(phase_component(parameters.robin_phase_by_id,
                                             id,
                                             parameters.robin_phase,
                                             imaginary)))
            invalid = 1;
        robin_detail::assert_valid_boundary_data(invalid,
                                                 geometry.mpi_communicator());
        const auto forcing = [&](const auto &row, const bool imaginary) {
          KnownSource<dealii::Tensor<1, spacedim>, spacedim> source(
            [&parameters, id, imaginary](const auto &context) {
              Value        result;
              const double phase = phase_component(parameters.robin_phase_by_id,
                                                   id,
                                                   parameters.robin_phase,
                                                   imaginary);
              for (unsigned int c = 0; c < Value::dimension; ++c)
                result[c] =
                  -phase * parameters.get_robin_bc(id).value(context.point, c);
              return result;
            });
          known_term(source, test(row)).on(geometry).add(builder);
        };
        forcing(fields.real, false);
        forcing(fields.imaginary, true);
      }
    const Support spring(std::move(weighted), communicator);
    if (spring.measure() > 0.)
      {
        weak_term(value(fields.real), test(fields.real))
          .on(spring)
          .add(builder);
        weak_term(value(fields.imaginary), test(fields.imaginary))
          .on(spring)
          .add(builder);
        // The same clipped, beta-weighted boundary support must also anchor
        // the positive surrogate when no essential boundary is present.
        add_surrogate_terms(fields, spring, 0., 0., 1.);
      }
  }
} // namespace ImmersX::harmonic_detail
#endif
