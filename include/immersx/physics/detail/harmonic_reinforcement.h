// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_detail_harmonic_reinforcement_h
#define immersx_detail_harmonic_reinforcement_h

#include <deal.II/fe/fe_nothing.h>

#include <immersx/core/lifted_integration_support.h>
#include <immersx/core/observable_lift.h>
#include <immersx/physics/elasticity_freq_contributor.h>

#include <map>
#include <set>

namespace ImmersX::harmonic_detail
{
  /** A material-labelled volume support, including independent full meshes. */
  template <int spacedim>
  class ReinforcementSupport
  {
  public:
    static constexpr int intrinsic_dimension = spacedim;
    static constexpr int ambient_dimension   = spacedim;
    struct Point
    {
      dealii::Point<spacedim>    point;
      double                     weight;
      std::uint64_t              stable_id;
      unsigned int               owner_rank;
      dealii::types::material_id material_id;
    };
    ReinforcementSupport(std::vector<Point> points, MPI_Comm communicator)
      : points_(std::make_shared<const std::vector<Point>>(std::move(points)))
      , communicator_(communicator)
    {}
    const auto &
    locally_owned_quadrature_points() const
    {
      return *points_;
    }
    MPI_Comm
    mpi_communicator() const
    {
      return communicator_;
    }
    ReinforcementSupport
    with_material(const dealii::types::material_id id) const
    {
      std::vector<Point> points;
      for (const auto &p : *points_)
        if (p.material_id == id)
          points.push_back(p);
      return {std::move(points), communicator_};
    }
    std::set<dealii::types::material_id>
    material_ids() const
    {
      std::set<dealii::types::material_id> local, global;
      for (const auto &p : *points_)
        local.insert(p.material_id);
      for (const auto &part :
           dealii::Utilities::MPI::all_gather(communicator_, local))
        global.insert(part.begin(), part.end());
      return global;
    }
    double
    measure() const
    {
      double local = 0.;
      for (const auto &p : *points_)
        local += p.weight;
      return dealii::Utilities::MPI::sum(local, communicator_);
    }

  private:
    std::shared_ptr<const std::vector<Point>> points_;
    MPI_Comm                                  communicator_;
  };

  template <int dim, int spacedim, typename Lift>
  auto
  make_reinforcement_support(
    const ElasticityFreqProblem<dim, spacedim> &problem,
    const Lift                                 *lift)
  {
    using Support = ReinforcementSupport<spacedim>;
    std::vector<typename Support::Point> points;
    if constexpr (dim < spacedim)
      {
        std::map<dealii::types::global_cell_index, dealii::types::material_id>
          ids;
        for (const auto &cell : problem.triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            ids[cell->global_active_cell_index()] = cell->material_id();
        const auto geometry = problem.geometric_domain().lift(*lift);
        for (const auto &p : geometry.locally_owned_quadrature_points())
          points.push_back({p.point,
                            p.weight,
                            p.stable_id,
                            p.owner_rank,
                            ids.at(p.source_entity_id)});
      }
    else
      {
        const auto rank =
          dealii::Utilities::MPI::this_mpi_process(problem.communicator());
        const auto reference =
          detail::uniform_reference_cell(problem.triangulation());
        AssertThrow(reference.has_value(),
                    dealii::ExcMessage("The reinforcement mesh has no cells."));
        const auto quadrature = compatibility::gauss_quadrature<dim>(
          *reference, problem.parameters().fe_degree + 2);
        const dealii::FE_Nothing<dim, spacedim> fe(*reference);
        dealii::FEValues<dim, spacedim>         values(
          compatibility::linear_mapping<dim, spacedim>(*reference),
          fe,
          quadrature,
          dealii::update_quadrature_points | dealii::update_JxW_values);
        for (const auto &cell : problem.triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              values.reinit(cell);
              for (const auto q : values.quadrature_point_indices())
                points.push_back({values.quadrature_point(q),
                                  values.JxW(q),
                                  static_cast<std::uint64_t>(
                                    cell->global_active_cell_index()) *
                                      quadrature.size() +
                                    q,
                                  rank,
                                  cell->material_id()});
            }
      }
    return Support(std::move(points), problem.communicator());
  }

  template <typename Builder, int dim, int spacedim>
  void
  add_shared_reinforcement(
    Builder                                              &builder,
    const ElasticityFreqFields<spacedim, spacedim>       &fields,
    const ElasticityFreqProblemParameters<dim, spacedim> &parameters,
    const ReinforcementSupport<spacedim>                 &geometry,
    const double                                          omega)
  {
    for (const auto id : geometry.material_ids())
      {
        const auto support = geometry.with_material(id);
        add_material(builder,
                     fields,
                     parameters.get_material_properties(id),
                     support,
                     omega);
        const auto forcing = [&](const auto &row, const bool imaginary) {
          KnownSource<dealii::Tensor<1, spacedim>, spacedim> source(
            [&parameters, id, imaginary](const auto &context) {
              dealii::Tensor<1, spacedim> result;
              const double factor = phase_component(parameters.rhs_phase_by_id,
                                                    id,
                                                    parameters.rhs_phase,
                                                    imaginary);
              for (unsigned int d = 0; d < spacedim; ++d)
                result[d] =
                  -factor * parameters.get_rhs(id).value(context.point, d);
              return result;
            });
          known_term(source, test(row)).on(support).add(builder);
        };
        forcing(fields.real, false);
        forcing(fields.imaginary, true);
      }
  }

  /** Constant normal extension; each ambient component has its own equation. */
  template <typename Builder,
            typename MatrixField,
            typename FiberField,
            typename Multiplier,
            typename Lift>
  auto
  add_lifted_continuity(Builder           &builder,
                        const MatrixField &matrix,
                        const FiberField  &fiber,
                        const Multiplier  &multiplier,
                        const Lift        &geometry,
                        const double       section_measure)
  {
    constexpr int spacedim      = FiberField::spacedimension();
    using Scalar                = dealii::FEValuesExtractors::Scalar;
    const auto matrix_component = [&](unsigned int c) {
      return matrix.space().field(matrix.id(), matrix.name(), Scalar(c));
    };
    const auto fiber_component = [&](unsigned int c) {
      return fiber.space().field(fiber.id(), fiber.name(), Scalar(c));
    };
    const auto multiplier_component = [&](unsigned int c) {
      return multiplier.space().field(multiplier.name(), Scalar(c));
    };
    const auto bulk_term = [&](unsigned int c) {
      return weak_term(value(matrix_component(c)),
                       test(lift(value(multiplier_component(c)), geometry)));
    };
    const auto fiber_term = [&](unsigned int c) {
      return weak_term(section_measure * value(fiber_component(c)),
                       test(multiplier_component(c)));
    };
    auto terms = bulk_term(0) - fiber_term(0);
    for (unsigned int c = 1; c < spacedim; ++c)
      {
        terms.add(bulk_term(c));
        terms.add(fiber_term(c), -1.);
      }
    const auto metric = detail::make_multiplier_metric(builder, multiplier);
    const auto registered =
      make_constraint(std::move(terms)).add(builder, &metric);
    return multiplier.with_id(registered.multiplier.id());
  }
} // namespace ImmersX::harmonic_detail
#endif
