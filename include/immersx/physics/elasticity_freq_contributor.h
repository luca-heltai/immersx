// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#ifndef immersx_elasticity_freq_contributor_h
#define immersx_elasticity_freq_contributor_h

#include <boost/serialization/set.hpp>

#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/known_term.h>
#include <immersx/core/weak_term.h>
#include <immersx/physics/elasticity_freq.h>
#include <immersx/physics/static_elasticity_term.h>

#include <cmath>
#include <memory>
#include <set>
#include <utility>

namespace ImmersX
{
  template <int dim, int spacedim = dim>
  struct ElasticityFreqFields
  {
    using Space = FiniteElementSpaceView<dim, spacedim>;
    using VectorField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;
    VectorField                  real;
    VectorField                  imaginary;
    std::shared_ptr<const Space> real_space;
    std::shared_ptr<const Space> imaginary_space;
  };

  namespace harmonic_detail
  {
    template <typename Map, typename Key>
    double
    phase_component(const Map   &phases,
                    const Key    id,
                    const double fallback,
                    const bool   imaginary)
    {
      const auto   entry = phases.find(id);
      const double phase = (entry == phases.end() ? fallback : entry->second) *
                           std::acos(-1.) / 180.;
      return imaginary ? std::sin(phase) : std::cos(phase);
    }
    template <int spacedim>
    class RobinAmplitude : public dealii::Function<spacedim>
    {
    public:
      RobinAmplitude(const dealii::Function<spacedim> &function,
                     const double                      factor)
        : dealii::Function<spacedim>(function.n_components)
        , function(function)
        , factor(factor)
      {}
      double
      value(const dealii::Point<spacedim> &point,
            const unsigned int             component = 0) const override
      {
        return factor * function.value(point, component);
      }
      double
      scale(const double) const
      {
        return 1.;
      }

    private:
      const dealii::Function<spacedim> &function;
      const double                      factor;
    };

    template <int dim, int spacedim>
    struct RobinParametersView
    {
      using Ids = std::set<dealii::types::boundary_id>;
      const Ids                                        &robin_ids;
      const Ids                                        &dirichlet_ids;
      const Ids                                        &neumann_ids;
      mutable dealii::Functions::ZeroFunction<spacedim> robin_coefficient{1};
      mutable dealii::Functions::ZeroFunction<spacedim> robin_rhs{spacedim};
      std::map<dealii::types::boundary_id,
               std::unique_ptr<RobinAmplitude<spacedim>>>
        coefficients, loads;
      RobinParametersView(
        const ElasticityFreqProblemParameters<dim, spacedim> &parameters,
        const double                                          weight,
        const bool                                            imaginary)
        : robin_ids(parameters.robin_ids)
        , dirichlet_ids(parameters.dirichlet_ids)
        , neumann_ids(parameters.neumann_ids)
      {
        parameters.set_boundary_condition_times(0.);
        for (const auto id : robin_ids)
          {
            coefficients.emplace(id,
                                 std::make_unique<RobinAmplitude<spacedim>>(
                                   parameters.get_robin_coefficient(id),
                                   weight));
            loads.emplace(id,
                          std::make_unique<RobinAmplitude<spacedim>>(
                            parameters.get_robin_bc(id),
                            weight *
                              phase_component(parameters.robin_phase_by_id,
                                              id,
                                              parameters.robin_phase,
                                              imaginary)));
          }
      }
      const RobinAmplitude<spacedim> &
      get_robin_coefficient(const dealii::types::boundary_id id) const
      {
        return *coefficients.at(id);
      }
      const RobinAmplitude<spacedim> &
      get_robin_rhs(const dealii::types::boundary_id id) const
      {
        return *loads.at(id);
      }
    };

    /** Add the same constitutive law on a native or independent support. */
    template <typename Builder, typename Fields, typename Support>
    void
    add_material(Builder                  &builder,
                 const Fields             &fields,
                 const MaterialProperties &material,
                 const Support            &support,
                 const double              omega,
                 const double              weight = 1.)
    {
      const auto &real       = fields.real;
      const auto &imaginary  = fields.imaginary;
      const auto  elasticity = [&](const auto  &source,
                                  const auto  &row,
                                  const double factor,
                                  const bool keep_constraint_diagonal = false) {
        const double shear = factor * 2. * material.Lame_mu;
        const double bulk  = factor * material.Lame_lambda;
        if (shear != 0. || keep_constraint_diagonal)
          (weak_term(shear * symmetric_gradient(source),
                     symmetric_gradient(test(row))) *
           support)
            .add(builder);
        if (bulk != 0.)
          (weak_term(bulk * divergence(source), divergence(test(row))) *
           support)
            .add(builder);
      };
      const auto inertia =
        [&](const auto &source, const auto &row, const double factor) {
          const double mass = factor * material.rho;
          if (mass != 0.)
            (weak_term(mass * value(source), test(row)) * support).add(builder);
        };
      const auto damping =
        [&](const auto &source, const auto &row, const double factor) {
          elasticity(source, row, factor * material.rayleigh_beta);
          inertia(source, row, factor * material.rayleigh_alpha);
          const double viscous = factor * material.neta;
          if (viscous != 0.)
            (weak_term(viscous * symmetric_gradient(source),
                       symmetric_gradient(test(row))) *
             support)
              .add(builder);
        };
      // Native essential rows are supplied by a diagonal weak term. Retain
      // one zero shear action per field when all diagonal coefficients vanish,
      // including purely viscous models whose physical terms are all
      // off-diagonal.
      const bool keep_constraint_diagonal =
        weight * 2. * material.Lame_mu == 0. &&
        weight * material.Lame_lambda == 0. &&
        -weight * omega * omega * material.rho == 0.;
      elasticity(real, real, weight, keep_constraint_diagonal);
      elasticity(imaginary, imaginary, weight, keep_constraint_diagonal);
      inertia(real, real, -weight * omega * omega);
      inertia(imaginary, imaginary, -weight * omega * omega);
      damping(imaginary, real, -weight * omega);
      damping(real, imaginary, weight * omega);
    }
  } // namespace harmonic_detail

  /** Register (K-omega^2 M+i omega D)u=f with the exp(i omega t) convention.
   * Real and imaginary displacements are algebraic Fields. Their support and
   * essential constraints stay with the Problem; the adapter owns block
   * indices.
   */
  template <typename Builder, int dim, int spacedim>
  ElasticityFreqFields<dim, spacedim>
  contribute(Builder                                    &builder,
             const ElasticityFreqProblem<dim, spacedim> &problem,
             const double                                volume_weight = 1.)
  {
    using namespace dealii;
    AssertThrow(std::isfinite(volume_weight) && volume_weight > 0.,
                ExcMessage(
                  "The harmonic integration weight must be positive."));
    const auto &parameters = problem.parameters();
    AssertThrow(parameters.weak_dirichlet_ids.empty(),
                ExcMessage(
                  "The harmonic execution-adapter path requires strong "
                  "Dirichlet data; weak Dirichlet data remain available "
                  "in the standalone elasticity_freq application."));
    const double omega = 2. * std::acos(-1.) * problem.frequency();
    using Fields       = ElasticityFreqFields<dim, spacedim>;
    using Space        = typename Fields::Space;
    auto real_space =
      std::make_shared<Space>(fe_space(problem.dof_handler(),
                                       problem.mapping(),
                                       problem.real_constraints(),
                                       &problem.locally_relevant_dofs()));
    auto imaginary_space =
      std::make_shared<Space>(fe_space(problem.dof_handler(),
                                       problem.mapping(),
                                       problem.imaginary_part_constraints(),
                                       &problem.locally_relevant_dofs()));
    const auto real_id =
      builder.algebraic_field("real",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    const auto imaginary_id =
      builder.algebraic_field("imaginary",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    const auto real = real_space->field(real_id,
                                        "displacement_real",
                                        FEValuesExtractors::Vector(0));
    const auto imaginary =
      imaginary_space->field(imaginary_id,
                             "displacement_imaginary",
                             FEValuesExtractors::Vector(0));
    std::set<types::material_id> local_ids;
    for (const auto &cell : problem.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        local_ids.insert(cell->material_id());
    std::set<types::material_id> ids;
    for (const auto &part :
         Utilities::MPI::all_gather(problem.communicator(), local_ids))
      ids.insert(part.begin(), part.end());
    for (const auto id : ids)
      {
        const auto &material = parameters.get_material_properties(id);
        const IntegrationSupport<dim, spacedim> support(
          problem.dof_handler().get_triangulation(), id);
        harmonic_detail::add_material(
          builder,
          ElasticityFreqFields<dim, spacedim>{
            real, imaginary, real_space, imaginary_space},
          material,
          support,
          omega,
          volume_weight);
      }
    const auto forcing = [&](const auto &row, const bool imaginary_part) {
      KnownSource<Tensor<1, spacedim>, spacedim> body(
        [&parameters, imaginary_part, volume_weight](const auto &context) {
          Tensor<1, spacedim> result;
          const auto &function = parameters.get_rhs(context.material_id);
          const auto  factor =
            harmonic_detail::phase_component(parameters.rhs_phase_by_id,
                                             context.material_id,
                                             parameters.rhs_phase,
                                             imaginary_part);
          for (unsigned int d = 0; d < spacedim; ++d)
            result[d] =
              -volume_weight * factor * function.value(context.point, d);
          return result;
        });
      known_term(body, test(row)).add(builder);
      for (const auto id : parameters.neumann_ids)
        {
          KnownSource<Tensor<1, spacedim>, spacedim> traction(
            [&parameters, imaginary_part, id, volume_weight](
              const auto &context) {
              Tensor<1, spacedim> result;
              const auto         &function = parameters.get_neumann_bc(id);
              const auto          factor =
                harmonic_detail::phase_component(parameters.neumann_phase_by_id,
                                                 id,
                                                 parameters.neumann_phase,
                                                 imaginary_part);
              for (unsigned int d = 0; d < spacedim; ++d)
                result[d] =
                  -volume_weight * factor * function.value(context.point, d);
              return result;
            });
          known_term(traction, test(row)).on_boundary(id).add(builder);
        }
    };
    const harmonic_detail::RobinParametersView<dim, spacedim> real_robin(
      parameters, volume_weight, false);
    const harmonic_detail::RobinParametersView<dim, spacedim> imaginary_robin(
      parameters, volume_weight, true);
    static_elasticity_robin(real, real_robin)(builder);
    static_elasticity_robin(imaginary, imaginary_robin)(builder);
    forcing(real, false);
    forcing(imaginary, true);
    return {real, imaginary, std::move(real_space), std::move(imaginary_space)};
  }
} // namespace ImmersX
#endif
