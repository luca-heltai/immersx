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
      const auto &real      = fields.real;
      const auto &imaginary = fields.imaginary;
      const auto  elasticity =
        [&](const auto &source, const auto &row, const double factor) {
          (weak_term((factor * 2. * material.Lame_mu) *
                       symmetric_gradient(source),
                     symmetric_gradient(test(row))) *
           support)
            .add(builder);
          (weak_term((factor * material.Lame_lambda) * divergence(source),
                     divergence(test(row))) *
           support)
            .add(builder);
        };
      const auto inertia =
        [&](const auto &source, const auto &row, const double factor) {
          (weak_term((factor * material.rho) * value(source), test(row)) *
           support)
            .add(builder);
        };
      const auto damping =
        [&](const auto &source, const auto &row, const double factor) {
          elasticity(source, row, factor * material.rayleigh_beta);
          inertia(source, row, factor * material.rayleigh_alpha);
          (weak_term((factor * material.neta) * symmetric_gradient(source),
                     symmetric_gradient(test(row))) *
           support)
            .add(builder);
        };
      elasticity(real, real, weight);
      elasticity(imaginary, imaginary, weight);
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
    forcing(real, false);
    forcing(imaginary, true);
    return {real, imaginary, std::move(real_space), std::move(imaginary_space)};
  }
} // namespace ImmersX
#endif
