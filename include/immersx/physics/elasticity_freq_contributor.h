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
#include <immersx/core/robin_term.h>
#include <immersx/core/weak_term.h>
#include <immersx/physics/elasticity_freq.h>

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace ImmersX
{
  namespace harmonic_detail
  {
    /** One positive surrogate and inverse shared by the complex components. */
    struct SurrogatePreconditioner
    {
      using Matrix   = ImmersXLA::MPI::SparseMatrix;
      using Operator = dealii::LinearOperator<ImmersXLA::MPI::Vector>;
      std::function<std::shared_ptr<Matrix>()>         make_matrix;
      std::function<Operator(std::shared_ptr<Matrix>)> make_inverse;
      std::vector<std::function<void(Matrix &)>>       additions;
      std::optional<Operator>                          inverse;

      std::shared_ptr<Matrix>
      matrix() const
      {
        auto result = make_matrix();
        for (const auto &add : additions)
          add(*result);
        return result;
      }

      Operator
      get()
      {
        if (!inverse)
          inverse = make_inverse(matrix());
        return *inverse;
      }
    };
  } // namespace harmonic_detail

  template <int dim, int spacedim = dim>
  struct ElasticityFreqFields
  {
    using Space = FiniteElementSpaceView<dim, spacedim>;
    using VectorField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Vector>;
    VectorField                                               real;
    VectorField                                               imaginary;
    std::shared_ptr<const Space>                              real_space;
    std::shared_ptr<const Space>                              imaginary_space;
    std::shared_ptr<harmonic_detail::SurrogatePreconditioner> surrogate =
      nullptr;
  };

  namespace harmonic_detail
  {
    template <int dim, int spacedim>
    auto
    make_surrogate_inverse(const ElasticityFreqProblem<dim, spacedim> &problem,
                           std::shared_ptr<ImmersXLA::MPI::SparseMatrix> matrix)
    {
      using Vector                  = ImmersXLA::MPI::Vector;
      using Operator                = dealii::LinearOperator<Vector>;
      const auto       owned        = problem.locally_owned_dofs();
      const auto       communicator = problem.communicator();
      dealii::IndexSet constrained(owned.size());
      for (const auto &line : problem.real_constraints().get_lines())
        if (owned.is_element(line.index))
          {
            constrained.add_index(line.index);
            matrix->set(line.index, line.index, 1.);
          }
      constrained.compress();
      matrix->compress(dealii::VectorOperation::insert);
      const auto amg  = problem.make_surrogate_amg(*matrix);
      auto     memory = std::make_shared<dealii::GrowingVectorMemory<Vector>>();
      Operator result;
      result.reinit_range_vector = [owned, communicator](Vector &v,
                                                         const bool) {
        v.reinit(owned, communicator);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [matrix, amg, constrained, memory](Vector       &dst,
                                                        const Vector &src) {
        typename dealii::VectorMemory<Vector>::Pointer rhs(*memory);
        rhs->reinit(src);
        *rhs = src;
        for (const auto index : constrained)
          (*rhs)[index] = 0.;
        dst               = 0.;
        const double norm = rhs->l2_norm();
        AssertThrow(std::isfinite(norm),
                    dealii::ExcMessage(
                      "Non-finite harmonic surrogate right-hand side."));
        if (norm > 0.)
          {
            // Match the standalone surrogate's reduction and iteration cap.
            // The outer FGMRES checks convergence of the physical system.
            dealii::IterationNumberControl control(30, .1 * norm, false, false);
            dealii::SolverCG<Vector>       solver(control, *memory);
            solver.solve(*matrix, dst, *rhs, *amg);
          }
        for (const auto index : constrained)
          dst[index] = src[index];
        AssertThrow(std::isfinite(dst.l2_norm()),
                    dealii::ExcMessage(
                      "Non-finite harmonic surrogate correction."));
      };
      result.Tvmult    = result.vmult;
      result.vmult_add = [apply = result.vmult, memory](Vector       &dst,
                                                        const Vector &src) {
        typename dealii::VectorMemory<Vector>::Pointer temporary(*memory);
        temporary->reinit(dst);
        apply(*temporary, src);
        dst += *temporary;
      };
      result.Tvmult_add = result.vmult_add;
      return result;
    }

    /** Add positive bilinear terms on their physical integration support. */
    template <typename Fields, typename Support>
    void
    add_surrogate_terms(const Fields  &fields,
                        const Support &support,
                        const double   shear,
                        const double   bulk,
                        const double   mass)
    {
      fields.surrogate->inverse.reset();
      // Field holds a non-owning FE view; retain its owner for lazy assembly.
      fields.surrogate->additions.push_back([field = fields.real,
                                             space = fields.real_space,
                                             support,
                                             shear,
                                             bulk,
                                             mass](auto &matrix) {
        using Vector = ImmersXLA::MPI::Vector;
        using Matrix = ImmersXLA::MPI::SparseMatrix;
        StateLayout                         layout;
        SemiDiscreteModel<Vector, Matrix>   model;
        SemidiscreteBuilder<Vector, Matrix> builder(layout, model);
        const auto id    = builder.algebraic_field("surrogate",
                                                field.locally_owned_dofs(),
                                                field.locally_relevant_dofs());
        const auto trial = field.with_id(id);
        if (shear != 0.)
          weak_term(shear * symmetric_gradient(trial),
                    symmetric_gradient(test(trial)))
            .on(support)
            .add(builder);
        if (bulk != 0.)
          weak_term(bulk * divergence(trial), divergence(test(trial)))
            .on(support)
            .add(builder);
        if (mass != 0.)
          weak_term(mass * value(trial), test(trial)).on(support).add(builder);
        Vector zero;
        zero.reinit(field.locally_owned_dofs(),
                    field.space().mpi_communicator());
        StateView<Vector> state(layout, 0.);
        state.bind(id, zero);
        const EvaluationContext<Vector> context(0., state);
        const auto contribution = model.state_matrix_operator(id, id, context);
        if (contribution)
          matrix.add(1., *contribution->matrix());
      });
    }

    /** Extend the surrogate on the same support as a shared material term. */
    template <typename Fields, typename Support>
    void
    add_surrogate_material(const Fields             &fields,
                           const MaterialProperties &material,
                           const Support            &support,
                           const double              omega)
    {
      const double shear =
        2. * material.Lame_mu * (1. + omega * material.rayleigh_beta) +
        omega * material.neta;
      const double bulk =
        material.Lame_lambda * (1. + omega * material.rayleigh_beta);
      const double mass =
        material.rho * (omega * omega + omega * material.rayleigh_alpha);
      add_surrogate_terms(fields, support, shear, bulk, mass);
    }

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
    parameters.set_boundary_condition_times(0.);
    const auto add_robin = [&](const auto &row, const bool imaginary_part) {
      robin_boundary_term(
        row,
        parameters.robin_ids,
        [&parameters](const auto id) -> const Function<spacedim> & {
          return parameters.get_robin_coefficient(id);
        },
        [&parameters](const auto id) -> const Function<spacedim> & {
          return parameters.get_robin_bc(id);
        },
        [&parameters, imaginary_part, volume_weight](const auto id) {
          return volume_weight *
                 harmonic_detail::phase_component(parameters.robin_phase_by_id,
                                                  id,
                                                  parameters.robin_phase,
                                                  imaginary_part);
        },
        volume_weight)(builder);
    };
    add_robin(real, false);
    add_robin(imaginary, true);
    forcing(real, false);
    forcing(imaginary, true);
    auto surrogate =
      std::make_shared<harmonic_detail::SurrogatePreconditioner>();
    surrogate->make_matrix = [&problem, volume_weight] {
      return problem.make_surrogate_matrix(volume_weight);
    };
    surrogate->make_inverse = [&problem](auto matrix) {
      return harmonic_detail::make_surrogate_inverse(problem,
                                                     std::move(matrix));
    };
    const auto local_inverse = [surrogate](const auto &, const auto &) {
      return surrogate->get();
    };
    builder.preconditioner(real_id, local_inverse);
    builder.preconditioner(imaginary_id, local_inverse);
    return {real,
            imaginary,
            std::move(real_space),
            std::move(imaginary_space),
            std::move(surrogate)};
  }
} // namespace ImmersX
#endif
