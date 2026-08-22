// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_elastodynamics_semidiscrete_h
#define immersx_elastodynamics_semidiscrete_h

#include <immersx/core/contributor.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/physics/elastodynamics.h>

#include <string>
#include <utility>

namespace ImmersX
{
  /** Semantic fields contributed by one elastodynamics Problem. */
  struct ElastodynamicsFields
  {
    FieldId     displacement;
    FieldId     velocity;
    std::string prefix;

    std::string
    term(const char *local_name) const
    {
      return prefix + "." + local_name;
    }
  };

  /** A solver-neutral elastodynamics contributor. */
  template <int dim, int spacedim = dim>
  class ElastodynamicsContributor
  {
  public:
    using Solver     = ElastodynamicsSolver<dim, spacedim>;
    using VectorType = typename Solver::VectorType;

    explicit ElastodynamicsContributor(const Solver        &problem,
                                       const HistoryGroupId history_group = {})
      : problem_(problem)
      , history_group_(history_group)
    {}

    template <typename Builder>
    ElastodynamicsFields
    operator()(Builder &builder, const std::string &prefix) const
    {
      const auto displacement =
        builder.add_field(prefix,
                          "displacement",
                          TimeRole::differential,
                          problem_.locally_owned_dofs(),
                          problem_.locally_relevant_dofs(),
                          history_group_);
      const auto velocity = builder.add_field(prefix,
                                              "velocity",
                                              TimeRole::differential,
                                              problem_.locally_owned_dofs(),
                                              problem_.locally_relevant_dofs(),
                                              history_group_);

      ElastodynamicsFields fields{displacement, velocity, prefix};
      const auto           mass =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.mass_matrix()));
      const auto stiffness =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.stiffness_matrix()));
      const auto damping =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.damping_matrix()));

      builder.add_residual(
        displacement,
        fields.term("kinematic"),
        [fields, &problem = problem_, mass](const auto &context) {
          const auto &d_dot =
            context.state_derivative()->field(fields.displacement,
                                              context.time());
          const auto &v =
            context.state().field(fields.velocity, context.time());
          return semidiscrete_detail::constrained_operation(
            mass * d_dot - mass * v, problem.constraints());
        });

      builder.add_residual(
        velocity,
        fields.term("dynamics"),
        [fields, &problem = problem_, mass, stiffness, damping](
          const auto &context) {
          const auto &v_dot =
            context.state_derivative()->field(fields.velocity, context.time());
          const auto &d =
            context.state().field(fields.displacement, context.time());
          const auto &v =
            context.state().field(fields.velocity, context.time());
          auto result = mass * v_dot + stiffness * d + damping * v;
          typename SemiDiscreteModel<VectorType>::Operation forcing;
          forcing.reinit_vector = [v_dot](VectorType &vector, const bool omit) {
            vector.reinit(v_dot, omit);
          };
          forcing.apply = [&problem,
                           time = context.time()](VectorType &vector) {
            problem.body_force_at_time(time, vector);
          };
          forcing.apply_add = [&problem,
                               time = context.time()](VectorType &vector) {
            VectorType force;
            problem.body_force_at_time(time, force);
            vector += force;
          };
          return semidiscrete_detail::constrained_operation(
            result - forcing, problem.velocity_constraints());
        });

      builder.add_state_operator(displacement,
                                 velocity,
                                 fields.term("kinematic"),
                                 semidiscrete_detail::constrained_operator(
                                   -1. * mass, problem_.constraints()));
      builder.add_derivative_operator(displacement,
                                      displacement,
                                      fields.term("kinematic"),
                                      semidiscrete_detail::constrained_operator(
                                        mass, problem_.constraints()));
      builder.add_state_operator(velocity,
                                 displacement,
                                 fields.term("elasticity"),
                                 semidiscrete_detail::constrained_operator(
                                   stiffness, problem_.velocity_constraints()));
      builder.add_state_operator(velocity,
                                 velocity,
                                 fields.term("damping"),
                                 semidiscrete_detail::constrained_operator(
                                   damping, problem_.velocity_constraints()));
      builder.add_derivative_operator(velocity,
                                      velocity,
                                      fields.term("inertia"),
                                      semidiscrete_detail::constrained_operator(
                                        mass, problem_.velocity_constraints()));
      return fields;
    }

  private:
    const Solver  &problem_;
    HistoryGroupId history_group_;
  };

  template <int dim, int spacedim = dim>
  ElastodynamicsContributor<dim, spacedim>
  elastodynamics(const ElastodynamicsSolver<dim, spacedim> &problem,
                 const HistoryGroupId                       history_group = {})
  {
    return ElastodynamicsContributor<dim, spacedim>(problem, history_group);
  }

  /**
   * Register one elastodynamics Problem in caller-owned semantic objects.
   *
   * The returned FieldIds refer to @p layout, which must outlive every model
   * callback using them. The Problem is not owned and must outlive the model
   * terms. Prefixes and history groups are composition decisions made by the
   * caller, so several Problems of the same type can coexist.
   */
  template <int dim, int spacedim = dim>
  ElastodynamicsFields
  register_elastodynamics_fields(
    StateLayout                               &layout,
    const ElastodynamicsSolver<dim, spacedim> &problem,
    const std::string                         &prefix,
    const HistoryGroupId                       history_group)
  {
    AssertThrow(!prefix.empty(),
                dealii::ExcMessage(
                  "An elastodynamics field prefix cannot be empty."));

    FieldDescriptor displacement_descriptor;
    displacement_descriptor.name             = prefix + ".displacement";
    displacement_descriptor.time_role        = TimeRole::differential;
    displacement_descriptor.history_group    = history_group;
    displacement_descriptor.locally_owned    = problem.locally_owned_dofs();
    displacement_descriptor.locally_relevant = problem.locally_relevant_dofs();

    FieldDescriptor velocity_descriptor;
    velocity_descriptor.name             = prefix + ".velocity";
    velocity_descriptor.time_role        = TimeRole::differential;
    velocity_descriptor.history_group    = history_group;
    velocity_descriptor.locally_owned    = problem.locally_owned_dofs();
    velocity_descriptor.locally_relevant = problem.locally_relevant_dofs();

    ElastodynamicsFields fields;
    fields.prefix       = prefix;
    fields.displacement = layout.add_field(std::move(displacement_descriptor));
    fields.velocity     = layout.add_field(std::move(velocity_descriptor));
    return fields;
  }

  /** Add the additive M/K/D/body-force residual terms of one Problem. */
  template <int dim, int spacedim = dim>
  void
  add_elastodynamics_terms(
    SemiDiscreteModel<typename ElastodynamicsSolver<dim, spacedim>::VectorType>
                                              &model,
    const ElastodynamicsSolver<dim, spacedim> &problem,
    const ElastodynamicsFields                &fields)
  {
    using namespace semidiscrete_detail;

    model.add_term(
      fields.term("kinematic"),
      [fields, &problem](const auto &context, auto &residual) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage(
                      "Elastodynamics kinematics needs state_dot."));
        const auto &d_dot =
          context.state_derivative()->field(fields.displacement,
                                            context.time());
        const auto &v = context.state().field(fields.velocity, context.time());
        auto       &row = residual.field(fields.displacement);
        add_matrix_product(problem.mass_matrix(), d_dot, row);
        add_matrix_product(problem.mass_matrix(), v, row, -1.);
        zero_constrained(problem.constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        const double a   = linearization.state_weight();
        const double b   = linearization.derivative_weight();
        const double t   = linearization.evaluation().time();
        auto        &row = residual.field(fields.displacement);
        add_matrix_product(problem.mass_matrix(),
                           increment.field(fields.displacement, t),
                           row,
                           b);
        add_matrix_product(problem.mass_matrix(),
                           increment.field(fields.velocity, t),
                           row,
                           -a);
        zero_constrained(problem.constraints(), row);
      });

    model.add_term(
      fields.term("inertia"),
      [fields, &problem](const auto &context, auto &residual) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage(
                      "Elastodynamics inertia needs state_dot."));
        const auto &v_dot =
          context.state_derivative()->field(fields.velocity, context.time());
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.mass_matrix(), v_dot, row);
        zero_constrained(problem.velocity_constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        const double b   = linearization.derivative_weight();
        const double t   = linearization.evaluation().time();
        auto        &row = residual.field(fields.velocity);
        add_matrix_product(problem.mass_matrix(),
                           increment.field(fields.velocity, t),
                           row,
                           b);
        zero_constrained(problem.velocity_constraints(), row);
      });

    model.add_term(
      fields.term("elasticity"),
      [fields, &problem](const auto &context, auto &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.stiffness_matrix(),
                           context.state().field(fields.displacement,
                                                 context.time()),
                           row);
        zero_constrained(problem.velocity_constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.stiffness_matrix(),
                           increment.field(fields.displacement,
                                           linearization.evaluation().time()),
                           row,
                           linearization.state_weight());
        zero_constrained(problem.velocity_constraints(), row);
      });

    model.add_term(
      fields.term("damping"),
      [fields, &problem](const auto &context, auto &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.damping_matrix(),
                           context.state().field(fields.velocity,
                                                 context.time()),
                           row);
        zero_constrained(problem.velocity_constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.damping_matrix(),
                           increment.field(fields.velocity,
                                           linearization.evaluation().time()),
                           row,
                           linearization.state_weight());
        zero_constrained(problem.velocity_constraints(), row);
      });

    using VectorType = typename ElastodynamicsSolver<dim, spacedim>::VectorType;
    model.add_term(fields.term("forcing"),
                   [fields, &problem](const auto &context, auto &residual) {
                     VectorType force;
                     problem.body_force_at_time(context.time(), force);
                     auto &row = residual.field(fields.velocity);
                     row -= force;
                     zero_constrained(problem.velocity_constraints(), row);
                   });
  }

  /**
   * Convenience standalone owner for the composable elastodynamics adapter.
   *
   * This class owns a StateLayout and SemiDiscreteModel for the traditional
   * one-Problem API. New compositions should call the registration functions
   * directly with caller-owned objects.
   */
  template <int dim, int spacedim = dim>
  class ElastodynamicsSemiDiscreteModel
  {
  public:
    using Solver     = ElastodynamicsSolver<dim, spacedim>;
    using VectorType = typename Solver::VectorType;
    using Model      = SemiDiscreteModel<VectorType>;

    explicit ElastodynamicsSemiDiscreteModel(const Solver &solver)
      : solver_(solver)
      , fields_(register_elastodynamics_fields(layout_,
                                               solver_,
                                               "solid",
                                               HistoryGroupId(0)))
    {
      add_elastodynamics_terms(model_, solver_, fields_);
    }

    ElastodynamicsSemiDiscreteModel(const ElastodynamicsSemiDiscreteModel &) =
      delete;
    ElastodynamicsSemiDiscreteModel &
    operator=(const ElastodynamicsSemiDiscreteModel &) = delete;

    const StateLayout &
    layout() const
    {
      return layout_;
    }

    const Model &
    model() const
    {
      return model_;
    }

    Model &
    model()
    {
      return model_;
    }

    FieldId
    displacement_field() const
    {
      return fields_.displacement;
    }

    FieldId
    velocity_field() const
    {
      return fields_.velocity;
    }

  private:
    const Solver        &solver_;
    StateLayout          layout_;
    ElastodynamicsFields fields_;
    Model                model_;
  };
} // namespace ImmersX

#endif // immersx_elastodynamics_semidiscrete_h
