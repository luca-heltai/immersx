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
