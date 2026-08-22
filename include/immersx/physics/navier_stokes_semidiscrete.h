// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_navier_stokes_semidiscrete_h
#define immersx_navier_stokes_semidiscrete_h

#include <immersx/core/contributor.h>
#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/physics/navier_stokes.h>

#include <string>
#include <utility>

namespace ImmersX
{
  /** Semantic fields contributed by one unsteady-Stokes Problem. */
  struct NavierStokesFields
  {
    FieldId     velocity;
    FieldId     pressure;
    std::string prefix;

    std::string
    term(const char *local_name) const
    {
      return prefix + "." + local_name;
    }
  };

  /** Solver-neutral contributor for the existing linear unsteady Stokes model.
   */
  template <int dim, int spacedim = dim>
  class NavierStokesContributor
  {
  public:
    using Solver     = NavierStokesSolver<dim, spacedim>;
    using VectorType = typename Solver::VectorType;

    explicit NavierStokesContributor(const Solver        &problem,
                                     const HistoryGroupId history_group = {})
      : problem_(problem)
      , history_group_(history_group)
    {}

    template <typename Builder>
    NavierStokesFields
    operator()(Builder &builder, const std::string &prefix) const
    {
      const auto velocity =
        builder.add_field(prefix,
                          "velocity",
                          TimeRole::differential,
                          problem_.locally_owned_dofs_by_block()[0],
                          problem_.locally_relevant_dofs_by_block()[0],
                          history_group_);
      const auto pressure =
        builder.add_field(prefix,
                          "pressure",
                          TimeRole::algebraic,
                          problem_.locally_owned_dofs_by_block()[1],
                          problem_.locally_relevant_dofs_by_block()[1],
                          history_group_);
      NavierStokesFields fields{velocity, pressure, prefix};

      const auto mass =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.velocity_mass_matrix())) *
        problem_.density();
      const auto viscosity =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.continuous_operator().block(0, 0)));
      const auto pressure_operator =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.continuous_operator().block(0, 1)));
      const auto divergence =
        ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
          problem_.continuous_operator().block(1, 0)));

      builder.add_residual(
        velocity,
        fields.term("stokes"),
        [fields, &problem = problem_, mass, viscosity, pressure_operator](
          const auto &context) {
          const auto &v_dot =
            context.state_derivative()->field(fields.velocity, context.time());
          const auto &v =
            context.state().field(fields.velocity, context.time());
          const auto &p =
            context.state().field(fields.pressure, context.time());
          auto result = mass * v_dot + viscosity * v + pressure_operator * p;
          typename SemiDiscreteModel<VectorType>::Operation forcing;
          forcing.reinit_vector = [v_dot](VectorType &vector, const bool omit) {
            vector.reinit(v_dot, omit);
          };
          forcing.apply = [&problem,
                           time = context.time()](VectorType &vector) {
            problem.velocity_forcing_at_time(time, vector);
            vector *= problem.density();
          };
          forcing.apply_add = [&problem,
                               time = context.time()](VectorType &vector) {
            VectorType force;
            problem.velocity_forcing_at_time(time, force);
            force *= problem.density();
            vector += force;
          };
          return semidiscrete_detail::constrained_operation(
            result - forcing, problem.constraints());
        });

      builder.add_residual(
        pressure,
        fields.term("incompressibility"),
        [fields, &problem = problem_, divergence](const auto &context) {
          const auto &v =
            context.state().field(fields.velocity, context.time());
          return semidiscrete_detail::mixed_constrained_operation(
            divergence * v,
            problem.constraints(),
            problem.velocity_block_size());
        });

      builder.add_state_operator(velocity,
                                 velocity,
                                 fields.term("viscosity"),
                                 semidiscrete_detail::constrained_operator(
                                   viscosity, problem_.constraints()));
      builder.add_state_operator(velocity,
                                 pressure,
                                 fields.term("pressure"),
                                 semidiscrete_detail::constrained_operator(
                                   pressure_operator, problem_.constraints()));
      builder.add_state_operator(
        pressure,
        velocity,
        fields.term("incompressibility"),
        semidiscrete_detail::mixed_constrained_operator(
          divergence, problem_.constraints(), problem_.velocity_block_size()));
      builder.add_derivative_operator(velocity,
                                      velocity,
                                      fields.term("mass"),
                                      semidiscrete_detail::constrained_operator(
                                        mass, problem_.constraints()));
      return fields;
    }

  private:
    const Solver  &problem_;
    HistoryGroupId history_group_;
  };

  template <int dim, int spacedim = dim>
  NavierStokesContributor<dim, spacedim>
  unsteady_stokes(const NavierStokesSolver<dim, spacedim> &problem,
                  const HistoryGroupId                     history_group = {})
  {
    return NavierStokesContributor<dim, spacedim>(problem, history_group);
  }

  /** Register fluid velocity and algebraic pressure in caller-owned state. */
  template <int dim, int spacedim = dim>
  NavierStokesFields
  register_navier_stokes_fields(
    StateLayout                             &layout,
    const NavierStokesSolver<dim, spacedim> &problem,
    const std::string                       &prefix,
    const HistoryGroupId                     history_group)
  {
    AssertThrow(!prefix.empty(),
                dealii::ExcMessage(
                  "A Navier-Stokes field prefix cannot be empty."));
    AssertThrow(problem.locally_owned_dofs_by_block().size() == 2,
                dealii::ExcMessage(
                  "Navier-Stokes semantic fields need two DoF blocks."));

    FieldDescriptor velocity_descriptor;
    velocity_descriptor.name          = prefix + ".velocity";
    velocity_descriptor.time_role     = TimeRole::differential;
    velocity_descriptor.history_group = history_group;
    velocity_descriptor.locally_owned =
      problem.locally_owned_dofs_by_block()[0];
    velocity_descriptor.locally_relevant =
      problem.locally_relevant_dofs_by_block()[0];

    FieldDescriptor pressure_descriptor;
    pressure_descriptor.name          = prefix + ".pressure";
    pressure_descriptor.time_role     = TimeRole::algebraic;
    pressure_descriptor.history_group = history_group;
    pressure_descriptor.locally_owned =
      problem.locally_owned_dofs_by_block()[1];
    pressure_descriptor.locally_relevant =
      problem.locally_relevant_dofs_by_block()[1];

    NavierStokesFields fields;
    fields.prefix   = prefix;
    fields.velocity = layout.add_field(std::move(velocity_descriptor));
    fields.pressure = layout.add_field(std::move(pressure_descriptor));
    return fields;
  }

  /** Add the current linear unsteady-Stokes terms of one Problem. */
  template <int dim, int spacedim = dim>
  void
  add_navier_stokes_terms(
    SemiDiscreteModel<typename NavierStokesSolver<dim, spacedim>::VectorType>
                                            &model,
    const NavierStokesSolver<dim, spacedim> &problem,
    const NavierStokesFields                &fields)
  {
    using namespace semidiscrete_detail;
    using VectorType = typename NavierStokesSolver<dim, spacedim>::VectorType;

    model.add_term(
      fields.term("mass"),
      [fields, &problem](const auto &context, auto &residual) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage("Stokes mass needs state_dot."));
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.velocity_mass_matrix(),
                           context.state_derivative()->field(fields.velocity,
                                                             context.time()),
                           row,
                           problem.density());
        zero_constrained(problem.constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.velocity_mass_matrix(),
                           increment.field(fields.velocity,
                                           linearization.evaluation().time()),
                           row,
                           problem.density() *
                             linearization.derivative_weight());
        zero_constrained(problem.constraints(), row);
      });

    model.add_term(
      fields.term("viscosity"),
      [fields, &problem](const auto &context, auto &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.continuous_operator().block(0, 0),
                           context.state().field(fields.velocity,
                                                 context.time()),
                           row);
        zero_constrained(problem.constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.continuous_operator().block(0, 0),
                           increment.field(fields.velocity,
                                           linearization.evaluation().time()),
                           row,
                           linearization.state_weight());
        zero_constrained(problem.constraints(), row);
      });

    model.add_term(
      fields.term("pressure"),
      [fields, &problem](const auto &context, auto &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.continuous_operator().block(0, 1),
                           context.state().field(fields.pressure,
                                                 context.time()),
                           row);
        zero_constrained(problem.constraints(), row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        auto &row = residual.field(fields.velocity);
        add_matrix_product(problem.continuous_operator().block(0, 1),
                           increment.field(fields.pressure,
                                           linearization.evaluation().time()),
                           row,
                           linearization.state_weight());
        zero_constrained(problem.constraints(), row);
      });

    model.add_term(
      fields.term("incompressibility"),
      [fields, &problem](const auto &context, auto &residual) {
        auto &row = residual.field(fields.pressure);
        add_matrix_product(problem.continuous_operator().block(1, 0),
                           context.state().field(fields.velocity,
                                                 context.time()),
                           row);
        zero_mixed_constrained(problem.constraints(),
                               problem.velocity_block_size(),
                               row);
      },
      [fields, &problem](const auto &linearization,
                         const auto &increment,
                         auto       &residual) {
        auto &row = residual.field(fields.pressure);
        add_matrix_product(problem.continuous_operator().block(1, 0),
                           increment.field(fields.velocity,
                                           linearization.evaluation().time()),
                           row,
                           linearization.state_weight());
        zero_mixed_constrained(problem.constraints(),
                               problem.velocity_block_size(),
                               row);
      });

    model.add_term(fields.term("forcing"),
                   [fields, &problem](const auto &context, auto &residual) {
                     VectorType force;
                     problem.velocity_forcing_at_time(context.time(), force);
                     force *= problem.density();
                     auto &row = residual.field(fields.velocity);
                     row -= force;
                     zero_constrained(problem.constraints(), row);
                   });
  }

  /** Convenience standalone owner for the composable Stokes adapter. */
  template <int dim, int spacedim = dim>
  class NavierStokesSemiDiscreteModel
  {
  public:
    using Solver     = NavierStokesSolver<dim, spacedim>;
    using VectorType = typename Solver::VectorType;
    using Model      = SemiDiscreteModel<VectorType>;

    explicit NavierStokesSemiDiscreteModel(const Solver &solver)
      : solver_(solver)
      , fields_(register_navier_stokes_fields(layout_,
                                              solver_,
                                              "fluid",
                                              HistoryGroupId(0)))
    {
      add_navier_stokes_terms(model_, solver_, fields_);
    }

    NavierStokesSemiDiscreteModel(const NavierStokesSemiDiscreteModel &) =
      delete;
    NavierStokesSemiDiscreteModel &
    operator=(const NavierStokesSemiDiscreteModel &) = delete;

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
    velocity_field() const
    {
      return fields_.velocity;
    }

    FieldId
    pressure_field() const
    {
      return fields_.pressure;
    }

  private:
    const Solver      &solver_;
    StateLayout        layout_;
    NavierStokesFields fields_;
    Model              model_;
  };
} // namespace ImmersX

#endif // immersx_navier_stokes_semidiscrete_h
