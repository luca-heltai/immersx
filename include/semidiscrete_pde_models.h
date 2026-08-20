// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_semidiscrete_pde_models_h
#define immersx_semidiscrete_pde_models_h

#include "elastodynamics.h"
#include "navier_stokes.h"
#include "time_residual.h"

namespace ImmersX
{
  namespace semidiscrete_terms
  {
    inline constexpr const char *solid_kinematic  = "solid.kinematic";
    inline constexpr const char *solid_inertia    = "solid.inertia";
    inline constexpr const char *solid_elasticity = "solid.elasticity";
    inline constexpr const char *solid_damping    = "solid.damping";
    inline constexpr const char *solid_forcing    = "solid.forcing";

    inline constexpr const char *fluid_mass      = "fluid.mass";
    inline constexpr const char *fluid_viscosity = "fluid.viscosity";
    inline constexpr const char *fluid_pressure  = "fluid.pressure";
    inline constexpr const char *fluid_incompressibility =
      "fluid.incompressibility";
    inline constexpr const char *fluid_forcing = "fluid.forcing";
  } // namespace semidiscrete_terms

  namespace detail
  {
    template <typename VectorType, typename MatrixType>
    void
    add_matrix_product(const MatrixType &matrix,
                       const VectorType &source,
                       VectorType       &destination,
                       const double      factor = 1.)
    {
      VectorType product;
      product.reinit(destination);
      matrix.vmult(product, source);
      if (factor == 1.)
        destination += product;
      else
        {
          product *= factor;
          destination += product;
        }
    }

    template <typename VectorType>
    void
    zero_constrained(const dealii::AffineConstraints<double> &constraints,
                     VectorType                              &vector)
    {
      for (const auto index : vector.locally_owned_elements())
        if (constraints.is_constrained(index))
          vector(index) = 0.;
    }

    template <typename VectorType>
    void
    zero_mixed_constrained(const dealii::AffineConstraints<double> &constraints,
                           const dealii::types::global_dof_index block_offset,
                           VectorType                           &vector)
    {
      for (const auto index : vector.locally_owned_elements())
        if (constraints.is_constrained(block_offset + index))
          vector(index) = 0.;
    }
  } // namespace detail


  /** Semantic first-order residual view of ElastodynamicsSolver. */
  template <int dim, int spacedim = dim>
  class ElastodynamicsSemiDiscreteModel
  {
  public:
    using Solver     = ::ElastodynamicsSolver<dim, spacedim>;
    using VectorType = typename Solver::VectorType;
    using Model      = SemiDiscreteModel<VectorType>;

    explicit ElastodynamicsSemiDiscreteModel(const Solver &solver)
      : solver_(solver)
    {
      FieldDescriptor displacement_descriptor;
      displacement_descriptor.name             = "solid.displacement";
      displacement_descriptor.time_role        = TimeRole::differential;
      displacement_descriptor.locally_owned    = solver.locally_owned_dofs();
      displacement_descriptor.locally_relevant = solver.locally_relevant_dofs();
      displacement_ = layout_.add_field(std::move(displacement_descriptor));

      FieldDescriptor velocity_descriptor;
      velocity_descriptor.name             = "solid.velocity";
      velocity_descriptor.time_role        = TimeRole::differential;
      velocity_descriptor.locally_owned    = solver.locally_owned_dofs();
      velocity_descriptor.locally_relevant = solver.locally_relevant_dofs();
      velocity_ = layout_.add_field(std::move(velocity_descriptor));

      add_terms();
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
      return displacement_;
    }

    FieldId
    velocity_field() const
    {
      return velocity_;
    }

  private:
    void
    add_terms()
    {
      model_.add_term(
        semidiscrete_terms::solid_kinematic,
        [this](const auto &context, auto &residual) {
          const auto &state = context.state();
          AssertThrow(context.has_state_derivative(),
                      dealii::ExcMessage(
                        "Elastodynamics kinematics needs state_dot."));
          const auto &d = state.field(displacement_, context.time());
          const auto &v = state.field(velocity_, context.time());
          const auto &d_dot =
            context.state_derivative()->field(displacement_, context.time());
          auto &row = residual.field(displacement_);
          detail::add_matrix_product(solver_.mass_matrix(), d_dot, row);
          detail::add_matrix_product(solver_.mass_matrix(), v, row, -1.);
          detail::zero_constrained(solver_.constraints(), row);
          (void)d;
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          const double a   = linearization.state_weight();
          const double b   = linearization.derivative_weight();
          const double t   = linearization.evaluation().time();
          auto        &row = residual.field(displacement_);
          detail::add_matrix_product(solver_.mass_matrix(),
                                     increment.field(displacement_, t),
                                     row,
                                     b);
          detail::add_matrix_product(solver_.mass_matrix(),
                                     increment.field(velocity_, t),
                                     row,
                                     -a);
          detail::zero_constrained(solver_.constraints(), row);
        });

      model_.add_term(
        semidiscrete_terms::solid_inertia,
        [this](const auto &context, auto &residual) {
          AssertThrow(context.has_state_derivative(),
                      dealii::ExcMessage(
                        "Elastodynamics inertia needs state_dot."));
          const auto &v_dot =
            context.state_derivative()->field(velocity_, context.time());
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(solver_.mass_matrix(), v_dot, row);
          detail::zero_constrained(solver_.velocity_constraints(), row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          const double b   = linearization.derivative_weight();
          const double t   = linearization.evaluation().time();
          auto        &row = residual.field(velocity_);
          detail::add_matrix_product(solver_.mass_matrix(),
                                     increment.field(velocity_, t),
                                     row,
                                     b);
          detail::zero_constrained(solver_.velocity_constraints(), row);
        });

      model_.add_term(
        semidiscrete_terms::solid_elasticity,
        [this](const auto &context, auto &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(solver_.stiffness_matrix(),
                                     context.state().field(displacement_,
                                                           context.time()),
                                     row);
          detail::zero_constrained(solver_.velocity_constraints(), row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(
            solver_.stiffness_matrix(),
            increment.field(displacement_, linearization.evaluation().time()),
            row,
            linearization.state_weight());
          detail::zero_constrained(solver_.velocity_constraints(), row);
        });

      model_.add_term(
        semidiscrete_terms::solid_damping,
        [this](const auto &context, auto &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(solver_.damping_matrix(),
                                     context.state().field(velocity_,
                                                           context.time()),
                                     row);
          detail::zero_constrained(solver_.velocity_constraints(), row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(
            solver_.damping_matrix(),
            increment.field(velocity_, linearization.evaluation().time()),
            row,
            linearization.state_weight());
          detail::zero_constrained(solver_.velocity_constraints(), row);
        });

      model_.add_term(semidiscrete_terms::solid_forcing,
                      [this](const auto &context, auto &residual) {
                        VectorType force;
                        solver_.body_force_at_time(context.time(), force);
                        auto &row = residual.field(velocity_);
                        row -= force;
                        detail::zero_constrained(solver_.velocity_constraints(),
                                                 row);
                      });
    }

    const Solver &solver_;
    StateLayout   layout_;
    FieldId       displacement_;
    FieldId       velocity_;
    Model         model_;
  };


  /** Semantic unsteady-Stokes residual view of NavierStokesSolver. */
  template <int dim, int spacedim = dim>
  class NavierStokesSemiDiscreteModel
  {
  public:
    using Solver     = ::NavierStokesSolver<dim, spacedim>;
    using VectorType = typename Solver::VectorType;
    using Model      = SemiDiscreteModel<VectorType>;

    explicit NavierStokesSemiDiscreteModel(const Solver &solver)
      : solver_(solver)
    {
      AssertThrow(solver.locally_owned_dofs_by_block().size() == 2,
                  dealii::ExcMessage(
                    "Navier-Stokes semantic fields need two DoF blocks."));

      FieldDescriptor velocity_descriptor;
      velocity_descriptor.name      = "fluid.velocity";
      velocity_descriptor.time_role = TimeRole::differential;
      velocity_descriptor.locally_owned =
        solver.locally_owned_dofs_by_block()[0];
      velocity_descriptor.locally_relevant =
        solver.locally_relevant_dofs_by_block()[0];
      velocity_ = layout_.add_field(std::move(velocity_descriptor));

      FieldDescriptor pressure_descriptor;
      pressure_descriptor.name      = "fluid.pressure";
      pressure_descriptor.time_role = TimeRole::algebraic;
      pressure_descriptor.locally_owned =
        solver.locally_owned_dofs_by_block()[1];
      pressure_descriptor.locally_relevant =
        solver.locally_relevant_dofs_by_block()[1];
      pressure_ = layout_.add_field(std::move(pressure_descriptor));

      add_terms();
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
      return velocity_;
    }

    FieldId
    pressure_field() const
    {
      return pressure_;
    }

  private:
    void
    add_terms()
    {
      model_.add_term(
        semidiscrete_terms::fluid_mass,
        [this](const auto &context, auto &residual) {
          AssertThrow(context.has_state_derivative(),
                      dealii::ExcMessage("Stokes mass needs state_dot."));
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(
            solver_.velocity_mass_matrix(),
            context.state_derivative()->field(velocity_, context.time()),
            row,
            solver_.density());
          detail::zero_constrained(solver_.constraints(), row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(
            solver_.velocity_mass_matrix(),
            increment.field(velocity_, linearization.evaluation().time()),
            row,
            solver_.density() * linearization.derivative_weight());
          detail::zero_constrained(solver_.constraints(), row);
        });

      model_.add_term(
        semidiscrete_terms::fluid_viscosity,
        [this](const auto &context, auto &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(solver_.continuous_operator().block(0, 0),
                                     context.state().field(velocity_,
                                                           context.time()),
                                     row);
          detail::zero_constrained(solver_.constraints(), row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(
            solver_.continuous_operator().block(0, 0),
            increment.field(velocity_, linearization.evaluation().time()),
            row,
            linearization.state_weight());
          detail::zero_constrained(solver_.constraints(), row);
        });

      model_.add_term(
        semidiscrete_terms::fluid_pressure,
        [this](const auto &context, auto &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(solver_.continuous_operator().block(0, 1),
                                     context.state().field(pressure_,
                                                           context.time()),
                                     row);
          detail::zero_constrained(solver_.constraints(), row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          auto &row = residual.field(velocity_);
          detail::add_matrix_product(
            solver_.continuous_operator().block(0, 1),
            increment.field(pressure_, linearization.evaluation().time()),
            row,
            linearization.state_weight());
          detail::zero_constrained(solver_.constraints(), row);
        });

      model_.add_term(
        semidiscrete_terms::fluid_incompressibility,
        [this](const auto &context, auto &residual) {
          auto &row = residual.field(pressure_);
          detail::add_matrix_product(solver_.continuous_operator().block(1, 0),
                                     context.state().field(velocity_,
                                                           context.time()),
                                     row);
          detail::zero_mixed_constrained(solver_.constraints(),
                                         solver_.velocity_block_size(),
                                         row);
        },
        [this](const auto &linearization,
               const auto &increment,
               auto       &residual) {
          auto &row = residual.field(pressure_);
          detail::add_matrix_product(
            solver_.continuous_operator().block(1, 0),
            increment.field(velocity_, linearization.evaluation().time()),
            row,
            linearization.state_weight());
          detail::zero_mixed_constrained(solver_.constraints(),
                                         solver_.velocity_block_size(),
                                         row);
        });

      model_.add_term(semidiscrete_terms::fluid_forcing,
                      [this](const auto &context, auto &residual) {
                        VectorType force;
                        solver_.velocity_forcing_at_time(context.time(), force);
                        force *= solver_.density();
                        auto &row = residual.field(velocity_);
                        row -= force;
                        detail::zero_constrained(solver_.constraints(), row);
                      });
    }

    const Solver &solver_;
    StateLayout   layout_;
    FieldId       velocity_;
    FieldId       pressure_;
    Model         model_;
  };
} // namespace ImmersX

#endif // immersx_semidiscrete_pde_models_h
