// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_fiber_reinforced_elastodynamics_h
#define immersx_coral_fiber_reinforced_elastodynamics_h

#include <deal.II/base/parameter_acceptor.h>

#include <immersx/physics/fiber_reinforced_elastodynamics.h>

#include <memory>
#include <string>

namespace ImmersX::Coral
{
  template <int dim>
  class FiberMatrixProblem;

  template <int dim>
  class FiberEmbeddedProblem;

  template <int dim>
  class FiberVelocityContinuity;

  template <int dim>
  class FiberExecutionAdapter;

  namespace detail
  {
    template <int dim>
    struct FiberGraphState
    {
      using Parameters = ImmersX::FiberReinforcedElastodynamicsParameters<dim>;
      using Workflow   = ImmersX::FiberReinforcedElastodynamics<dim>;

      explicit FiberGraphState(const std::string &parameter_file)
        : parameters(std::make_shared<Parameters>())
        , workflow(std::make_shared<Workflow>(*parameters))
      {
        dealii::ParameterAcceptor::initialize(parameter_file);
      }

      std::shared_ptr<Parameters> parameters;
      std::shared_ptr<Workflow>   workflow;
    };

    template <int dim>
    void
    assert_same_state(const std::shared_ptr<FiberGraphState<dim>> &lhs,
                      const std::shared_ptr<FiberGraphState<dim>> &rhs,
                      const char                                  *message)
    {
      AssertThrow(lhs && rhs && lhs == rhs, dealii::ExcMessage(message));
    }
  } // namespace detail

  /** Shared graph-owned state for the matrix/fiber/interaction composition. */
  template <int dim>
  class FiberReinforcedElastodynamicsGraph
  {
  public:
    explicit FiberReinforcedElastodynamicsGraph(
      const std::string &parameter_file)
      : implementation(
          std::make_shared<detail::FiberGraphState<dim>>(parameter_file))
    {}

    FiberMatrixProblem<dim>
    matrix_problem() const
    {
      return FiberMatrixProblem<dim>(implementation);
    }

    FiberEmbeddedProblem<dim>
    embedded_problem() const
    {
      return FiberEmbeddedProblem<dim>(implementation);
    }

    FiberVelocityContinuity<dim>
    velocity_continuity() const
    {
      return FiberVelocityContinuity<dim>(implementation);
    }

    FiberExecutionAdapter<dim>
    execution_adapter() const
    {
      return FiberExecutionAdapter<dim>(implementation);
    }

  private:
    std::shared_ptr<detail::FiberGraphState<dim>> implementation;
  };

  /** Coral façade for the matrix Problem in the fiber composition. */
  template <int dim>
  class FiberMatrixProblem
  {
  public:
    FiberMatrixProblem() = default;

    void
    prepare()
    {
      AssertThrow(implementation,
                  dealii::ExcMessage("The matrix Problem is not connected."));
      implementation->workflow->prepare_matrix_problem();
    }

  private:
    explicit FiberMatrixProblem(
      std::shared_ptr<detail::FiberGraphState<dim>> state)
      : implementation(std::move(state))
    {}

    std::shared_ptr<detail::FiberGraphState<dim>> implementation;

    friend class FiberReinforcedElastodynamicsGraph<dim>;
    friend class FiberVelocityContinuity<dim>;
    friend class FiberExecutionAdapter<dim>;
  };

  /** Coral façade for the one-dimensional embedded fiber Problem. */
  template <int dim>
  class FiberEmbeddedProblem
  {
  public:
    FiberEmbeddedProblem() = default;

    void
    prepare()
    {
      AssertThrow(implementation,
                  dealii::ExcMessage("The fiber Problem is not connected."));
      implementation->workflow->prepare_fiber_problem();
    }

  private:
    explicit FiberEmbeddedProblem(
      std::shared_ptr<detail::FiberGraphState<dim>> state)
      : implementation(std::move(state))
    {}

    std::shared_ptr<detail::FiberGraphState<dim>> implementation;

    friend class FiberReinforcedElastodynamicsGraph<dim>;
    friend class FiberVelocityContinuity<dim>;
    friend class FiberExecutionAdapter<dim>;
  };

  /** Coral façade for the reusable matrix/fiber velocity Interaction. */
  template <int dim>
  class FiberVelocityContinuity
  {
  public:
    FiberVelocityContinuity() = default;

    void
    prepare(const FiberMatrixProblem<dim>   &matrix,
            const FiberEmbeddedProblem<dim> &fiber)
    {
      detail::assert_same_state(
        implementation,
        matrix.implementation,
        "The velocity interaction and matrix Problem belong to different "
        "fiber compositions.");
      detail::assert_same_state(
        implementation,
        fiber.implementation,
        "The velocity interaction and fiber Problem belong to different "
        "fiber compositions.");
      implementation->workflow->prepare_velocity_continuity();
    }

  private:
    explicit FiberVelocityContinuity(
      std::shared_ptr<detail::FiberGraphState<dim>> state)
      : implementation(std::move(state))
    {}

    std::shared_ptr<detail::FiberGraphState<dim>> implementation;

    friend class FiberReinforcedElastodynamicsGraph<dim>;
    friend class FiberExecutionAdapter<dim>;
  };

  /** Coral façade for the coupled transient execution adapter. */
  template <int dim>
  class FiberExecutionAdapter
  {
  public:
    FiberExecutionAdapter() = default;

    void
    run(const FiberMatrixProblem<dim>      &matrix,
        const FiberEmbeddedProblem<dim>    &fiber,
        const FiberVelocityContinuity<dim> &velocity_continuity)
    {
      detail::assert_same_state(
        implementation,
        matrix.implementation,
        "The execution adapter and matrix Problem belong to different "
        "fiber compositions.");
      detail::assert_same_state(
        implementation,
        fiber.implementation,
        "The execution adapter and fiber Problem belong to different "
        "fiber compositions.");
      detail::assert_same_state(
        implementation,
        velocity_continuity.implementation,
        "The execution adapter and velocity interaction belong to different "
        "fiber compositions.");
      implementation->workflow->run_execution();
    }

    bool
    state_is_finite() const
    {
      AssertThrow(implementation,
                  dealii::ExcMessage(
                    "The execution adapter is not connected."));
      return implementation->workflow->matrix_problem().state_is_finite() &&
             implementation->workflow->fiber_problem().state_is_finite();
    }

    double
    matrix_velocity_residual() const
    {
      AssertThrow(implementation,
                  dealii::ExcMessage(
                    "The execution adapter is not connected."));
      return implementation->workflow->residuals().matrix_velocity;
    }

    double
    fiber_velocity_residual() const
    {
      AssertThrow(implementation,
                  dealii::ExcMessage(
                    "The execution adapter is not connected."));
      return implementation->workflow->residuals().fiber_velocity;
    }

    double
    velocity_constraint_residual() const
    {
      AssertThrow(implementation,
                  dealii::ExcMessage(
                    "The execution adapter is not connected."));
      return implementation->workflow->residuals().velocity_constraint;
    }

    double
    displacement_compatibility() const
    {
      AssertThrow(implementation,
                  dealii::ExcMessage(
                    "The execution adapter is not connected."));
      return implementation->workflow->residuals().displacement_compatibility;
    }

  private:
    explicit FiberExecutionAdapter(
      std::shared_ptr<detail::FiberGraphState<dim>> state)
      : implementation(std::move(state))
    {}

    std::shared_ptr<detail::FiberGraphState<dim>> implementation;

    friend class FiberReinforcedElastodynamicsGraph<dim>;
  };
} // namespace ImmersX::Coral

#endif // immersx_coral_fiber_reinforced_elastodynamics_h
