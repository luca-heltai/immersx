#ifndef immersx_solver_controls_h
#define immersx_solver_controls_h

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/solver_control.h>

namespace ImmersX
{
  using namespace dealii;

  /**
   * SolverControl variant that accumulates all completed solve iterations.
   */
  class CumulativeSolverControl : public SolverControl
  {
  public:
    CumulativeSolverControl(const unsigned int n           = 100,
                            const double       tol         = 1.e-10,
                            const bool         log_history = false,
                            const bool         log_result  = false)
      : SolverControl(n, tol, log_history, log_result)
    {}

    explicit CumulativeSolverControl(const SolverControl &control)
      : SolverControl(control.max_steps(),
                      control.tolerance(),
                      control.log_history(),
                      control.log_result())
    {}

    State
    check(const unsigned int step, const double check_value) override
    {
      const auto state = SolverControl::check(step, check_value);
      if (state != SolverControl::iterate)
        {
          total_iterations += step;
          ++completed_solves;
        }
      return state;
    }

    unsigned long long
    total_step_count() const
    {
      return total_iterations;
    }

    unsigned int
    solve_count() const
    {
      return completed_solves;
    }

  private:
    unsigned long long total_iterations = 0;
    unsigned int       completed_solves = 0;
  };



  /**
   * ReductionControl variant that accumulates all completed solve iterations.
   */
  class CumulativeReductionControl : public ReductionControl
  {
  public:
    explicit CumulativeReductionControl(const ReductionControl &control)
      : ReductionControl(control.max_steps(),
                         control.tolerance(),
                         control.reduction(),
                         control.log_history(),
                         control.log_result())
    {}

    State
    check(const unsigned int step, const double check_value) override
    {
      const auto state = ReductionControl::check(step, check_value);
      if (state != SolverControl::iterate)
        {
          total_iterations += step;
          ++completed_solves;
        }
      return state;
    }

    unsigned long long
    total_step_count() const
    {
      return total_iterations;
    }

    unsigned int
    solve_count() const
    {
      return completed_solves;
    }

  private:
    unsigned long long total_iterations = 0;
    unsigned int       completed_solves = 0;
  };



  template <typename OuterControlType,
            typename AugmentedControlType,
            typename MassControlType>
  void
  output_augmented_lagrangian_iteration_summary(
    ConditionalOStream         &pcout,
    const OuterControlType     &outer_control,
    const AugmentedControlType &augmented_control,
    const MassControlType      &mass_control)
  {
    pcout << "   Iteration summary: outer FGMRES = "
          << outer_control.last_step() << ", inner augmented/displacement = "
          << augmented_control.total_step_count() << " total over "
          << augmented_control.solve_count()
          << " solves (last = " << augmented_control.last_step()
          << "), inner mass = " << mass_control.total_step_count()
          << " total over " << mass_control.solve_count()
          << " solves (last = " << mass_control.last_step() << ")" << std::endl;
  }

} // namespace ImmersX

#endif
