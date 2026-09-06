// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_fiber_reinforced_elastodynamics_h
#define immersx_fiber_reinforced_elastodynamics_h

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/affine_constraints.h>

#include <immersx/algebra/lagrange_multiplier_schur_solver.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/time_parameters.h>
#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#endif
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Parameters for the full-order matrix-plus-excess-fiber application. */
  template <int dim>
  class FiberReinforcedElastodynamicsParameters
    : public dealii::ParameterAcceptor
  {
  public:
    explicit FiberReinforcedElastodynamicsParameters(
      const std::string &subsection = "/Fiber Reinforced Elastodynamics/");

    TimeParameters time_parameters;

    ElastodynamicsParameters<dim> matrix_parameters;
    ElastodynamicsParameters<dim> fiber_parameters;

    std::string output_directory = "./output/fiber_reinforced_elastodynamics";
    std::string multiplier_output_name = "velocity_multiplier";

    /** Zero selects the fiber degree; positive values select an independent
     * multiplier FE degree. */
    unsigned int multiplier_degree = 0;

    unsigned int schur_max_steps                 = 200;
    double       schur_tolerance                 = 1.e-10;
    double       block_tolerance                 = 1.e-12;
    double       initial_compatibility_tolerance = 1.e-10;
  };


  /** Norms of the three residual rows of the coupled velocity solve. */
  struct FiberReinforcedElastodynamicsResiduals
  {
    double matrix_velocity            = 0.;
    double fiber_velocity             = 0.;
    double velocity_constraint        = 0.;
    double displacement_compatibility = 0.;
  };


  /**
   * Standalone full-order coupled driver for a matrix and an excess fiber.
   *
   * Both Problems are `ElastodynamicsSolver<dim, dim>` objects on independent
   * full-dimensional meshes.  The fiber mesh is geometrically embedded in the
   * matrix mesh, but it is not a reduced `<1,3>` representation.  The driver
   * owns the coupled time loop. With SUNDIALS enabled, it uses the public IDA
   * execution adapter; without SUNDIALS, it uses the explicit
   * Schur-complement backward-Euler path.
   */
  template <int dim>
  class FiberReinforcedElastodynamics
  {
  public:
    using Parameters = FiberReinforcedElastodynamicsParameters<dim>;
    using Problem    = ElastodynamicsSolver<dim, dim>;
    using VectorType = typename Problem::VectorType;
    using MatrixType = typename Problem::MatrixType;
#ifdef DEAL_II_WITH_SUNDIALS
    using GlobalVectorType = ImmersXLA::MPI::BlockVector;
    using IDAAdapterType   = IDAAdapter<VectorType, GlobalVectorType>;
#endif

    explicit FiberReinforcedElastodynamics(const Parameters &parameters);

    /** Create both meshes and assemble both Problems.
     *
     * The coupling is prepared by initialization or by the execution adapter.
     */
    void
    setup();

    /** Set both configured initial states and check displacement compatibility.
     */
    void
    set_initial_conditions();

    /** Advance one coupled backward-Euler step. */
    void
    advance_one_timestep();

    /** Run setup, initialization, output, and the coupled time loop. */
    void
    run();

    const Problem &
    matrix_problem() const
    {
      return matrix_problem_storage;
    }

    const Problem &
    fiber_problem() const
    {
      return fiber_problem_storage;
    }

    const VectorType &
    multiplier() const;

    const FiberReinforcedElastodynamicsResiduals &
    residuals() const
    {
      return residuals_storage;
    }

    /** Current coupled time and accepted-step number. */
    double
    current_time() const
    {
      return current_time_storage;
    }

    unsigned int
    time_step_number() const
    {
      return time_step_number_storage;
    }

    /** Elastic energy carried by the additive/excess fiber Problem. */
    double
    fiber_excess_elastic_energy() const;

    /** Difference from a matrix-only backward-Euler reference response. */
    double
    matrix_only_displacement_difference() const;

  private:
    void
    build_effective_matrices(double dt);

    void
    build_effective_rhs(const Problem    &problem,
                        const VectorType &previous_displacement,
                        const VectorType &previous_velocity,
                        double            time,
                        double            dt,
                        VectorType       &rhs) const;

    void
    update_diagnostics(const VectorType &matrix_rhs,
                       const VectorType &fiber_rhs);

    void
    output_results() const;

    void
    assemble_coupling_matrices();

#ifdef DEAL_II_WITH_SUNDIALS
    void
    run_with_ida();

    void
    setup_ida();

    void
    update_from_ida_state(const GlobalVectorType &state,
                          const GlobalVectorType &state_dot,
                          const double            time,
                          const unsigned int      step);

    void
    initialize_ida_derivative(GlobalVectorType &state_dot);
#endif

    const Parameters &parameters;
    Problem           matrix_problem_storage;
    Problem           fiber_problem_storage;

    using SchurSolver =
      LagrangeMultiplierSchurSolver<MatrixType,
                                    VectorType,
                                    ImmersXLA::MPI::PreconditionJacobi>;

    std::unique_ptr<dealii::FESystem<dim>>   multiplier_fe_storage;
    std::unique_ptr<dealii::DoFHandler<dim>> multiplier_dof_handler_storage;
    std::unique_ptr<dealii::IndexSet>        multiplier_relevant_storage;
    std::unique_ptr<dealii::AffineConstraints<double>>
                                           multiplier_constraints_storage;
    std::unique_ptr<FESpaceView<dim, dim>> matrix_space_storage;
    std::unique_ptr<FESpaceView<dim, dim>> fiber_space_storage;
    std::unique_ptr<FESpaceView<dim, dim>> multiplier_space_storage;
    std::shared_ptr<MatrixType>            matrix_to_multiplier_storage;
    std::shared_ptr<MatrixType>            fiber_to_multiplier_storage;
    std::shared_ptr<MatrixType>            matrix_coupling_storage;
    std::unique_ptr<SchurSolver>           schur_solver;
#ifdef DEAL_II_WITH_SUNDIALS
    std::unique_ptr<IDAAdapterType> ida_storage;

    ElastodynamicsFields matrix_fields_storage;
    ElastodynamicsFields fiber_fields_storage;
    ConstraintFields     coupling_fields_storage;
#endif

    MatrixType matrix_effective_matrix;
    MatrixType fiber_effective_matrix;
    VectorType multiplier_storage;
    VectorType matrix_only_displacement_storage;
    mutable std::vector<std::pair<double, std::string>>
      multiplier_output_records_storage;

    double       current_time_storage     = 0.;
    unsigned int time_step_number_storage = 0;
    bool         setup_complete           = false;
    bool         initial_conditions_set   = false;
    bool         effective_matrices_valid = false;
    double       effective_time_step      = 0.;

    FiberReinforcedElastodynamicsResiduals residuals_storage;
  };

} // namespace ImmersX

#endif // immersx_fiber_reinforced_elastodynamics_h
