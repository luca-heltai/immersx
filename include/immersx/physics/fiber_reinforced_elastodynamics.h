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

#include <immersx/algebra/lagrange_multiplier_schur_solver.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/time_parameters.h>
#include <immersx/io/output_handler.h>
#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#endif
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>

#include <memory>
#include <string>
#include <utility>

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

    TimeIntervalParameters time_parameters;
    FixedStepParameters    fixed_step_parameters;
    IDAParameters          ida_parameters;

    ElastodynamicsParameters<dim, dim>   matrix_parameters;
    ElastodynamicsParameters<1, dim>     fiber_parameters;
    FiniteElementSpaceParameters<1, dim> multiplier_parameters;

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
   * Standalone coupled driver for a matrix and an embedded excess fiber.
   *
   * The matrix is an `ElastodynamicsSolver<dim, dim>` and the fiber is an
   * independently meshed `ElastodynamicsSolver<1, dim>`. The driver owns the
   * coupled time loop. With SUNDIALS enabled, it uses the public IDA execution
   * adapter; without SUNDIALS, it uses the explicit Schur-complement
   * backward-Euler path.
   */
  template <int dim>
  class FiberReinforcedElastodynamics
  {
  public:
    using Parameters    = FiberReinforcedElastodynamicsParameters<dim>;
    using MatrixProblem = ElastodynamicsSolver<dim, dim>;
    using FiberProblem  = ElastodynamicsSolver<1, dim>;
    using Problem       = MatrixProblem;
    using VectorType    = typename MatrixProblem::VectorType;
    using MatrixType    = typename MatrixProblem::MatrixType;
    using MultiplierOutputHandler = OutputHandler<1, dim, VectorType>;
#ifdef DEAL_II_WITH_SUNDIALS
    using GlobalVectorType = ImmersXLA::MPI::BlockVector;
    using IDAAdapterType   = IDAAdapter<VectorType, GlobalVectorType>;
#endif

    explicit FiberReinforcedElastodynamics(const Parameters &parameters);

    /** Prepare only the matrix Elastodynamics Problem. */
    void
    prepare_matrix_problem();

    /** Prepare only the embedded fiber Elastodynamics Problem. */
    void
    prepare_fiber_problem();

    /** Prepare the multiplier space and velocity-continuity coupling. */
    void
    prepare_velocity_continuity();

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

    /** Run a composition whose Problems and interaction were prepared first. */
    void
    run_execution();

#ifdef DEAL_II_WITH_SUNDIALS
    /** Run the prepared matrix/fiber/interaction composition through IDA. */
    void
    run_ida_execution();
#endif

    const MatrixProblem &
    matrix_problem() const
    {
      return matrix_problem_storage;
    }

    const FiberProblem &
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

  private:
    void
    build_effective_matrices(double dt);

    template <int problem_dim, int spacedim>
    void
    build_effective_rhs(
      const ElastodynamicsSolver<problem_dim, spacedim> &problem,
      const VectorType                                  &previous_displacement,
      const VectorType                                  &previous_velocity,
      double                                             time,
      double                                             dt,
      VectorType                                        &rhs) const;

    void
    update_diagnostics(const VectorType &matrix_rhs,
                       const VectorType &fiber_rhs);

    void
    output_results() const;

    void
    assemble_coupling_matrices();

    double
    continuity_defect(const VectorType &matrix_values,
                      const VectorType &fiber_values) const;


    const Parameters &parameters;
    MatrixProblem     matrix_problem_storage;
    FiberProblem      fiber_problem_storage;

    using SchurSolver =
      LagrangeMultiplierSchurSolver<MatrixType,
                                    VectorType,
                                    ImmersXLA::MPI::PreconditionJacobi>;

    std::unique_ptr<FiniteElementSpace<1, dim>>     multiplier_space_owner;
    std::unique_ptr<FiniteElementSpaceView<1, dim>> multiplier_space_storage;
    StateLayout                              multiplier_output_layout_storage;
    FieldId                                  multiplier_output_field_id_storage;
    std::unique_ptr<MultiplierOutputHandler> multiplier_output_storage;
    std::shared_ptr<MatrixType>              matrix_to_multiplier_storage;
    std::shared_ptr<MatrixType>              fiber_to_multiplier_storage;
    std::shared_ptr<MatrixType>              matrix_coupling_storage;
    std::shared_ptr<MatrixType>              outside_mass_storage;
    std::unique_ptr<SchurSolver>             schur_solver;


    MatrixType matrix_effective_matrix;
    MatrixType fiber_effective_matrix;
    VectorType multiplier_storage;

    double       current_time_storage     = 0.;
    unsigned int time_step_number_storage = 0;
    bool         setup_complete           = false;
    bool         matrix_setup_complete    = false;
    bool         fiber_setup_complete     = false;
    bool         coupling_setup_complete  = false;
    bool         initial_conditions_set   = false;
    bool         effective_matrices_valid = false;
    double       effective_time_step      = 0.;

    FiberReinforcedElastodynamicsResiduals residuals_storage;
  };

} // namespace ImmersX

#endif // immersx_fiber_reinforced_elastodynamics_h
