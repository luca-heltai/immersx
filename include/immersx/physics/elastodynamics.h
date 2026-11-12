// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 2.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the LICENSE.md file at the top
// level of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_elastodynamics_h
#define immersx_elastodynamics_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/domain.h>
#include <immersx/core/robin_boundary.h>
#include <immersx/core/second_order_preconditioner.h>
#include <immersx/core/time_parameters.h>

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ImmersX
{
  namespace LA
  {
    using namespace ImmersXLA;
  } // namespace LA

  /** Isotropic elastic, Rayleigh, and Kelvin--Voigt material coefficients. */
  struct ElastodynamicsMaterialCoefficients
  {
    double density     = 1.0;
    double lame_mu     = 1.0;
    double lame_lambda = 1.0;

    double rayleigh_alpha = 0.0;
    double rayleigh_beta  = 0.0;
    double viscosity_eta  = 0.0;
  };


  /** Register one named material below Material properties. */
  class ElastodynamicsMaterialParameters : public dealii::ParameterAcceptor
  {
  public:
    explicit ElastodynamicsMaterialParameters(const std::string &subsection);

    ElastodynamicsMaterialCoefficients coefficients;
  };

  template <int dim, int spacedim = dim>
  /**
   * Parameters for a standalone first-order linear elastodynamics problem.
   *
   * The public name deliberately uses *Elastodynamics*: the repository already
   * contains an `ElasticityProblem` for immersed coupling, and this class is a
   * separate uncoupled volumetric problem. The state is
   * \f$[d,v]\f$, with displacement \f$d\f$ and velocity \f$v\f$.
   */
  class ElastodynamicsParameters : public dealii::ParameterAcceptor
  {
  private:
    std::unique_ptr<TimeIntervalParameters> owned_time_parameters;
    std::unique_ptr<FixedStepParameters>    owned_fixed_step_parameters;
    std::unique_ptr<IDAParameters>          owned_ida_parameters;

  public:
    /** Construct the parameters below a configurable subsection. */
    explicit ElastodynamicsParameters(
      const std::string      &subsection                   = "/Elastodynamics/",
      TimeIntervalParameters *shared_time_parameters       = nullptr,
      FixedStepParameters    *shared_fixed_step_parameters = nullptr,
      IDAParameters          *shared_ida_parameters        = nullptr);

    std::string                     output_directory = ".";
    std::string                     output_name      = "elastodynamics";
    unsigned int                    fe_degree        = 1;
    DomainParameters<dim, spacedim> domain_parameters;
    unsigned int                    n_refinement_cycles = 1;

    std::set<dealii::types::boundary_id> dirichlet_ids{0};
    std::set<dealii::types::boundary_id> neumann_ids;
    std::set<dealii::types::boundary_id> robin_ids;

    ElastodynamicsMaterialParameters default_material;

    std::map<dealii::types::material_id, std::string>
      material_tags_by_material_id;

    std::map<std::string, std::unique_ptr<ElastodynamicsMaterialParameters>>
      material_parameters_by_tag;

    /** Return the mapped material, falling back to the default for unmapped
     * ids. */
    ElastodynamicsMaterialCoefficients
    get_material_properties(dealii::types::material_id material_id) const;

    /** Solver-independent interval used by this Problem. */
    TimeIntervalParameters &time_parameters;
    /** Fixed-step policy used by the standalone driver. */
    FixedStepParameters &fixed_step_parameters;
    /** IDA policy used by the adapter application. */
    IDAParameters &ida_parameters;

    /** Body force and boundary data.
     *
     * `neumann_boundary` is a vector traction integrated against the test
     * functions on `neumann_ids`. The supplied body force therefore follows
     * the residual convention `M u_tt + K u + D u_t - f = 0`, with the
     * boundary traction included in `f`.
     */
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      body_force;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      displacement_boundary;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      neumann_boundary;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      robin_coefficient;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      robin_rhs;
    /**
     * Optional explicit velocity data for the semidiscrete adapter.
     *
     * The standalone time integrator derives the velocity constraint from
     * displacement boundary values and the selected time-stepping rule, so a
     * velocity boundary subsection is not required there.
     */
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      velocity_boundary;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      initial_displacement;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      initial_velocity;

    /** Optional manufactured displacement used for convergence tables. */
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      exact_solution;

    mutable dealii::ParameterAcceptorProxy<dealii::ReductionControl>
      solver_control;

    /** Convergence table for the displacement at the final time. */
    mutable dealii::ParsedConvergenceTable convergence_table;

  private:
    template <int, int>
    friend class ElastodynamicsSolver;

    bool robin_coefficient_depends_on_time     = true;
    bool robin_rhs_depends_on_time             = true;
    bool body_force_depends_on_time            = true;
    bool neumann_boundary_depends_on_time      = true;
    bool body_force_is_zero                    = false;
    bool neumann_boundary_is_zero              = false;
    bool displacement_boundary_depends_on_time = true;
    bool velocity_boundary_depends_on_time     = true;
  };


  template <int dim, int spacedim = dim>
  /**
   * Standalone first-order-in-time linear elastodynamics solver.
   *
   * The spatial domain may be embedded in a higher-dimensional ambient space,
   * with `1 <= dim <= spacedim <= 3`. The spatial finite-element space is one
   * vector-valued `FESystem<dim>(FE_Q<dim>, spacedim)` on hypercube meshes, or
   * `FESystem<dim>(FE_SimplexP<dim>, spacedim)` on simplex meshes, represented
   * by one DoFHandler. Simplex meshes need the fullydistributed Domain
   * triangulation. Displacement and velocity are separate algebraic vectors on
   * that same space. The solver exposes separate mass, stiffness, and optional
   * damping operators and advances the state with the configured standalone
   * time-stepping strategy.
   *
   * Coupling, particles, multipliers, moving geometry, nonlinear materials,
   * and SUNDIALS policy belong outside this class.
   */
  class ElastodynamicsSolver : public dealii::EnableObserverPointer
  {
    static_assert(dim >= 1 && dim <= spacedim && spacedim <= 3,
                  "ElastodynamicsSolver requires 1 <= dim <= spacedim <= 3.");

  public:
    using VectorType = LA::MPI::Vector;
    using MatrixType = LA::MPI::SparseMatrix;

    explicit ElastodynamicsSolver(
      const ElastodynamicsParameters<dim, spacedim> &par);

    /** Build the distributed mesh from the configured grid generator. */
    void
    make_grid();

    /** Build the mesh through a coupled Domain preparation policy.
     * The callback must prepare and finalize the Domain before returning.
     * This runs before attaching the application DoFHandler.
     */
    void
    make_grid(const std::function<void(Domain<dim, spacedim> &)> &initialize);

    /** Create the vector FE, quadratures, and mapping for the mesh cells. */
    void
    setup_fe();

    /** Distribute DoFs, constraints, sparsity patterns, matrices, and vectors.
     */
    void
    setup_system();

    /** Assemble `M`, `K`, `D` once after setup_system(). Repeated calls reuse
     * the operators until the next spatial setup.
     */
    void
    assemble_operators();

    /** Interpolate and constrain the configured state at the initial time. */
    void
    set_initial_conditions();

    /** Refine the mesh for the next spatial convergence cycle. */
    void
    refine_global();

    /** Compute the physically consistent initial acceleration. */
    void
    initial_acceleration(VectorType &acceleration) const;

    /** Advance one configured time-integration step. */
    void
    advance_one_timestep();

    /** Alias for one standalone time-integration solve step. */
    void
    solve();

    /** Write displacement and velocity output for the current state. */
    void
    output_results() const;

    /** Add the current displacement error to the configured error table. */
    void
    compute_error() const;

    /** Run setup, initialization, and the configured time loop. */
    void
    run();

    /** Replace displacement while preserving its constraints and ghost state.
     */
    void
    set_displacement(const VectorType &new_displacement);

    /** Replace velocity while preserving its constraints and ghost state. */
    void
    set_velocity(const VectorType &new_velocity);

    /**
     * Atomically accept an externally solved state.
     *
     * This seam is for application-level coupled drivers. It updates the
     * problem time and accepted-step counter together with displacement and
     * velocity, then reapplies the current homogeneous/inhomogeneous
     * constraints and ghost values. It does not assemble or solve a standalone
     * time-integration system.
     */
    void
    accept_state(const VectorType &new_displacement,
                 const VectorType &new_velocity,
                 double            time,
                 unsigned int      step_number);

    /** Return the global number of spatial DoFs. */
    dealii::types::global_dof_index
    n_dofs() const;

    /** Return whether both current state vectors have finite norms. */
    bool
    state_is_finite() const;

    /** Return the distributed mesh. */
    const dealii::parallel::TriangulationBase<dim, spacedim> &
    triangulation() const;

    /** Homogeneous mesh constraints, without prescribed boundary values. */
    const dealii::AffineConstraints<double> &
    hanging_node_constraints() const;

    /** Return the vector-valued finite element. */
    const dealii::FiniteElement<dim, spacedim> &
    fe() const;

    /** Return the mapping used by this problem. */
    const dealii::Mapping<dim, spacedim> &
    mapping() const;

    /** Return the spatial DoFHandler. */
    const dealii::DoFHandler<dim, spacedim> &
    dof_handler() const;

    /** Return displacement constraints, including hanging-node constraints. */
    const dealii::AffineConstraints<double> &
    constraints() const;

    /** Return the current velocity constraints, including hanging nodes. */
    const dealii::AffineConstraints<double> &
    velocity_constraints() const;

    /** Return locally owned spatial DoFs. */
    const dealii::IndexSet &
    locally_owned_dofs() const;

    /** Return locally relevant spatial DoFs. */
    const dealii::IndexSet &
    locally_relevant_dofs() const;

    /** Return the consistent mass operator with mesh slaves eliminated.
     * Prescribed-value boundary columns remain. Hanging-node rows and columns
     * are zero; the adapter enforces constraint inhomogeneities in these slots.
     * Physical state and output reconstruct mesh-slave values.
     */
    const MatrixType &
    mass_matrix() const;

    /** Return the diagonal of mass_matrix() as a matrix with a diagonal
     * sparsity pattern. The kinematic rows D (u' - v) = 0 use it instead of
     * the consistent mass: any positive diagonal gives u' = v, and the
     * Jacobian then contains the mass pattern only once. Hanging-node rows are
     * zero, as in mass_matrix(). Row-sum lumping is not used because it gives
     * negative vertex entries for quadratic simplices.
     */
    const MatrixType &
    diagonal_mass_matrix() const;

    /** Return the stiffness operator with the same mesh condensation as mass.
     */
    const MatrixType &
    stiffness_matrix() const;

    /** Return the Rayleigh and Kelvin--Voigt damping operator, possibly
     * identically zero. */
    const MatrixType &
    damping_matrix() const;

    /** Homogeneously condensed spatial operators. The mass operator has
     * identity rows on constrained DoFs; stiffness and damping have zero rows.
     */
    const MatrixType &
    homogeneous_mass_matrix() const;

    const MatrixType &
    homogeneous_stiffness_matrix() const;

    const MatrixType &
    homogeneous_damping_matrix() const;

    /** Return the body-force vector assembled at the current time. */
    const VectorType &
    body_force_vector() const;

    /** Return the load at an externally supplied time. Time-independent body
     * forces and tractions are reused until the next setup_system().
     */
    void
    body_force_at_time(double time, VectorType &destination) const;

    /** Update time-dependent essential constraints without changing state. */
    void
    update_constraints(double time) const;

    /** Update boundary stiffness at a candidate time without accepting state.
     */
    void
    update_boundary_operators(double time) const;

    /** Counts of actual assembly/update work over this object's lifetime. */
    struct AssemblyStatistics
    {
      unsigned long spatial_operators  = 0;
      unsigned long robin_operators    = 0;
      unsigned long body_force         = 0;
      unsigned long constraints        = 0;
      unsigned long time_system_matrix = 0;
    };

    /** Return monotone counters; spatial setup invalidates caches, not counts.
     */
    const AssemblyStatistics &
    assembly_statistics() const;

    /** Return the internal time-integration matrix from the last step. */
    const MatrixType &
    system_matrix() const;

    /** Return the right-hand side from the last time-integration step. */
    const VectorType &
    system_rhs() const;

    /** Return the current owned displacement vector. */
    const VectorType &
    displacement() const;

    /** Return the current owned velocity vector. */
    const VectorType &
    velocity() const;

    /** Return the current physical time. */
    double
    current_time() const;

    /** Return the time step used by the most recent step. */
    double
    time_step() const;

    /** Return the number of accepted time-integration steps. */
    unsigned int
    time_step_number() const;

  private:
    void
    assemble_body_force(double time);

    void
    update_constraints(double time, bool update_velocity) const;

    void
    prepare_time_integration_matrix(double dt, double theta);

    void
    assemble_time_integration_rhs(const VectorType &previous_displacement,
                                  const VectorType &previous_velocity,
                                  const VectorType &previous_body_force,
                                  double            dt,
                                  double            theta);

    void
    assemble_backward_euler_system(const VectorType &previous_displacement,
                                   const VectorType &previous_velocity,
                                   double            dt);

    void
    assemble_trapezoidal_system(const VectorType &previous_displacement,
                                const VectorType &previous_velocity,
                                const VectorType &previous_body_force,
                                double            dt);

    void
    advance_one_timestep(double dt);

    void
    solve_time_integration_system();

    void
    update_locally_relevant_state();

    void
    run_time_integration();

    static void
    copy_constraints(const dealii::AffineConstraints<double> &source,
                     dealii::AffineConstraints<double>       &target,
                     dealii::types::global_dof_index          shift);

    void
    rebuild_combined_constraints() const;

    void
    update_discrete_velocity_constraints(
      const dealii::AffineConstraints<double> &previous_displacement,
      double                                   previous_time,
      double                                   current_time) const;

    void
    update_trapezoidal_velocity_constraints(
      const dealii::AffineConstraints<double> &previous_displacement,
      const dealii::AffineConstraints<double> &previous_velocity,
      double                                   previous_time,
      double                                   current_time) const;

    const ElastodynamicsParameters<dim, spacedim> &par;

    MPI_Comm                    mpi_communicator;
    dealii::ConditionalOStream  pcout;
    mutable dealii::TimerOutput computing_timer;

    Domain<dim, spacedim>                                 domain;
    dealii::parallel::TriangulationBase<dim, spacedim>   *tria;
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe_storage;
    std::unique_ptr<dealii::Quadrature<dim>>              quadrature;
    std::unique_ptr<dealii::Quadrature<dim - 1>>          face_quadrature;
    std::unique_ptr<dealii::Mapping<dim, spacedim>>       mapping_storage;
    dealii::DoFHandler<dim, spacedim>                     dh;

    dealii::IndexSet                          owned_dofs;
    dealii::IndexSet                          relevant_dofs;
    dealii::IndexSet                          combined_owned_dofs;
    dealii::IndexSet                          combined_relevant_dofs;
    dealii::AffineConstraints<double>         hanging_node_constraints_storage;
    mutable dealii::AffineConstraints<double> displacement_constraints_storage;
    mutable dealii::AffineConstraints<double> velocity_constraints_storage;
    mutable dealii::AffineConstraints<double> combined_constraints_storage;

    MatrixType                    mass_matrix_storage;
    MatrixType                    diagonal_mass_matrix_storage;
    mutable MatrixType            stiffness_matrix_storage;
    MatrixType                    damping_matrix_storage;
    MatrixType                    system_matrix_storage;
    MatrixType                    homogeneous_mass_matrix_storage;
    mutable MatrixType            homogeneous_stiffness_matrix_storage;
    mutable std::optional<double> robin_operator_time;
    mutable VectorType            cached_robin_force;
    mutable std::optional<double> robin_force_time;
    VectorType                    previous_stiffness_action;
    MatrixType                    homogeneous_damping_matrix_storage;

    mutable std::unique_ptr<
      SecondOrderBlockPreconditioner<VectorType, MatrixType>>
      time_preconditioner;

    bool                          spatial_operators_ready = false;
    mutable std::optional<double> system_matrix_time_step;
    double                        system_matrix_theta = 0.;
    mutable std::optional<double> displacement_constraints_time;
    mutable std::optional<double> velocity_constraints_time;
    mutable bool       velocity_constraints_are_boundary_data = false;
    mutable bool       combined_constraints_initialized       = false;
    mutable VectorType cached_body_force;
    mutable VectorType cached_volume_force;
    mutable VectorType cached_traction_force;
    mutable std::optional<double> volume_force_cache_time;
    mutable std::optional<double> traction_force_cache_time;
    mutable AssemblyStatistics    assembly_statistics_storage;

    VectorType body_force_storage;
    VectorType displacement_storage;
    VectorType velocity_storage;
    VectorType locally_relevant_displacement;
    VectorType locally_relevant_velocity;
    VectorType system_rhs_storage;

    mutable std::vector<std::pair<double, std::string>> cycles_and_solutions;

    double       current_time_storage     = 0.0;
    double       current_time_step        = 0.0;
    unsigned int time_step_number_storage = 0;
    unsigned int refinement_cycle_storage = 0;
  };

} // namespace ImmersX

#endif // immersx_elastodynamics_h
