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
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/algebra/linear_algebra.h>

#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ImmersX
{
  namespace LA
  {
    using namespace ImmersXLA;
  } // namespace LA


  template <int dim, int spacedim = dim>
  /**
   * Parameters for a standalone first-order linear elastodynamics problem.
   *
   * The public name deliberately uses *Elastodynamics*: the repository already
   * contains an `ElasticityProblem` for immersed coupling, and this class is a
   * separate uncoupled volumetric problem. The state is
   * $[d,v]$, with displacement $d$ and velocity $v$.
   */
  class ElastodynamicsParameters : public dealii::ParameterAcceptor
  {
  public:
    /** Construct the parameters below a configurable subsection. */
    explicit ElastodynamicsParameters(
      const std::string &subsection = "/Elastodynamics/");

    std::string  output_directory    = ".";
    std::string  output_name         = "elastodynamics";
    unsigned int fe_degree           = 1;
    unsigned int initial_refinement  = 2;
    unsigned int output_frequency    = 1;
    unsigned int n_refinement_cycles = 1;

    std::set<dealii::types::boundary_id> dirichlet_ids{0};
    std::string                          name_of_grid       = "hyper_cube";
    std::string                          arguments_for_grid = "-1: 1: false";
    std::string                          triangulation_type = "distributed";

    double density       = 1.0;
    double lame_mu       = 1.0;
    double lame_lambda   = 1.0;
    double damping_shear = 0.0;
    double damping_bulk  = 0.0;

    double       initial_time    = 0.0;
    double       final_time      = 0.1;
    double       time_step       = 1.e-2;
    unsigned int number_of_steps = 0;

    /** Body force, displacement boundary data, and velocity boundary data. */
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      body_force;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      displacement_boundary;
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
  };


  template <int dim, int spacedim = dim>
  /**
   * Standalone first-order-in-time linear elastodynamics solver.
   *
   * The supported problems are full-dimensional volumetric elasticity in two
   * and three dimensions (`dim == spacedim`). The spatial finite-element space
   * is one vector-valued `FESystem<dim>(FE_Q<dim>, spacedim)`, represented by
   * one DoFHandler. Displacement and velocity are separate algebraic vectors on
   * that same space.
   *
   * The continuous semidiscrete equations are
   * [
   *   M\dot d - Mv = 0, \qquad
   *   M\dot v + Kd + Dv = f.
   * ]
   * Here `M` is the consistent physical mass matrix, also used for the weak
   * kinematic equation. `K` is the linear isotropic elasticity operator and `D`
   * is the optional Kelvin--Voigt damping operator. These three matrices remain
   * separate public operators; the backward-Euler matrix is only a standalone
   * driver.
   *
   * No coupling, particles, multipliers, moving geometry, nonlinear materials,
   * or SUNDIALS dependency belongs to this class. Its intended next use is as a
   * clean source of a `SemiDiscreteModel` residual for SUNDIALS and future FSI.
   */
  class ElastodynamicsSolver : public dealii::EnableObserverPointer
  {
    static_assert(dim == spacedim && (dim == 2 || dim == 3),
                  "ElastodynamicsSolver supports only 2D and 3D volumetric "
                  "problems with dim == spacedim.");

  public:
    using VectorType = LA::MPI::Vector;
    using MatrixType = LA::MPI::SparseMatrix;

    explicit ElastodynamicsSolver(
      const ElastodynamicsParameters<dim, spacedim> &par);

    /** Build the distributed mesh from the configured grid generator. */
    void
    make_grid();

    /** Create the vector FE and quadrature. */
    void
    setup_fe();

    /** Distribute DoFs, constraints, sparsity patterns, matrices, and vectors.
     */
    void
    setup_system();

    /** Assemble the continuous spatial operators `M`, `K`, `D`. */
    void
    assemble_operators();

    /** Interpolate and constrain the configured state at the initial time. */
    void
    set_initial_conditions();

    /** Advance one backward-Euler step using the configured time step. */
    void
    advance_one_timestep();

    /** Alias for one standalone backward-Euler solve step. */
    void
    solve();

    /** Write displacement and velocity output for the current state. */
    void
    output_results() const;

    /** Run setup, initialization, and the configured backward-Euler time loop.
     */
    void
    run();

    /** Replace displacement while preserving its constraints and ghost state.
     */
    void
    set_displacement(const VectorType &new_displacement);

    /** Replace velocity while preserving its constraints and ghost state. */
    void
    set_velocity(const VectorType &new_velocity);

    /** Return the global number of spatial DoFs. */
    dealii::types::global_dof_index
    n_dofs() const;

    /** Return whether both current state vectors have finite norms. */
    bool
    state_is_finite() const;

    /** Return the distributed mesh. */
    const dealii::parallel::TriangulationBase<dim, spacedim> &
    triangulation() const;

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

    /**
     * Return velocity constraints, including its explicit velocity boundary.
     *
     * The displacement and velocity boundary functions are independent parsed
     * functions. The solver does not numerically differentiate displacement
     * data: callers prescribing a moving boundary should provide consistent
     * values in both subsections.
     */
    const dealii::AffineConstraints<double> &
    velocity_constraints() const;

    /** Return locally owned spatial DoFs. */
    const dealii::IndexSet &
    locally_owned_dofs() const;

    /** Return locally relevant spatial DoFs. */
    const dealii::IndexSet &
    locally_relevant_dofs() const;

    /** Return the consistent mass operator. */
    const MatrixType &
    mass_matrix() const;

    /** Return the linear isotropic elasticity stiffness operator. */
    const MatrixType &
    stiffness_matrix() const;

    /** Return the Kelvin--Voigt damping operator, possibly identically zero. */
    const MatrixType &
    damping_matrix() const;

    /** Return the body-force vector assembled at the current time. */
    const VectorType &
    body_force_vector() const;

    /** Assemble the body-force row at an externally supplied time. */
    void
    body_force_at_time(double time, VectorType &destination) const;

    /** Return the internal backward-Euler matrix from the last step. */
    const MatrixType &
    system_matrix() const;

    /** Return the right-hand side from the last backward-Euler step. */
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

    /** Return the number of accepted backward-Euler steps. */
    unsigned int
    time_step_number() const;

  private:
    using DistributedTriangulation =
      dealii::parallel::distributed::Triangulation<dim, spacedim>;
    using FullyDistributedTriangulation =
      dealii::parallel::fullydistributed::Triangulation<dim, spacedim>;
    using TriangulationVariant =
      std::variant<DistributedTriangulation, FullyDistributedTriangulation>;

    static TriangulationVariant
    make_triangulation_storage(MPI_Comm mpi_communicator);

    bool
    uses_fully_distributed_triangulation() const;

    void
    update_constraints(double time);

    void
    assemble_body_force(double time);

    void
    assemble_backward_euler_system(const VectorType &previous_displacement,
                                   const VectorType &previous_velocity,
                                   double            dt);

    void
    advance_one_timestep(double dt);

    void
    solve_backward_euler_system();

    void
    update_locally_relevant_state();

    void
    run_time_integration();

    static void
    copy_constraints(const dealii::AffineConstraints<double> &source,
                     dealii::AffineConstraints<double>       &target,
                     dealii::types::global_dof_index          shift);

    const ElastodynamicsParameters<dim, spacedim> &par;

    MPI_Comm                    mpi_communicator;
    dealii::ConditionalOStream  pcout;
    mutable dealii::TimerOutput computing_timer;

    TriangulationVariant                                  triangulation_storage;
    dealii::parallel::TriangulationBase<dim, spacedim>   *tria;
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe_storage;
    std::unique_ptr<dealii::Quadrature<dim>>              quadrature;
    std::unique_ptr<dealii::Mapping<dim, spacedim>>       mapping_storage;
    dealii::DoFHandler<dim, spacedim>                     dh;

    dealii::IndexSet                  owned_dofs;
    dealii::IndexSet                  relevant_dofs;
    dealii::IndexSet                  combined_owned_dofs;
    dealii::IndexSet                  combined_relevant_dofs;
    dealii::AffineConstraints<double> displacement_constraints_storage;
    dealii::AffineConstraints<double> velocity_constraints_storage;
    dealii::AffineConstraints<double> combined_constraints_storage;

    MatrixType mass_matrix_storage;
    MatrixType stiffness_matrix_storage;
    MatrixType damping_matrix_storage;
    MatrixType system_matrix_storage;

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
