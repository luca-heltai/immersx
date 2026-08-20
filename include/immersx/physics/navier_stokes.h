// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_navier_stokes_h
#define immersx_navier_stokes_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <immersx/algebra/linear_algebra.h>

#include <list>
#include <memory>
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


  /** Parameters for a standalone transient incompressible fluid problem.
   *
   * The problem solved by NavierStokesSolver is
   * \f[
   *   \rho(\partial_t u + (u\cdot\nabla)u)
   *     - \operatorname{div}(2\nu\varepsilon(u)) + \nabla p = \rho f,
   *   \qquad \operatorname{div}u=0.
   * \f]
   *
   * The first implementation uses backward Euler for the mass and viscous
   * terms. When `include_convective_term` is true, the convective term is
   * evaluated with the accepted velocity from the previous time step and moved
   * explicitly to the right-hand side. Setting it to false gives unsteady
   * Stokes.
   */
  template <int dim, int spacedim = dim>
  class NavierStokesParameters : public dealii::ParameterAcceptor
  {
  public:
    /** Construct a parameter object below @p subsection. */
    explicit NavierStokesParameters(
      const std::string &subsection = "/Navier-Stokes/");

    std::string  output_directory = ".";
    std::string  output_name      = "navier_stokes";
    unsigned int output_frequency = 1;

    unsigned int                          velocity_degree    = 2;
    unsigned int                          pressure_degree    = 1;
    unsigned int                          initial_refinement = 1;
    std::list<dealii::types::boundary_id> dirichlet_ids{0};

    std::string name_of_grid       = "hyper_cube";
    std::string arguments_for_grid = "-1: 1: false";
    std::string triangulation_type = "distributed";

    double density                 = 1.0;
    double viscosity               = 1.0;
    bool   include_convective_term = true;

    std::string  time_step_policy     = "number_of_steps";
    double       initial_time         = 0.0;
    double       final_time           = 0.1;
    double       time_step            = 0.01;
    unsigned int number_of_time_steps = 10;

    /** Optional exact field expression used by external error postprocessing.
     */
    std::string analytical_solution_expression =
      dim == 2 ? "0; 0; 0" : "0; 0; 0; 0";

    /** Parameter-driven convergence/error table for manufactured solutions. */
    mutable dealii::ParsedConvergenceTable convergence_table;

    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      rhs;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      bc;
    mutable dealii::ParameterAcceptorProxy<
      dealii::Functions::ParsedFunction<spacedim>>
      initial_condition;

    mutable dealii::ParameterAcceptorProxy<dealii::ReductionControl>
                 solver_control;
    unsigned int inner_solver_max_steps = 200;
    double       inner_solver_tolerance = 1.e-10;
    bool         log_solver_iterations  = false;

    /** Update the time used by all parsed functions. */
    void
    set_time(const double time) const;
  };


  /**
   * Distributed mixed finite-element solver for transient
   * Stokes/Navier--Stokes.
   *
   * The solver deliberately follows the lifecycle of PoissonSolver: grid
   * creation, finite-element setup, algebraic setup, assembly, solve, output,
   * and a small explicit time-stepping driver. A single Problem owns the native
   * two-block algebraic system `[velocity, pressure]`.
   *
   * The default finite-element pair is Taylor--Hood Q_k/Q_{k-1}. The pressure
   * is normalized by constraining one pressure degree of freedom to zero. This
   * is a direct algebraic pressure normalization; it removes the constant
   * nullspace without adding a third Lagrange-multiplier block.
   *
   * This first implementation supports full-dimensional two- and three-
   * dimensional meshes only (`dim == spacedim`). It has no elasticity,
   * immersed-boundary, coupling, ALE, or external time-integrator dependency.
   */
  template <int dim, int spacedim = dim>
  class NavierStokesSolver : public dealii::EnableObserverPointer
  {
    static_assert(dim == spacedim,
                  "NavierStokesSolver currently supports full-dimensional "
                  "meshes only.");
    static_assert(dim == 2 || dim == 3,
                  "NavierStokesSolver supports only 2D and 3D meshes.");

  public:
    using VectorType      = LA::MPI::Vector;
    using BlockVectorType = LA::MPI::BlockVector;

    explicit NavierStokesSolver(
      const NavierStokesParameters<dim, spacedim> &par);

    void
    make_grid();

    void
    setup_fe();

    void
    setup_system();

    /** Assemble the time-discrete system at the current time. */
    void
    assemble_system();

    /** Solve the currently assembled two-block saddle-point system. */
    void
    solve();

    void
    output_results() const;

    /** Advance one backward-Euler step and accept its solution. */
    void
    advance_one_timestep();

    /** Run the complete transient lifecycle. */
    void
    run();

    dealii::types::global_dof_index
    n_dofs() const;

    unsigned int
    n_time_steps() const;

    unsigned int
    timestep_number() const;

    double
    solution_l2_norm() const;

    bool
    solution_is_finite() const;

    double
    system_residual_l2_norm() const;

    double
    divergence_l2_norm() const;

    double
    current_time() const;

    double
    time_step() const;

    const dealii::parallel::TriangulationBase<dim, spacedim> &
    triangulation() const;

    const dealii::DoFHandler<dim, spacedim> &
    dof_handler() const;

    const dealii::FiniteElement<dim, spacedim> &
    finite_element() const;

    const dealii::Mapping<dim, spacedim> &
    mapping() const;

    const dealii::AffineConstraints<double> &
    constraints() const;

    const LA::MPI::BlockSparseMatrix &
    system_matrix() const;

    const LA::MPI::BlockSparseMatrix &
    mass_matrix() const;

    /** Return the continuous spatial Stokes operator [A,-B^T;-B,0]. */
    const LA::MPI::BlockSparseMatrix &
    continuous_operator() const;

    /** Return the physical velocity mass matrix without the density factor. */
    const LA::MPI::SparseMatrix &
    velocity_mass_matrix() const;

    /** Return the pressure metric used only by the native preconditioner. */
    const LA::MPI::SparseMatrix &
    pressure_metric_matrix() const;

    /** Assemble the unscaled velocity forcing vector at an external time. */
    void
    velocity_forcing_at_time(double time, LA::MPI::Vector &destination) const;

    const BlockVectorType &
    system_rhs() const;

    const BlockVectorType &
    solution() const;

    const BlockVectorType &
    previous_solution() const;

    const BlockVectorType &
    locally_relevant_solution() const;

    const dealii::IndexSet &
    locally_owned_dofs() const;

    const dealii::IndexSet &
    locally_relevant_dofs() const;

    const std::vector<dealii::IndexSet> &
    locally_owned_dofs_by_block() const;

    const std::vector<dealii::IndexSet> &
    locally_relevant_dofs_by_block() const;

    const dealii::FEValuesExtractors::Vector &
    velocity_extractor() const;

    const dealii::FEValuesExtractors::Scalar &
    pressure_extractor() const;

    const dealii::ComponentMask &
    velocity_component_mask() const;

    /** Return the physical density used by the continuous velocity equation. */
    double
    density() const;

    /** Return the global mixed-block offset of the pressure block. */
    dealii::types::global_dof_index
    velocity_block_size() const;

    /** Replace the accepted state and use it as the next-step history. */
    void
    set_solution(const BlockVectorType &new_solution);

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
    update_constraints();

    void
    interpolate_initial_condition();

    void
    update_locally_relevant_solution();

    void
    initialize_time_control();

    const NavierStokesParameters<dim, spacedim> &par;

    MPI_Comm                    mpi_communicator;
    dealii::ConditionalOStream  pcout;
    mutable dealii::TimerOutput computing_timer;

    TriangulationVariant                                  triangulation_storage;
    dealii::parallel::TriangulationBase<dim, spacedim>   *tria;
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe;
    std::unique_ptr<dealii::Quadrature<dim>>              quadrature;
    dealii::MappingQ1<dim, spacedim>                      mapping_storage;
    dealii::DoFHandler<dim, spacedim>                     dh;

    dealii::IndexSet                             owned_dofs;
    dealii::IndexSet                             relevant_dofs;
    std::vector<dealii::IndexSet>                owned_dofs_by_block;
    std::vector<dealii::IndexSet>                relevant_dofs_by_block;
    std::vector<dealii::types::global_dof_index> dofs_per_block;
    dealii::AffineConstraints<double>            constraints_storage;

    LA::MPI::BlockSparseMatrix system_matrix_storage;
    LA::MPI::BlockSparseMatrix mass_matrix_storage;
    LA::MPI::BlockSparseMatrix continuous_operator_storage;
    BlockVectorType            solution_storage;
    BlockVectorType            previous_solution_storage;
    BlockVectorType            system_rhs_storage;
    BlockVectorType            locally_relevant_solution_storage;

    LA::MPI::PreconditionAMG velocity_preconditioner;
    LA::MPI::PreconditionAMG pressure_preconditioner;

    dealii::FEValuesExtractors::Vector velocity;
    dealii::FEValuesExtractors::Scalar pressure;
    dealii::ComponentMask              velocity_mask;

    double       current_time_storage    = 0.0;
    double       time_step_storage       = 0.0;
    unsigned int n_time_steps_storage    = 0;
    unsigned int timestep_number_storage = 0;
    mutable std::vector<std::pair<double, std::string>> times_and_names;
    mutable unsigned int                                output_cycle = 0;
  };

} // namespace ImmersX

#endif // immersx_navier_stokes_h
