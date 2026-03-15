// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

/* ---------------------------------------------------------------------
 */
#ifndef dealii_distributed_lagrange_multiplier_h
#define dealii_distributed_lagrange_multiplier_h

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
// #define MATRIX_FREE_PATH

#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#ifdef MATRIX_FREE_PATH
#  include <deal.II/matrix_free/operators.h>

#  include <deal.II/multigrid/mg_coarse.h>
#  include <deal.II/multigrid/mg_matrix.h>
#  include <deal.II/multigrid/mg_smoother.h>
#  include <deal.II/multigrid/mg_tools.h>
#  include <deal.II/multigrid/mg_transfer_matrix_free.h>
#  include <deal.II/multigrid/multigrid.h>
#endif

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include <matrix_free_utils.h>

#include "inclusions.h"


#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>


template <int dim, int spacedim = dim>
/**
 * Parameter set for the immersed Poisson benchmark.
 */
class ProblemParameters : public ParameterAcceptor
{
public:
  /**
   * Register all parameters required by the immersed Poisson benchmark.
   */
  ProblemParameters();

  /**
   * Output, mesh, discretization, and boundary-condition settings.
   */
  /// @{
  std::string                   output_directory = "."; ///< Output folder.
  std::string                   output_name      = "solution"; ///< Output stem.
  unsigned int                  fe_degree        = 1;          ///< FE degree.
  unsigned int                  initial_refinement = 5; ///< Global refinements.
  std::list<types::boundary_id> dirichlet_ids{0}; ///< Dirichlet boundary ids.
  std::string  name_of_grid = "hyper_cube"; ///< Grid generator/input name.
  std::string  arguments_for_grid  = "-1: 1: false";   ///< Grid arguments.
  std::string  refinement_strategy = "fixed_fraction"; ///< Adaptivity strategy.
  double       coarsening_fraction = 0.0;              ///< Coarsening fraction.
  double       refinement_fraction = 0.3;              ///< Refinement fraction.
  unsigned int n_refinement_cycles = 1;                ///< Adapt cycles.
  unsigned int max_cells           = 20000;            ///< Cell cap.
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    rhs; ///< RHS function.
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    bc; ///< Dirichlet data.
  /// @}

  /**
   * Inner/outer Krylov solver controls.
   */
  /// @{
  mutable ParameterAcceptorProxy<ReductionControl> inner_control;
  mutable ParameterAcceptorProxy<ReductionControl>
    outer_control; ///< Outer solver control.
  /// @}

  /**
   * Emit output once before linear-system solution.
   */
  bool output_results_before_solving = false;

  /**
   * Convergence table used to report global error and timing quantities.
   */
  mutable ParsedConvergenceTable convergence_table;
};



template <int dim, int spacedim>
ProblemParameters<dim, spacedim>::ProblemParameters()
  : ParameterAcceptor("/Immersed Problem/")
  , rhs("/Immersed Problem/Right hand side")
  , bc("/Immersed Problem/Dirichlet boundary conditions")
  , inner_control("/Immersed Problem/Solver/Inner control")
  , outer_control("/Immersed Problem/Solver/Outer control")
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  enter_subsection("Grid generation");
  {
    add_parameter("Grid generator", name_of_grid);
    add_parameter("Grid generator arguments", arguments_for_grid);
  }
  leave_subsection();
  enter_subsection("Refinement and remeshing");
  {
    add_parameter("Strategy",
                  refinement_strategy,
                  "",
                  this->prm,
                  Patterns::Selection("fixed_fraction|fixed_number|global"));
    add_parameter("Coarsening fraction", coarsening_fraction);
    add_parameter("Refinement fraction", refinement_fraction);
    add_parameter("Maximum number of cells", max_cells);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
  }
  leave_subsection();

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();
}



template <int dim, int spacedim = dim>
/**
 * Solver for Poisson problems with reduced Lagrange multiplier coupling.
 */
class PoissonProblem : public Subscriptor
{
public:
  /**
   * Build the problem from parsed parameters.
   */
  PoissonProblem(const ProblemParameters<dim, spacedim> &par);
  /**
   * Create or import the computational mesh.
   */
  void
  make_grid();
  /**
   * Initialize finite element and quadrature objects.
   */
  void
  setup_fe();
  /**
   * Distribute DoFs and initialize matrices/vectors.
   */
  void
  setup_dofs();
#ifndef MATRIX_FREE_PATH
  /**
   * Assemble the bulk Poisson matrix and forcing block.
   */
  void
  assemble_poisson_system();
#else
  /**
   * Assemble only the right-hand side when using matrix-free operators.
   */
  void
  assemble_rhs();
#endif
  /**
   * Assemble immersed coupling matrix and multiplier right-hand side.
   */
  void
  assemble_coupling();
  /**
   * Run the full solve/refinement pipeline.
   */
  void
  run();

  /**
   * Builds coupling sparsity, and returns locally relevant inclusion dofs.
   */
  IndexSet
  assemble_coupling_sparsity(DynamicSparsityPattern &dsp) const;

  /**
   * Solve the coupled linear system.
   */
  void
  solve();

  /**
   * Perform adaptive refinement and transfer block vectors.
   */
  void
  refine_and_transfer();

  /**
   * Return output base filename for the current cycle.
   */
  std::string
  output_solution() const;

  /**
   * Write current solution fields to disk.
   */
  void
  output_results() const;

  /**
   * Print selected runtime and parameter values.
   */
  void
  print_parameters() const;

private:
  /**
   * Parsed parameter object.
   */
  const ProblemParameters<dim, spacedim> &par;
  /**
   * MPI communicator used by distributed data structures.
   */
  MPI_Comm mpi_communicator;
  /**
   * Stream that prints only on rank zero.
   */
  ConditionalOStream pcout;
  /**
   * Timer collecting wall times by section.
   */
  mutable TimerOutput computing_timer;
  /**
   * Distributed bulk triangulation.
   */
  parallel::distributed::Triangulation<spacedim> tria;
  /**
   * Finite element used for the background field.
   */
  std::unique_ptr<FiniteElement<spacedim>> fe;
  /**
   * Inclusion geometry and reduced basis data.
   */
  Inclusions<spacedim> inclusions;
  /**
   * Quadrature on the background mesh.
   */
  std::unique_ptr<Quadrature<spacedim>> quadrature;

  /**
   * DoF metadata for the background field.
   */
  /// @{
  DoFHandler<spacedim>  dh;
  std::vector<IndexSet> owned_dofs;    ///< Locally-owned DoF sets per block.
  std::vector<IndexSet> relevant_dofs; ///< Locally-relevant DoF sets per block.
  /// @}

  /**
   * Constraints in bulk and multiplier blocks.
   */
  /// @{
  AffineConstraints<double> constraints;
  AffineConstraints<double>
    inclusion_constraints; ///< Inclusion multiplier constraints.
  /// @}

  /**
   * Assembled sparse operators and mapping.
   */
  /// @{
  LA::MPI::SparseMatrix coupling_matrix;
  LA::MPI::SparseMatrix inclusion_matrix; ///< Multiplier mass/scaling matrix.
  LA::MPI::SparseMatrix mass_matrix;      ///< Bulk mass matrix.
  MappingQ<spacedim>    mapping; ///< Geometry mapping for coupling points.
  /// @}
#ifdef MATRIX_FREE_PATH
  /**
   * Vector and operator types for matrix-free mode.
   */
  /// @{
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;
  std::unique_ptr<CouplingOperator<spacedim, double, 1>> coupling_operator;
  MatrixFreeOperators::LaplaceOperator<spacedim, -1>     stiffness_matrix;
  using LevelMatrixType = MatrixFreeOperators::LaplaceOperator<
    spacedim,
    -1,
    -1,
    1,
    LinearAlgebra::distributed::Vector<float>>;
  MGLevelObject<LevelMatrixType> mg_matrices;
  MGConstrainedDoFs              mg_constrained_dofs;
  /// @}
#else
  /**
   * Matrix/vector types for assembled sparse mode.
   */
  /// @{
  LA::MPI::SparseMatrix stiffness_matrix;
  using VectorType      = LA::MPI::Vector;      ///< Monolithic vector type.
  using BlockVectorType = LA::MPI::BlockVector; ///< Two-block vector type.
                                                /// @}
#endif

  /**
   * Solution and right-hand side vectors.
   */
  /// @{
  BlockVectorType solution;
  BlockVectorType locally_relevant_solution; ///< Ghosted solution copy.
  BlockVectorType system_rhs;                ///< Coupled rhs vector.
  /// @}
  /**
   * Process-local coverings of the background mesh for particle insertion.
   */
  std::vector<std::vector<BoundingBox<spacedim>>> global_bounding_boxes;
  /**
   * Current adaptive-refinement cycle index.
   */
  unsigned int cycle = 0;
};


#endif
