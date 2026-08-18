// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_poisson_h
#define immersx_poisson_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#define FORCE_USE_OF_TRILINOS
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define IMMERSX_POISSON_USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>


template <int dim, int spacedim = dim>
/**
 * Parameters for the standalone background Poisson problem.
 *
 * The class deliberately contains only data needed for
 * \f$-\Delta u=f\f$ on the background domain.
 */
class PoissonParameters : public dealii::ParameterAcceptor
{
public:
  PoissonParameters();

  std::string                           output_directory   = ".";
  std::string                           output_name        = "solution";
  unsigned int                          fe_degree          = 1;
  unsigned int                          initial_refinement = 5;
  std::list<dealii::types::boundary_id> dirichlet_ids{0};
  std::string                           name_of_grid       = "hyper_cube";
  std::string                           arguments_for_grid = "-1: 1: false";

  std::string  refinement_strategy = "fixed_fraction";
  double       coarsening_fraction = 0.0;
  double       refinement_fraction = 0.3;
  unsigned int n_refinement_cycles = 1;
  unsigned int max_cells           = 20000;

  mutable dealii::ParameterAcceptorProxy<
    dealii::Functions::ParsedFunction<spacedim>>
    rhs;
  mutable dealii::ParameterAcceptorProxy<
    dealii::Functions::ParsedFunction<spacedim>>
    bc;

  mutable dealii::ParameterAcceptorProxy<dealii::ReductionControl>
    solver_control;

  bool output_results_before_solving = false;
  bool estimate_condition_number     = false;

  mutable dealii::ParsedConvergenceTable convergence_table;
};


template <int dim, int spacedim = dim>
/**
 * Standalone distributed finite-element solver for \f$-\Delta u=f\f$.
 *
 * This class is the standalone background Poisson baseline: it owns one scalar
 * finite-element space, one assembled stiffness matrix, and one distributed
 * solution.
 */
class PoissonSolver : public dealii::EnableObserverPointer
{
public:
  explicit PoissonSolver(const PoissonParameters<dim, spacedim> &par);

  void
  make_grid();

  void
  setup_fe();

  void
  setup_system();

  void
  assemble_system();

  void
  solve();

  void
  output_results() const;

  void
  refine_grid();

  void
  run();

  /** Return the global number of background degrees of freedom. */
  dealii::types::global_dof_index
  n_dofs() const;

  /** Return the global L2 norm of the most recently computed solution. */
  double
  solution_l2_norm() const;

  /** Return whether the most recently computed solution has a finite norm. */
  bool
  solution_is_finite() const;

private:
  void
  update_locally_relevant_solution();

  const PoissonParameters<dim, spacedim> &par;

  MPI_Comm                    mpi_communicator;
  dealii::ConditionalOStream  pcout;
  mutable dealii::TimerOutput computing_timer;

  dealii::parallel::distributed::Triangulation<spacedim> tria;
  std::unique_ptr<dealii::FiniteElement<spacedim>>       fe;
  std::unique_ptr<dealii::Quadrature<spacedim>>          quadrature;
  dealii::DoFHandler<spacedim>                           dh;

  dealii::IndexSet                  owned_dofs;
  dealii::IndexSet                  relevant_dofs;
  dealii::AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix stiffness_matrix;
  using VectorType = LA::MPI::Vector;
  VectorType solution;
  VectorType system_rhs;
  VectorType locally_relevant_solution;

  mutable std::vector<std::pair<double, std::string>> cycles_and_solutions;
  unsigned int                                        cycle = 0;
};

#endif // immersx_poisson_h
