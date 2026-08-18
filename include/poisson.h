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

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

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

#include "linear_algebra.h"

namespace LA
{
  using namespace ImmersXLA;
} // namespace LA

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>


template <int dim, int spacedim = dim>
/**
 * Parameters for a scalar Poisson or Laplace--Beltrami problem.
 *
 * The equation is
 * \f$-\Delta_\Omega u=f\f$ on a \f$dim\f$-dimensional mesh embedded in
 * \f$\mathbb{R}^{spacedim}\f$. For \f$dim=spacedim\f$ this is the usual
 * Poisson equation; for \f$dim<spacedim\f$ the same finite-element assembly
 * represents the Laplace--Beltrami operator induced by the mesh geometry.
 * Functions are evaluated in the embedding space, so right-hand sides and
 * Dirichlet data use \f$spacedim\f$ coordinates.
 */
class PoissonParameters : public dealii::ParameterAcceptor
{
public:
  /**
   * Construct a parameter object below @p subsection.
   *
   * The default keeps the standalone application layout (`/Poisson/`).  A
   * caller embedding more than one Poisson problem can pass another section,
   * for example `/Bulk Poisson/` and `/Surface Poisson/`.
   */
  explicit PoissonParameters(const std::string &subsection = "/Poisson/");

  std::string                           output_directory   = ".";
  std::string                           output_name        = "solution";
  unsigned int                          fe_degree          = 1;
  unsigned int                          initial_refinement = 5;
  std::list<dealii::types::boundary_id> dirichlet_ids{0};
  std::string                           name_of_grid       = "hyper_cube";
  std::string                           arguments_for_grid = "-1: 1: false";
  std::string                           triangulation_type = "distributed";

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
 * Distributed finite-element solver for a scalar Poisson problem.
 *
 * The solver follows the standard deal.II step-6 lifecycle:
 * `make_grid()`, `setup_fe()`, `setup_system()`, `assemble_system()`,
 * `solve()`, `output_results()`, and optional `refine_grid()` are orchestrated
 * by `run()`. It owns one scalar finite-element space, one assembled stiffness
 * matrix, and one distributed solution vector.
 *
 * The template parameters describe the mesh dimension and its embedding:
 * `PoissonSolver<1>`, `PoissonSolver<1, 2>`, `PoissonSolver<1, 3>`,
 * `PoissonSolver<2>`, `PoissonSolver<2, 3>`, and `PoissonSolver<3>` are the
 * supported combinations. Embedded meshes use the geometric Jacobians
 * supplied by deal.II, so the same class also provides a Laplace--Beltrami
 * discretization.
 *
 * The triangulation backend is selected by
 * `PoissonParameters::triangulation_type`. The usual distributed backend is
 * used by default for dimensions 2 and 3. The fully distributed backend can
 * be selected for any supported pair and is forced for `dim == 1`, where
 * deal.II does not provide a distributed 1D triangulation. Fully distributed
 * meshes are constructed from a serial mesh and are not currently refined
 * after construction; use one initial mesh with that backend.
 */
class PoissonSolver : public dealii::EnableObserverPointer
{
public:
  using VectorType = LA::MPI::Vector;

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

  /** Return the global number of degrees of freedom. */
  dealii::types::global_dof_index
  n_dofs() const;

  /** Return the global L2 norm of the most recently computed solution. */
  double
  solution_l2_norm() const;

  /** Return whether the most recently computed solution has a finite norm. */
  bool
  solution_is_finite() const;

  /** Return the triangulation used by this problem. */
  const dealii::parallel::TriangulationBase<dim, spacedim> &
  triangulation() const;

  /** Return the finite-element DoFHandler used by this problem. */
  const dealii::DoFHandler<dim, spacedim> &
  dof_handler() const;

  /** Return the homogeneous and inhomogeneous algebraic constraints. */
  const dealii::AffineConstraints<double> &
  constraints() const;

  /** Return the assembled Poisson operator without mutable access. */
  const LA::MPI::SparseMatrix &
  system_matrix() const;

  /** Return the assembled right-hand side without mutable access. */
  const VectorType &
  system_rhs() const;

  /** Return the locally owned algebraic degrees of freedom. */
  const dealii::IndexSet &
  locally_owned_dofs() const;

  /** Return the locally relevant algebraic degrees of freedom. */
  const dealii::IndexSet &
  locally_relevant_dofs() const;

  /** Return the current locally owned solution vector. */
  const VectorType &
  solution() const;

  /**
   * Replace the algebraic state computed by an external solver.
   *
   * Constraints are distributed and the ghosted state used by output and
   * error routines is refreshed, so a coupled algebraic solver can hand its
   * result back to an otherwise independent Poisson problem.
   */
  void
  set_solution(const VectorType &new_solution);

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
  update_locally_relevant_solution();

  const PoissonParameters<dim, spacedim> &par;

  MPI_Comm                    mpi_communicator;
  dealii::ConditionalOStream  pcout;
  mutable dealii::TimerOutput computing_timer;

  TriangulationVariant                                  triangulation_storage;
  dealii::parallel::TriangulationBase<dim, spacedim>   *tria;
  std::unique_ptr<dealii::FiniteElement<dim, spacedim>> fe;
  std::unique_ptr<dealii::Quadrature<dim>>              quadrature;
  dealii::DoFHandler<dim, spacedim>                     dh;

  dealii::IndexSet                  owned_dofs;
  dealii::IndexSet                  relevant_dofs;
  dealii::AffineConstraints<double> constraints_storage;

  LA::MPI::SparseMatrix stiffness_matrix;
  VectorType            solution_storage;
  VectorType            system_rhs_storage;
  VectorType            locally_relevant_solution;

  mutable std::vector<std::pair<double, std::string>> cycles_and_solutions;
  unsigned int                                        cycle = 0;
};

#endif // immersx_poisson_h
