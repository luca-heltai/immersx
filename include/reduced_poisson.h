// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
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
#ifndef dealii_reduced_poisson_h
#define dealii_reduced_poisson_h

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

#include <deal.II/numerics/matrix_tools.h>

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

#ifdef DEAL_II_WITH_VTK

#  include "reduced_coupling.h"

#  ifdef DEAL_II_WITH_OPENCASCADE
#    include <TopoDS.hxx>
#  endif
#  include <cmath>
#  include <fstream>
#  include <iostream>
#  include <memory>


template <int spacedim>
class ReducedPoissonParameters : public ParameterAcceptor
{
public:
  ReducedPoissonParameters();

  std::string                   output_directory = ".";
  std::string                   output_name      = "solution";
  unsigned int                  fe_degree        = 1;
  std::list<types::boundary_id> dirichlet_ids{0};
  std::string                   name_of_grid        = "hyper_cube";
  std::string                   arguments_for_grid  = "-1: 1: false";
  std::string                   refinement_strategy = "fixed_fraction";
  double                        coarsening_fraction = 0.0;
  double                        refinement_fraction = 0.3;
  unsigned int                  n_refinement_cycles = 1;
  unsigned int                  max_cells           = 20000;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> bc;

  mutable ParameterAcceptorProxy<ReductionControl> inner_control;
  mutable ParameterAcceptorProxy<ReductionControl> outer_control;

  bool        output_results_before_solving = false;
  std::string solver_name                   = "AL";
  bool        assemble_full_AL_system       = false;

  mutable ParsedConvergenceTable convergence_table;

  ReducedCouplingParameters<1, 2, spacedim, 1> reduced_coupling_parameters;
};


/**
 * Solver for a Poisson problem in a bulk domain coupled to a reduced
 * Lagrange-multiplier space supported on an immersed tubular interface.
 *
 * Let $\Omega \subset \mathbb{R}^{\texttt{spacedim}}$ be the background domain
 * and let $\gamma$ be the reduced manifold read from
 * `reduced_coupling_parameters.tensor_product_space_parameters.reduced_grid_name`.
 * The class `ReducedCoupling<1,2,spacedim,1>` builds an interface
 * $\Gamma_\gamma$ by sweeping a one-dimensional reference cross section $D$
 * along $\gamma$. A scalar multiplier on $\Gamma_\gamma$ is approximated in the
 * reduced tensor-product form \f[ \lambda(s,y) \approx \sum_{i=1}^N
 * \lambda_i(s)\,\phi_i(y), \qquad (s,y)\in \gamma\times D, \f] where $\phi_i$
 * are the selected cross-section modes provided by `ReferenceCrossSection`, and
 * $\lambda_i$ are finite element unknowns on $\gamma$.
 *
 * The bulk unknown is the scalar field $u : \Omega \to \mathbb{R}$ solving \f[
 * -\Delta u = f \qquad \text{in } \Omega, \f] with Dirichlet data $u = u_D$ on
 * the boundary ids listed in `dirichlet_ids`. The immersed constraint is not
 * imposed pointwise on the full interface. Instead, the trace of $u$ on
 * $\Gamma_\gamma$ is required to match a prescribed reduced datum \f[ g(s,y) =
 * \sum_{i=1}^N \bar g_i(s)\,\phi_i(y), \f] where the coefficient functions
 * $\bar g_i$ are given by
 * `reduced_coupling_parameters.coupling_rhs_expressions`.
 *
 * The formulation implemented by this class is: find $(u,\lambda)\in V\times
 * Q_{\mathrm{red}}$ such that
 *
 * \f[ \int_\Omega \nabla u \cdot \nabla v \, dx
 * + \int_\gamma \int_D \lambda(s,y)\, v(X(s,y)) \, dy\, ds = \int_\Omega
 *   f\,v\,dx \qquad \forall v \in V_0, \f] and \f[ \int_\gamma \int_D
 *   u(X(s,y))\, \mu(s,y) \, dy\, ds = \int_\gamma \int_D
 *   g(s,y)\,\mu(s,y)\,dy\,ds \qquad \forall \mu \in Q_{\mathrm{red}}, \f]
 *
 * where $X(s,y)$ is the geometric lifting from the reduced manifold and the
 * reference cross section to the immersed interface $\Gamma_\gamma$.
 *
 * Because both $g$ and $\mu$ are expanded in the same transverse basis
 * $\{\phi_i\}_{i=1}^N$, the second equation is assembled in reduced coordinates
 * as \f[ \sum_{i=1}^N \int_\gamma |D_s|\, \bar g_i(s)\,\mu_i(s)\,ds, \f] with
 * $|D_s|$ the measure of the local cross section. This is what
 * `ReducedCoupling::assemble_reduced_rhs()` computes.
 *
 * After discretization, the class solves the saddle-point linear system
 *
 * \f[ \begin{bmatrix} A & B^T \\
 * B & 0 \end{bmatrix} \begin{bmatrix} u \\
 * \lambda \end{bmatrix}
 * =
 * \begin{bmatrix} f \\
 * g \end{bmatrix}, \f]
 *
 * where:
 * - $A$ is the bulk Poisson stiffness matrix assembled in
 *   `assemble_poisson_system()`;
 * - $B$ is the reduced trace/projection operator assembled in
 *   `ReducedCoupling::assemble_coupling_matrix()`;
 * - $g$ is the reduced right-hand side assembled in
 *   `ReducedCoupling::assemble_reduced_rhs()`.
 *
 * Therefore, `ReducedPoisson` computes a harmonic field in the bulk whose trace
 * on the immersed interface matches, in the reduced modal sense defined by
 * `TensorProductSpace`, the prescribed lower-dimensional data.
 */
template <int dim, int spacedim = dim>
class ReducedPoisson : public Subscriptor
{
public:
  ReducedPoisson(const ReducedPoissonParameters<spacedim> &par);
  void
  make_grid();
  void
  setup_fe();
  void
  setup_dofs();
#  ifndef MATRIX_FREE_PATH
  void
  assemble_poisson_system();
#  else
  void
                        assemble_rhs();
#  endif
  void
  run();

  void
  solve();

  void
  refine_and_transfer();

  std::string
  output_solution() const;

  void
  output_results() const;

  void
  print_parameters() const;

private:
  const ReducedPoissonParameters<spacedim>      &par;
  MPI_Comm                                       mpi_communicator;
  ConditionalOStream                             pcout;
  mutable TimerOutput                            computing_timer;
  parallel::distributed::Triangulation<spacedim> tria;
  std::unique_ptr<FiniteElement<spacedim>>       fe;

  std::unique_ptr<Quadrature<spacedim>> quadrature;

  DoFHandler<spacedim> dh;

  ReducedCoupling<1, 2, spacedim, 1> reduced_coupling;

  std::vector<IndexSet> owned_dofs;
  std::vector<IndexSet> relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix coupling_matrix;
  LA::MPI::SparseMatrix coupling_matrix_transpose;
  LA::MPI::SparseMatrix inclusion_matrix;
  MappingQ<spacedim>    mapping;
#  ifdef MATRIX_FREE_PATH
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
#  else
  LA::MPI::SparseMatrix stiffness_matrix;
  using VectorType      = LA::MPI::Vector;
  using BlockVectorType = LA::MPI::BlockVector;
#  endif

  BlockVectorType                                 solution;
  BlockVectorType                                 locally_relevant_solution;
  BlockVectorType                                 system_rhs;
  std::vector<std::vector<BoundingBox<spacedim>>> global_bounding_boxes;
  unsigned int                                    cycle = 0;
};


#endif

#endif // DEAL_II_WITH_VTK
