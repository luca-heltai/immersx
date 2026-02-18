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
#ifndef dealii_distributed_lagrange_multiplier_elasticity_h
#define dealii_distributed_lagrange_multiplier_elasticity_h

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
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
#include <deal.II/base/work_stream.h>

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
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

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
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

// #include <deal.II/trilinos/parameter_acceptor.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/simple.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/matrix_tools.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include "inclusions.h"


#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif
#include <deal.II/base/hdf5.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>

#include "elasticity_problem_parameters.h"
#include "material_properties.h"

/**
 * Definition of the Rigid body motions for linear elasticity.
 */
template <int dim>
class RigidBodyMotion : public Function<dim>
{
public:
  RigidBodyMotion(const unsigned int type_);

  virtual double
  value(const Point<dim> &p, const unsigned int component) const override;

private:
  const unsigned int type;
};

/**
 * Finite element model for (optionally) time-dependent elasticity, coupled to
 * immersed 1D/0D inclusions via reduced Lagrange multipliers.
 *
 * This class assembles a small set of global sparse matrices and then solves
 * either:
 * - a pure displacement problem (no inclusions), or
 * - a saddle-point problem for displacement and Lagrange multipliers, or
 * - a displacement problem driven by inclusion "pressure" data (when
 *   `ElasticityProblemParameters::pressure_coupling` is enabled).
 *
 * The intent of this documentation is to state *exactly* which continuous and
 * discrete equations correspond to the current implementation in
 * `source/elasticity.cc` (primarily `assemble_elasticity_system()`,
 * `assemble_coupling()`, and `solve()`), and to list the constants and options
 * that select between different model variants.
 *
 * @note The bulk constitutive model is selected via
 * `ElasticityProblemParameters::elasticity_model` (inferred from material
 * parameters). Rayleigh damping is determined by the per-material parameters
 * `MaterialProperties::{rayleigh_alpha,rayleigh_beta}`.
 *
 * @tparam dim Topological dimension of the mesh (typically equal to
 * `spacedim`).
 * @tparam spacedim Embedding space dimension. The displacement has
 * `spacedim` components.
 *
 * @section elasticity_unknowns Unknowns, discrete spaces, and blocks
 *
 * Let \f$\Omega \subset \mathbb{R}^{spacedim}\f$ be the computational domain.
 *
 * - Primary unknown: displacement \f$u:\Omega\to\mathbb{R}^{spacedim}\f$.
 * - Optional secondary unknown: reduced Lagrange multipliers
 *   \f$\lambda \in \mathbb{R}^{N_\lambda}\f$ attached to immersed inclusions
 *   (`Inclusions<spacedim>`).
 *
 * In the code, the global solution is stored as a 2-block vector:
 * - `solution.block(0)` \f$\equiv u_h\f$,
 * - `solution.block(1)` \f$\equiv \lambda_h\f$ (may be size 0 if no
 * inclusions).
 *
 * The displacement is discretized with a continuous \f$Q_p\f$ vector element:
 * `FESystem<spacedim>(FE_Q<spacedim>(fe_degree), spacedim)`.
 *
 * @section elasticity_operators Assembled matrices and right-hand sides
 *
 * The following sparse matrices are assembled explicitly:
 *
 * - `stiffness_matrix` \f$\equiv A\f$
 * - `mass_matrix` \f$\equiv C\f$
 * - `damping_matrix` \f$\equiv D\f$
 * - `coupling_matrix` \f$\equiv B^T\f$ (maps \f$\lambda\f$ to displacement
 * dofs)
 * - `inclusion_matrix` \f$\equiv M\f$ (multiplier mass / scaling matrix)
 *
 * Right-hand side blocks are:
 * - `system_rhs.block(0)` \f$\equiv f\f$ (volumetric + boundary loads and
 *   penalty terms),
 * - `system_rhs.block(1)` \f$\equiv g\f$ (inclusion data projected onto the
 *   multiplier basis),
 * - `system_rhs_f.block(0)` \f$\equiv f_f\f$ (computed in `solve()` as
 *   \f$f_f=B^T\lambda\f$ in some branches; it is not assembled in
 *   `assemble_elasticity_system()`).
 *
 * @subsection elasticity_constants Constants and where they come from
 *
 * Material constants are taken per-cell from `MaterialProperties` (see
 * `include/material_properties.h`):
 * - \f$\mu\f$  = `MaterialProperties::Lame_mu`
 * - \f$\lambda\f$ = `MaterialProperties::Lame_lambda`
 * - \f$\rho\f$ = `MaterialProperties::rho`
 * - \f$\eta\f$ = `MaterialProperties::neta` (Kelvin–Voigt viscosity)
 * - \f$\alpha\f$ = `MaterialProperties::rayleigh_alpha`
 * - \f$\beta\f$  = `MaterialProperties::rayleigh_beta`
 *
 * Model/solver constants are taken from `ElasticityProblemParameters` (see
 * `include/elasticity_problem_parameters.h`):
 * - penalty coefficient \f$\gamma_p\f$ =
 * `ElasticityProblemParameters::penalty_term`
 * - "wave" amplitude \f$a_w\f$ = `ElasticityProblemParameters::wave_ampltiude`
 * - time step \f$\Delta t\f$ = `ElasticityProblemParameters::dt`
 * - Newmark parameters \f$\beta_N\f$ and \f$\gamma_N\f$ =
 *   `ElasticityProblemParameters::beta` and
 * `ElasticityProblemParameters::gamma`
 *
 * A single length scale \f$h\f$ used in the penalty term is computed as
 * `GridTools::minimal_cell_diameter(tria)` in `assemble_elasticity_system()`.
 *
 * @section elasticity_bulk Bulk constitutive forms (what goes into A, C, D)
 *
 * Define the symmetric gradient \f$\varepsilon(u)=\frac12(\nabla u+\nabla
 * u^T)\f$ and divergence \f$\nabla\cdot u\f$.
 *
 * Let \f$\{\varphi_i\}\f$ be the displacement FE basis functions
 * (vector-valued), and let \f$\langle \cdot, \cdot \rangle\f$ denote the
 * Euclidean inner product on vectors/tensors. The entries assembled in
 * `assemble_elasticity_system()` are:
 *
 * Mass matrix (always assembled):
 * \f[
 *   C_{ij} = \int_\Omega \rho\, \varphi_i \cdot \varphi_j\,dx.
 * \f]
 *
 * Stiffness matrix contributions (assembled depending on flags):
 *
 * - If `ElasticityProblemParameters::elasticity_model ==
 * ElasticityModel::LinearElasticity`: \f[ A_{ij} \mathrel{+}= \int_\Omega
 * \Big(2\mu\,\varepsilon(\varphi_i):\varepsilon(\varphi_j)
 *   + \lambda\,(\nabla\cdot \varphi_i)(\nabla\cdot \varphi_j)\Big)\,dx.
 * \f]
 *
 * - If `ElasticityProblemParameters::elasticity_model ==
 * ElasticityModel::KelvinVoigt` (note: the current implementation uses only the
 * \f$\mu\f$ term for stiffness, without a \f$\lambda\f$ contribution): \f[
 *   A_{ij} \mathrel{+}= \int_\Omega
 * \mu\,\varepsilon(\varphi_i):\varepsilon(\varphi_j)\,dx. \f]
 *
 * Damping matrix contributions:
 *
 * - If `ElasticityProblemParameters::elasticity_model ==
 * ElasticityModel::KelvinVoigt`: \f[ D_{ij} \mathrel{+}= \int_\Omega
 * \eta\,\varepsilon(\varphi_i):\varepsilon(\varphi_j)\,dx. \f]
 *
 * - If any material has nonzero Rayleigh parameters, the code builds
 * (intended) Rayleigh damping of the form \f[ D \approx \beta\,A + \alpha\,C,
 * \f]
 * with \f$\alpha=\f$ `MaterialProperties::rayleigh_alpha` and
 * \f$\beta=\f$ `MaterialProperties::rayleigh_beta`.
 *
 * Body force right-hand side (always assembled):
 * \f[
 *   f_i \mathrel{+}= \int_\Omega \varphi_i \cdot f_{\text{rhs}}(x)\,dx,
 * \f]
 * where `f_rhs` comes from `ElasticityProblemParameters::rhs`.
 *
 * @section elasticity_bc Boundary conditions (as implemented)
 *
 * Strong Dirichlet boundary conditions:
 * - On boundary ids in `ElasticityProblemParameters::dirichlet_ids`, the code
 *   applies \f$u = u_D\f$ strongly via
 *   `VectorTools::interpolate_boundary_values(..., par.bc, constraints)`.
 *
 * Normal-flux constraints:
 * - On boundary ids in `ElasticityProblemParameters::normal_flux_ids`, the code
 *   adds constraints via
 *   `VectorTools::compute_nonzero_normal_flux_constraints(...)` using
 *   `ElasticityProblemParameters::Neumann_bc`. (This constrains the net normal
 *   flux; see the deal.II function documentation for the exact constraint
 *   meaning.)
 *
 * Neumann boundary ids:
 * - On boundary ids in `ElasticityProblemParameters::neumann_ids`, the code
 *   adds a scalar traction-like load computed as
 *   \f$t_n(x)=\frac{1}{spacedim}\,(\texttt{Neumann\_bc}(x)\cdot n(x))\f$ and
 *   assembled as
 * \f[
 *   f_i \mathrel{+}= -\int_{\Gamma_N} t_n\, \varphi_i\,ds.
 * \f]
 *
 * Weak Dirichlet ids (penalty term):
 * - For cells that have at least one boundary face with id in
 *   `ElasticityProblemParameters::weak_dirichlet_ids`, the code adds a penalty
 *   term using \f$h = \f$ `GridTools::minimal_cell_diameter(tria)`:
 * \f[
 *   A_{ij} \mathrel{+}= \frac{\gamma_p}{h}\int_{\Omega_{\text{marked}}}
 *   \varphi_i\cdot\varphi_j\,dx,\qquad
 *   f_i \mathrel{+}=
 * \frac{\gamma_p}{h}\,a_w\int_{\Omega_{\text{marked}}}\varphi_i\,dx. \f] with
 * \f$\gamma_p=\f$ `penalty_term` and \f$a_w=\f$ `wave_ampltiude`.
 *
 * @section elasticity_inclusions Inclusion coupling (B, M, g) and equation
 * types
 *
 * The immersed inclusions define a reduced multiplier space with basis values
 * (Fourier/harmonic modes) evaluated at quadrature points on each inclusion.
 * In the code, `assemble_coupling()` builds:
 *
 * - Coupling matrix \f$B^T\f$ (`coupling_matrix`, size \f$N_u\times
 * N_\lambda\f$),
 * - Inclusion matrix \f$M\f$ (`inclusion_matrix`, size \f$N_\lambda\times
 * N_\lambda\f$, assembled *diagonally*),
 * - Multiplier right-hand side \f$g\f$ (`system_rhs.block(1)`).
 *
 * Conceptually, the saddle-point branch (`pressure_coupling == false`) solves
 * the discrete constrained system
 * \f[
 * \begin{bmatrix}
 *   A & B^T \\
 *   B & 0
 * \end{bmatrix}
 * \begin{bmatrix}
 *   u \\
 *   \lambda
 * \end{bmatrix}
 * =
 * \begin{bmatrix}
 *   f \\
 *   g
 * \end{bmatrix}.
 * \f]
 *
 * The `pressure_coupling == true` branch does *not* solve the saddle-point
 * problem. Instead it computes
 * \f$\lambda \approx M^{-1}g\f$ and then forms an additional load
 * \f$f_f = B^T\lambda\f$ (stored in `system_rhs_f.block(0)`).
 *
 * Multiplier right-hand side \f$g\f$ (as assembled in `assemble_coupling()`) is
 * a normalized quadrature projection of the inclusion boundary data onto the
 * selected Fourier coefficients. In particular, for each inclusion and each
 * selected mode index \f$j\f$, the code adds terms of the form
 * \f[
 *   g_j \mathrel{+}= \frac{1}{|\Gamma|}\sum_q \psi_j(q)\,d_j(q)\,ds_q,
 * \f]
 * with \f$|\Gamma|=\f$ `section_measure`, basis value
 * \f$\psi_j(q)=\f$ `inclusion_fe_values[j]`, and \f$d_j(q)\f$ taken either from
 * `Inclusions::inclusions_rhs` or from `Inclusions::inclusions_data` (see
 * `source/elasticity.cc`, `assemble_coupling()`).
 *
 * @note In the stationary case (`initial_time == final_time`) with
 * `pressure_coupling == true`, the current implementation computes
 * \f$f_f=B^T\lambda\f$ but then solves \f$A u = f\f$ (i.e., \f$f_f\f$ is not
 * used in the stationary solve path).
 *
 * @section elasticity_time Time integration (Newmark predictor/corrector)
 *
 * If `ElasticityProblemParameters::initial_time != final_time`, the solver uses
 * Newmark time integration for the (semi-discrete) second-order system
 * \f[
 *   C\,\ddot u + D\,\dot u + A\,u = \big(f + f_f\big)\,\sin(t),
 * \f]
 * where \f$f_f=B^T\lambda\f$ is present only in the `pressure_coupling == true`
 * branch (as implemented in `solve()`).
 *
 * Given \f$u^n, v^n=\dot u^n, a^n=\ddot u^n\f$, the code computes:
 * Predictor:
 * \f[
 *   u^{pred} = u^n + \Delta t\,v^n + \frac{\Delta t^2}{2}(1-2\beta_N)a^n,\qquad
 *   v^{pred} = v^n + \Delta t(1-\gamma_N)a^n.
 * \f]
 *
 * Effective (acceleration) solve:
 * - Linear elasticity:
 *   \f[
 *     \big(C + \beta_N\Delta t^2 A\big)a^{n+1}
 *     = (f+f_f)\sin(t^n) - A\,u^{pred}.
 *   \f]
 * - With damping (Rayleigh and/or Kelvin–Voigt):
 *   \f[
 *     \big(C + \beta_N\Delta t^2 A + \gamma_N\Delta t\,D\big)a^{n+1}
 *     = (f+f_f)\sin(t^n) - D\,v^{pred} - A\,u^{pred}.
 *   \f]
 * Corrector:
 * \f[
 *   u^{n+1} = u^{pred} + \beta_N\Delta t^2 a^{n+1},\qquad
 *   v^{n+1} = v^{pred} + \gamma_N\Delta t\, a^{n+1}.
 * \f]
 *
 * @section elasticity_predictor_corrector_matrices Predictor/corrector matrices
 *
 * For matrix-based solvers/preconditioners that work on assembled matrices, the
 * key "effective" (acceleration) matrix used in the Newmark solve is:
 * - \f$A_n = C + \beta_N\Delta t^2 A\f$ (undamped)
 * - \f$A_n = C + \beta_N\Delta t^2 A + \gamma_N\Delta t\,D\f$ (damped)
 *
 * In the current implementation these combinations are built as
 * `LinearOperator` sums in `solve()` (not as explicit sparse matrices).
 */
template <int dim, int spacedim = dim>
class ElasticityProblem : public Subscriptor
{
public:
  ElasticityProblem(const ElasticityProblemParameters<dim, spacedim> &par);

  void
  make_grid();
  void
  setup_fe();
  void
  setup_dofs();
  void
  assemble_elasticity_system();
  void
  assemble_forcing_terms();
  void
  compute_system_rhs();
  void
  assemble_coupling();
  void
  run();

  /**
   * Internal run implementations selected by `time_mode`.
   *
   * - `run_static()`: stationary solve with optional refinement cycles.
   * - `run_quasistatic()`: time-dependent run without inertia
   *   (currently `solve_quasistatic()` is not implemented).
   * - `run_newmark()`: time-dependent run with inertia (Newmark).
   */
  void
  run_static();

  void
  run_quasistatic();

  void
  run_newmark();

  /**
   * Builds coupling sparsity, and returns locally relevant inclusion dofs.
   */
  IndexSet
  assemble_coupling_sparsity(DynamicSparsityPattern &dsp);

  void
  solve();

  /**
   * Internal solver implementations selected by `time_mode`.
   */
  void
  solve_static();

  void
  solve_quasistatic();

  void
  solve_newmark();

  void
  refine_and_transfer();

  void
  refine_and_transfer_around_inclusions();

  void
  execute_actual_refine_and_transfer();

  std::string
  output_solution() const;

  void
  output_results() const;

  void
  print_parameters() const;

  void
  compute_internal_and_boundary_stress(bool) const;

  void
  output_lambda() const;

  std::string
  output_stresses() const;

  // void
  // compute_face_stress();

  // protected:
  const ElasticityProblemParameters<dim, spacedim> &par;
  MPI_Comm                                          mpi_communicator;
  ConditionalOStream                                pcout;
  mutable TimerOutput                               computing_timer;
  parallel::distributed::Triangulation<spacedim>    tria;
  std::unique_ptr<FiniteElement<spacedim>>          fe;
  Inclusions<spacedim>                              inclusions;
  std::unique_ptr<Quadrature<spacedim>>             quadrature;
  std::unique_ptr<Quadrature<spacedim - 1>>         face_quadrature_formula;
  DoFHandler<spacedim>                              dh;
  std::vector<IndexSet>                             owned_dofs;
  std::vector<IndexSet>                             relevant_dofs;

  AffineConstraints<double> constraints;
  AffineConstraints<double> inclusion_constraints;
  AffineConstraints<double> mean_value_constraints;

  LA::MPI::SparseMatrix stiffness_matrix;
  LA::MPI::SparseMatrix newmark_matrix;
  LA::MPI::SparseMatrix mass_matrix;
  LA::MPI::SparseMatrix coupling_matrix;
  LA::MPI::SparseMatrix damping_matrix;

  LA::MPI::PreconditionAMG prec_A;
  LA::MPI::PreconditionAMG prec_newmark;
  LA::MPI::PreconditionAMG prec_C;
  LA::MPI::PreconditionAMG prec_M;

  LA::MPI::SparseMatrix                           inclusion_matrix;
  LA::MPI::BlockVector                            solution;
  LA::MPI::BlockVector                            velocity;
  LA::MPI::BlockVector                            acceleration;
  LA::MPI::BlockVector                            predictor;
  LA::MPI::BlockVector                            corrector;
  LA::MPI::BlockVector                            locally_relevant_solution;
  LA::MPI::BlockVector                            force_rhs;
  LA::MPI::BlockVector                            bc_rhs;
  LA::MPI::BlockVector                            neumann_bc_rhs;
  LA::MPI::BlockVector                            system_rhs;
  std::vector<std::vector<BoundingBox<spacedim>>> global_bounding_boxes;
  unsigned int                                    cycle     = 0;
  unsigned int                                    time_step = 0;

  FEValuesExtractors::Vector displacement;

  // Postprocessing values
  std::map<types::boundary_id, Tensor<1, spacedim>> forces;
  std::map<types::boundary_id, Tensor<1, spacedim>> average_displacements;
  std::map<types::boundary_id, Tensor<1, spacedim>> average_normals;
  std::map<types::boundary_id, double>              areas;
  TrilinosWrappers::MPI::Vector                     sigma_n;
  // std::vector<BaseClass::BlockType>                 pressure_records;

  // Time dependency.
  double current_time = 0.0;

  class Postprocessor;
};

#endif
