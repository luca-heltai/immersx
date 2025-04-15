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

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include "inclusions.h"


#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif
#include <deal.II/base/hdf5.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>


template <int dim, int spacedim = dim>
class ElasticityProblemParameters : public ParameterAcceptor
{
public:
  ElasticityProblemParameters();

  std::string                   output_directory   = ".";
  std::string                   output_name        = "solution";
  unsigned int                  fe_degree          = 1;
  unsigned int                  initial_refinement = 5;
  std::list<types::boundary_id> dirichlet_ids{0};
  std::list<types::boundary_id> neumann_ids{};
  std::set<types::boundary_id>  normal_flux_ids{};
  std::string                   domain_type         = "generate";
  std::string                   name_of_grid        = "hyper_cube";
  std::string                   arguments_for_grid  = "-1: 1: false";
  std::string                   refinement_strategy = "fixed_fraction";
  double                        coarsening_fraction = 0.0;
  double                        refinement_fraction = 0.3;
  unsigned int                  n_refinement_cycles = 1;
  unsigned int                  max_cells           = 20000;
  bool                          output_pressure     = false;

  double Lame_mu     = 1;
  double Lame_lambda = 1;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    exact_solution;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> bc;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    Neumann_bc;

  std::string weight_expression = "1.";

  mutable ParameterAcceptorProxy<ReductionControl> inner_control;
  mutable ParameterAcceptorProxy<ReductionControl> outer_control;

  bool output_results_before_solving = false;

  mutable ParsedConvergenceTable convergence_table;

  // Time dependency.
  double initial_time = 0.0;
  double final_time   = 0.0;
  double dt           = 5e-3;
};



template <int dim, int spacedim>
ElasticityProblemParameters<dim, spacedim>::ElasticityProblemParameters()
  : ParameterAcceptor("/Immersed Problem/")
  , rhs("/Immersed Problem/Right hand side", spacedim)
  , exact_solution("/Immersed Problem/Exact solution", spacedim)
  , bc("/Immersed Problem/Dirichlet boundary conditions", spacedim)
  , Neumann_bc("/Immersed Problem/Neumann boundary conditions", spacedim)
  , inner_control("/Immersed Problem/Solver/Inner control")
  , outer_control("/Immersed Problem/Solver/Outer control")
  , convergence_table(std::vector<std::string>(spacedim, "u"))
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output name", output_name);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Dirichlet boundary ids", dirichlet_ids);
  add_parameter("Neumann boundary ids", neumann_ids);
  add_parameter("Normal flux boundary ids", normal_flux_ids);
  add_parameter("Output pressure", output_pressure);
  enter_subsection("Grid generation");
  {
    add_parameter("Domain type",
                  domain_type,
                  "",
                  this->prm,
                  Patterns::Selection("generate|file|cheese|cylinder"));
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
  enter_subsection("Physical constants");
  {
    add_parameter("Lame mu", Lame_mu);
    add_parameter("Lame lambda", Lame_lambda);
  }
  leave_subsection();
  enter_subsection("Exact solution");
  {
    add_parameter("Weight expression", weight_expression);
  }
  leave_subsection();
  enter_subsection("Time dependency");
  {
    add_parameter("Initial time", initial_time);
    add_parameter("Final time", final_time);
    add_parameter("Time step", dt);
  }
  leave_subsection();

  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();

  auto reset_function = [this]() {
    this->prm.set("Function expression", (spacedim == 2 ? "0; 0" : "0; 0; 0"));
  };
  rhs.declare_parameters_call_back.connect(reset_function);
  exact_solution.declare_parameters_call_back.connect(reset_function);
  Neumann_bc.declare_parameters_call_back.connect(reset_function);
  bc.declare_parameters_call_back.connect(reset_function);
}



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
  assemble_coupling();
  void
  run();
  void
  check_boundary_ids();

  /**
   * Builds coupling sparsity, and returns locally relevant inclusion dofs.
   */
  IndexSet
  assemble_coupling_sparsity(DynamicSparsityPattern &dsp);

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

  void
  compute_internal_and_boundary_stress(
    bool openfilefirsttime) const; // make const

  void
  output_pressure(bool openfilefirsttime) const;

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

  LA::MPI::SparseMatrix                           stiffness_matrix;
  LA::MPI::SparseMatrix                           coupling_matrix;
  LA::MPI::SparseMatrix                           inclusion_matrix;
  LA::MPI::BlockVector                            solution;
  LA::MPI::BlockVector                            locally_relevant_solution;
  LA::MPI::BlockVector                            system_rhs;
  std::vector<std::vector<BoundingBox<spacedim>>> global_bounding_boxes;
  unsigned int                                    cycle = 0;

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
