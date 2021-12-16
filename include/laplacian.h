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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>

#include "reference_inclusion.h"


#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>


template <int dim, int spacedim = dim>
class ProblemParameters : public ParameterAcceptor
{
public:
  ProblemParameters();

  std::string                   output_directory       = ".";
  unsigned int                  fe_degree              = 1;
  unsigned int                  initial_refinement     = 5;
  unsigned int                  rtree_extraction_level = 1;
  std::list<types::boundary_id> dirichlet_ids{0};
  std::string                   name_of_grid        = "hyper_cube";
  std::string                   arguments_for_grid  = "-1: 1: false";
  std::string                   refinement_strategy = "fixed_fraction";
  double                        coarsening_fraction = 0.3;
  double                        refinement_fraction = 0.3;
  unsigned int                  n_refinement_cycles = 1;
  unsigned int                  max_cells           = 20000;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs;
  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> bc;

  mutable ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    inclusions_rhs;

  /**
   * Each inclusion has: cx, cy, R
   */
  std::vector<std::vector<double>> inclusions = {{-.2, -.2, .3}};

  /**
   * $\alpha_1$ is the Dirichlet coefficient.
   */
  double alpha1 = 1.0;

  /**
   * $\alpha_2$ is the Neumann coefficient.
   */
  double alpha2 = 0.0;

  unsigned int inclusions_refinement  = 1000;
  unsigned int n_fourier_coefficients = 1;

  mutable ParameterAcceptorProxy<ReductionControl> inner_control;
  mutable ParameterAcceptorProxy<ReductionControl> outer_control;

  bool output_results_before_solving = false;

  mutable ParsedConvergenceTable convergence_table;
};



template <int dim, int spacedim>
ProblemParameters<dim, spacedim>::ProblemParameters()
  : ParameterAcceptor("/Immersed Problem/")
  , rhs("/Immersed Problem/Right hand side")
  , bc("/Immersed Problem/Dirichlet boundary conditions")
  , inclusions_rhs("/Immersed Problem/Immersed inclusions/Boundary data")
  , inner_control("/Immersed Problem/Solver/Inner control")
  , outer_control("/Immersed Problem/Solver/Outer control")
{
  add_parameter("FE degree", fe_degree, "", this->prm, Patterns::Integer(1));
  add_parameter("Output directory", output_directory);
  add_parameter("Output results also before solving",
                output_results_before_solving);
  add_parameter("Initial refinement", initial_refinement);
  add_parameter("Bounding boxes extraction level", rtree_extraction_level);
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
  enter_subsection("Immersed inclusions");
  {
    add_parameter("alpha1 (Dirichlet factor)", alpha1);
    add_parameter("alpha2 (Neumann factor)", alpha2);
    add_parameter("Inclusions", inclusions);
    add_parameter("Inclusions refinement", inclusions_refinement);
    add_parameter("Number of fourier coefficients", n_fourier_coefficients);
  }
  leave_subsection();
  this->prm.enter_subsection("Error");
  convergence_table.add_parameters(this->prm);
  this->prm.leave_subsection();
}



template <int dim, int spacedim = dim>
class PoissonProblem : public Subscriptor
{
public:
  PoissonProblem(const ProblemParameters<dim, spacedim> &par);
  void
  make_grid();
  void
  setup_inclusions_particles();
  void
  setup_fe();
  void
  setup_dofs();
  void
  assemble_poisson_system();
  void
  assemble_coupling();
  void
  run();

  /**
   * Builds coupling sparsity, and returns locally relevant inclusion dofs.
   */
  IndexSet
  assemble_coupling_sparsity(DynamicSparsityPattern &dsp) const;

  void
  solve();

  void
  refine_and_transfer();

  std::string
  output_solution() const;

  std::string
  output_particles() const;

  void
  output_results() const;

  void
  print_parameters() const;

  types::global_dof_index
  n_inclusions_dofs() const;

private:
  const ProblemParameters<dim, spacedim> &       par;
  MPI_Comm                                       mpi_communicator;
  ConditionalOStream                             pcout;
  mutable TimerOutput                            computing_timer;
  parallel::distributed::Triangulation<spacedim> tria;
  std::unique_ptr<FiniteElement<spacedim>>       fe;
  std::unique_ptr<ReferenceInclusion<spacedim>>  inclusion;
  std::unique_ptr<Quadrature<spacedim>>          quadrature;
  DoFHandler<spacedim>                           dh;
  std::vector<IndexSet>                          owned_dofs;
  std::vector<IndexSet>                          relevant_dofs;

  Particles::ParticleHandler<spacedim> inclusions_as_particles;

  AffineConstraints<double> constraints;
  AffineConstraints<double> inclusion_constraints;

  LA::MPI::SparseMatrix                           stiffness_matrix;
  LA::MPI::SparseMatrix                           coupling_matrix;
  LA::MPI::SparseMatrix                           inclusion_matrix;
  LA::MPI::BlockVector                            solution;
  LA::MPI::BlockVector                            locally_relevant_solution;
  LA::MPI::BlockVector                            system_rhs;
  std::vector<std::vector<BoundingBox<spacedim>>> global_bounding_boxes;
  unsigned int                                    cycle = 0;
};


#endif