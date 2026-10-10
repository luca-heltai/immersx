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
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

/* ---------------------------------------------------------------------
 */
#ifndef immersx_elasticity_freq_h
#define immersx_elasticity_freq_h

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <variant>
#define FORCE_USE_OF_TRILINOS
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

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

#ifdef Handle
#  pragma push_macro("Handle")
#  undef Handle
#  define IMMERSX_RESTORE_OPENCASCADE_HANDLE
#endif
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/simple.h>
#ifdef IMMERSX_RESTORE_OPENCASCADE_HANDLE
#  pragma pop_macro("Handle")
#  undef IMMERSX_RESTORE_OPENCASCADE_HANDLE
#endif

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/matrix_tools.h>

#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>



#ifdef DEAL_II_WITH_VTK
#  include <immersx/coupling/reduced_coupling.h>
#endif


#ifdef DEAL_II_WITH_OPENCASCADE
#  include <TopoDS.hxx>
#endif
#include <deal.II/base/hdf5.h>

#include <immersx/physics/material_properties.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "elasticity_freq_problem_parameters.h"


namespace ImmersX
{
  namespace LA
  {
    using namespace dealii::LinearAlgebraTrilinos;
  }
  template <int dim, int spacedim = dim>
  class ElasticityFreqProblem : public EnableObserverPointer
  {
  public:
    explicit ElasticityFreqProblem(
      const ElasticityFreqProblemParameters<dim, spacedim> &par);
    void
    run();
    void
    make_grid();
    void
    setup_fe();
    void
    setup_dofs();
    void
    setup_constraints();
    void
    refine_and_transfer();
    void
    execute_actual_refine_and_transfer();
    void
    print_parameters() const;
    void
    compute_internal_and_boundary_stress(bool openfilefirsttime) const;

  private:
    bool
    uses_fully_distributed_triangulation() const;
    void
    assemble_frequency_part(double                           omega,
                            bool                             imaginary,
                            bool                             build_matrix,
                            const AffineConstraints<double> &fc,
                            LA::MPI::SparseMatrix           &matrix,
                            LA::MPI::Vector                 &rhs);
    void
    output_frequency(const LA::MPI::Vector &real,
                     const LA::MPI::Vector &imag) const;
    const ElasticityFreqProblemParameters<dim, spacedim> &par;
    MPI_Comm                                              mpi_communicator;
    ConditionalOStream                                    pcout;
    mutable TimerOutput                                   computing_timer;
    using DistributedTriangulation =
      parallel::distributed::Triangulation<spacedim>;
    using FullyDistributedTriangulation =
      parallel::fullydistributed::Triangulation<spacedim>;
    using TriangulationVariant =
      std::variant<DistributedTriangulation, FullyDistributedTriangulation>;
    TriangulationVariant                      triangulation_storage;
    parallel::TriangulationBase<spacedim>    *tria;
    std::unique_ptr<FiniteElement<spacedim>>  fe;
    std::unique_ptr<Quadrature<spacedim>>     quadrature;
    std::unique_ptr<Quadrature<spacedim - 1>> face_quadrature_formula;
    DoFHandler<spacedim>                      dh;
    std::vector<IndexSet>                     owned_dofs, relevant_dofs;
    AffineConstraints<double>  constraints, imaginary_constraints;
    LA::MPI::BlockVector       solution, locally_relevant_solution;
    unsigned int               cycle = 0, time_step = 0;
    FEValuesExtractors::Vector displacement;
    double                     current_time = 0.0, current_frequency = 0.0;
    std::string                stress_part = "real";
  };
} // namespace ImmersX
#endif
