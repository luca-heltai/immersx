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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

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
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
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

#include <immersx/algebra/linear_algebra.h>
#include <immersx/compatibility/reference_cell.h>
#include <immersx/core/domain.h>

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
#include <deal.II/numerics/solution_transfer.h>
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
    using namespace ImmersXLA;
  }
  template <int dim, int spacedim = dim>
  class ElasticityFreqProblem : public EnableObserverPointer
  {
    static_assert(dim >= 1 && dim <= spacedim && spacedim <= 3,
                  "ElasticityFreqProblem requires 1 <= dim <= spacedim <= 3.");

  public:
    using VectorType = ImmersXLA::MPI::Vector;
    using MatrixType = ImmersXLA::MPI::SparseMatrix;

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

    /** Set the frequency in Hz before registering this algebraic Problem. */
    void
    prepare_frequency(double frequency_hz);
    /** Accept an external harmonic solve and reconstruct essential values. */
    void
    accept_frequency_solution(const VectorType &real,
                              const VectorType &imaginary);
    /** Write the accepted complex displacement using a sweep output index. */
    void
    output_results(unsigned int index = 0);
    const ElasticityFreqProblemParameters<dim, spacedim> &
    parameters() const
    {
      return par;
    }
    const parallel::TriangulationBase<dim, spacedim> &
    triangulation() const
    {
      AssertThrow(tria != nullptr, ExcNotInitialized());
      return *tria;
    }
    const Domain<dim, spacedim> &
    geometric_domain() const
    {
      AssertThrow(domain != nullptr, ExcNotInitialized());
      return *domain;
    }
    const Mapping<dim, spacedim> &
    mapping() const
    {
      AssertThrow(fe != nullptr, ExcNotInitialized());
      return compatibility::linear_mapping<dim, spacedim>(fe->reference_cell());
    }
    MPI_Comm
    communicator() const
    {
      return mpi_communicator;
    }
    double
    frequency() const
    {
      AssertThrow(
        frequency_prepared,
        ExcMessage(
          "Call prepare_frequency before registering the harmonic Problem."));
      return current_frequency;
    }

    /** Owned coefficients of the most recently solved complex displacement. */
    const VectorType &
    real_displacement() const
    {
      return real_displacement_storage;
    }
    const VectorType &
    imaginary_displacement() const
    {
      return imaginary_displacement_storage;
    }

    /** Spatial FE support, independent of the harmonic execution layout. */
    const DoFHandler<dim, spacedim> &
    dof_handler() const
    {
      return dh;
    }
    const IndexSet &
    locally_owned_dofs() const
    {
      return owned_dofs[0];
    }
    const IndexSet &
    locally_relevant_dofs() const
    {
      return relevant_dofs[0];
    }
    const AffineConstraints<double> &
    real_constraints() const
    {
      return constraints;
    }
    const AffineConstraints<double> &
    imaginary_part_constraints() const
    {
      return imaginary_constraints;
    }

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
    std::unique_ptr<Domain<dim, spacedim>>                domain;
    parallel::TriangulationBase<dim, spacedim>           *tria = nullptr;
    std::unique_ptr<FiniteElement<dim, spacedim>>         fe;
    std::unique_ptr<Quadrature<dim>>                      quadrature;
    std::unique_ptr<Quadrature<dim - 1>> face_quadrature_formula;
    DoFHandler<dim, spacedim>            dh;
    std::vector<IndexSet>                owned_dofs, relevant_dofs;
    AffineConstraints<double>            constraints, imaginary_constraints;
    LA::MPI::BlockVector                 solution, locally_relevant_solution;
    VectorType   real_displacement_storage, imaginary_displacement_storage;
    unsigned int cycle = 0, frequency_index = 0;
    FEValuesExtractors::Vector displacement;
    double                     current_frequency  = 0.0;
    bool                       frequency_prepared = false;
    std::string                stress_part        = "real";
  };
} // namespace ImmersX
#endif
