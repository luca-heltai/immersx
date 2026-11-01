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
#ifndef immersx_elasticity_explicit_h
#define immersx_elasticity_explicit_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <variant>
#define FORCE_USE_OF_TRILINOS

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <immersx/compatibility/reference_cell.h>
#include <immersx/core/integration_support.h>
#include <immersx/physics/elasticity_problem_parameters.h>

#include <memory>
#include <string>
#include <vector>

namespace ImmersX
{
  namespace LA
  {
#ifdef DEAL_II_WITH_TRILINOS
    using namespace dealii::LinearAlgebraTrilinos;
#else
#  error "elasticity_explicit requires Trilinos"
#endif
  } // namespace LA


  /** Uncoupled forward Euler elasticity; independent of ElasticityProblem. */
  template <int dim, int spacedim = dim>
  class ElasticityExplicit
  {
  public:
    explicit ElasticityExplicit(
      const ElasticityProblemParameters<dim, spacedim> &par);
    void
    run();

  private:
    const Mapping<spacedim> &
    geometry_mapping() const
    {
      AssertThrow(fe != nullptr, ExcNotInitialized());
      return compatibility::linear_mapping<spacedim>(fe->reference_cell());
    }

    void
    make_grid();
    void
    setup_fe();
    void
    setup_dofs();
    void
    setup_constraints();
    void
    setup_boundary_kinematics(const double time);
    void
    assemble_elasticity_system();
    void
    assemble_forcing_terms(const double evaluation_time);
    void
    solve_forward_euler(const double dt);
    bool
    uses_fully_distributed_triangulation() const;
    std::string
    output_solution() const;
    void
    output_results() const;

    const ElasticityProblemParameters<dim, spacedim> &par;
    MPI_Comm                                          mpi_communicator;
    ConditionalOStream                                pcout;
    mutable TimerOutput                               computing_timer;
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
    AffineConstraints<double>                 constraints;
    AffineConstraints<double>                 operator_constraints;
    AffineConstraints<double>                 velocity_constraints;
    AffineConstraints<double>                 acceleration_constraints;
    AffineConstraints<double>                 homogeneous_constraints;
    LA::MPI::SparseMatrix                     stiffness_matrix, damping_matrix;
    LA::MPI::Vector                           lumped_mass;
    LA::MPI::BlockVector                      solution, velocity, acceleration;
    LA::MPI::BlockVector                      locally_relevant_solution;
    LA::MPI::BlockVector                      system_rhs;
    unsigned int                              time_step = 0;
    FEValuesExtractors::Vector                displacement;
    double                                    current_time = 0.0;
  };
} // namespace ImmersX
#endif
