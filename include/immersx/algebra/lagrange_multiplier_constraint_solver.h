// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_lagrange_multiplier_constraint_solver_h
#define immersx_lagrange_multiplier_constraint_solver_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

namespace ImmersX
{
  /**
   * Algebraic solver for a single PDE with a prescribed multiplier constraint.
   *
   * For a background operator A and a coupling matrix C with background rows
   * and multiplier columns, this class solves
   *
   * @code
   *   [ A  C ] [u     ] = [f    ],
   *   [ C^T 0 ] [lambda]   [g_Q  ].
   * @endcode
   *
   * The solver is independent of the Problem and Representation layers.  Its
   * Schur complement is applied matrix-free and the only preconditioner it
   * needs is the one for the PDE block A.
   */
  template <typename MatrixType,
            typename VectorType,
            typename PreconditionerType>
  class LagrangeMultiplierConstraintSolver
  {
  private:
    class SchurComplementOperator
    {
    public:
      explicit SchurComplementOperator(
        const LagrangeMultiplierConstraintSolver &solver)
        : solver(solver)
      {
        background_coupling.reinit(solver.background_owned_dofs,
                                   solver.mpi_communicator);
        background_inverse.reinit(solver.background_owned_dofs,
                                  solver.mpi_communicator);
      }

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        background_coupling = 0.;
        solver.coupling_matrix.vmult(background_coupling, src);
        solver.solve_block(background_inverse,
                           background_coupling,
                           solver.background_matrix,
                           solver.background_preconditioner);
        dst = 0.;
        solver.coupling_matrix.Tvmult(dst, background_inverse);
      }

      dealii::types::global_dof_index
      m() const
      {
        return solver.multiplier_owned_dofs.size();
      }

      dealii::types::global_dof_index
      n() const
      {
        return solver.multiplier_owned_dofs.size();
      }

    private:
      const LagrangeMultiplierConstraintSolver &solver;
      mutable VectorType                        background_coupling;
      mutable VectorType                        background_inverse;
    };

  public:
    LagrangeMultiplierConstraintSolver(
      const MatrixType       &background_matrix,
      const MatrixType       &coupling_matrix,
      const dealii::IndexSet &background_owned_dofs,
      const dealii::IndexSet &multiplier_owned_dofs,
      const MPI_Comm          mpi_communicator,
      const unsigned int      max_steps       = 200,
      const double            tolerance       = 1.e-10,
      const double            block_tolerance = 1.e-12)
      : background_matrix(background_matrix)
      , coupling_matrix(coupling_matrix)
      , background_owned_dofs(background_owned_dofs)
      , multiplier_owned_dofs(multiplier_owned_dofs)
      , mpi_communicator(mpi_communicator)
      , schur_solver_control(max_steps, tolerance)
      , block_solver_control(max_steps, block_tolerance)
    {}

    /** Solve for the PDE state and multiplier. */
    void
    solve(VectorType       &background_solution,
          VectorType       &multiplier,
          const VectorType &background_rhs,
          const VectorType &constraint_rhs) const
    {
      initialize_preconditioner();

      VectorType background_inverse_rhs;
      VectorType multiplier_rhs;
      VectorType background_correction;
      background_inverse_rhs.reinit(background_owned_dofs, mpi_communicator);
      multiplier_rhs.reinit(multiplier_owned_dofs, mpi_communicator);
      background_correction.reinit(background_owned_dofs, mpi_communicator);

      solve_block(background_inverse_rhs,
                  background_rhs,
                  background_matrix,
                  background_preconditioner);

      multiplier_rhs = 0.;
      coupling_matrix.Tvmult(multiplier_rhs, background_inverse_rhs);
      multiplier_rhs -= constraint_rhs;

      multiplier.reinit(multiplier_owned_dofs, mpi_communicator);
      multiplier = 0.;
      SchurComplementOperator      schur_operator(*this);
      dealii::SolverCG<VectorType> schur_solver(schur_solver_control);
      dealii::PreconditionIdentity identity;
      schur_solver.solve(schur_operator, multiplier, multiplier_rhs, identity);

      coupling_matrix.vmult(background_correction, multiplier);
      background_correction *= -1.;
      background_correction += background_rhs;
      solve_block(background_solution,
                  background_correction,
                  background_matrix,
                  background_preconditioner);
    }

  private:
    void
    initialize_preconditioner() const
    {
      if (preconditioner_initialized)
        return;

      typename PreconditionerType::AdditionalData data;
#ifdef IMMERSX_USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      background_preconditioner.initialize(background_matrix, data);
      preconditioner_initialized = true;
    }

    void
    solve_block(VectorType         &solution,
                const VectorType   &rhs,
                const MatrixType   &matrix,
                PreconditionerType &preconditioner) const
    {
      solution.reinit(rhs);
      solution = 0.;
      dealii::SolverControl fresh_control(block_solver_control.max_steps(),
                                          block_solver_control.tolerance());
      dealii::SolverCG<VectorType> solver(fresh_control);
      solver.solve(matrix, solution, rhs, preconditioner);
    }

    const MatrixType &background_matrix;
    const MatrixType &coupling_matrix;

    const dealii::IndexSet background_owned_dofs;
    const dealii::IndexSet multiplier_owned_dofs;
    const MPI_Comm         mpi_communicator;

    mutable PreconditionerType    background_preconditioner;
    mutable bool                  preconditioner_initialized = false;
    mutable dealii::SolverControl schur_solver_control;
    mutable dealii::SolverControl block_solver_control;
  };

} // namespace ImmersX

#endif // immersx_lagrange_multiplier_constraint_solver_h
