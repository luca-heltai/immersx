// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_lagrange_multiplier_schur_solver_h
#define immersx_lagrange_multiplier_schur_solver_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

namespace ImmersX
{
  /**
   * Algebraic Schur-complement solver for one two-field multiplier system.
   *
   * Given matrices A_bulk, A_embedded, C, and M with orientations
   *
   * @code
   *   C : bulk x multiplier,   M : multiplier x embedded,
   * @endcode
   *
   * this class solves
   *
   * @code
   *   A_bulk u + C lambda       = f,
   *   A_embedded w - M^T lambda = g,
   *   C^T u - M w               = 0.
   * @endcode
   *
   * It owns no PDE object and has no knowledge of how any of the four matrices
   * were assembled. The two diagonal inverses use the supplied AMG
   * preconditioner type; the multiplier Schur operator is applied matrix-free.
   */
  template <typename MatrixType,
            typename VectorType,
            typename PreconditionerType>
  class LagrangeMultiplierSchurSolver
  {
  private:
    class SchurComplementOperator
    {
    public:
      explicit SchurComplementOperator(
        const LagrangeMultiplierSchurSolver &solver)
        : solver(solver)
      {
        bulk_coupling.reinit(solver.bulk_owned_dofs, solver.mpi_communicator);
        bulk_inverse.reinit(solver.bulk_owned_dofs, solver.mpi_communicator);
        embedded_coupling.reinit(solver.embedded_owned_dofs,
                                 solver.mpi_communicator);
        embedded_inverse.reinit(solver.embedded_owned_dofs,
                                solver.mpi_communicator);
      }

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        // Some distributed vector backends do not guarantee that a vector
        // handed to a transpose multiply is initialized.  The first term
        // below is conceptually an assignment, while the embedded term is an
        // additive contribution; make that contract explicit before applying
        // the Schur operator.
        dst = 0.;
        solver.coupling_matrix.vmult(bulk_coupling, src);
        solver.solve_block(bulk_inverse,
                           bulk_coupling,
                           solver.bulk_matrix,
                           solver.bulk_preconditioner);
        solver.coupling_matrix.Tvmult(dst, bulk_inverse);

        solver.mass_matrix.Tvmult(embedded_coupling, src);
        solver.solve_block(embedded_inverse,
                           embedded_coupling,
                           solver.embedded_matrix,
                           solver.embedded_preconditioner);
        solver.mass_matrix.vmult_add(dst, embedded_inverse);
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
      const LagrangeMultiplierSchurSolver &solver;
      mutable VectorType                   bulk_coupling;
      mutable VectorType                   bulk_inverse;
      mutable VectorType                   embedded_coupling;
      mutable VectorType                   embedded_inverse;
    };

  public:
    LagrangeMultiplierSchurSolver(const MatrixType       &bulk_matrix,
                                  const MatrixType       &embedded_matrix,
                                  const MatrixType       &coupling_matrix,
                                  const MatrixType       &mass_matrix,
                                  const dealii::IndexSet &bulk_owned_dofs,
                                  const dealii::IndexSet &embedded_owned_dofs,
                                  const dealii::IndexSet &multiplier_owned_dofs,
                                  const MPI_Comm          mpi_communicator,
                                  const unsigned int      max_steps = 200,
                                  const double            tolerance = 1.e-10,
                                  const double block_tolerance      = 1.e-12)
      : bulk_matrix(bulk_matrix)
      , embedded_matrix(embedded_matrix)
      , coupling_matrix(coupling_matrix)
      , mass_matrix(mass_matrix)
      , bulk_owned_dofs(bulk_owned_dofs)
      , embedded_owned_dofs(embedded_owned_dofs)
      , multiplier_owned_dofs(multiplier_owned_dofs)
      , mpi_communicator(mpi_communicator)
      , schur_solver_control(max_steps, tolerance)
      , block_solver_control(max_steps, block_tolerance)
    {}

    /** Solve for u, w, and lambda from the two independent right-hand sides. */
    void
    solve(VectorType       &bulk_solution,
          VectorType       &embedded_solution,
          VectorType       &multiplier,
          const VectorType &bulk_rhs,
          const VectorType &embedded_rhs) const
    {
      initialize_preconditioners();

      VectorType bulk_inverse_rhs;
      VectorType embedded_inverse_rhs;
      VectorType multiplier_rhs;
      VectorType multiplier_correction;
      VectorType bulk_correction;
      VectorType embedded_correction;
      bulk_inverse_rhs.reinit(bulk_owned_dofs, mpi_communicator);
      embedded_inverse_rhs.reinit(embedded_owned_dofs, mpi_communicator);
      multiplier_rhs.reinit(multiplier_owned_dofs, mpi_communicator);
      multiplier_correction.reinit(multiplier_owned_dofs, mpi_communicator);
      bulk_correction.reinit(bulk_owned_dofs, mpi_communicator);
      embedded_correction.reinit(embedded_owned_dofs, mpi_communicator);

      solve_block(bulk_inverse_rhs, bulk_rhs, bulk_matrix, bulk_preconditioner);
      solve_block(embedded_inverse_rhs,
                  embedded_rhs,
                  embedded_matrix,
                  embedded_preconditioner);

      coupling_matrix.Tvmult(multiplier_rhs, bulk_inverse_rhs);
      mass_matrix.vmult(multiplier_correction, embedded_inverse_rhs);
      multiplier_rhs -= multiplier_correction;

      multiplier.reinit(multiplier_owned_dofs, mpi_communicator);
      multiplier = 0.;
      SchurComplementOperator schur_operator(*this);
      // The algebraic Schur operator is symmetric positive definite for the
      // ideal unconstrained pairing.  Distributed constraint elimination and
      // backend-specific sparse row operations can, however, leave a tiny
      // nonsymmetric component. GMRES preserves the same matrix-free Schur
      // reuse without making the coupled driver fail on that harmless backend
      // asymmetry.
      dealii::SolverGMRES<VectorType> schur_solver(schur_solver_control);
      dealii::PreconditionIdentity    identity;
      schur_solver.solve(schur_operator, multiplier, multiplier_rhs, identity);

      coupling_matrix.vmult(bulk_correction, multiplier);
      bulk_correction *= -1.;
      bulk_correction += bulk_rhs;
      solve_block(bulk_solution,
                  bulk_correction,
                  bulk_matrix,
                  bulk_preconditioner);

      mass_matrix.Tvmult(embedded_correction, multiplier);
      embedded_correction += embedded_rhs;
      solve_block(embedded_solution,
                  embedded_correction,
                  embedded_matrix,
                  embedded_preconditioner);
    }

  private:
    void
    initialize_preconditioners() const
    {
      if (preconditioners_initialized)
        return;

      typename PreconditionerType::AdditionalData data;
#ifdef IMMERSX_USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      bulk_preconditioner.initialize(bulk_matrix, data);
      embedded_preconditioner.initialize(embedded_matrix, data);
      preconditioners_initialized = true;
    }

    void
    solve_block(VectorType         &solution,
                const VectorType   &rhs,
                const MatrixType   &matrix,
                PreconditionerType &preconditioner) const
    {
      solution.reinit(rhs);
      solution = 0.;
      dealii::SolverCG<VectorType> solver(block_solver_control);
      solver.solve(matrix, solution, rhs, preconditioner);
    }

    const MatrixType &bulk_matrix;
    const MatrixType &embedded_matrix;
    const MatrixType &coupling_matrix;
    const MatrixType &mass_matrix;

    const dealii::IndexSet bulk_owned_dofs;
    const dealii::IndexSet embedded_owned_dofs;
    const dealii::IndexSet multiplier_owned_dofs;
    const MPI_Comm         mpi_communicator;

    mutable PreconditionerType    bulk_preconditioner;
    mutable PreconditionerType    embedded_preconditioner;
    mutable bool                  preconditioners_initialized = false;
    mutable dealii::SolverControl schur_solver_control;
    mutable dealii::SolverControl block_solver_control;
  };

} // namespace ImmersX

#endif // immersx_lagrange_multiplier_schur_solver_h
