// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_constraint_equation_h
#define immersx_constraint_equation_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "linear_algebra.h"

/** Orientation of a matrix contribution in a constraint equation. */
enum class ConstraintContributionOrientation
{
  direct,
  transpose
};


/** A signed matrix contribution to a constraint equation. */
struct ConstraintEquationContribution
{
  using MatrixType = ImmersXLA::MPI::SparseMatrix;
  using BlockId    = std::size_t;

  BlockId                           block_id;
  const MatrixType                 *matrix;
  ConstraintContributionOrientation orientation;
  double                            sign;
};


/**
 * A single algebraic relation
 *
 * @f[
 *   \sum_i C_i u_i = d.
 * @f]
 *
 * The matrices are non-owning references to interaction-owned algebraic
 * contributions.  This object therefore describes one interaction and does
 * not become a global problem graph.  `block_id` identifies the corresponding
 * Problem/state block supplied to `apply()` or `residual()`.
 *
 * The optional multiplier metric is deliberately separate from any solver
 * preconditioning metric.  For the current Lagrange-multiplier interaction it
 * is the physical multiplier mass matrix M; a future solver may use a
 * different W without changing this equation.
 */
class ConstraintEquation
{
public:
  using MatrixType   = ImmersXLA::MPI::SparseMatrix;
  using VectorType   = ImmersXLA::MPI::Vector;
  using BlockId      = std::size_t;
  using Contribution = ConstraintEquationContribution;

  ConstraintEquation(const dealii::IndexSet &multiplier_locally_owned_dofs,
                     const MPI_Comm          mpi_communicator)
    : multiplier_owned_dofs_storage(multiplier_locally_owned_dofs)
    , mpi_communicator(mpi_communicator)
  {
    rhs_storage.reinit(this->multiplier_owned_dofs_storage,
                       this->mpi_communicator);
    rhs_storage = 0.;
  }

  /** Add a signed contribution to the constraint equation. */
  void
  add_contribution(const BlockId                           block_id,
                   const MatrixType                       &matrix,
                   const ConstraintContributionOrientation orientation,
                   const double                            sign = 1.)
  {
    AssertThrow(std::isfinite(sign),
                dealii::ExcMessage("A constraint contribution sign must be "
                                   "finite."));
    contributions.push_back({block_id, &matrix, orientation, sign});
  }

  /** Remove all matrix contributions while keeping the multiplier space. */
  void
  clear_contributions()
  {
    contributions.clear();
  }

  /** Set the multiplier-dual right-hand side d. */
  void
  set_rhs(const VectorType &rhs)
  {
    AssertDimension(rhs.size(), rhs_storage.size());
    rhs_storage = rhs;
  }

  /** Reset d to zero. */
  void
  clear_rhs()
  {
    rhs_storage = 0.;
  }

  /** Associate the physical multiplier metric M with this equation. */
  void
  set_multiplier_metric(const MatrixType &metric)
  {
    multiplier_metric_storage = &metric;
  }

  /** Return the multiplier-space ownership. */
  const dealii::IndexSet &
  multiplier_locally_owned_dofs() const
  {
    return multiplier_owned_dofs_storage;
  }

  /** Return all signed matrix contributions in insertion order. */
  const std::vector<Contribution> &
  contributions_view() const
  {
    return contributions;
  }

  /** Return d. */
  const VectorType &
  rhs() const
  {
    return rhs_storage;
  }

  /** Return whether a multiplier metric has been attached. */
  bool
  has_multiplier_metric() const
  {
    return multiplier_metric_storage != nullptr;
  }

  /** Return the multiplier metric M. */
  const MatrixType &
  multiplier_metric() const
  {
    AssertThrow(multiplier_metric_storage != nullptr,
                dealii::ExcMessage("This constraint has no multiplier "
                                   "metric."));
    return *multiplier_metric_storage;
  }

  /** Apply sum_i C_i u_i to a multiplier-space vector. */
  void
  apply(const std::vector<const VectorType *> &states, VectorType &value) const
  {
    value.reinit(multiplier_owned_dofs_storage, mpi_communicator);
    value = 0.;

    for (const auto &contribution : contributions)
      {
        AssertIndexRange(contribution.block_id, states.size());
        AssertThrow(states[contribution.block_id] != nullptr,
                    dealii::ExcMessage(
                      "A constraint state block pointer cannot be null."));

        VectorType contribution_value;
        contribution_value.reinit(multiplier_owned_dofs_storage,
                                  mpi_communicator);
        if (contribution.orientation ==
            ConstraintContributionOrientation::direct)
          contribution.matrix->vmult(contribution_value,
                                     *states[contribution.block_id]);
        else
          contribution.matrix->Tvmult(contribution_value,
                                      *states[contribution.block_id]);

        contribution_value *= contribution.sign;
        value += contribution_value;
      }
  }

  /** Apply the constraint and subtract d. */
  void
  residual(const std::vector<const VectorType *> &states,
           VectorType                            &value) const
  {
    apply(states, value);
    value -= rhs_storage;
  }

private:
  const dealii::IndexSet    multiplier_owned_dofs_storage;
  const MPI_Comm            mpi_communicator;
  std::vector<Contribution> contributions;
  const MatrixType         *multiplier_metric_storage = nullptr;
  VectorType                rhs_storage;
};

#endif // immersx_constraint_equation_h
