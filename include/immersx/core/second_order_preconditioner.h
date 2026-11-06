// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_second_order_preconditioner_h
#define immersx_second_order_preconditioner_h

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>

#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/matrix_operator.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>

namespace ImmersX
{
  /**
   * Inexact block factorization for [a M, -b M; c K, a M + d D].
   *
   * The free-variable Schur complement is a M + d D + (b c / a) K.
   * Both spatial solves use CG with AMG. Their stopping criteria make this a
   * variable preconditioner, so the outer solver must be flexible (FGMRES).
   * Constrained rows and columns are projected symmetrically for the inner
   * CG solves; the outer system retains its own constraint equations.
   *
   * Matrices already condensed for hanging nodes may be supplied. Constrained
   * diagonal entries are normalized internally and are never passed to CG as
   * nonzero right-hand sides. Spatial operators and mass AMG are retained;
   * changing the integration coefficients rebuilds only the Schur combination
   * and its AMG hierarchy.
   */
  template <typename VectorType,
            typename MatrixType = ImmersXLA::MPI::SparseMatrix>
  class SecondOrderBlockPreconditioner
  {
  public:
    using ReinitVector = std::function<void(VectorType &, bool)>;
    using AMGData = typename ImmersXLA::MPI::PreconditionAMG::AdditionalData;

    void
    initialize(const MatrixType       &mass,
               const MatrixType       &stiffness,
               const MatrixType       &damping,
               const dealii::IndexSet &constrained,
               ReinitVector            reinit_vector,
               const AMGData          &amg_data = {})
    {
      AssertDimension(mass.m(), stiffness.m());
      AssertDimension(mass.m(), damping.m());
      AssertDimension(mass.m(), constrained.size());
      constrained_   = constrained;
      reinit_vector_ = std::move(reinit_vector);
      reinit_vector_(prototype_, false);
      amg_data_  = amg_data;
      mass_      = detail::clone_matrix(mass);
      stiffness_ = detail::clone_matrix(stiffness);
      damping_   = detail::clone_matrix(damping);
      project(*mass_, 1.);
      project(*stiffness_, 0.);
      project(*damping_, 0.);
      // Cartesian mass components are independent. Reserve the elastic
      // rigid-body seeds for the Schur hierarchy, rather than enlarging
      // the mass coarse spaces with rotational modes.
      auto mass_amg_data = amg_data_;
      mass_amg_data.constant_modes.clear();
      mass_amg_data.constant_modes_values.clear();
      mass_amg_             = make_amg_preconditioner<VectorType>(*mass_,
                                                      reinit_vector_,
                                                      mass_amg_data);
      static_stiffness_amg_ = dealii::LinearOperator<VectorType>();
      static_stiffness_.reset();
      coefficients_set_ = false;
    }

    /** Prepare the row-swapped factorization of [0, -M; K, D].
     * The free stiffness block must be invertible, as with Robin support.
     * D need not be invertible. The M/K AMG cycles must be symmetric, as for
     * the CG solves with positive derivative weights. Their fixed actions
     * are reused as local adjoints; the complete pair transpose reverses
     * their order. Positive weights retain their separate factorization.
     */
    void
    prepare_static_inverse()
    {
      AssertThrow(mass_, dealii::ExcNotInitialized());
      if (static_stiffness_)
        return;
      static_stiffness_ = detail::clone_matrix(*stiffness_);
      project(*static_stiffness_, 1.);
      static_stiffness_amg_ =
        make_amg_preconditioner<VectorType>(*static_stiffness_,
                                            reinit_vector_,
                                            amg_data_);
    }

    /** Apply the complete static pair inverse, with identity constrained rows.
     * A fixed AMG action is linear and can be used inside a Schur operator;
     * residual-stopped CG is reserved for a flexible outer preconditioner.
     */
    void
    static_vmult(VectorType       &destination_displacement,
                 VectorType       &destination_velocity,
                 const VectorType &source_displacement,
                 const VectorType &source_velocity,
                 const bool        fixed_linear_action = false) const
    {
      AssertThrow(static_stiffness_, dealii::ExcNotInitialized());
      static_spatial_inverse(*mass_,
                             mass_amg_,
                             destination_velocity,
                             source_displacement,
                             fixed_linear_action);
      destination_velocity *= -1.;
      typename dealii::VectorMemory<VectorType>::Pointer right_hand_side(
        vector_memory_);
      right_hand_side->reinit(prototype_, true);
      damping_->vmult(*right_hand_side, destination_velocity);
      right_hand_side->sadd(-1., 1., source_velocity);
      static_spatial_inverse(*static_stiffness_,
                             static_stiffness_amg_,
                             destination_displacement,
                             *right_hand_side,
                             fixed_linear_action);
      restore_constrained(destination_displacement, source_displacement);
      restore_constrained(destination_velocity, source_velocity);
    }

    /** Transposed static inverse; the M, D and K actions reverse order. */
    void
    static_Tvmult(VectorType       &destination_displacement,
                  VectorType       &destination_velocity,
                  const VectorType &source_displacement,
                  const VectorType &source_velocity,
                  const bool        fixed_linear_action = false) const
    {
      AssertThrow(static_stiffness_, dealii::ExcNotInitialized());
      static_spatial_inverse(*static_stiffness_,
                             static_stiffness_amg_,
                             destination_velocity,
                             source_displacement,
                             fixed_linear_action);
      typename dealii::VectorMemory<VectorType>::Pointer right_hand_side(
        vector_memory_);
      right_hand_side->reinit(prototype_, true);
      damping_->Tvmult(*right_hand_side, destination_velocity);
      *right_hand_side -= source_velocity;
      static_spatial_inverse(*mass_,
                             mass_amg_,
                             destination_displacement,
                             *right_hand_side,
                             fixed_linear_action);
      restore_constrained(destination_displacement, source_displacement);
      restore_constrained(destination_velocity, source_velocity);
    }

    /** Configure one time-discrete Jacobian without spatial quadrature. */
    void
    set_coefficients(const double a,
                     const double b,
                     const double c,
                     const double d)
    {
      AssertThrow(mass_ && std::isfinite(a) && a > 0. && std::isfinite(b) &&
                    b >= 0. && std::isfinite(c) && c >= 0. &&
                    std::isfinite(d) && d >= 0.,
                  dealii::ExcMessage(
                    "Second-order block coefficients require a > 0 and "
                    "nonnegative finite b, c, d."));
      const std::array<double, 4> coefficients{{a, b, c, d}};
      if (coefficients_set_ && coefficients == coefficients_)
        return;
      coefficients_ = coefficients;
      schur_        = detail::clone_matrix(*mass_);
      *schur_ *= a;
      if (d != 0.)
        schur_->add(d, *damping_);
      if (b * c != 0.)
        schur_->add(b * c / a, *stiffness_);
      schur_amg_ =
        make_amg_preconditioner<VectorType>(*schur_, reinit_vector_, amg_data_);
      coefficients_set_ = true;
    }

    /** Approximate (a M)^{-1} on free entries, identity on constrained ones. */
    void
    mass_inverse(VectorType &destination, const VectorType &source) const
    {
      AssertThrow(coefficients_set_, dealii::ExcNotInitialized());
      solve(*mass_, mass_amg_, destination, source);
      destination /= coefficients_[0];
      restore_constrained(destination, source);
    }

    /** Approximate the physical Schur inverse on free entries. */
    void
    schur_inverse(VectorType &destination, const VectorType &source) const
    {
      AssertThrow(coefficients_set_, dealii::ExcNotInitialized());
      solve(*schur_, schur_amg_, destination, source);
      restore_constrained(destination, source);
    }

    /** Fixed linear AMG action for use inside a Schur matrix operator. */
    void
    apply_mass_amg(VectorType &destination, const VectorType &source) const
    {
      AssertThrow(coefficients_set_, dealii::ExcNotInitialized());
      apply_projected_amg(mass_amg_,
                          destination,
                          source,
                          1. / coefficients_[0]);
    }

    /** Fixed linear AMG approximation of the physical Schur inverse. */
    void
    apply_schur_amg(VectorType &destination, const VectorType &source) const
    {
      AssertThrow(coefficients_set_, dealii::ExcNotInitialized());
      apply_projected_amg(schur_amg_, destination, source, 1.);
    }

    /** Apply the complete lower-diagonal-upper block inverse. */
    void
    vmult(VectorType       &destination_displacement,
          VectorType       &destination_velocity,
          const VectorType &source_displacement,
          const VectorType &source_velocity) const
    {
      mass_inverse(destination_displacement, source_displacement);
      typename dealii::VectorMemory<VectorType>::Pointer right_hand_side(
        vector_memory_);
      typename dealii::VectorMemory<VectorType>::Pointer stiffness_action(
        vector_memory_);
      right_hand_side->reinit(prototype_, true);
      stiffness_action->reinit(prototype_, true);
      *right_hand_side = source_velocity;
      stiffness_->vmult(*stiffness_action, destination_displacement);
      right_hand_side->add(-coefficients_[2], *stiffness_action);
      schur_inverse(destination_velocity, *right_hand_side);
      destination_displacement.add(coefficients_[1] / coefficients_[0],
                                   destination_velocity);
      restore_constrained(destination_displacement, source_displacement);
    }

  private:
    void
    project(MatrixType &matrix, const double constrained_diagonal) const
    {
      for (const auto row : matrix.locally_owned_range_indices())
        {
          const bool constrained_row = constrained_.is_element(row);
          if (matrix.row_length(row) != 0)
            for (auto entry = matrix.begin(row), end = matrix.end(row);
                 entry != end;
                 ++entry)
              if (constrained_row || constrained_.is_element(entry->column()))
                matrix.set(row, entry->column(), 0.);
          if (constrained_row)
            matrix.set(row, row, constrained_diagonal);
        }
      matrix.compress(dealii::VectorOperation::insert);
    }

    void
    restore_constrained(VectorType &destination, const VectorType &source) const
    {
      for (const auto index : source.locally_owned_elements())
        if (constrained_.is_element(index))
          destination(index) = source(index);
    }

    void
    apply_projected_amg(const dealii::LinearOperator<VectorType> &amg,
                        VectorType                               &destination,
                        const VectorType                         &source,
                        const double                              scale,
                        const bool restore_identity = true) const
    {
      typename dealii::VectorMemory<VectorType>::Pointer right_hand_side(
        vector_memory_);
      right_hand_side->reinit(prototype_, true);
      *right_hand_side = source;
      for (const auto index : source.locally_owned_elements())
        if (constrained_.is_element(index))
          (*right_hand_side)(index) = 0.;
      amg.vmult(destination, *right_hand_side);
      destination *= scale;
      if (restore_identity)
        restore_constrained(destination, source);
      else
        for (const auto index : source.locally_owned_elements())
          if (constrained_.is_element(index))
            destination(index) = 0.;
    }

    void
    static_spatial_inverse(const MatrixType                         &matrix,
                           const dealii::LinearOperator<VectorType> &amg,
                           VectorType       &destination,
                           const VectorType &source,
                           const bool        fixed_linear_action) const
    {
      if (fixed_linear_action)
        // M, K and their AMG cycles are symmetric, as required by inner CG.
        // ML does not provide a transpose apply; reverse the pair factors in
        // static_Tvmult rather than requesting an unsupported local action.
        apply_projected_amg(amg, destination, source, 1., false);
      else
        // The physical M and K are symmetric; only the factor order changes.
        solve(matrix, amg, destination, source);
    }

    void
    solve(const MatrixType                         &matrix,
          const dealii::LinearOperator<VectorType> &amg,
          VectorType                               &destination,
          const VectorType                         &source) const
    {
      typename dealii::VectorMemory<VectorType>::Pointer right_hand_side(
        vector_memory_);
      right_hand_side->reinit(prototype_, true);
      *right_hand_side = source;
      for (const auto index : source.locally_owned_elements())
        if (constrained_.is_element(index))
          (*right_hand_side)(index) = 0.;
      destination       = 0.;
      const double norm = right_hand_side->l2_norm();
      if (norm == 0.)
        return;
      dealii::SolverControl        control(1000, 1.e-2 * norm);
      dealii::SolverCG<VectorType> solver(control, vector_memory_);
      solver.solve(matrix, destination, *right_hand_side, amg);
    }

    dealii::IndexSet                   constrained_;
    ReinitVector                       reinit_vector_;
    VectorType                         prototype_;
    AMGData                            amg_data_;
    std::shared_ptr<MatrixType>        mass_, stiffness_, damping_, schur_;
    std::shared_ptr<MatrixType>        static_stiffness_;
    dealii::LinearOperator<VectorType> mass_amg_, schur_amg_;
    dealii::LinearOperator<VectorType> static_stiffness_amg_;
    std::array<double, 4>              coefficients_{};
    bool                               coefficients_set_ = false;
    mutable dealii::GrowingVectorMemory<VectorType> vector_memory_;
  };
} // namespace ImmersX

#endif // immersx_second_order_preconditioner_h
