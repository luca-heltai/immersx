// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coupling_operator_h
#define immersx_coupling_operator_h

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

namespace ImmersX
{
  /** Non-owning target test-space storage information for a coupling. */
  template <typename VectorType>
  class CouplingSpace
  {
  public:
    explicit CouplingSpace(const VectorType &prototype)
      : prototype_(&prototype)
    {}

    void
    reinit(VectorType &vector, const bool omit_ghost_values = false) const
    {
      vector.reinit(*prototype_, omit_ghost_values);
    }

  private:
    const VectorType *prototype_;
  };

  /**
   * A value-type map from represented quantity values to target residuals.
   *
   * The quantity and target vector types are intentionally separate.  The
   * operator owns the target-space reinitialization and the weak-form action;
   * it has no knowledge of source Fields or Representations.
   */
  template <typename QuantityVectorType, typename TargetVectorType>
  class CouplingOperator
  {
  public:
    using Operator =
      dealii::LinearOperator<TargetVectorType, QuantityVectorType>;

    CouplingOperator(const Operator                        &operator_view,
                     const CouplingSpace<TargetVectorType> &target_space)
      : operator_view_(operator_view)
      , target_space_(target_space)
    {}

    /** Apply the weak-form action C q to a represented quantity q. */
    TargetVectorType
    apply(const QuantityVectorType &quantity) const
    {
      TargetVectorType result;
      target_space_.reinit(result);
      linearize().vmult(result, quantity);
      return result;
    }

    /** Return C, the derivative of the residual with respect to q. */
    Operator
    linearize() const
    {
      Operator result            = operator_view_;
      result.reinit_range_vector = [space = target_space_](TargetVectorType &v,
                                                           const bool omit) {
        space.reinit(v, omit);
      };
      return result;
    }

    /** Build a residual operation for contributor registration. */
    dealii::PackagedOperation<TargetVectorType>
    residual(const QuantityVectorType &quantity) const
    {
      const auto coupling = linearize();
      const auto value    = quantity;

      dealii::PackagedOperation<TargetVectorType> result;
      result.reinit_vector = [coupling](TargetVectorType &vector,
                                        const bool        omit) {
        coupling.reinit_range_vector(vector, omit);
      };
      result.apply =
        [coupling, value, space = target_space_](TargetVectorType &vector) {
          space.reinit(vector);
          coupling.vmult(vector, value);
        };
      result.apply_add = [coupling, value](TargetVectorType &vector) {
        coupling.vmult_add(vector, value);
      };
      return result;
    }

  private:
    Operator                        operator_view_;
    CouplingSpace<TargetVectorType> target_space_;
  };
} // namespace ImmersX

#endif // immersx_coupling_operator_h
