// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_field_contributor_h
#define immersx_field_contributor_h

#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>

#include <utility>

namespace ImmersX
{
  /** Describe an existing FE field as an algebraic execution contributor. */
  template <typename FieldType>
  class AlgebraicFieldContributor
  {
  public:
    explicit AlgebraicFieldContributor(FieldType field)
      : field_(std::move(field))
    {}

    template <typename VectorType, typename MatrixType>
    FieldType
    operator()(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      const auto id = builder.algebraic_field(field_.name(),
                                              field_.locally_owned_dofs(),
                                              field_.locally_relevant_dofs());
      return field_.with_id(id);
    }

    const FieldType &
    field() const
    {
      return field_;
    }

  private:
    FieldType field_;
  };

  template <typename FieldType>
  AlgebraicFieldContributor<FieldType>
  algebraic_field(FieldType field)
  {
    return AlgebraicFieldContributor<FieldType>(std::move(field));
  }
} // namespace ImmersX

#endif // immersx_field_contributor_h
