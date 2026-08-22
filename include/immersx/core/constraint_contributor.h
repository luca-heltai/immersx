// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// ---------------------------------------------------------------------

#ifndef immersx_constraint_contributor_h
#define immersx_constraint_contributor_h

#include <immersx/core/constraint_equation.h>
#include <immersx/core/contributor.h>

#include <array>
#include <string>

namespace ImmersX
{
  struct ConstraintEquationFields
  {
    FieldId first;
    FieldId second;
    FieldId multiplier;
  };

  /** Translate a two-part ConstraintEquation into semantic terms. */
  template <typename Builder>
  ConstraintEquationFields
  contribute_constraint_equation(Builder                  &builder,
                                 const ConstraintEquation &equation,
                                 const FieldId             first,
                                 const FieldId             second)
  {
    using VectorType = typename ConstraintEquation::VectorType;
    using Operator   = dealii::LinearOperator<VectorType, VectorType>;

    const auto multiplier =
      builder.field("lambda",
                    TimeRole::algebraic,
                    equation.multiplier_locally_owned_dofs(),
                    equation.multiplier_locally_owned_dofs());

    const std::array<FieldId, 2> participants = {first, second};
    for (const auto &entry : equation.contributions_view())
      {
        AssertIndexRange(entry.block_id, participants.size());
        const auto     participant = participants[entry.block_id];
        const Operator map         = ImmersX::payload_free(
          dealii::linear_operator<VectorType, VectorType>(*entry.matrix));
        const Operator to_multiplier =
          entry.orientation == ConstraintContributionOrientation::direct ?
            map :
            dealii::transpose_operator(map);
        const Operator from_multiplier =
          entry.orientation == ConstraintContributionOrientation::direct ?
            dealii::transpose_operator(map) :
            map;
        const std::string term =
          builder.prefix() + ".constraint." + std::to_string(entry.block_id);
        const double sign = entry.sign;

        builder.add_residual(
          participant,
          term + ".participant",
          [multiplier, from_multiplier, sign](const auto &context) {
            return sign * (from_multiplier * context.state(multiplier));
          });
        builder.add_state_operator(participant,
                                   multiplier,
                                   term + ".participant",
                                   sign * from_multiplier);

        builder.add_residual(
          multiplier,
          term + ".multiplier",
          [participant, to_multiplier, sign](const auto &context) {
            return sign * (to_multiplier * context.state(participant));
          });
        builder.add_state_operator(multiplier,
                                   participant,
                                   term + ".multiplier",
                                   sign * to_multiplier);
      }

    if (equation.rhs().size() != 0)
      builder.add_residual(multiplier,
                           builder.prefix() + ".constraint.rhs",
                           [multiplier, &equation](const auto &context) {
                             const auto &state = context.state(multiplier);
                             dealii::PackagedOperation<VectorType> result;
                             result.reinit_vector = [state](VectorType &vector,
                                                            const bool  omit) {
                               vector.reinit(state, omit);
                             };
                             result.apply = [&equation](VectorType &vector) {
                               vector -= equation.rhs();
                             };
                             result.apply_add =
                               [&equation](VectorType &vector) {
                                 vector -= equation.rhs();
                               };
                             return result;
                           });

    return {first, second, multiplier};
  }
} // namespace ImmersX

#endif // immersx_constraint_contributor_h
