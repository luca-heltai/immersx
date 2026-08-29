// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// ---------------------------------------------------------------------

#ifndef immersx_constraint_contributor_h
#define immersx_constraint_contributor_h

#include <immersx/core/constraint_equation.h>
#include <immersx/core/contributor.h>

#include <string>
#include <vector>

namespace ImmersX
{
  struct ConstraintFields
  {
    FieldId multiplier;
  };

  /**
   * Translate an arbitrary participant list into semantic constraint terms.
   * Each equation block is mapped to the participant with the same position
   * in `participants`; the equation itself remains agnostic about the number
   * of participants.
   */
  template <typename Builder>
  ConstraintFields
  contribute_constraint_equation(Builder                    &builder,
                                 const ConstraintEquation   &equation,
                                 const std::vector<FieldId> &participants)
  {
    using VectorType = typename ConstraintEquation::VectorType;
    using Operator   = dealii::LinearOperator<VectorType, VectorType>;

    const auto multiplier =
      builder.algebraic_field("lambda",
                              equation.multiplier_locally_owned_dofs(),
                              equation.multiplier_locally_relevant_dofs());
    if (equation.has_multiplier_metric())
      builder.saddle_point(multiplier,
                           participants,
                           builder.matrix_operator(
                             equation.multiplier_metric()));
    else
      builder.saddle_point(multiplier, participants);

    for (const auto &entry : equation.contributions_view())
      {
        AssertIndexRange(entry.block_id, participants.size());
        const auto participant = participants[entry.block_id];
        const auto map         = builder.matrix_operator(*entry.matrix);
        const auto to_multiplier =
          entry.orientation == ConstraintContributionOrientation::direct ?
            map :
            ImmersX::transpose_operator(map);
        const auto from_multiplier =
          entry.orientation == ConstraintContributionOrientation::direct ?
            ImmersX::transpose_operator(map) :
            map;
        const double sign   = entry.sign;
        const auto   suffix = "constraint." + std::to_string(entry.block_id);

        auto reaction = builder.term(participant, suffix + ".participant");
        reaction
          .residual([multiplier, from_multiplier, sign](const auto &context) {
            return sign * (from_multiplier.view * context.state(multiplier));
          })
          .state(multiplier, sign * from_multiplier);

        auto equation_row = builder.term(multiplier, suffix + ".multiplier");
        equation_row
          .residual([participant, to_multiplier, sign](const auto &context) {
            return sign * (to_multiplier.view * context.state(participant));
          })
          .state(participant, sign * to_multiplier);
      }

    if (equation.rhs().size() != 0)
      builder.term(multiplier, "constraint.rhs")
        .residual([&equation](const auto &) {
          dealii::PackagedOperation<VectorType> result;
          result.reinit_vector = [&equation](VectorType &vector, const bool) {
            vector.reinit(equation.rhs());
          };
          result.apply = [&equation](VectorType &vector) {
            vector = equation.rhs();
            vector *= -1.;
          };
          result.apply_add = [&equation](VectorType &vector) {
            vector -= equation.rhs();
          };
          return result;
        });

    return {multiplier};
  }
} // namespace ImmersX

#endif // immersx_constraint_contributor_h
