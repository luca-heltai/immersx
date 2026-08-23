// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_load_interaction_h
#define immersx_load_interaction_h

#include <immersx/core/contributor.h>
#include <immersx/core/representation.h>
#include <immersx/coupling/coupling_operator.h>

#include <utility>

namespace ImmersX
{
  /**
   * A residual contribution that maps a represented pressure/load to a
   * Problem-owned force row.
   *
   * The interaction owns neither state nor an execution block.  It composes
   * the Representation derivative with a CouplingOperator that maps quantity
   * values into the target Field's residual space.
   */
  template <typename RepresentationType, typename TargetVectorType>
  class RepresentationLoadInteraction
  {
  public:
    using QuantityVectorType = typename RepresentationType::value_type;
    using CouplingType = CouplingOperator<QuantityVectorType, TargetVectorType>;

    RepresentationLoadInteraction(const RepresentationType &quantity,
                                  const FieldId             target,
                                  const CouplingType       &coupling)
      : quantity_(quantity)
      , target_(target)
      , coupling_(coupling)
    {}

    const RepresentationType &
    quantity() const
    {
      return quantity_;
    }

    FieldId
    target() const
    {
      return target_;
    }

    const CouplingType &
    coupling() const
    {
      return coupling_;
    }

  private:
    RepresentationType quantity_;
    FieldId            target_;
    CouplingType       coupling_;
  };

  /** This interaction introduces no auxiliary semantic fields. */
  struct RepresentationLoadFields
  {};

  /** Register a representation-driven load residual and its dF/dy term. */
  template <typename Builder,
            typename RepresentationType,
            typename TargetVectorType>
  RepresentationLoadFields
  contribute(Builder                                               &builder,
             const RepresentationLoadInteraction<RepresentationType,
                                                 TargetVectorType> &interaction)
  {
    using OperatorFactory = typename Builder::Model::OperatorFactory;

    const auto quantity = interaction.quantity();
    const auto coupling = interaction.coupling();

    builder.term(interaction.target(), "pressure-load")
      .residual([quantity, coupling](const auto &context) {
        return coupling.residual(quantity.evaluate(context));
      })
      .state(quantity.source(),
             OperatorFactory([quantity, coupling](const auto &context) {
               return coupling.linearize() * quantity.linearize(context);
             }));

    return {};
  }
} // namespace ImmersX

#endif // immersx_load_interaction_h
