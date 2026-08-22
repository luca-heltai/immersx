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

#include <utility>

namespace ImmersX
{
  /**
   * A residual contribution that maps a represented pressure/load to a
   * Problem-owned force row.
   *
   * The interaction owns neither state nor an execution block.  Its load
   * operator maps the source Representation's vector space into the target
   * Field's vector space; the same operator is registered as the state
   * Jacobian.  PressureLoadInteraction is the application-facing name for
   * this first representation-driven load specialization.
   */
  template <typename VectorType>
  class RepresentationLoadInteraction
  {
  public:
    using RepresentationType = Representation<VectorType>;
    using Operator           = dealii::LinearOperator<VectorType, VectorType>;

    RepresentationLoadInteraction(const RepresentationType &pressure,
                                  const FieldId             target,
                                  Operator                  load_operator)
      : pressure_(pressure)
      , target_(target)
      , load_operator_(std::move(load_operator))
    {}

    const RepresentationType &
    pressure() const
    {
      return pressure_;
    }

    FieldId
    target() const
    {
      return target_;
    }

    const Operator &
    load_operator() const
    {
      return load_operator_;
    }

  private:
    RepresentationType pressure_;
    FieldId            target_;
    Operator           load_operator_;
  };

  template <typename VectorType>
  using PressureLoadInteraction = RepresentationLoadInteraction<VectorType>;

  /** This interaction introduces no auxiliary semantic fields. */
  struct RepresentationLoadFields
  {};

  /** Register a representation-driven load residual and its dF/dy term. */
  template <typename Builder, typename VectorType>
  RepresentationLoadFields
  contribute(Builder                                         &builder,
             const RepresentationLoadInteraction<VectorType> &interaction)
  {
    using OperatorFactory = typename Builder::Model::OperatorFactory;

    const auto pressure = interaction.pressure();
    const auto load     = interaction.load_operator();

    builder.term(interaction.target(), "pressure-load")
      .residual([pressure, load](const auto &context) {
        return load * pressure.evaluate(context);
      })
      .state(pressure.source(),
             OperatorFactory([pressure, load](const auto &context) {
               return load * pressure.linearize(context);
             }));

    return {};
  }
} // namespace ImmersX

#endif // immersx_load_interaction_h
