// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_local_co_refinement_parameters_h
#define immersx_local_co_refinement_parameters_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>

#include <cmath>
#include <string>
#include <utility>

namespace ImmersX
{
  using dealii::ParameterAcceptor;
  /**
   * Controls local refinement synchronization between bulk and embedded meshes.
   *
   * For a zero-dimensional representative domain there is no embedded mesh to
   * refine or synchronize. In that case only the global bulk refinement
   * parameters are exposed.
   */
  template <int reduced_dim>
  struct RefinementParameters : public ParameterAcceptor
  {
    /**
     * Register local refinement configuration parameters.
     */
    RefinementParameters()
      : RefinementParameters("Local refinement parameters")
    {}

    /** Register a separate, explicitly named refinement parameter group. */
    explicit RefinementParameters(const std::string &subsection)
      : ParameterAcceptor(subsection)
    {
      this->add_parameter("Refinement strategy",
                          refinement_strategy,
                          "",
                          this->prm,
                          dealii::Patterns::Selection("space|embedded|both"));
      this->add_parameter(
        "Embedded partitioning policy",
        embedded_partitioning_policy,
        "Final embedded ownership: default or background-aligned.",
        this->prm,
        dealii::Patterns::Selection("default|immersed"));
      this->add_parameter("Space post-refinement cycles",
                          space_post_refinement_cycles);
      this->add_parameter("Embedded post-refinement cycles",
                          embedded_post_refinement_cycles);
      this->add_parameter("Space pre-refinement cycles",
                          space_pre_refinement_cycles);
      this->add_parameter("Embedded pre-refinement cycles",
                          embedded_pre_refinement_cycles);
      this->add_parameter("Lower factor", lower_factor);
      this->add_parameter("Upper factor", upper_factor);
      this->add_parameter("Refinement factor",
                          refinement_factor,
                          "Deprecated single-mesh factor; zero disables it.");
      this->add_parameter("Max refinement level", max_refinement_level);
    }

    /** Validate configuration and resolve the deprecated single-mesh factor.
     * A positive Refinement factor replaces lower by its reciprocal for space,
     * or upper by its value for embedded. It is not supported with both.
     * Zero disables this compatibility parameter.
     */
    std::pair<double, double>
    diameter_bounds() const
    {
      using namespace dealii;
      AssertThrow(refinement_strategy == "space" ||
                    refinement_strategy == "embedded" ||
                    refinement_strategy == "both",
                  ExcMessage(
                    "Refinement strategy must be space, embedded, or both."));
      AssertThrow(
        embedded_partitioning_policy == "default" ||
          embedded_partitioning_policy == "immersed",
        ExcMessage(
          "Embedded partitioning policy must be default or immersed."));
      AssertThrow(max_refinement_level >= 0,
                  ExcMessage("Max refinement level must be non-negative."));
      AssertThrow(
        std::isfinite(refinement_factor) && refinement_factor >= 0.,
        ExcMessage(
          "The deprecated refinement factor must be non-negative and finite."));
      double lower = lower_factor;
      double upper = upper_factor;
      if (refinement_factor > 0.)
        {
          AssertThrow(
            refinement_strategy != "both",
            ExcMessage(
              "Use Lower factor and Upper factor with strategy both."));
          if (refinement_strategy == "space")
            lower = 1. / refinement_factor;
          else if (refinement_strategy == "embedded")
            upper = refinement_factor;
        }
      AssertThrow(
        std::isfinite(lower) && std::isfinite(upper) && 0. < lower &&
          lower < upper,
        ExcMessage(
          "Factors must satisfy 0 < Lower factor < Upper factor and be finite."));
      return {lower, upper};
    }

    /**
     * Select which meshes may be refined: space, embedded, or both.
     */
    std::string refinement_strategy = "space";
    /** Final ownership in LocalCoRefinement: default or background-aligned.
     * The immersed policy requires a full-dimensional background and a
     * fullydistributed embedded Domain.
     */
    std::string embedded_partitioning_policy = "default";
    /**
     * Number of local space passes around intersecting embedded cell boxes.
     */
    unsigned int space_post_refinement_cycles = 0;
    /**
     * Number of global post-refinement cycles for the embedded grid.
     */
    unsigned int embedded_post_refinement_cycles = 0;
    /**
     * Number of global pre-refinement cycles for the space grid.
     */
    unsigned int space_pre_refinement_cycles = 0;
    /**
     * Number of global pre-refinement cycles for the embedded grid.
     */
    unsigned int embedded_pre_refinement_cycles = 0;
    /**
     * Deprecated single-mesh factor; zero disables this compatibility input.
     */
    double refinement_factor = 0.0;
    /** Fixed bounds for h_embedded / h_space, independent of strategy. */
    double lower_factor = 0.5;
    double upper_factor = 2.0;
    /**
     * Level cap for ratio-driven passes only (not initial, pre, or post).
     */
    int max_refinement_level = 10;
  };

  template <>
  struct RefinementParameters<0> : public ParameterAcceptor
  {
    RefinementParameters()
      : ParameterAcceptor("Local refinement parameters")
    {
      this->add_parameter("Refinement factor", refinement_factor);
      this->add_parameter("Max refinement level", max_refinement_level);
      this->add_parameter("Space post-refinement cycles",
                          space_post_refinement_cycles);
      this->add_parameter("Space pre-refinement cycles",
                          space_pre_refinement_cycles);
    }

    double       refinement_factor            = 1.0;
    int          max_refinement_level         = 10;
    unsigned int space_post_refinement_cycles = 0;
    unsigned int space_pre_refinement_cycles  = 0;
  };

} // namespace ImmersX
#endif
