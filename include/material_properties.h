// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at the top
// level of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#ifndef reduced_lagrange_multipliers_material_properties_h
#define reduced_lagrange_multipliers_material_properties_h

#include <deal.II/base/parameter_acceptor.h>

#include <string>

using namespace dealii;

/**
 * Material constants and derived elastic moduli used by the bulk model.
 */
struct MaterialProperties : public ParameterAcceptor
{
  /**
   * Construct and register one material-parameter subsection.
   */
  MaterialProperties(const std::string &material_tag = "default")
    : ParameterAcceptor("/Immersed Problem/Material properties/" +
                        material_tag + "/")
    , material_tag(material_tag)
  {
    add_parameter("Lame mu", Lame_mu);
    add_parameter("Lame lambda", Lame_lambda);
    add_parameter("Density", rho);
    add_parameter("Viscosity eta", neta);
    add_parameter("Rayleigh alpha", rayleigh_alpha);
    add_parameter("Rayleigh beta", rayleigh_beta);

    parse_parameters_call_back.connect([this]() {
      // Elastic modulus for isotropic materials in 3D
      elastic_modulus =
        Lame_mu * (3 * Lame_lambda + 2 * Lame_mu) / (Lame_lambda + Lame_mu);

      // Poisson ratio for isotropic materials in 3D
      poisson_ratio = Lame_lambda / (2 * (Lame_lambda + Lame_mu));

      // Bulk modulus for isotropic materials in 3D
      bulk_modulus = Lame_lambda + (2.0 / 3.0) * Lame_mu;

      // Shear modulus is just Lame mu
      shear_modulus = Lame_mu;
    });
  }

  /**
   * User-facing material identifier used in parameter subsections.
   */
  std::string material_tag = "default";
  /**
   * First Lame parameter \f$\mu\f$.
   */
  double Lame_mu = 1.0;
  /**
   * Second Lame parameter \f$\lambda\f$.
   */
  double Lame_lambda = 1.0;
  /**
   * Mass density.
   */
  double rho = 0.0;
  /**
   * Kelvin-Voigt viscosity coefficient.
   */
  double neta = 0.0;
  /**
   * Rayleigh damping mass coefficient.
   */
  double rayleigh_alpha = 0.0;
  /**
   * Rayleigh damping stiffness coefficient.
   */
  double rayleigh_beta = 0.0;
  /**
   * Derived Young's modulus.
   */
  double elastic_modulus = 0.0;
  /**
   * Derived Poisson ratio.
   */
  double poisson_ratio = 0.0;
  /**
   * Derived bulk modulus.
   */
  double bulk_modulus = 0.0;
  /**
   * Derived shear modulus.
   */
  double shear_modulus = 0.0;
};

#endif
