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

struct MaterialProperties : public ParameterAcceptor
{
  MaterialProperties(const std::string &material_tag = "default")
    : ParameterAcceptor("/Immersed Problem/Material properties/" +
                        material_tag + "/")
    , material_tag(material_tag)
  {
    add_parameter("Lame mu", Lame_mu);
    add_parameter("Lame lambda", Lame_lambda);
    add_parameter("Density", rho);
    add_parameter("Viscosity eta", neta);
    add_parameter("Relaxation time", relaxation_time);
    add_parameter("Rayleigh alpha", rayleigh_alpha);
    add_parameter("Rayleigh beta", rayleigh_beta);
  }

  std::string material_tag    = "default";
  double      Lame_mu         = 1.0;
  double      Lame_lambda     = 1.0;
  double      rho             = 0.0;
  double      neta            = 0.0;
  double      relaxation_time = 0.0;
  double      rayleigh_alpha  = 0.0;
  double      rayleigh_beta   = 0.0;
};

#endif
