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
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

#include "laplacian.h"
#include "tests.h"

// Test some integral properties of the fourier coefficients.
template <int dim>
void
test(const std::vector<std::vector<double>> &inclusions)
{
  const unsigned int Nq = 100;
  const unsigned int Nc = 3;
  Inclusions<dim>    inclusion;
  inclusion.set_n_q_points(Nq);
  inclusion.set_n_coefficients(Nc);
  inclusion.inclusions = inclusions;
  inclusion.initialize();

  // Test integrals
  // Int_gamma 1 = 2 pi
  {
    std::vector<double> integrals(Nc, 0.0);
    for (unsigned int q = 0; q < Nq; ++q)
      {
        const auto &values = inclusion.get_fe_values(0);
        for (unsigned int i = 0; i < Nc; ++i)
          integrals[i] += values[i] * inclusion.get_JxW(0);
      }
    deallog << "integral 0: " << integrals[0] << std::endl
            << "integral 1: " << integrals[1] << std::endl
            << "integral 2: " << integrals[2] << std::endl;
  }
  {
    // Now test integrals VS cos(theta)
    std::vector<double> integrals(Nc, 0.0);
    for (unsigned int q = 0; q < Nq; ++q)
      {
        const auto &values = inclusion.get_fe_values(0);
        const auto  theta  = q * 2 * numbers::PI / Nq;
        for (unsigned int i = 0; i < Nc; ++i)
          integrals[i] += std::cos(theta) * values[i] * inclusion.get_JxW(0);
      }
    deallog << "cos(theta)*phi_0: " << integrals[0] << std::endl
            << "cos(theta)*phi_1: " << integrals[1] << std::endl
            << "cos(theta)*phi_2: " << integrals[2] << std::endl;
  }
  {
    // Now test integrals VS cos(theta)
    std::vector<double> integrals(Nc, 0.0);
    for (unsigned int q = 0; q < Nq; ++q)
      {
        const auto &values = inclusion.get_fe_values(0);
        const auto  theta  = q * 2 * numbers::PI / Nq;
        for (unsigned int i = 0; i < Nc; ++i)
          integrals[i] += std::sin(theta) * values[i] * inclusion.get_JxW(0);
      }
    deallog << "sin(theta)*phi_0: " << integrals[0] << std::endl
            << "sin(theta)*phi_1: " << integrals[1] << std::endl
            << "sin(theta)*phi_2: " << integrals[2] << std::endl;
  }
}

int
main()
{
  initlog();
  // Reference circles, Dirichlet data
  test<2>({{0, 0, 1.0}});
  // Half a radius circle, Dirichlet data
  test<2>({{0, 0, .5}});
}
