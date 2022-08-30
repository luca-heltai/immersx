//-----------------------------------------------------------
//
//    Copyright (C) 2020 by the deal.II authors
//
//    This file is subject to LGPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------

#include "laplacian.h"
#include "tests.h"

// Test some integral properties of the fourier coefficients.
template <int dim>
void
test(const std::vector<std::vector<double>> &inclusions)
{
  const unsigned int      Nq = 100;
  const unsigned int      Nc = 3;
  Inclusions<dim> inclusion;
  inclusion.n_q_points = Nq;
  inclusion.n_coefficients = Nc;
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
          integrals[i] += values[i];
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
        const auto theta = q * 2 * numbers::PI / Nq;
        for (unsigned int i = 0; i < Nc; ++i)
          integrals[i] += std::cos(theta) * values[i];
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
        const auto theta = q * 2 * numbers::PI / Nq;
        for (unsigned int i = 0; i < Nc; ++i)
          integrals[i] += std::sin(theta) * values[i];
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
