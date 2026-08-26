// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parameter_acceptor.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>

#include "coupled_poisson_elasticity.h"

using namespace dealii;


TEST(TractionParameters, IndependentParticleCouplingRoots) // NOLINT
{
  ParameterAcceptor::clear();

  CoupledPoissonElasticity::Traction traction_a("/Traction A/");
  CoupledPoissonElasticity::Traction traction_b("/Traction B/");

  ImmersX::initialize_parameters_from_string(R"(
    subsection Traction A
      subsection Particle coupling
        set RTree extraction level = 1
      end
    end
    subsection Traction B
      subsection Particle coupling
        set RTree extraction level = 3
      end
    end
  )");

  EXPECT_EQ(traction_a.particle_coupling_parameters().rtree_extraction_level,
            1u);
  EXPECT_EQ(traction_b.particle_coupling_parameters().rtree_extraction_level,
            3u);
}
