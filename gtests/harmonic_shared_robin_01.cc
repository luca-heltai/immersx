// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/fiber_reinforced_elasticity_freq.h>

#include <complex>

#include "test_paths.h"

namespace
{
  template <typename Problem, typename Supports, typename Parameters>
  bool
  check_shared_robin_surrogate(const Problem    &problem,
                               const Supports   &supports,
                               const Parameters &parameters)
  {
    using namespace ImmersX;
    using V = ImmersXLA::MPI::Vector;
    using B = ImmersXLA::MPI::BlockVector;
    LinearAdapterParameters options("/Shared spring oracle/");
    LinearAdapter<V, B>     adapter(options, problem.communicator());
    const auto              fields = adapter.add(problem, "matrix").fields();
    adapter.add(
      [&](auto &builder) {
        harmonic_detail::add_shared_robin(builder,
                                          fields,
                                          supports,
                                          parameters);
        return fields;
      },
      "spring");
    const auto surrogate = fields.surrogate->matrix();
    V          translation, action;
    translation.reinit(problem.locally_owned_dofs(), problem.communicator());
    action.reinit(translation);
    translation = 1.;
    surrogate->vmult(action, translation);
    double local = 0.;
    for (const auto &[id, support] : supports)
      for (const auto &point : support.locally_owned_quadrature_points())
        local += point.weight *
                 parameters.get_robin_coefficient(id).value(point.point);
    const double expected =
      dealii::Utilities::MPI::sum(local, problem.communicator()) *
      std::decay_t<decltype(fields.real)>::spacedimension();
    EXPECT_GT(expected, 0.);
    const double energy = translation * action;
    EXPECT_NEAR(energy, expected, 1.e-10 * expected);
    return expected > 0. && std::abs(energy - expected) <= 1.e-10 * expected;
  }

  template <int dim, int spacedim>
  void
  shared_robin_translation(const unsigned int refinement   = 0,
                           const bool         per_id_phase = false,
                           const bool         iterative    = false)
  {
    using namespace ImmersX;
    using namespace dealii;
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<dim, spacedim> parameters;
    parameters.formulation      = "shared";
    parameters.output_directory = TestPaths::output_directory(
      "shared-robin-" + std::to_string(dim) + "-" + std::to_string(spacedim) +
      "-" + std::to_string(refinement));
    parameters.matrix_parameters.dirichlet_ids.clear();
    parameters.matrix_parameters.initial_refinement = refinement;
    parameters.fiber_parameters.initial_refinement  = refinement;
    parameters.fiber_parameters.arguments_for_grid  = "-0.5: 0.5: true";
    for (unsigned int id = 0; id < 2 * dim; ++id)
      parameters.fiber_parameters.robin_ids.insert(id);
    parameters.matrix_parameters.default_material_properties.rho = 0.;
    parameters.fiber_parameters.default_material_properties.rho  = 0.;
    parameters.fiber_parameters.robin_phase                      = 30.;
    if constexpr (dim < spacedim)
      parameters.geometry->thickness = "0.25";
    std::string force;
    for (unsigned int c = 0; c < spacedim; ++c)
      {
        if (c)
          force += ";";
        force += std::to_string(c + 1) + "*(2+x)";
      }
    std::string override_force;
    for (unsigned int c = 0; c < spacedim; ++c)
      {
        if (c)
          override_force += ";";
        override_force += "-" + std::to_string(c + 1) + "*(3+x)";
      }
    const std::string phase_override =
      per_id_phase ? " set Robin phase by boundary id = 1:210\n" : "";
    const std::string function_override =
      per_id_phase ?
        " subsection Robin coefficient 1\n set Function expression = 3+x\n end\n"
        " subsection Robin boundary conditions 1\n set Function expression = " +
          override_force + "\n end\n" :
        "";
    initialize_parameters_from_string(
      "subsection Fiber Reinforced Elasticity Frequency\n"
      " set Formulation = shared\n"
      " subsection Fiber\n subsection Frequency domain\n set Robin phase = 30\n" +
      phase_override +
      " end\n subsection Functions\n"
      " subsection Robin coefficient\n set Function expression = 2+x\n end\n"
      " subsection Robin boundary conditions\n set Function expression = " +
      force + "\n end\n" + function_override + " end\n end\nend\n");
    if (per_id_phase)
      {
        EXPECT_DOUBLE_EQ(parameters.fiber_parameters.get_robin_coefficient(1)
                           .value(Point<spacedim>()),
                         3.);
        EXPECT_DOUBLE_EQ(parameters.fiber_parameters.get_robin_bc(1).value(
                           Point<spacedim>(), 0),
                         -3.);
      }
    FiberReinforcedElasticityFreq<dim, spacedim> driver(parameters);
    driver.setup();
    const auto                &matrix = driver.matrix_problem();
    GridTools::Cache<spacedim> background(matrix.triangulation(),
                                          matrix.mapping());
    const auto supports = harmonic_detail::make_reinforcement_boundary_supports(
      driver.fiber_problem(), parameters.geometry.get(), background);
    const double face_measure = std::pow(.5, spacedim - dim);
    for (const auto &[id, support] : supports)
      {
        EXPECT_NEAR(support.measure(), face_measure, 1.e-12) << id;
      }
    if (iterative)
      {
        ASSERT_TRUE(check_shared_robin_surrogate(matrix,
                                                 supports,
                                                 parameters.fiber_parameters));
        parameters.solver_parameters.solver = LinearSolver::iterative;
        parameters.solver_parameters.preconditioner =
          LinearPreconditioner::block_diagonal;
      }
    driver.solve_frequency(0.);
    const std::complex<double> amplitude = std::polar(1., std::acos(-1.) / 6.);
    const auto                &fe        = matrix.dof_handler().get_fe();
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : matrix.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (matrix.locally_owned_dofs().is_element(indices[i]))
              {
                const auto expected =
                  (fe.system_to_component_index(i).first + 1.) * amplitude;
                EXPECT_NEAR(matrix.real_displacement()[indices[i]],
                            expected.real(),
                            2.e-8);
                EXPECT_NEAR(matrix.imaginary_displacement()[indices[i]],
                            expected.imag(),
                            2.e-8);
              }
        }
    EXPECT_LT(driver.residual_norm(), 1.e-9);
  }
} // namespace

TEST(FiberReinforcedElasticityFreq, BOTH_SharedRobinPhysicalBoundaryTranslation)
{
  shared_robin_translation<1, 1>();
  shared_robin_translation<1, 2>();
  shared_robin_translation<1, 3>();
  shared_robin_translation<2, 2>();
  shared_robin_translation<2, 3>();
  shared_robin_translation<3, 3>();
}

TEST(FiberReinforcedElasticityFreq, BOTH_SharedRobinIterativeSurrogate)
{
  shared_robin_translation<1, 1>(0, false, true);
  shared_robin_translation<1, 3>(0, true, true);
  shared_robin_translation<2, 2>(0, true, true);
}

TEST(FiberReinforcedElasticityFreq, BOTH_SharedRobinAcrossRefinedMatrix)
{
  shared_robin_translation<1, 2>(1, true);
}

TEST(FiberReinforcedElasticityFreq, BOTH_SharedRobinZeroLimitAndInvalidData)
{
  using namespace ImmersX;
  using namespace dealii;
  for (unsigned int mode = 0; mode < 3; ++mode)
    {
      ParameterAcceptor::clear();
      FiberReinforcedElasticityFreqParameters<1, 2> parameters;
      parameters.formulation      = "shared";
      parameters.output_directory = TestPaths::output_directory(
        "shared-robin-limit-" + std::to_string(mode));
      parameters.matrix_parameters.initial_refinement = 1;
      parameters.fiber_parameters.initial_refinement  = 0;
      parameters.fiber_parameters.arguments_for_grid  = "-0.5:0.5:true";
      parameters.fiber_parameters.robin_ids           = {0, 1};
      parameters.geometry->thickness                  = "0.25";
      const std::string beta                          = mode == 0 ? "0" :
                                                        mode == 1 ? "if(x<0,-1,1)" :
                                                                    "1";
      const std::string load = mode == 2 ? "exp(2000*x);0" : "0;0";
      initialize_parameters_from_string(
        "subsection Fiber Reinforced Elasticity Frequency\n"
        " set Formulation = shared\n subsection Matrix\n subsection Functions\n"
        " subsection Dirichlet boundary conditions\n set Function expression = 1;1\n end\n"
        " end\n end\n subsection Fiber\n subsection Functions\n"
        " subsection Robin coefficient\n set Function expression = " +
        beta +
        "\n end\n subsection Robin boundary conditions\n set Function expression = " +
        load + "\n end\n end\n end\nend\n");
      FiberReinforcedElasticityFreq<1, 2> driver(parameters);
      driver.setup();
      if (mode != 0)
        {
          EXPECT_THROW(driver.solve_frequency(0.), ExceptionBase);
          continue;
        }
      driver.solve_frequency(0.);
      const auto &matrix = driver.matrix_problem();
      for (const auto index : matrix.locally_owned_dofs())
        {
          EXPECT_NEAR(matrix.real_displacement()[index], 1., 2.e-8);
          EXPECT_NEAR(matrix.imaginary_displacement()[index], 0., 2.e-8);
        }
      EXPECT_LT(driver.residual_norm(), 1.e-9);
    }
}

TEST(FiberReinforcedElasticityFreq,
     BOTH_SharedRobinBoundaryClipsPhysicalSection)
{
  using namespace ImmersX;
  using namespace dealii;
  ParameterAcceptor::clear();
  FiberReinforcedElasticityFreqParameters<1, 3> parameters;
  parameters.formulation                    = "shared";
  parameters.matrix_parameters.name_of_grid = "subdivided_hyper_rectangle";
  parameters.matrix_parameters.arguments_for_grid =
    "1,1,1: -1,0,-1: 1,1,1: false";
  parameters.matrix_parameters.initial_refinement = 0;
  parameters.fiber_parameters.arguments_for_grid  = "-0.5:0.5:true";
  parameters.fiber_parameters.initial_refinement  = 0;
  parameters.fiber_parameters.robin_ids           = {0, 1};
  parameters.geometry->thickness                  = "0.25";
  parameters.geometry->section.n_q_points         = 2;
  parameters.matrix_parameters.dirichlet_ids.clear();
  parameters.matrix_parameters.default_material_properties.rho = 0.;
  initialize_parameters_from_string(
    "subsection Fiber Reinforced Elasticity Frequency\n"
    " subsection Fiber\n subsection Functions\n subsection Robin coefficient\n"
    " set Function expression = 2+x\n end\n end\n end\n end\n");
  FiberReinforcedElasticityFreq<1, 3> driver(parameters);
  driver.setup();
  const auto         &matrix = driver.matrix_problem();
  GridTools::Cache<3> background(matrix.triangulation(), matrix.mapping());
  const auto supports = harmonic_detail::make_reinforcement_boundary_supports(
    driver.fiber_problem(), parameters.geometry.get(), background);
  ASSERT_EQ(supports.size(), 2u);
  // The rod has length1 and a0.5-by-0.5 section, half inside the matrix.
  EXPECT_NEAR(driver.reinforcement_measure(), .125, 1.e-12);
  EXPECT_NEAR(supports.at(0).measure(), .125, 1.e-12);
  EXPECT_NEAR(supports.at(1).measure(), .125, 1.e-12);
  EXPECT_TRUE(check_shared_robin_surrogate(matrix,
                                           supports,
                                           parameters.fiber_parameters));
}
