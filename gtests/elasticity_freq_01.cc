// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#include <deal.II/base/function_parser.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_freq.h>
#include <immersx/physics/elasticity_freq_contributor.h>

#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>

#include "test_paths.h"

namespace
{
  template <int dim, int spacedim>
  void
  check_dimension()
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElasticityFreqProblemParameters<dim, spacedim> parameters;
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("harmonic-dimensions");
    parameters.initial_refinement = 0;
    parameters.dirichlet_ids.clear();
    ImmersX::initialize_parameters();
    ImmersX::ElasticityFreqProblem<dim, spacedim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    EXPECT_EQ(problem.dof_handler().get_triangulation().n_global_active_cells(),
              1u);
    EXPECT_EQ(problem.dof_handler().n_dofs(), (1u << dim) * spacedim);
  }
} // namespace

TEST(ElasticityFreq, BOTH_Dimensions)
{
  check_dimension<1, 1>();
  check_dimension<1, 2>();
  check_dimension<1, 3>();
  check_dimension<2, 2>();
  check_dimension<2, 3>();
  check_dimension<3, 3>();
}

TEST(ElasticityFreq, BOTH_IndependentParameterSubsections)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::ElasticityFreqProblemParameters<2>    matrix("/Harmonic/Matrix/");
  ImmersX::ElasticityFreqProblemParameters<1, 2> fiber("/Harmonic/Fiber/");
  ImmersX::initialize_parameters_from_string(
    "subsection Harmonic\n"
    " subsection Matrix\n"
    "  subsection Material properties\n"
    "   subsection default\n"
    "    set Lame mu = 7\n"
    "   end\n"
    "  end\n"
    " end\n"
    " subsection Fiber\n"
    "  subsection Material properties\n"
    "   subsection default\n"
    "    set Lame mu = 13\n"
    "   end\n"
    "  end\n"
    " end\nend\n");
  EXPECT_DOUBLE_EQ(matrix.default_material_properties.Lame_mu, 7.);
  EXPECT_DOUBLE_EQ(fiber.default_material_properties.Lame_mu, 13.);
}

namespace
{
  template <int dim, int spacedim>
  void
  check_translation()
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElasticityFreqProblemParameters<dim, spacedim> parameters;
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("harmonic-translation");
    parameters.output_name =
      "translation_" + std::to_string(dim) + std::to_string(spacedim);
    parameters.initial_refinement = 1;
    parameters.dirichlet_ids.clear();
    parameters.frequencies                                = {0.25};
    parameters.default_material_properties.rho            = 2.;
    parameters.default_material_properties.rayleigh_alpha = 0.4;
    std::string force                                     = "1";
    for (unsigned int d = 1; d < spacedim; ++d)
      force += "; 1";
    ImmersX::initialize_parameters_from_string(
      "subsection Functions\n subsection Right hand side\n set Function expression = " +
      force + "\n end\nend\n");
    ImmersX::ElasticityFreqProblem<dim, spacedim> problem(parameters);
    problem.run();
    const double omega = 0.5 * std::acos(-1.);
    const auto   expected =
      1. / std::complex<double>(-2. * omega * omega, 0.8 * omega);
    for (const auto i : problem.locally_owned_dofs())
      {
        EXPECT_NEAR(problem.real_displacement()[i], expected.real(), 1.e-8);
        EXPECT_NEAR(problem.imaginary_displacement()[i],
                    expected.imag(),
                    1.e-8);
      }
  }
} // namespace

TEST(ElasticityFreq, BOTH_DampedRigidTranslation)
{
  check_translation<1, 1>();
  check_translation<1, 2>();
  check_translation<1, 3>();
  check_translation<2, 2>();
  check_translation<2, 3>();
  check_translation<3, 3>();
}

namespace
{
  template <int dim, int spacedim>
  void
  check_adapter_affine_displacement(const bool simplex = false)
  {
    SCOPED_TRACE("dim=" + std::to_string(dim) +
                 ", spacedim=" + std::to_string(spacedim));
    using namespace ImmersX;
    using namespace dealii;
    ParameterAcceptor::clear();
    ElasticityFreqProblemParameters<dim, spacedim> parameters;
    parameters.initial_refinement = 1;
    if (simplex)
      {
        parameters.initial_refinement = 0;
        parameters.triangulation_type = "fullydistributed";
        parameters.name_of_grid       = TestPaths::data_filename(
          dim == 2 ? "tests/simplex_square.msh" : "tests/simplex_cube.msh");
      }
    // A generated interval retains endpoint ids 0 and 1 even without
    // colorization.
    parameters.dirichlet_ids.clear();
    for (unsigned int id = 0; id < 2 * dim; ++id)
      parameters.dirichlet_ids.insert(id);
    parameters.output_directory =
      TestPaths::output_directory("harmonic-adapter-affine");
    parameters.default_material_properties.rho            = 2.;
    parameters.default_material_properties.rayleigh_alpha = 0.4;
    parameters.default_material_properties.rayleigh_beta  = 0.05;
    parameters.default_material_properties.neta           = 0.3;
    const double               frequency                  = 0.25;
    const double               omega = 2. * std::acos(-1.) * frequency;
    const double               phase = std::acos(-1.) / 6.;
    const std::complex<double> dynamic_coefficient(-2. * omega * omega,
                                                   0.8 * omega);
    const auto                 displacement_factor = std::polar(1., phase);
    parameters.dirichlet_phase                     = 30.;
    parameters.rhs_phase =
      (std::arg(dynamic_coefficient) + phase) * 180. / std::acos(-1.);
    std::ostringstream force, boundary;
    force << std::setprecision(17);
    for (unsigned int d = 0; d < spacedim; ++d)
      {
        if (d)
          {
            force << "; ";
            boundary << "; ";
          }
        force << std::abs(dynamic_coefficient) * (d + 1) << "*(2+x)";
        boundary << d + 1 << "*(2+x)";
      }
    initialize_parameters_from_string(
      "subsection Functions\n subsection Right hand side\n set Function expression = " +
      force.str() +
      "\n end\n subsection Dirichlet boundary conditions\n set Function expression = " +
      boundary.str() + "\n end\nend\n");
    ElasticityFreqProblem<dim, spacedim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    problem.prepare_frequency(frequency);
    if constexpr (dim == 1)
      {
        IndexSet physical_endpoints(problem.dof_handler().n_dofs());
        for (const auto &cell : problem.dof_handler().active_cell_iterators())
          if (!cell->is_artificial())
            for (const auto face : cell->face_indices())
              if (cell->at_boundary(face))
                for (unsigned int component = 0; component < spacedim;
                     ++component)
                  physical_endpoints.add_index(
                    cell->vertex_dof_index(face, component));
        physical_endpoints.compress();
        // Artificial-cell endpoints must not prescribe interior unknowns.
        for (const auto &line : problem.real_constraints().get_lines())
          EXPECT_TRUE(physical_endpoints.is_element(line.index));
        for (const auto &line :
             problem.imaginary_part_constraints().get_lines())
          EXPECT_TRUE(physical_endpoints.is_element(line.index));
      }
    LinearAdapterParameters solver_parameters;
    solver_parameters.solver = LinearSolver::direct;
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
      solver_parameters, problem.communicator());
    const auto fields = adapter.add(problem, "solid").fields();
    auto       state  = adapter.make_state();
    adapter.solve(state);
    auto residual = adapter.make_state();
    adapter.evaluate_residual(state, residual);
    EXPECT_LT(residual.l2_norm(), 1.e-9);
    problem.accept_frequency_solution(adapter.field(state, fields.real),
                                      adapter.field(state, fields.imaginary));
    const auto                          &fe = problem.dof_handler().get_fe();
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : problem.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (problem.locally_owned_dofs().is_element(indices[i]))
              {
                const auto point =
                  problem.mapping().transform_unit_to_real_cell(
                    cell, fe.get_unit_support_points()[i]);
                const auto component = fe.system_to_component_index(i).first;
                const auto expected =
                  displacement_factor * (component + 1.) * (2. + point[0]);
                EXPECT_NEAR(problem.real_displacement()[indices[i]],
                            expected.real(),
                            1.e-9);
                EXPECT_NEAR(problem.imaginary_displacement()[indices[i]],
                            expected.imag(),
                            1.e-9);
              }
        }
  }
} // namespace

TEST(ElasticityFreq, BOTH_AdapterAffineDisplacement)
{
  check_adapter_affine_displacement<1, 1>();
  check_adapter_affine_displacement<1, 2>();
  check_adapter_affine_displacement<1, 3>();
  check_adapter_affine_displacement<2, 2>();
  check_adapter_affine_displacement<2, 3>();
  check_adapter_affine_displacement<3, 3>();
}

TEST(ElasticityFreq, BOTH_SimplexAdapterAffineDisplacement)
{
  check_adapter_affine_displacement<2, 2>(true);
  check_adapter_affine_displacement<2, 3>(true);
  check_adapter_affine_displacement<3, 3>(true);
}

TEST(ElasticityFreq, BOTH_ZeroMaterialPreservesInhomogeneousConstraints)
{
  using namespace ImmersX;
  using namespace dealii;
  for (const double viscosity : {0., 0.7})
    {
      ParameterAcceptor::clear();
      ElasticityFreqProblemParameters<2> parameters;
      parameters.initial_refinement = 0;
      parameters.name_of_grid       = "hyper_cube";
      parameters.arguments_for_grid = "-1:1:false";
      parameters.fe_degree          = 1;
      parameters.dirichlet_ids      = {0};
      parameters.dirichlet_phase    = 30.;
      parameters.output_directory =
        TestPaths::output_directory("harmonic-zero-material");
      initialize_parameters_from_string(
        "subsection Functions\n"
        " subsection Right hand side\n set Function expression = 0; 0\n end\n"
        " subsection Dirichlet boundary conditions\n"
        "  set Function expression = 2+x; 3-y\n end\nend\n");
      auto &material          = parameters.default_material_properties;
      material.Lame_mu        = 0.;
      material.Lame_lambda    = 0.;
      material.rho            = 0.;
      material.rayleigh_alpha = 0.;
      material.rayleigh_beta  = 0.;
      material.neta           = viscosity;
      ElasticityFreqProblem<2> problem(parameters);
      problem.make_grid();
      problem.setup_fe();
      problem.setup_dofs();
      problem.prepare_frequency(viscosity == 0. ? 0. : 0.25);
      ASSERT_EQ(problem.dof_handler().n_dofs(), 8u);
      unsigned int local_free = 0;
      for (const auto index : problem.locally_owned_dofs())
        local_free += !problem.real_constraints().is_constrained(index);
      ASSERT_EQ(Utilities::MPI::sum(local_free, MPI_COMM_WORLD), 0u);
      LinearAdapterParameters options;
      options.solver = LinearSolver::direct;
      LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
                 adapter(options, problem.communicator());
      const auto fields = adapter.add(problem, "solid").fields();
      auto       state  = adapter.make_state();
      adapter.solve(state);
      FunctionParser<2> exact(2);
      exact.initialize("x,y", "2+x;3-y", {});
      auto expected = adapter.field(state, fields.real);
      VectorTools::interpolate(problem.mapping(),
                               problem.dof_handler(),
                               exact,
                               expected);
      const double phase      = std::acos(-1.) / 6.;
      auto         real_error = adapter.field(state, fields.real);
      real_error.add(-std::cos(phase), expected);
      auto imaginary_error = adapter.field(state, fields.imaginary);
      imaginary_error.add(-std::sin(phase), expected);
      EXPECT_LT(real_error.l2_norm(), 1.e-12);
      EXPECT_LT(imaginary_error.l2_norm(), 1.e-12);
      auto residual = adapter.make_state();
      adapter.evaluate_residual(state, residual);
      EXPECT_LT(residual.l2_norm(), 1.e-12);
    }
}

TEST(ElasticityFreq, BOTH_RefinementParameterRoundTrip)
{
  std::string saved_parameters;
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElasticityFreqProblemParameters<2> parameters;
    ImmersX::initialize_parameters_from_string(
      "subsection Immersed Problem\n set Initial refinement = 1\n set Output directory = " +
      ImmersX::TestPaths::output_directory("harmonic-refinement-roundtrip") +
      "\nend\n");
    EXPECT_EQ(parameters.initial_refinement, 1u);
    std::ostringstream saved;
    dealii::ParameterAcceptor::prm.print_parameters(
      saved, dealii::ParameterHandler::PRM | dealii::ParameterHandler::Short);
    saved_parameters = saved.str();
  }
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElasticityFreqProblemParameters<2> parameters;
    ImmersX::initialize_parameters_from_string(saved_parameters);
    EXPECT_EQ(parameters.initial_refinement, 1u);
  }
}

namespace
{
  template <int dim, int spacedim>
  void
  check_robin_translation(const bool adapter_path,
                          const bool mixed   = false,
                          const bool simplex = false)
  {
    using namespace ImmersX;
    using namespace dealii;
    ParameterAcceptor::clear();
    ElasticityFreqProblemParameters<dim, spacedim> parameters;
    parameters.initial_refinement = 1;
    parameters.dirichlet_ids.clear();
    for (unsigned id = 0; id < 2 * dim; ++id)
      parameters.robin_ids.insert(id);
    if (mixed)
      {
        parameters.dirichlet_ids.insert(0);
        parameters.robin_ids.erase(0);
      }
    if (simplex)
      {
        parameters.initial_refinement = 0;
        parameters.triangulation_type = "fullydistributed";
        parameters.name_of_grid       = TestPaths::data_filename(
          dim == 2 ? "tests/simplex_square.msh" : "tests/simplex_cube.msh");
      }
    parameters.frequencies                                = {0.25};
    parameters.robin_phase                                = 30.;
    parameters.dirichlet_phase                            = 30.;
    parameters.rhs_phase                                  = -150.;
    parameters.default_material_properties.rho            = 2.;
    parameters.default_material_properties.rayleigh_alpha = 0.;
    parameters.default_material_properties.rayleigh_beta  = .7;
    parameters.output_directory = TestPaths::output_directory("harmonic-robin");
    const double       omega    = .5 * std::acos(-1.);
    std::ostringstream force, load, prescribed;
    force << std::setprecision(17);
    for (unsigned d = 0; d < spacedim; ++d)
      {
        if (d)
          {
            force << ';';
            load << ';';
            prescribed << ';';
          }
        force << 2. * omega * omega * (d + 1.);
        load << "(2+x)*" << d + 1;
        prescribed << d + 1;
      }
    initialize_parameters_from_string(
      "subsection Functions\n subsection Dirichlet boundary conditions\n set Function expression = " +
      prescribed.str() +
      "\n end\n subsection Robin coefficient\n set Function expression = 2+x\n end\n"
      " subsection Robin boundary conditions\n set Function expression = " +
      load.str() +
      "\n end\n subsection Right hand side\n set Function expression = " +
      force.str() + "\n end\nend\n");
    ElasticityFreqProblem<dim, spacedim> problem(parameters);
    if (adapter_path)
      {
        problem.make_grid();
        problem.setup_fe();
        problem.setup_dofs();
        problem.prepare_frequency(.25);
        LinearAdapterParameters control;
        control.solver = LinearSolver::direct;
        LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
                   adapter(control, MPI_COMM_WORLD);
        const auto fields = adapter.add(problem, "solid").fields();
        auto       state  = adapter.make_state();
        adapter.solve(state);
        auto residual = adapter.make_state();
        adapter.evaluate_residual(state, residual);
        EXPECT_LT(residual.l2_norm(), 1.e-9);
        problem.accept_frequency_solution(adapter.field(state, fields.real),
                                          adapter.field(state,
                                                        fields.imaginary));
      }
    else
      problem.run();
    const auto                          &fe = problem.dof_handler().get_fe();
    std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
    for (const auto &cell : problem.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ids);
          for (unsigned i = 0; i < ids.size(); ++i)
            if (problem.locally_owned_dofs().is_element(ids[i]))
              {
                const double magnitude =
                  fe.system_to_component_index(i).first + 1.;
                EXPECT_NEAR(problem.real_displacement()[ids[i]],
                            magnitude * std::sqrt(3.) / 2.,
                            1.e-8);
                EXPECT_NEAR(problem.imaginary_displacement()[ids[i]],
                            magnitude / 2.,
                            1.e-8);
              }
        }
  }
} // namespace
TEST(ElasticityFreq, BOTH_RobinTranslationAndPhase)
{
  check_robin_translation<1, 1>(false);
  check_robin_translation<1, 2>(false);
  check_robin_translation<1, 3>(false);
  check_robin_translation<2, 2>(false);
  check_robin_translation<2, 3>(false);
  check_robin_translation<3, 3>(false);
}
TEST(ElasticityFreq, BOTH_AdapterRobinTranslationAndPhase)
{
  check_robin_translation<1, 1>(true);
  check_robin_translation<1, 2>(true);
  check_robin_translation<1, 3>(true);
  check_robin_translation<2, 2>(true);
  check_robin_translation<2, 3>(true);
  check_robin_translation<3, 3>(true);
}

TEST(ElasticityFreq, BOTH_MixedRobinAndSimplexTranslation)
{
  check_robin_translation<2, 2>(false, true);
  check_robin_translation<3, 3>(true, true);
  check_robin_translation<2, 2>(false, false, true);
  check_robin_translation<2, 3>(true, false, true);
  check_robin_translation<3, 3>(true, false, true);
}
