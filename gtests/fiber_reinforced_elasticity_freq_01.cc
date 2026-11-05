// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#include <deal.II/base/function_parser.h>

#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/fiber_reinforced_elasticity_freq.h>

#include <complex>
#include <sstream>
#include <vector>

#include "test_paths.h"

namespace
{
  using namespace ImmersX;
  using namespace dealii;

  template <typename Problem, typename Expected>
  void
  check_coefficients(const Problem &problem, const Expected &expected)
  {
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
                const auto value     = expected(point, component);
                EXPECT_NEAR(problem.real_displacement()[indices[i]],
                            value.real(),
                            1.e-9);
                EXPECT_NEAR(problem.imaginary_displacement()[indices[i]],
                            value.imag(),
                            1.e-9);
              }
        }
  }

  template <int dim>
  void
  check_dynamic_translation()
  {
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<1, dim> parameters;
    parameters.output_directory = TestPaths::output_directory(
      "harmonic-fiber-translation-" + std::to_string(dim));
    parameters.geometry->thickness                  = "0.5";
    parameters.matrix_parameters.initial_refinement = 0;
    parameters.fiber_parameters.initial_refinement  = 1;
    parameters.matrix_parameters.dirichlet_ids.clear();
    parameters.fiber_parameters.dirichlet_ids.clear();
    parameters.fiber_parameters.arguments_for_grid = "-0.75: 0.75: true";
    parameters.matrix_parameters.default_material_properties.rho = 2.;
    parameters.matrix_parameters.default_material_properties.rayleigh_alpha =
      0.4;
    parameters.fiber_parameters.default_material_properties.rho = 3.;
    parameters.fiber_parameters.default_material_properties.rayleigh_alpha =
      0.7;
    std::string force = "1";
    std::string zero  = "0";
    for (unsigned int d = 1; d < dim; ++d)
      {
        force += "; " + std::to_string(d + 1);
        zero += "; 0";
      }
    initialize_parameters_from_string(
      "subsection Fiber Reinforced Elasticity Frequency\n"
      " subsection Matrix\n subsection Functions\n subsection Right hand side\n set Function expression = " +
      force +
      "\n end\n end\n end\n subsection Fiber\n subsection Functions\n subsection Right hand side\n set Function expression = " +
      zero + "\n end\n end\n end\nend\n");
    FiberReinforcedElasticityFreq<1, dim> driver(parameters);
    driver.setup();
    const double              volume = std::pow(2., dim);
    const std::vector<double> frequencies =
      dim == 2 ? std::vector<double>{0.25, 0.4} : std::vector<double>{0.25};
    for (const double frequency : frequencies)
      {
        driver.solve_frequency(frequency, frequency == 0.25 ? 0 : 1);
        const double omega = 2. * std::acos(-1.) * frequency;
        const auto   matrix =
          std::complex<double>(-2. * omega * omega, 0.8 * omega);
        const auto fiber =
          std::complex<double>(-3. * omega * omega, 2.1 * omega);
        const auto amplitude = volume / (volume * matrix + 1.5 * fiber);
        const auto expected  = [amplitude](const auto &,
                                          const unsigned int component) {
          return (component + 1.) * amplitude;
        };
        check_coefficients(driver.matrix_problem(), expected);
        check_coefficients(driver.fiber_problem(), expected);
        EXPECT_LT(driver.residual_norm(), 1.e-9);
        EXPECT_LT(driver.continuity_norm(), 1.e-10);
        EXPECT_GT(std::hypot(driver.real_multiplier().l2_norm(),
                             driver.imaginary_multiplier().l2_norm()),
                  1.e-5);
      }
  }

} // namespace

TEST(FiberReinforcedElasticityFreq, BOTH_DampedTranslationAndFrequencySweep)
{
  check_dynamic_translation<2>();
  check_dynamic_translation<3>();
}


namespace
{
  template <int dim>
  void
  check_viscoelastic_fiber()
  {
    using namespace ImmersX;
    using namespace dealii;
    ParameterAcceptor::clear();
    ElasticityFreqProblemParameters<1, dim> parameters("/Harmonic fiber/");
    parameters.output_directory = TestPaths::output_directory(
      "harmonic-fiber-viscoelastic-" + std::to_string(dim));
    parameters.fe_degree          = 2;
    parameters.initial_refinement = 1;
    parameters.dirichlet_ids      = {0, 1};
    auto &material                = parameters.default_material_properties;
    material.Lame_mu              = 2.;
    material.Lame_lambda          = 3.;
    material.rho                  = 0.;
    material.rayleigh_beta        = 0.2;
    material.neta                 = 0.4;
    std::string force             = "1";
    for (unsigned int d = 1; d < dim; ++d)
      force += "; 0";
    initialize_parameters_from_string(
      "subsection Harmonic fiber\n subsection Functions\n subsection Right hand side\n set Function expression = " +
      force + "\n end\n end\nend\n");
    ElasticityFreqProblem<1, dim> fiber(parameters);
    fiber.make_grid();
    fiber.setup_fe();
    fiber.setup_dofs();
    fiber.prepare_frequency(0.5);
    LinearAdapterParameters options;
    options.solver = LinearSolver::direct;
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
      options, fiber.communicator());
    const auto fields = adapter.add(fiber, "fiber").fields();
    auto       state  = adapter.make_state();
    adapter.solve(state);
    fiber.accept_frequency_solution(adapter.field(state, fields.real),
                                    adapter.field(state, fields.imaginary));
    // The axial modulus is lambda+2*mu=7; the viscous modulus is
    // beta*7+eta=1.8. With unit body load and zero endpoint data,
    // u=(1-x^2)/(2*(7+i*pi*1.8)).
    const auto amplitude =
      1. / (2. * std::complex<double>(7., std::acos(-1.) * 1.8));
    const auto expected = [amplitude](const auto        &point,
                                      const unsigned int component) {
      return component == 0 ? amplitude * (1. - point[0] * point[0]) :
                              std::complex<double>();
    };
    check_coefficients(fiber, expected);
  }
} // namespace

TEST(FiberReinforcedElasticityFreq, BOTH_ViscoelasticFiberResponse)
{
  check_viscoelastic_fiber<2>();
  check_viscoelastic_fiber<3>();
}

TEST(FiberReinforcedElasticityFreq, BOTH_CollectiveMetadataFailure)
{
  using namespace ImmersX;
  dealii::ParameterAcceptor::clear();
  FiberReinforcedElasticityFreqParameters<1, 2> parameters;
  parameters.output_directory =
    TestPaths::output_directory("harmonic-fiber-metadata-error");
  parameters.matrix_parameters.initial_refinement = 0;
  parameters.fiber_parameters.initial_refinement  = 1;
  initialize_parameters();
  std::filesystem::create_directories(parameters.output_directory +
                                      "/frequencies.csv");
  FiberReinforcedElasticityFreq<1, 2> driver(parameters);
  EXPECT_THROW(driver.run(), dealii::ExceptionBase);
}

TEST(FiberReinforcedElasticityFreq, BOTH_CommonFrequencyParameters)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::FiberReinforcedElasticityFreqParameters<1, 2> parameters;
  parameters.output_directory =
    ImmersX::TestPaths::output_directory("harmonic-shared-frequencies");
  ImmersX::initialize_parameters_from_string(
    "subsection Fiber Reinforced Elasticity Frequency\n set Frequencies = 0.1, 0.2\nend\n");
  EXPECT_EQ(parameters.matrix_parameters.frequencies,
            (std::vector<double>{0.1, 0.2}));
  EXPECT_EQ(parameters.fiber_parameters.frequencies,
            (std::vector<double>{0.1, 0.2}));
  std::ostringstream saved;
  dealii::ParameterAcceptor::prm.print_parameters(
    saved, dealii::ParameterHandler::PRM | dealii::ParameterHandler::Short);
  const auto text  = saved.str();
  const auto first = text.find("set Frequencies");
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(text.find("set Frequencies", first + 1), std::string::npos);
}

namespace
{
  template <int dim, int spacedim>
  void
  check_reinforcement_modes(
    const std::string         &formulation,
    const LinearPreconditioner choice = LinearPreconditioner::automatic)
  {
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<dim, spacedim> parameters;
    parameters.formulation = formulation;
    if (choice != LinearPreconditioner::automatic)
      {
        parameters.solver_parameters.solver         = LinearSolver::iterative;
        parameters.solver_parameters.preconditioner = choice;
        parameters.solver_parameters.tolerance      = 1.e-11;
      }
    parameters.output_directory = TestPaths::output_directory(
      "harmonic-modes-" + std::to_string(dim) + "-" + std::to_string(spacedim) +
      "-" + formulation + "-" + std::to_string(static_cast<int>(choice)));
    auto &matrix              = parameters.matrix_parameters;
    auto &fiber               = parameters.fiber_parameters;
    matrix.initial_refinement = 0;
    fiber.initial_refinement  = formulation == "independent" ? 1 : 0;
    matrix.arguments_for_grid = "-1: 1: true";
    fiber.arguments_for_grid  = "-1: 1: true";
    matrix.dirichlet_ids      = {0};
    matrix.neumann_ids        = {1};
    matrix.default_material_properties.Lame_mu     = 2.;
    matrix.default_material_properties.Lame_lambda = 0.;
    fiber.default_material_properties.Lame_mu      = 5.;
    fiber.default_material_properties.Lame_lambda  = 0.;
    if constexpr (dim < spacedim)
      parameters.geometry->thickness = "0.5";
    std::string traction = "1";
    for (unsigned int d = 1; d < spacedim; ++d)
      traction += "; 0";
    std::string essential = "0.25";
    for (unsigned int d = 1; d < spacedim; ++d)
      essential += "; 0";
    initialize_parameters_from_string(
      "subsection Fiber Reinforced Elasticity Frequency\n subsection Matrix\n subsection Functions\n subsection Neumann boundary conditions\n set Function expression = " +
      traction +
      "\nend\nsubsection Dirichlet boundary conditions\nset Function expression = " +
      essential +
      "\nend\nend\nsubsection Frequency domain\nset Dirichlet phase = 20\nset Neumann phase = 20\nend\nend\nend\n");
    FiberReinforcedElasticityFreq<dim, spacedim> driver(parameters);
    driver.setup();
    EXPECT_NEAR(driver.reinforcement_measure(), std::pow(2., dim), 1.e-10);
    driver.solve_frequency(0.);
    // Parallel axial moduli carry different bulk and reinforcement areas.
    const double matrix_area = std::pow(2., spacedim - 1);
    const double fiber_area  = std::pow(2., dim - 1);
    const double slope    = matrix_area / (4. * matrix_area + 10. * fiber_area);
    const auto   expected = [slope](const auto &p, const unsigned int c) {
      return c == 0 ?
                 std::polar(0.25 + slope * (p[0] + 1.), std::acos(-1.) / 9.) :
                 std::complex<double>();
    };
    check_coefficients(driver.matrix_problem(), expected);
    if (formulation == "independent")
      {
        check_coefficients(driver.fiber_problem(), expected);
        EXPECT_LT(driver.continuity_norm(), 1.e-9);
      }
    EXPECT_LT(driver.residual_norm(), 1.e-8);
  }

  template <int dim, int spacedim>
  void
  check_shared_dynamic_response(
    const bool                 iterative = false,
    const LinearPreconditioner choice    = LinearPreconditioner::block_diagonal)
  {
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<dim, spacedim> parameters;
    parameters.formulation = "shared";
    if (iterative)
      {
        parameters.solver_parameters.solver         = LinearSolver::iterative;
        parameters.solver_parameters.preconditioner = choice;
        parameters.solver_parameters.tolerance      = 1.e-11;
      }
    parameters.output_directory = TestPaths::output_directory(
      "harmonic-shared-dynamic-" + std::to_string(dim) + "-" +
      std::to_string(spacedim) + "-" +
      std::to_string(static_cast<int>(choice)));
    auto &matrix              = parameters.matrix_parameters;
    auto &fiber               = parameters.fiber_parameters;
    matrix.initial_refinement = 0;
    fiber.initial_refinement  = dim == 1 && spacedim == 3 ? 1 : 0;
    matrix.dirichlet_ids.clear();
    fiber.arguments_for_grid                          = "-0.75: 0.75: true";
    matrix.default_material_properties.rho            = 2.;
    matrix.default_material_properties.rayleigh_alpha = 0.4;
    fiber.default_material_properties.rho             = 3.;
    fiber.default_material_properties.rayleigh_alpha  = 0.7;
    if constexpr (dim < spacedim)
      parameters.geometry->thickness = "0.25";
    std::string force = "1", extra = "2";
    for (unsigned int d = 1; d < spacedim; ++d)
      {
        force += "; " + std::to_string(d + 1);
        extra += "; " + std::to_string(2 * (d + 1));
      }
    initialize_parameters_from_string(
      "subsection Fiber Reinforced Elasticity Frequency\nsubsection Matrix\nsubsection Functions\nsubsection Right hand side\nset Function expression = " +
      force +
      "\nend\nend\nend\nsubsection Fiber\nsubsection Functions\nsubsection Right hand side\nset Function expression = " +
      extra +
      "\nend\nend\nsubsection Frequency domain\nset Rhs phase = 30\nend\nend\nend\n");
    FiberReinforcedElasticityFreq<dim, spacedim> driver(parameters);
    driver.setup();
    const double volume = std::pow(2., spacedim);
    const double reinforcement =
      std::pow(1.5, dim) * std::pow(0.5, spacedim - dim);
    EXPECT_NEAR(driver.reinforcement_measure(), reinforcement, 1.e-10);
    // One codimension-two representative checks frequency refresh; all six
    // dimension pairs retain the nonzero dynamic-mass and phased-load oracle.
    const std::vector<double> frequencies = dim == 1 && spacedim == 3 ?
                                              std::vector<double>{0.25, 0.4} :
                                              std::vector<double>{0.25};
    for (const double frequency : frequencies)
      {
        driver.solve_frequency(frequency, frequency == 0.25 ? 0 : 1);
        const double omega = 2. * std::acos(-1.) * frequency;
        const auto   amplitude =
          (volume + 2. * reinforcement * std::polar(1., std::acos(-1.) / 6.)) /
          (volume * std::complex<double>(-2. * omega * omega, 0.8 * omega) +
           reinforcement *
             std::complex<double>(-3. * omega * omega, 2.1 * omega));
        const auto expected = [amplitude](const auto &, const unsigned int c) {
          return (c + 1.) * amplitude;
        };
        check_coefficients(driver.matrix_problem(), expected);
        EXPECT_LT(driver.residual_norm(), 1.e-8);
      }
  }
} // namespace

TEST(FiberReinforcedElasticityFreq, BOTH_AllDimensionsAndFormulations)
{
  check_reinforcement_modes<1, 1>("independent");
  check_reinforcement_modes<1, 2>("independent");
  check_reinforcement_modes<1, 3>("independent");
  check_reinforcement_modes<2, 2>("independent");
  check_reinforcement_modes<2, 3>("independent");
  check_reinforcement_modes<3, 3>("independent");
  check_reinforcement_modes<1, 3>("shared");
  check_reinforcement_modes<2, 3>("shared");
  check_reinforcement_modes<3, 3>("shared");
}

TEST(FiberReinforcedElasticityFreq, BOTH_SharedDynamicMassAndPhasedForcing)
{
  check_shared_dynamic_response<1, 1>();
  check_shared_dynamic_response<1, 2>();
  check_shared_dynamic_response<1, 3>();
  check_shared_dynamic_response<2, 2>();
  check_shared_dynamic_response<2, 3>();
  check_shared_dynamic_response<3, 3>();
}

TEST(FiberReinforcedElasticityFreq, BOTH_SurrogatePreconditionedSharedResponse)
{
  check_shared_dynamic_response<1, 1>(true);
  check_shared_dynamic_response<1, 2>(true);
  check_shared_dynamic_response<1, 3>(true);
  check_shared_dynamic_response<2, 2>(true);
  check_shared_dynamic_response<2, 3>(true);
  check_shared_dynamic_response<3, 3>(true);
}

TEST(FiberReinforcedElasticityFreq, BOTH_NormalPreconditionedSharedResponse)
{
  check_shared_dynamic_response<1, 3>(true, LinearPreconditioner::normal);
  check_reinforcement_modes<1, 3>("shared", LinearPreconditioner::normal);
}

#if defined(DEAL_II_WITH_MUMPS) && !defined(IMMERSX_USE_PETSC_LA)
TEST(FiberReinforcedElasticityFreq, BOTH_MumpsBlrPreconditionedResponse)
{
  check_shared_dynamic_response<1, 3>(true, LinearPreconditioner::mumps_blr);
  check_reinforcement_modes<1, 3>("shared", LinearPreconditioner::mumps_blr);
  check_reinforcement_modes<1, 1>("independent",
                                  LinearPreconditioner::mumps_blr);
}
#endif

TEST(FiberReinforcedElasticityFreq, BOTH_NormalRequiresSharedFormulation)
{
  ParameterAcceptor::clear();
  FiberReinforcedElasticityFreqParameters<1, 3> parameters;
  parameters.formulation                      = "independent";
  parameters.solver_parameters.preconditioner = LinearPreconditioner::normal;
  initialize_parameters_from_string("");
  FiberReinforcedElasticityFreq<1, 3> driver(parameters);
  EXPECT_THROW(driver.setup(), ExceptionBase);
}

TEST(FiberReinforcedElasticityFreq, BOTH_PreconditionerParameters)
{
  ParameterAcceptor::clear();
  FiberReinforcedElasticityFreqParameters<1, 3> parameters;
  initialize_parameters_from_string(
    "subsection Fiber Reinforced Elasticity Frequency\n"
    " subsection Solver\n set Solver = iterative\n set Preconditioner = normal\n"
    " set FGMRES basis size = 75\n set Normal inner maximum iterations = 80\n"
    " set Normal inner reduction = 0.0002\n subsection MUMPS BLR\n"
    " set Tolerance = 0.00001\n set Memory limit MB per rank = 12000\n"
    " set Output details = true\n end\n end\n end\n");
  EXPECT_EQ(parameters.solver_parameters.preconditioner,
            LinearPreconditioner::normal);
  EXPECT_EQ(parameters.solver_parameters.fgmres_basis_size, 75u);
  EXPECT_EQ(parameters.normal_inner_maximum_iterations, 80u);
  EXPECT_DOUBLE_EQ(parameters.normal_inner_reduction, 0.0002);
  EXPECT_DOUBLE_EQ(parameters.solver_parameters.mumps_blr_tolerance, 0.00001);
  EXPECT_EQ(parameters.solver_parameters.mumps_memory_limit_mb, 12000u);
  EXPECT_TRUE(parameters.solver_parameters.mumps_output_details);
}

TEST(FiberReinforcedElasticityFreq, BOTH_SharedSurrogateIncludesFiberEnergy)
{
  using namespace ImmersX;
  using namespace dealii;
  ParameterAcceptor::clear();
  ElasticityFreqProblemParameters<2> bulk_parameters("/Bulk/"),
    fiber_parameters("/Fiber/");
  bulk_parameters.dirichlet_ids.clear();
  fiber_parameters.dirichlet_ids.clear();
  bulk_parameters.initial_refinement  = 1;
  fiber_parameters.initial_refinement = 0;
  bulk_parameters.arguments_for_grid  = "-1:1:false";
  fiber_parameters.arguments_for_grid = "-.5:.5:false";
  for (auto *material : {&bulk_parameters.default_material_properties,
                         &fiber_parameters.default_material_properties})
    {
      material->Lame_mu        = 2.;
      material->Lame_lambda    = 3.;
      material->rho            = 4.;
      material->rayleigh_alpha = .2;
      material->rayleigh_beta  = .3;
      material->neta           = .4;
    }
  initialize_parameters();
  ElasticityFreqProblem<2> bulk(bulk_parameters), fiber(fiber_parameters);
  bulk.make_grid();
  bulk.setup_fe();
  bulk.setup_dofs();
  const double frequency = .25, omega = 2. * std::acos(-1.) * frequency;
  bulk.prepare_frequency(frequency);
  fiber.make_grid();
  const auto support = harmonic_detail::make_reinforcement_support(
    fiber, static_cast<const std::monostate *>(nullptr));
  LinearAdapterParameters                                            options;
  LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector> adapter(
    options, bulk.communicator());
  const auto fields = adapter.add(bulk, "bulk").fields();
  adapter.add(
    [&](auto &builder) {
      harmonic_detail::add_shared_reinforcement(
        builder, fields, fiber_parameters, support, omega);
      return fields;
    },
    "reinforcement");
  const auto             matrix = fields.surrogate->matrix();
  ImmersXLA::MPI::Vector displacement, action;
  displacement.reinit(bulk.locally_owned_dofs(), bulk.communicator());
  action.reinit(displacement);
  FunctionParser<2> expression(2);
  expression.initialize("x,y", "x;0", {});
  VectorTools::interpolate(bulk.mapping(),
                           bulk.dof_handler(),
                           expression,
                           displacement);
  matrix->vmult(action, displacement);
  // For u=(x,0), the strain and divergence have unit squared norm. The
  // bulk has volume 4 and integral x^2=4/3; the fiber has 1 and 1/12.
  const auto  &material = bulk_parameters.default_material_properties;
  const double expected = ((1. + omega * material.rayleigh_beta) *
                             (2. * material.Lame_mu + material.Lame_lambda) +
                           omega * material.neta) *
                            5. +
                          (omega * omega + omega * material.rayleigh_alpha) *
                            material.rho * (4. / 3. + 1. / 12.);
  EXPECT_NEAR(displacement * action, expected, 1.e-10 * expected);
}

namespace
{
  void
  check_harmonic_overlap(const bool partial)
  {
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<1, 3> parameters;
    parameters.output_directory = TestPaths::output_directory(
      partial ? "harmonic-fiber-partial" : "harmonic-fiber-exterior");
    parameters.geometry->thickness          = "0.5";
    parameters.geometry->section.n_q_points = 2;
    auto &matrix                            = parameters.matrix_parameters;
    auto &fiber                             = parameters.fiber_parameters;
    matrix.initial_refinement               = 0;
    fiber.initial_refinement                = 1;
    matrix.dirichlet_ids.clear();
    fiber.dirichlet_ids.clear();
    if (partial)
      {
        matrix.name_of_grid       = "subdivided_hyper_rectangle";
        matrix.arguments_for_grid = "1, 1, 1: -1, 0, -1: 1, 1, 1: false";
      }
    fiber.arguments_for_grid = partial ? "-0.75:0.75:false" : "2:3:false";
    matrix.default_material_properties.rho            = 2.;
    matrix.default_material_properties.rayleigh_alpha = .4;
    fiber.default_material_properties.rho             = 3.;
    fiber.default_material_properties.rayleigh_alpha  = .7;
    initialize_parameters_from_string(R"(
subsection Fiber Reinforced Elasticity Frequency
 subsection Matrix
  subsection Functions
   subsection Right hand side
    set Function expression = 1; 2; 3
   end
  end
 end
 subsection Fiber
  subsection Functions
   subsection Right hand side
    set Function expression = 2; 4; 6
   end
  end
 end
end
)");
    FiberReinforcedElasticityFreq<1, 3> driver(parameters);
    driver.setup();
    const double frequency = .25;
    driver.solve_frequency(frequency);
    EXPECT_LT(driver.residual_norm(), 1.e-8);
    EXPECT_LT(driver.continuity_norm(), 1.e-8);
    const double multiplier =
      std::hypot(driver.real_multiplier().l2_norm(),
                 driver.imaginary_multiplier().l2_norm());
    if (partial)
      {
        EXPECT_GT(multiplier, 1.e-5);
        EXPECT_GT(driver.matrix_problem().real_displacement().l2_norm(), 1.e-5);
        EXPECT_GT(driver.fiber_problem().real_displacement().l2_norm(), 1.e-5);
      }
    else
      {
        EXPECT_LT(multiplier, 1.e-10);
        const double omega = 2. * std::acos(-1.) * frequency;
        const auto   matrix_amplitude =
          1. / std::complex<double>(-2. * omega * omega, .8 * omega);
        const auto fiber_amplitude =
          2. / std::complex<double>(-3. * omega * omega, 2.1 * omega);
        check_coefficients(driver.matrix_problem(),
                           [matrix_amplitude](const auto &, unsigned int c) {
                             return (c + 1.) * matrix_amplitude;
                           });
        check_coefficients(driver.fiber_problem(),
                           [fiber_amplitude](const auto &, unsigned int c) {
                             return (c + 1.) * fiber_amplitude;
                           });
      }
  }
} // namespace

TEST(FiberReinforcedElasticityFreq, BOTH_PartialLiftUsesExteriorMass)
{
  check_harmonic_overlap(true);
}

TEST(FiberReinforcedElasticityFreq, BOTH_ExteriorLiftPreservesUncoupledDynamics)
{
  check_harmonic_overlap(false);
}

namespace
{
  void
  check_shared_harmonic_overlap(const bool partial)
  {
    ParameterAcceptor::clear();
    FiberReinforcedElasticityFreqParameters<1, 3> parameters;
    parameters.formulation      = "shared";
    parameters.output_directory = TestPaths::output_directory(
      partial ? "harmonic-shared-partial" : "harmonic-shared-exterior");
    parameters.geometry->thickness          = "0.5";
    parameters.geometry->section.n_q_points = 2;
    auto &matrix                            = parameters.matrix_parameters;
    auto &fiber                             = parameters.fiber_parameters;
    matrix.initial_refinement               = 0;
    fiber.initial_refinement                = 1;
    matrix.dirichlet_ids.clear();
    fiber.dirichlet_ids.clear();
    if (partial)
      {
        matrix.name_of_grid       = "subdivided_hyper_rectangle";
        matrix.arguments_for_grid = "1, 1, 1: -1, 0, -1: 1, 1, 1: false";
      }
    fiber.arguments_for_grid = partial ? "-0.75:0.75:false" : "2:3:false";
    for (auto *material : {&matrix.default_material_properties,
                           &fiber.default_material_properties})
      {
        material->rho            = 2.;
        material->rayleigh_alpha = .4;
      }
    initialize_parameters_from_string(R"(
subsection Fiber Reinforced Elasticity Frequency
 subsection Matrix
  subsection Functions
   subsection Right hand side
    set Function expression = 1; 2; 3
   end
  end
 end
 subsection Fiber
  subsection Functions
   subsection Right hand side
    set Function expression = 1; 2; 3
   end
  end
 end
end
)");
    FiberReinforcedElasticityFreq<1, 3> driver(parameters);
    driver.setup();
    // Half of the symmetric section is inside y>=0; all points of the
    // translated exterior fiber are outside the cube.
    EXPECT_NEAR(driver.reinforcement_measure(), partial ? .75 : 0., 1.e-12);
    const double frequency = .25;
    driver.solve_frequency(frequency);
    const double omega = 2. * std::acos(-1.) * frequency;
    const auto   amplitude =
      1. / std::complex<double>(-2. * omega * omega, .8 * omega);
    // Matching density, damping and body load give an exact translation
    // pointwise, independent of the retained reinforcement geometry.
    check_coefficients(driver.matrix_problem(),
                       [amplitude](const auto &, unsigned int c) {
                         return (c + 1.) * amplitude;
                       });
    EXPECT_LT(driver.residual_norm(), 1.e-8);
  }
} // namespace

TEST(FiberReinforcedElasticityFreq, BOTH_SharedPartialLiftSkipsExteriorPoints)
{
  check_shared_harmonic_overlap(true);
}

TEST(FiberReinforcedElasticityFreq,
     BOTH_SharedExteriorLiftPreservesBulkDynamics)
{
  check_shared_harmonic_overlap(false);
}
