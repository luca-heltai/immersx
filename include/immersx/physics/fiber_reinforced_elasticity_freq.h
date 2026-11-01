// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#ifndef immersx_fiber_reinforced_elasticity_freq_h
#define immersx_fiber_reinforced_elasticity_freq_h

#include <deal.II/fe/fe_tools.h>

#include <immersx/core/constraint.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/output_handler.h>
#include <immersx/physics/detail/harmonic_reinforcement.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ImmersX
{
  /** Harmonic reinforcement parameters. dim is the reinforcement dimension;
   * the matrix always has dimension spacedim. */
  template <int dim, int spacedim = dim>
  class FiberReinforcedElasticityFreqParameters
    : public dealii::ParameterAcceptor
  {
  public:
    static_assert(1 <= dim && dim <= spacedim && spacedim <= 3,
                  "Reinforcement requires 1 <= dim <= spacedim <= 3.");
    using Lift = std::conditional_t<(dim < spacedim),
                                    TensorProductLift<dim, spacedim, spacedim>,
                                    std::monostate>;
    std::vector<double> frequencies{1.};

    explicit FiberReinforcedElasticityFreqParameters(
      const std::string &subsection = "/Fiber Reinforced Elasticity Frequency/")
      : ParameterAcceptor(MaterialProperties::normalize_subsection(subsection))
      , matrix_parameters(MaterialProperties::normalize_subsection(subsection) +
                            "Matrix/",
                          &frequencies)
      , fiber_parameters(MaterialProperties::normalize_subsection(subsection) +
                           "Fiber/",
                         &frequencies)
      , multiplier_parameters(MaterialProperties::normalize_subsection(
                                subsection) +
                                "Multiplier space/",
                              "auto")
      , solver_parameters(MaterialProperties::normalize_subsection(subsection) +
                          "Solver/")
    {
      add_parameter("Frequencies", frequencies);
      add_parameter("Output directory", output_directory);
      add_parameter(
        "Formulation",
        formulation,
        "Independent matrix/fiber displacements or a shared displacement.",
        prm,
        dealii::Patterns::Selection("independent|shared"));
      fiber_parameters.dirichlet_ids.clear();
      if constexpr (dim < spacedim)
        {
          geometry = std::make_unique<Lift>(
            MaterialProperties::normalize_subsection(subsection) + "Lift/");
          geometry->section.inclusion_type        = "hyper_cube";
          geometry->section.inclusion_degree      = 0;
          geometry->section.selected_coefficients = {0};
          geometry->section.refinement_level      = 0;
          geometry->section.n_q_points            = 3;
          geometry->representative_n_q_points     = 3;
          geometry->thickness                     = "0.1";
        }
      add_parameter("Multiplier FE degree",
                    multiplier_degree,
                    "Zero follows the fiber degree.",
                    prm,
                    dealii::Patterns::Integer(0));
      solver_parameters.solver = LinearSolver::direct;
      parse_parameters_call_back.connect([this]() {
        matrix_parameters.output_directory = output_directory + "/matrix";
        fiber_parameters.output_directory  = output_directory + "/fiber";
        matrix_parameters.output_name      = "matrix";
        fiber_parameters.output_name       = "fiber";
        AssertThrow(!frequencies.empty(),
                    dealii::ExcMessage("Provide at least one frequency."));
        for (const auto frequency : frequencies)
          AssertThrow(std::isfinite(frequency) && frequency >= 0.,
                      dealii::ExcMessage(
                        "Frequencies must be finite and nonnegative."));
      });
    }

    ElasticityFreqProblemParameters<spacedim, spacedim> matrix_parameters;
    ElasticityFreqProblemParameters<dim, spacedim>      fiber_parameters;
    FiniteElementSpaceParameters<dim, spacedim>         multiplier_parameters;
    LinearAdapterParameters                             solver_parameters;
    std::unique_ptr<Lift>                               geometry;
    std::string  formulation      = "independent";
    std::string  output_directory = "./output/fiber_reinforced_elasticity_freq";
    unsigned int multiplier_degree = 0;
  };

  /** Harmonic matrix reinforcement with independent or shared displacements.
   * Reduced reinforcement is extended normally through a constant section
   * mode. Independent displacements use real and imaginary continuity
   * constraints; shared displacement uses additional material integrals.
   */
  template <int dim, int spacedim = dim>
  class FiberReinforcedElasticityFreq
  {
    static_assert(1 <= dim && dim <= spacedim && spacedim <= 3,
                  "Reinforcement requires 1 <= dim <= spacedim <= 3.");

  public:
    using Parameters = FiberReinforcedElasticityFreqParameters<dim, spacedim>;
    using MatrixProblem    = ElasticityFreqProblem<spacedim, spacedim>;
    using FiberProblem     = ElasticityFreqProblem<dim, spacedim>;
    using VectorType       = typename MatrixProblem::VectorType;
    using GlobalVectorType = ImmersXLA::MPI::BlockVector;
    using Adapter          = LinearAdapter<VectorType, GlobalVectorType>;

    explicit FiberReinforcedElasticityFreq(const Parameters &parameters)
      : parameters(parameters)
      , matrix_storage(parameters.matrix_parameters)
      , fiber_storage(parameters.fiber_parameters)
    {}

    void
    setup()
    {
      AssertThrow(!initialized,
                  dealii::ExcMessage(
                    "The harmonic driver is already initialized."));
      AssertThrow(parameters.matrix_parameters.n_refinement_cycles == 1 &&
                    parameters.fiber_parameters.n_refinement_cycles == 1,
                  dealii::ExcMessage(
                    "The harmonic fiber driver uses one mesh per Problem; "
                    "set Number of refinement cycles to one."));
      AssertThrow(parameters.formulation == "independent" ||
                    parameters.formulation == "shared",
                  dealii::ExcMessage(
                    "Unknown harmonic reinforcement formulation."));
      if (parameters.formulation == "shared")
        AssertThrow(
          parameters.fiber_parameters.dirichlet_ids.empty() &&
            parameters.fiber_parameters.weak_dirichlet_ids.empty() &&
            parameters.fiber_parameters.neumann_ids.empty() &&
            parameters.fiber_parameters.normal_flux_ids.empty(),
          dealii::ExcMessage(
            "Shared displacement uses only matrix boundary conditions; clear the Fiber boundary ids."));
      matrix_storage.make_grid();
      matrix_storage.setup_fe();
      matrix_storage.setup_dofs();
      fiber_storage.make_grid();
      if constexpr (dim < spacedim)
        {
          AssertThrow(
            parameters.geometry->section.inclusion_type == "hyper_cube",
            dealii::ExcMessage(
              "Harmonic volume reinforcement requires a flat hyper_cube cross section."));
          AssertThrow(
            parameters.geometry->section.inclusion_degree == 0 &&
              parameters.geometry->section.selected_coefficients ==
                std::vector<unsigned int>{0},
            dealii::ExcMessage(
              "Harmonic reinforcement uses constant normal extension (degree zero and selected mode zero)."));
          TensorProductLiftSupport<dim, spacedim, spacedim> support(
            parameters.geometry->parameters());
          section_measure = support.section_measure(support.thickness());
        }
      volume_support =
        std::make_unique<harmonic_detail::ReinforcementSupport<spacedim>>(
          harmonic_detail::make_reinforcement_support(
            fiber_storage, parameters.geometry.get()));
      reinforcement_measure_storage = volume_support->measure();
      if (parameters.formulation == "shared")
        {
          initialized = true;
          return;
        }
      fiber_storage.setup_fe();
      fiber_storage.setup_dofs();
      const unsigned int degree =
        parameters.multiplier_degree == 0 ?
          fiber_storage.dof_handler().get_fe().degree :
          parameters.multiplier_degree;
      std::unique_ptr<dealii::FiniteElement<dim, spacedim>> element;
      if (parameters.multiplier_parameters.finite_element != "auto")
        element = dealii::FETools::get_fe_by_name<dim, spacedim>(
          parameters.multiplier_parameters.finite_element);
      else if (fiber_storage.dof_handler()
                 .get_fe()
                 .reference_cell()
                 .is_hyper_cube())
        element = std::make_unique<dealii::FESystem<dim, spacedim>>(
          dealii::FE_Q<dim, spacedim>(degree), spacedim);
      else if (fiber_storage.dof_handler()
                 .get_fe()
                 .reference_cell()
                 .is_simplex())
        element = std::make_unique<dealii::FESystem<dim, spacedim>>(
          dealii::FE_SimplexP<dim, spacedim>(degree), spacedim);
      multiplier_space = std::make_unique<FiniteElementSpace<dim, spacedim>>(
        fiber_storage.triangulation(), std::move(element));
      AssertThrow(multiplier_space->finite_element().n_components() == spacedim,
                  dealii::ExcMessage(
                    "The multiplier FE must have spacedim components."));
      auto &constraints = multiplier_space->constraints();
      constraints.reinit(multiplier_space->locally_owned_dofs(),
                         multiplier_space->locally_relevant_dofs());
      dealii::DoFTools::make_hanging_node_constraints(
        multiplier_space->dof_handler(), constraints);
      constraints.close();
      multiplier_view = std::make_unique<FiniteElementSpaceView<dim, spacedim>>(
        multiplier_space->view());
      initialized = true;
    }

    void
    solve_frequency(const double frequency, const unsigned int index = 0)
    {
      AssertThrow(initialized, dealii::ExcNotInitialized());
      matrix_storage.prepare_frequency(frequency);
      parameters.fiber_parameters.set_rhs_times(0.);
      Adapter    adapter(parameters.solver_parameters,
                      matrix_storage.communicator());
      const auto matrix = adapter.add(matrix_storage, "matrix").fields();
      if (parameters.formulation == "shared")
        {
          adapter.add(
            [&](auto &builder) {
              harmonic_detail::add_shared_reinforcement(
                builder,
                matrix,
                parameters.fiber_parameters,
                *volume_support,
                2. * std::acos(-1.) * frequency);
              return matrix;
            },
            "reinforcement");
          auto state = adapter.make_state();
          adapter.solve(state);
          accept_matrix(adapter, state, matrix, index);
          continuity_norm_storage = 0.;
        }
      else if (parameters.formulation == "independent")
        {
          fiber_storage.prepare_frequency(frequency);
          const auto fiber =
            adapter.add(fiber_storage, "fiber", section_measure).fields();
          const auto continuity = [&](const auto        &matrix_field,
                                      const auto        &fiber_field,
                                      const std::string &name) {
            const auto multiplier =
              multiplier_view->field(name,
                                     dealii::FEValuesExtractors::Vector(0));
            if constexpr (dim < spacedim)
              return adapter
                .add(
                  [&](auto &builder) {
                    return harmonic_detail::add_lifted_continuity(
                      builder,
                      matrix_field,
                      fiber_field,
                      multiplier,
                      *parameters.geometry,
                      section_measure);
                  },
                  name)
                .fields();
            else
              return adapter
                .add(make_continuity_constraint(matrix_field,
                                                fiber_field,
                                                multiplier),
                     name)
                .fields()
                .multiplier;
          };
          const auto lambda_real =
            continuity(matrix.real, fiber.real, "lambda_real");
          const auto lambda_imaginary =
            continuity(matrix.imaginary, fiber.imaginary, "lambda_imaginary");
          auto state = adapter.make_state();
          adapter.solve(state);
          accept_matrix(adapter, state, matrix, index);
          fiber_storage.accept_frequency_solution(
            adapter.field(state, fiber.real),
            adapter.field(state, fiber.imaginary));
          real_multiplier_storage      = adapter.field(state, lambda_real);
          imaginary_multiplier_storage = adapter.field(state, lambda_imaginary);
          auto residual                = adapter.make_state();
          adapter.evaluate_residual(state, residual);
          continuity_norm_storage =
            std::hypot(adapter.field(residual, lambda_real).l2_norm(),
                       adapter.field(residual, lambda_imaginary).l2_norm());
          fiber_storage.output_results(index);
          const auto directory =
            std::filesystem::path(parameters.output_directory) / "interaction" /
            ("frequency_" + std::to_string(index));
          OutputHandler<dim, spacedim, VectorType> real_output(lambda_real,
                                                               directory,
                                                               "lambda_real");
          OutputHandler<dim, spacedim, VectorType> imaginary_output(
            lambda_imaginary, directory, "lambda_imaginary");
          adapter.write_output(real_output, state, frequency);
          adapter.write_output(imaginary_output, state, frequency);
        }
    }

    void
    run()
    {
      setup();
      std::filesystem::create_directories(parameters.output_directory);
      unsigned int metadata_written = 1;
      if (dealii::Utilities::MPI::this_mpi_process(
            matrix_storage.communicator()) == 0)
        {
          std::ofstream metadata(parameters.output_directory +
                                 "/frequencies.csv");
          metadata << "index,frequency_hz\n";
          for (unsigned int i = 0; i < parameters.frequencies.size(); ++i)
            metadata << i << ',' << std::setprecision(17)
                     << parameters.frequencies[i] << '\n';
          metadata.close();
          metadata_written = metadata.good();
        }
      AssertThrow(
        dealii::Utilities::MPI::min(metadata_written,
                                    matrix_storage.communicator()) == 1,
        dealii::ExcMessage("Could not write harmonic frequency metadata."));
      for (unsigned int i = 0; i < parameters.frequencies.size(); ++i)
        solve_frequency(parameters.frequencies[i], i);
    }

    const MatrixProblem &
    matrix_problem() const
    {
      return matrix_storage;
    }
    const FiberProblem &
    fiber_problem() const
    {
      return fiber_storage;
    }
    const VectorType &
    real_multiplier() const
    {
      return real_multiplier_storage;
    }
    const VectorType &
    imaginary_multiplier() const
    {
      return imaginary_multiplier_storage;
    }
    double
    residual_norm() const
    {
      return residual_norm_storage;
    }
    double
    continuity_norm() const
    {
      return continuity_norm_storage;
    }

    double
    reinforcement_measure() const
    {
      return reinforcement_measure_storage;
    }

  private:
    template <typename Fields>
    void
    accept_matrix(Adapter                &adapter,
                  const GlobalVectorType &state,
                  const Fields           &matrix,
                  const unsigned int      index)
    {
      matrix_storage.accept_frequency_solution(
        adapter.field(state, matrix.real),
        adapter.field(state, matrix.imaginary));
      auto residual = adapter.make_state();
      adapter.evaluate_residual(state, residual);
      residual_norm_storage = residual.l2_norm();
      matrix_storage.output_results(index);
    }
    const Parameters                                      &parameters;
    MatrixProblem                                          matrix_storage;
    FiberProblem                                           fiber_storage;
    std::unique_ptr<FiniteElementSpace<dim, spacedim>>     multiplier_space;
    std::unique_ptr<FiniteElementSpaceView<dim, spacedim>> multiplier_view;
    VectorType real_multiplier_storage, imaginary_multiplier_storage;
    std::unique_ptr<harmonic_detail::ReinforcementSupport<spacedim>>
           volume_support;
    bool   initialized                   = false;
    double section_measure               = 1.;
    double reinforcement_measure_storage = 0.;
    double residual_norm_storage         = 0.;
    double continuity_norm_storage       = 0.;
  };
} // namespace ImmersX
#endif
