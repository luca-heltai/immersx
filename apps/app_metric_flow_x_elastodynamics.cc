// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/algebra/vessel_wall_interaction.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_vessel_wall_representation.h>

#  include <filesystem>
#  include <iostream>
#  include <memory>
#  include <stdexcept>
#  include <string>

namespace
{
  void
  make_output_directories(const std::string &flow_output_directory,
                          const std::string &solid_output_directory)
  {
    std::filesystem::create_directories(flow_output_directory);
    std::filesystem::create_directories(solid_output_directory);
  }

  std::string
  metric_flow_output_directory()
  {
    auto &prm = dealii::ParameterAcceptor::prm;
    prm.enter_subsection("MetricFlowSystem<1, 3>");
    const auto result = prm.get("Output directory");
    prm.leave_subsection();
    return result;
  }

  void
  run_coupled_problem(const std::string &parameter_file)
  {
    using SolidProblem = ImmersX::ElastodynamicsSolver<3>;
    using FlowProblem  = MetricFlowX::BloodFlowSystem<1, 3>;
    using FieldVector  = MetricFlowX::VectorType;
    using GlobalVector = ImmersX::ImmersXLA::MPI::BlockVector;
    using Adapter      = ImmersX::IDAAdapter<FieldVector, GlobalVector>;
    using SolidRepresentation =
      ImmersX::VectorFiniteElementRepresentation<3, 3>;
    using WallRepresentation =
      ImmersX::MetricFlowXAreaRadialDisplacementRepresentation;
    using Interaction =
      ImmersX::VesselWallInteraction<SolidRepresentation, WallRepresentation>;

    dealii::ParameterAcceptor::clear();
    FlowProblem flow_problem(MPI_COMM_WORLD);
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    ImmersX::TimeParameters flow_time;
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    ImmersX::ElastodynamicsParameters<3> solid_parameters("/Elastodynamics/",
                                                          &flow_time);
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    WallRepresentation::Lift wall_lift("/MetricFlowX vessel wall lift/");
    wall_lift.section.inclusion_degree      = 1;
    wall_lift.section.refinement_level      = 1;
    wall_lift.section.selected_coefficients = {3u, 7u};
    wall_lift.section.n_q_points            = 8;
    wall_lift.representative_n_q_points     = 2;
    ImmersX::ParticleCouplingParameters<3> search_parameters;
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);

    ImmersX::initialize_parameters(parameter_file);
    flow_problem.initialize_params(parameter_file);

    SolidProblem solid_problem(solid_parameters);
    solid_problem.make_grid();
    solid_problem.setup_fe();
    solid_problem.setup_system();
    solid_problem.assemble_operators();
    solid_problem.set_initial_conditions();

    flow_problem.setup();
    const auto flow_output_directory = metric_flow_output_directory();

    Adapter    adapter(flow_time, MPI_COMM_WORLD);
    const auto solid_fields = adapter.add(solid_problem, "elastodynamics");
    const auto flow_fields =
      adapter.add(ImmersX::metric_flow_x(flow_problem), "blood-flow");

    const SolidRepresentation solid_representation(
      solid_problem.triangulation(),
      solid_problem.dof_handler(),
      solid_problem.locally_owned_dofs(),
      solid_problem.locally_relevant_dofs(),
      solid_problem.constraints(),
      solid_problem.mapping(),
      dealii::FEValuesExtractors::Vector(0));
    const WallRepresentation wall_representation(flow_problem,
                                                 flow_fields.fields().area,
                                                 wall_lift);
    Interaction              interaction(solid_representation,
                            wall_representation,
                            search_parameters);
    interaction.assemble();
    const auto coupling_fields = adapter.add(interaction,
                                             "vessel-wall",
                                             solid_fields.fields().displacement,
                                             solid_fields.fields().velocity,
                                             flow_fields.fields().state);

    const auto output = [&solid_problem,
                         &solid_parameters,
                         &flow_problem,
                         &coupling_fields,
                         &interaction,
                         &adapter,
                         flow_fields,
                         flow_output_directory](const GlobalVector &state,
                                                const unsigned int  step,
                                                const double        time) {
      make_output_directories(flow_output_directory,
                              solid_parameters.output_directory);
      auto       &flow_state = adapter.field(state, flow_fields.fields().state);
      auto        pressure   = flow_problem.make_state();
      const auto &lambda =
        adapter.field(state, coupling_fields.fields().multiplier);
      const auto provider = interaction.make_external_pressure_provider(lambda);
      flow_problem.compute_pressure(flow_state, pressure, time, provider);
      flow_problem.output_results(flow_state, pressure, step);
      solid_problem.output_results();
      interaction.output_results(solid_parameters.output_directory +
                                   "/interaction",
                                 "pressure_jump",
                                 step,
                                 time);
    };

    adapter.set_output_step([&solid_problem,
                             &interaction,
                             &adapter,
                             &flow_time,
                             output,
                             solid_fields,
                             coupling_fields](const double        time,
                                              const GlobalVector &state,
                                              const GlobalVector &state_dot,
                                              const unsigned int  step) {
      solid_problem.accept_state(
        adapter.field(state, solid_fields.fields().displacement),
        adapter.field(state, solid_fields.fields().velocity),
        time,
        step);
      interaction.set_multiplier(
        adapter.field(state, coupling_fields.fields().multiplier));
      if (flow_time.output_frequency == 0 ||
          step % flow_time.output_frequency == 0 ||
          time >= flow_time.final_time)
        output(state, step, time);
      (void)state_dot;
    });

    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    adapter.field(state, solid_fields.fields().displacement) =
      solid_problem.displacement();
    adapter.field(state, solid_fields.fields().velocity) =
      solid_problem.velocity();
    adapter.field(state_dot, solid_fields.fields().displacement) =
      solid_problem.velocity();
    FieldVector acceleration;
    solid_problem.initial_acceleration(acceleration);
    adapter.field(state_dot, solid_fields.fields().velocity) = acceleration;
    flow_problem.initialize_state(adapter.field(state,
                                                flow_fields.fields().state),
                                  flow_time.initial_time);
    auto &flow_state = adapter.field(state, flow_fields.fields().state);
    for (const auto index :
         flow_problem.component_dofs(FlowProblem::Component::area))
      flow_state[index] = flow_problem.vessel_properties(0).a0;
    flow_problem.initialize_state_derivative(
      adapter.field(state_dot, flow_fields.fields().state),
      flow_time.initial_time);
    adapter.field(state, coupling_fields.fields().multiplier)     = 0.;
    adapter.field(state_dot, coupling_fields.fields().multiplier) = 0.;

    adapter.solve(state, state_dot);
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);
      if (argc != 2)
        throw std::invalid_argument(
          "usage: app_metric_flow_x_elastodynamics PARAMETER_FILE");
      run_coupled_problem(argv[1]);
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
    }
  return 0;
}

#else

int
main()
{
  return 0;
}

#endif // IMMERSX_WITH_METRIC_FLOW_X
