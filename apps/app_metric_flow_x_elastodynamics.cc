// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/algebra/metric_flow_x_vessel_wall_constraint.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_vessel_wall_observable.h>

#  include <filesystem>
#  include <fstream>
#  include <functional>
#  include <iostream>
#  include <map>
#  include <memory>
#  include <stdexcept>
#  include <string>

#  include "metric_flow_x_elastodynamics_mms.h"

namespace
{
  using namespace dealii;
  using namespace ImmersX;
  using namespace ImmersX::OneVesselMMS;

  class TutorialParameters : public ParameterAcceptor
  {
  public:
    TutorialParameters()
      : ParameterAcceptor("/MetricFlowX elastodynamics tutorial/")
    {
      add_parameter("MMS case", mms_case);
    }

    std::string mms_case = "none";
  };

  template <typename Function>
  void
  set_parsed_function(Function          &function,
                      const std::string &expression,
                      const std::string &constants)
  {
    auto &prm = ParameterAcceptor::prm;
    function.enter_my_subsection(prm);
    prm.set("Function constants", constants);
    prm.set("Function expression", expression);
    function.parse_parameters(prm);
    function.leave_my_subsection(prm);
  }

  std::string
  constant_flow_solution(const double area)
  {
    const auto value = number(area);
    return value + ";0;" + value + ";0";
  }

  std::filesystem::path
  write_metric_flow_bootstrap(const std::string &parameter_file)
  {
    const auto bootstrap =
      std::filesystem::temp_directory_path() /
      ("immersx-metric-flow-bootstrap-" +
       std::to_string(std::hash<std::string>{}(parameter_file)) + ".prm");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ofstream output(bootstrap);
        output
          << "subsection Metric Flow Parameters\n"
          << "  set Density (rho) = 1060.0\n"
          << "  set Reflection coefficient at outflow boundary (Rt) = 0.0\n"
          << "  set Tube law exponent (m) = 0.5\n"
          << "  set Viscosity coefficient (mu) = 0.0\n"
          << "  set Profile constant for friction term (xi) = 0.0\n"
          << "end\n\n"
          << "subsection Functions\n"
          << "  set Inflow function = 0.0\n"
          << "end\n\n"
          << "subsection MetricFlowSystem<1, 3>\n"
          << "  set Vtk file path for mesh input = mesh.vtk\n"
          << "end\n";
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return bootstrap;
  }

  template <typename Function>
  void
  in_metric_flow_runtime_directory(const std::string &parameter_file,
                                   Function         &&function)
  {
    const auto original_directory = std::filesystem::current_path();
    const auto runtime_directory =
      std::filesystem::temp_directory_path() /
      ("immersx-metric-flow-runtime-" +
       std::to_string(std::hash<std::string>{}(parameter_file)));
    std::filesystem::create_directories(runtime_directory);
    std::filesystem::current_path(runtime_directory);
    try
      {
        function();
      }
    catch (...)
      {
        std::filesystem::current_path(original_directory);
        throw;
      }
    std::filesystem::current_path(original_directory);
  }

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
    using SolidField = ImmersX::Field<3, 3, dealii::FEValuesExtractors::Vector>;
    using WallObservable = ImmersX::MetricFlowXAreaRadialDisplacementObservable;
    using Interaction =
      ImmersX::MetricFlowXVesselWallConstraint<SolidField, WallObservable>;

    dealii::ParameterAcceptor::clear();
    FlowProblem flow_problem(MPI_COMM_WORLD);
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    TutorialParameters tutorial_parameters;
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    ImmersX::TimeParameters flow_time;
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    ImmersX::ElastodynamicsParameters<3> solid_parameters("/Elastodynamics/",
                                                          &flow_time);
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);
    WallObservable::Lift wall_lift("/MetricFlowX vessel wall lift/");
    wall_lift.section.inclusion_degree      = 1;
    wall_lift.section.refinement_level      = 1;
    wall_lift.section.selected_coefficients = {3u, 7u};
    wall_lift.section.n_q_points            = 8;
    wall_lift.representative_n_q_points     = 2;
    ImmersX::ParticleCouplingParameters<3> search_parameters;
    ImmersX::reset_parameter_handler_to_root(dealii::ParameterAcceptor::prm);

    const auto flow_bootstrap = write_metric_flow_bootstrap(parameter_file);
    in_metric_flow_runtime_directory(parameter_file, [&]() {
      flow_problem.initialize_params(flow_bootstrap.string());
    });
    ImmersX::initialize_parameters(parameter_file);
    std::string mms_case = tutorial_parameters.mms_case;
    if (mms_case == "none")
      {
        const auto filename =
          std::filesystem::path(parameter_file).filename().string();
        mms_case =
          filename.find("01_wall_kinematics") != std::string::npos ?
            "kinematics" :
          filename.find("02_static_equilibrium") != std::string::npos ?
            "static_equilibrium" :
          filename.find("03_spatial_mms") != std::string::npos   ? "spatial" :
          filename.find("04_transient_mms") != std::string::npos ? "transient" :
                                                                   "none";
      }
    const bool kinematics = mms_case == "kinematics";
    const bool stationary =
      mms_case == "static_equilibrium" || mms_case == "spatial";
    const bool transient = mms_case == "transient";
    if (!(kinematics || stationary || transient || mms_case == "none"))
      throw std::runtime_error("Unknown MetricFlowX elastodynamics MMS case: " +
                               mms_case);

    const Parameters par;
    const auto       constants = function_constants();
    const auto       flow_solution =
      mms_case == "static_equilibrium" ?
              constant_flow_solution(std::stod(static_flow_area_expression())) :
            mms_case == "kinematics" ?
              constant_flow_solution(reference_area(par) + 1.e-8) :
            mms_case == "spatial" ?
              flow_area_expression(true) + ";0;" + flow_area_expression(true) + ";0" :
              flow_area_expression() + ";" + flow_velocity_expression() + ";" +
          flow_area_expression() + ";" + flow_velocity_expression();
    if (mms_case != "none")
      {
        auto &prm = ParameterAcceptor::prm;
        prm.enter_subsection("Functions");
        prm.set("RHS expression",
                mms_case == "spatial"   ? flow_rhs_expression(true) :
                mms_case == "transient" ? flow_rhs_expression() :
                                          "0;0");
        prm.set("Exact solution", flow_solution);
        prm.leave_subsection();
      }
    in_metric_flow_runtime_directory(parameter_file, [&]() {
      flow_problem.initialize_params();
    });

    if (mms_case != "none")
      {
        const std::string solid_displacement =
          mms_case == "static_equilibrium" ? static_displacement_expression() :
          mms_case == "spatial"            ? spatial_displacement_expression() :
          mms_case == "transient"          ? exact_displacement_expression() :
                                             "0;0;0";
        const std::string body_force =
          mms_case == "spatial"   ? spatial_body_force_expression() :
          mms_case == "transient" ? body_force_expression() :
                                    "0;0;0";
        set_parsed_function(solid_parameters.initial_displacement,
                            solid_displacement,
                            constants);
        set_parsed_function(solid_parameters.initial_velocity,
                            "0;0;0",
                            constants);
        set_parsed_function(solid_parameters.body_force, body_force, constants);
        set_parsed_function(solid_parameters.displacement_boundary,
                            "0;0;0",
                            constants);
        set_parsed_function(solid_parameters.velocity_boundary,
                            "0;0;0",
                            constants);
        set_parsed_function(solid_parameters.exact_solution,
                            solid_displacement,
                            constants);

        solid_parameters.dirichlet_ids.clear();
      }

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

    const auto solid_space =
      ImmersX::fe_space(solid_problem.dof_handler(),
                        solid_problem.mapping(),
                        solid_problem.constraints(),
                        solid_problem.locally_relevant_dofs());
    const SolidField solid_field =
      solid_space.field(solid_fields.fields().displacement,
                        "displacement",
                        dealii::FEValuesExtractors::Vector(0));
    const WallObservable wall_observable(flow_problem,
                                         flow_fields.fields().area,
                                         flow_fields.fields().area_components,
                                         wall_lift);
    Interaction interaction(solid_field, wall_observable, search_parameters);
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
    adapter.field(state_dot, solid_fields.fields().velocity) = 0.;
    auto &flow_state     = adapter.field(state, flow_fields.fields().state);
    auto &flow_state_dot = adapter.field(state_dot, flow_fields.fields().state);

    const auto set_flow_state = [&](const double time) {
      FieldVector exact_owned;
      exact_owned.reinit(flow_problem.dof_handler().locally_owned_dofs(),
                         MPI_COMM_WORLD);
      FunctionParser<3> exact(4);
      if (mms_case == "static_equilibrium")
        exact.initialize(FunctionParser<3>::default_variable_names() + ",t",
                         constant_flow_solution(
                           std::stod(static_flow_area_expression())),
                         {{"pi", numbers::PI}},
                         true);
      else if (mms_case == "kinematics")
        exact.initialize(FunctionParser<3>::default_variable_names() + ",t",
                         constant_flow_solution(reference_area(par) + 1.e-8),
                         {{"pi", numbers::PI}},
                         true);
      else
        initialize_exact_flow_function(exact, time, mms_case == "spatial");
      exact.set_time(time);
      VectorTools::interpolate(flow_problem.dof_handler(), exact, exact_owned);
      exact_owned.compress(VectorOperation::insert);
      flow_state = 0.;
      for (const auto index : flow_problem.dof_handler().locally_owned_dofs())
        flow_state[index] = exact_owned[index];
      flow_state.compress(VectorOperation::insert);
      flow_problem.initialize_trace_unknowns(flow_state, time);
    };

    const auto set_flow_state_derivative = [&](const double time) {
      flow_problem.initialize_state_derivative(flow_state_dot, time);
      if (mms_case == "transient")
        {
          FieldVector exact_owned;
          exact_owned.reinit(flow_problem.dof_handler().locally_owned_dofs(),
                             MPI_COMM_WORLD);
          FunctionParser<3> exact(4);
          exact.initialize(FunctionParser<3>::default_variable_names() + ",t",
                           "0;" + flow_velocity_time_derivative_expression() +
                             ";0;" + flow_velocity_time_derivative_expression(),
                           {{"pi", numbers::PI}},
                           true);
          exact.set_time(time);
          VectorTools::interpolate(flow_problem.dof_handler(),
                                   exact,
                                   exact_owned);
          exact_owned.compress(VectorOperation::insert);
          flow_state_dot = 0.;
          for (const auto index :
               flow_problem.dof_handler().locally_owned_dofs())
            flow_state_dot[index] = exact_owned[index];
          flow_state_dot.compress(VectorOperation::insert);
        }
    };

    set_flow_state(flow_time.initial_time);
    set_flow_state_derivative(flow_time.initial_time);

    if (mms_case == "transient")
      {
        FunctionParser<3> exact_acceleration(3);
        exact_acceleration.initialize(
          FunctionParser<3>::default_variable_names() + ",t",
          acceleration_expression(),
          function_constant_values(),
          true);
        exact_acceleration.set_time(flow_time.initial_time);
        SolidProblem::VectorType acceleration;
        acceleration.reinit(solid_problem.locally_owned_dofs(), MPI_COMM_WORLD);
        VectorTools::interpolate(solid_problem.dof_handler(),
                                 exact_acceleration,
                                 acceleration);
        acceleration.compress(VectorOperation::insert);
        adapter.field(state_dot, solid_fields.fields().velocity) = acceleration;
      }

    const auto set_multiplier = [&](const double time) {
      auto &multiplier =
        adapter.field(state, coupling_fields.fields().multiplier);
      multiplier = 0.;
      if (mms_case == "static_equilibrium")
        multiplier = std::stod(static_multiplier_expression());
      else if (mms_case == "spatial" || mms_case == "transient")
        {
          std::map<types::global_dof_index, Point<3>> support_points;
          MappingQ1<1, 3>                             mapping;
          DoFTools::map_dofs_to_support_points(mapping,
                                               flow_problem.dof_handler(),
                                               support_points);
          for (const auto &point : wall_observable.points())
            for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
              {
                const auto it = support_points.find(point.dof_indices[i]);
                if (it != support_points.end() &&
                    multiplier.locally_owned_elements().is_element(
                      point.multiplier_dof_indices[i]))
                  multiplier[point.multiplier_dof_indices[i]] =
                    mms_case == "spatial" ?
                      spatial_multiplier(par, it->second[0] + par.length / 2.) :
                      exact_multiplier(par,
                                       it->second[0] + par.length / 2.,
                                       time);
              }
        }
      multiplier.compress(VectorOperation::insert);
    };

    set_multiplier(flow_time.initial_time);
    adapter.field(state_dot, coupling_fields.fields().multiplier) = 0.;

    if (mms_case == "transient" || mms_case == "none")
      {
        const auto n_steps = adapter.solve(state, state_dot);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "MetricFlowX elastodynamics tutorial case " << mms_case
                    << " accepted IDA steps: " << n_steps << std::endl;
      }
    else
      {
        auto residual = adapter.make_state();
        adapter.solver().residual(flow_time.initial_time,
                                  state,
                                  state_dot,
                                  residual);
        solid_problem.accept_state(
          adapter.field(state, solid_fields.fields().displacement),
          adapter.field(state, solid_fields.fields().velocity),
          flow_time.initial_time,
          0);
        interaction.set_multiplier(
          adapter.field(state, coupling_fields.fields().multiplier));
        output(state, 0, flow_time.initial_time);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "MetricFlowX elastodynamics tutorial case " << mms_case
                    << " residual norm: " << residual.l2_norm() << std::endl;
      }
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
