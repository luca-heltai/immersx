// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <gtest/gtest.h>
#include <immersx/algebra/metric_flow_x_vessel_wall_constraint.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/metric_flow_x.h>
#include <immersx/physics/metric_flow_x_vessel_wall_observable.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "metric_flow_x_elastodynamics_mms.h"
#include "test_paths.h"

#if defined(IMMERSX_WITH_METRIC_FLOW_X) && defined(DEAL_II_WITH_SUNDIALS)

namespace
{
  using namespace ImmersX;
  using namespace dealii;
  using namespace ImmersX::OneVesselMMS;

  using FlowProblem    = MetricFlowX::BloodFlowSystem<1, 3>;
  using FlowVector     = MetricFlowX::VectorType;
  using GlobalVector   = ImmersXLA::MPI::BlockVector;
  using Adapter        = IDAAdapter<FlowVector, GlobalVector>;
  using SolidProblem   = ElastodynamicsSolver<3>;
  using SolidVector    = SolidProblem::VectorType;
  using SolidField     = Field<3, 3, dealii::FEValuesExtractors::Vector>;
  using WallObservable = MetricFlowXVesselWallGeometry;
  using Interaction =
    MetricFlowXVesselWallConstraint<SolidField, WallObservable>;
  using CouplingVector = typename Interaction::VectorType;

  struct ErrorRecord
  {
    unsigned int level             = 0;
    unsigned int n_steps           = 0;
    double       h_flow            = 0.;
    double       h_solid           = 0.;
    unsigned int flow_dofs         = 0;
    unsigned int solid_dofs        = 0;
    unsigned int multiplier_dofs   = 0;
    double       area_l2           = 0.;
    double       velocity_l2       = 0.;
    double       displacement_l2   = 0.;
    double       displacement_h1   = 0.;
    double       velocity_solid_l2 = 0.;
    double       multiplier_l2     = 0.;
    double       constraint        = 0.;
  };

  std::string
  function_constants()
  {
    std::ostringstream constants;
    constants << "a0=" << reference_area(Parameters{}) << ", "
              << "r0=" << Parameters{}.reference_r << ", "
              << "r1=" << Parameters{}.outer_r << ", "
              << "amp=" << Parameters{}.area_amplitude << ", "
              << "om=" << Parameters{}.omega << ", "
              << "k=" << wave_number(Parameters{}) << ", "
              << "half=" << Parameters{}.length / 2. << ", "
              << "rho=" << Parameters{}.solid_density << ", "
              << "mu=" << Parameters{}.shear_modulus << ", "
              << "lam=" << Parameters{}.lame_lambda;
    return constants.str();
  }

  std::string
  displacement_expression()
  {
    const std::string r2  = "(y*y+z*z)";
    const std::string r   = "sqrt(" + r2 + ")";
    const std::string a   = "(a0+amp*(1-cos(om*t))*sin(k*(x+half)))";
    const std::string d   = "(sqrt(" + a + "/pi)-r0)";
    const std::string phi = "(" + r2 + "<=r0*r0 ? " + r +
                            "/r0 : " + "r0/(r0*r0-r1*r1)*(" + r + "-r1*r1/" +
                            r + "))";
    const std::string radial = "(" + d + "*" + phi + ")";
    return "0;(" + r2 + "==0 ? 0 : " + radial + "*y/" + r + ");(" + r2 +
           "==0 ? 0 : " + radial + "*z/" + r + ")";
  }

  std::string
  spatial_displacement_expression()
  {
    const std::string r2  = "(y*y+z*z)";
    const std::string r   = "sqrt(" + r2 + ")";
    const std::string s   = "(x+half)";
    const std::string a   = "(a0+amp*sin(k*" + s + "))";
    const std::string d   = "(sqrt(" + a + "/pi)-r0)";
    const std::string phi = "(" + r2 + "<=r0*r0 ? " + r +
                            "/r0 : " + "r0/(r0*r0-r1*r1)*(" + r + "-r1*r1/" +
                            r + "))";
    const std::string radial = "(" + d + "*" + phi + ")";
    return "0;(" + r2 + "==0 ? 0 : " + radial + "*y/" + r + ");(" + r2 +
           "==0 ? 0 : " + radial + "*z/" + r + ")";
  }

  std::string
  acceleration_expression()
  {
    const std::string r2  = "(y*y+z*z)";
    const std::string r   = "sqrt(" + r2 + ")";
    const std::string s   = "(x+half)";
    const std::string a   = "(a0+amp*(1-cos(om*t))*sin(k*" + s + "))";
    const std::string at  = "(amp*om*sin(om*t)*sin(k*" + s + "))";
    const std::string att = "(amp*om*om*cos(om*t)*sin(k*" + s + "))";
    const std::string dtt = "(" + att + "/(2*sqrt(pi*" + a + "))-" + at + "*" +
                            at + "/(4*sqrt(pi)*" + a + "*sqrt(" + a + ")))";
    const std::string phi = "(" + r2 + "<=r0*r0 ? " + r +
                            "/r0 : " + "r0/(r0*r0-r1*r1)*(" + r + "-r1*r1/" +
                            r + "))";
    const std::string radial = "(" + dtt + "*" + phi + ")";
    return "0;(" + r2 + "==0 ? 0 : " + radial + "*y/" + r + ");(" + r2 +
           "==0 ? 0 : " + radial + "*z/" + r + ")";
  }

  std::string
  body_force_expression()
  {
    const std::string r2  = "(y*y+z*z)";
    const std::string r   = "sqrt(" + r2 + ")";
    const std::string s   = "(x+half)";
    const std::string q   = "(1-cos(om*t))";
    const std::string a   = "(a0+amp*" + q + "*sin(k*" + s + "))";
    const std::string as  = "(amp*" + q + "*k*cos(k*" + s + "))";
    const std::string ass = "(-amp*" + q + "*k*k*sin(k*" + s + "))";
    const std::string att = "(amp*om*om*cos(om*t)*sin(k*" + s + "))";
    const std::string dss = "(" + ass + "/(2*sqrt(pi*" + a + "))-" + as + "*" +
                            as + "/(4*sqrt(pi)*" + a + "*sqrt(" + a + ")))";
    const std::string dtt = "(" + att + "/(2*sqrt(pi*" + a + "))-" +
                            "(amp*om*sin(om*t)*sin(k*" + s +
                            "))*(amp*om*sin(om*t)*sin(k*" + s +
                            "))/(4*sqrt(pi)*" + a + "*sqrt(" + a + ")))";
    const std::string phi = "(" + r2 + "<=r0*r0 ? " + r +
                            "/r0 : " + "r0/(r0*r0-r1*r1)*(" + r + "-r1*r1/" +
                            r + "))";
    const std::string div = "(" + r2 + "<=r0*r0 ? 2/r0 : 2*r0/(r0*r0-r1*r1))";
    const std::string radial = "((rho*" + dtt + "-mu*" + dss + ")*" + phi + ")";
    const std::string ds     = "(" + as + "/(2*sqrt(pi*" + a + ")))";
    const std::string axial  = "(-(mu+lam)*" + ds + "*" + div + ")";
    return axial + ";(" + r2 + "==0 ? 0 : " + radial + "*y/" + r + ");(" + r2 +
           "==0 ? 0 : " + radial + "*z/" + r + ")";
  }

  std::string
  spatial_body_force_expression()
  {
    const std::string r2  = "(y*y+z*z)";
    const std::string r   = "sqrt(" + r2 + ")";
    const std::string s   = "(x+half)";
    const std::string a   = "(a0+amp*sin(k*" + s + "))";
    const std::string as  = "(amp*k*cos(k*" + s + "))";
    const std::string ass = "(-amp*k*k*sin(k*" + s + "))";
    const std::string dss = "(" + ass + "/(2*sqrt(pi*" + a + "))-" + as + "*" +
                            as + "/(4*sqrt(pi)*" + a + "*sqrt(" + a + ")))";
    const std::string phi = "(" + r2 + "<=r0*r0 ? " + r +
                            "/r0 : " + "r0/(r0*r0-r1*r1)*(" + r + "-r1*r1/" +
                            r + "))";
    const std::string div = "(" + r2 + "<=r0*r0 ? 2/r0 : 2*r0/(r0*r0-r1*r1))";
    const std::string radial = "(-mu*" + dss + "*" + phi + ")";
    const std::string ds     = "(" + as + "/(2*sqrt(pi*" + a + ")))";
    const std::string axial  = "(-(mu+lam)*" + ds + "*" + div + ")";
    return axial + ";(" + r2 + "==0 ? 0 : " + radial + "*y/" + r + ");(" + r2 +
           "==0 ? 0 : " + radial + "*z/" + r + ")";
  }

  std::string
  replace_symbol(std::string        expression,
                 const std::string &symbol,
                 const std::string &value)
  {
    std::size_t position = 0;
    while ((position = expression.find(symbol, position)) != std::string::npos)
      {
        expression.replace(position, symbol.size(), value);
        position += value.size();
      }
    return expression;
  }

  std::string
  number(const double value)
  {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
  }

  std::string
  flow_area_expression()
  {
    const Parameters par;
    return number(reference_area(par)) + "+" + number(par.area_amplitude) +
           "*(1-cos(" + number(par.omega) + "*t))*sin(" +
           number(wave_number(par)) + "*(x+" + number(par.length / 2.) + "))";
  }

  std::string
  flow_velocity_expression()
  {
    const Parameters  par;
    const std::string a = "(" + flow_area_expression() + ")";
    return "-" + number(par.area_amplitude) + "*" + number(par.omega) +
           "*sin(" + number(par.omega) + "*t)/" + number(wave_number(par)) +
           "*(1-cos(" + number(wave_number(par)) + "*(x+" +
           number(par.length / 2.) + ")))/" + a;
  }

  std::string
  spatial_flow_area_expression()
  {
    const Parameters par;
    return number(reference_area(par)) + "+" + number(par.area_amplitude) +
           "*sin(" + number(wave_number(par)) + "*(x+" +
           number(par.length / 2.) + "))";
  }

  std::string
  spatial_flow_rhs_expression()
  {
    const Parameters   par;
    const std::string  a     = "(a0+amp*sin(k*(x+half)))";
    const std::string  as    = "(amp*k*cos(k*(x+half)))";
    const double       kappa = traction_jump_coefficient(par);
    std::ostringstream ext;
    ext << std::setprecision(17) << kappa;
    std::string expression =
      "0;((4*sqrt(pi)*" + number(par.tube_E) + "*" + number(par.tube_h_wall) +
      "/(3*" + number(par.tube_a_d) + "*2*sqrt(" + a + "))*" + as + ")+ (" +
      ext.str() + "*(" + as + "/(2*sqrt(pi*" + a + "))))/1060)";
    expression = replace_symbol(expression, "a0", number(reference_area(par)));
    expression = replace_symbol(expression, "amp", number(par.area_amplitude));
    expression = replace_symbol(expression, "k", number(wave_number(par)));
    expression = replace_symbol(expression, "half", number(par.length / 2.));
    expression = replace_symbol(expression, "pi", number(numbers::PI));
    return expression;
  }

  std::string
  flow_velocity_time_derivative_expression()
  {
    const Parameters  par;
    const std::string a = "(" + flow_area_expression() + ")";
    const std::string h = "(1-cos(" + number(wave_number(par)) + "*(x+" +
                          number(par.length / 2.) + ")))";
    const std::string at = "(" + number(par.area_amplitude) + "*" +
                           number(par.omega) + "*sin(" + number(par.omega) +
                           "*t)*sin(" + number(wave_number(par)) + "*(x+" +
                           number(par.length / 2.) + ")))";
    return "-" + number(par.area_amplitude) + "*" + number(par.omega) + "*" +
           number(par.omega) + "*cos(" + number(par.omega) + "*t)/" +
           number(wave_number(par)) + "*" + h + "/" + a + "+" +
           number(par.area_amplitude) + "*" + number(par.omega) + "*sin(" +
           number(par.omega) + "*t)/" + number(wave_number(par)) + "*" + h +
           "*" + at + "/(" + a + "*" + a + ")";
  }

  std::string
  flow_rhs_expression()
  {
    // The native equations use U_t+U U_s+p_s/rho=RHS_U.  This is the
    // continuous manufactured source for the exact area/velocity pair.
    const Parameters  par;
    const std::string s  = "(x+half)";
    const std::string a  = "(a0+amp*(1-cos(om*t))*sin(k*" + s + "))";
    const std::string at = "(amp*om*sin(om*t)*sin(k*" + s + "))";
    const std::string as = "(amp*(1-cos(om*t))*k*cos(k*" + s + "))";
    const std::string h  = "(1-cos(k*" + s + "))";
    const std::string hs = "(k*sin(k*" + s + "))";
    const std::string u  = "(-amp*om*sin(om*t)/k*" + h + "/" + a + ")";
    const std::string ut = "(-amp*om*om*cos(om*t)/k*" + h + "/" + a +
                           "+amp*om*sin(om*t)/k*" + h + "*" + at + "/(" + a +
                           "*" + a + "))";
    const std::string us = "(-amp*om*sin(om*t)/k*(" + hs + "/" + a + "-" + h +
                           "*" + as + "/(" + a + "*" + a + ")))";
    const std::string tube_ps =
      "(4*sqrt(pi)*" + number(par.tube_E) + "*" + number(par.tube_h_wall) +
      "/(3*" + number(par.tube_a_d) + "*2*sqrt(" + a + "))*" + as + ")";
    const double       kappa = traction_jump_coefficient(par);
    std::ostringstream ext;
    ext << std::setprecision(17) << kappa;
    const std::string external_ps =
      "(" + ext.str() + "*(" + as + "/(2*sqrt(pi*" + a + "))))";
    std::string expression = "0;" + ut + "+" + u + "*" + us + "+(" + tube_ps +
                             "+" + external_ps + ")/1060";
    expression = replace_symbol(expression, "a0", number(reference_area(par)));
    expression = replace_symbol(expression, "amp", number(par.area_amplitude));
    expression = replace_symbol(expression, "om", number(par.omega));
    expression = replace_symbol(expression, "k", number(wave_number(par)));
    expression = replace_symbol(expression, "half", number(par.length / 2.));
    expression = replace_symbol(expression, "pi", number(numbers::PI));
    return expression;
  }

  void
  initialize_exact_flow_function(FunctionParser<3> &result,
                                 const double       time,
                                 const bool         spatial)
  {
    const auto area_expression =
      spatial ? spatial_flow_area_expression() : flow_area_expression();
    const auto velocity_expression = spatial ? "0" : flow_velocity_expression();
    result.initialize(FunctionParser<3>::default_variable_names() + ",t",
                      area_expression + ";" + velocity_expression + ";" +
                        area_expression + ";" + velocity_expression,
                      {{"pi", numbers::PI}},
                      true);
    result.set_time(time);
  }

  std::string
  write_mms_parameter_file(const unsigned int level,
                           const double       final_time,
                           const bool         spatial)
  {
    const auto filename = TestPaths::output_path(
      "metric-flow-x-elastodynamics-mms/parameters-" + std::to_string(level) +
      "-" + std::to_string(final_time) + ".prm");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(filename.parent_path());
        std::ofstream output(filename);
        output << "include "
               << TestPaths::parameter_path(
                    "gtests/parameters/metric_flow_x_elastodynamics.prm")
               << "\n";
        const auto area_expression =
          spatial ? spatial_flow_area_expression() : flow_area_expression();
        const auto velocity_expression =
          spatial ? "0" : flow_velocity_expression();
        output << "subsection Functions\n"
               << "  set RHS expression = "
               << (spatial ? spatial_flow_rhs_expression() :
                             flow_rhs_expression())
               << "\n"
               << "  set Exact solution = " << area_expression << ";"
               << velocity_expression << ";" << area_expression << ";"
               << velocity_expression << "\n"
               << "end\n";
        output << "subsection MetricFlowSystem<1, 3>\n"
               << "  set Number of global refinement = " << level << "\n"
               << "  set Vtk file path for mesh input = "
               << TestPaths::data_filename(
                    "metric_flow_x/single_vessel_centered.vtk")
               << "\nend\n";
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return filename.string();
  }

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

  struct MMSFixture
  {
    MMSFixture(const unsigned int flow_level,
               const double       final_time,
               const bool         spatial     = false,
               const unsigned int n_steps     = 1,
               const unsigned int solid_level = numbers::invalid_unsigned_int)
      : spatial_case(spatial)
    {
      ParameterAcceptor::clear();
      flow_problem = std::make_unique<FlowProblem>(MPI_COMM_WORLD);
      reset_parameter_handler_to_root(ParameterAcceptor::prm);
      flow_time = std::make_unique<TimeParameters>();
      reset_parameter_handler_to_root(ParameterAcceptor::prm);
      solid_parameters =
        std::make_unique<ElastodynamicsParameters<3>>("/Elastodynamics/",
                                                      flow_time.get());
      reset_parameter_handler_to_root(ParameterAcceptor::prm);
      wall_lift = std::make_unique<WallObservable::Lift>(
        "/MetricFlowX vessel wall lift/");
      reset_parameter_handler_to_root(ParameterAcceptor::prm);

      const auto parameter_file =
        write_mms_parameter_file(flow_level, final_time, spatial_case);
      initialize_parameters(parameter_file);
      flow_problem->initialize_params(parameter_file);

      const Parameters par;
      flow_time->initial_time      = 0.;
      flow_time->final_time        = final_time;
      flow_time->time_step         = final_time / n_steps;
      flow_time->number_of_steps   = n_steps;
      flow_time->initial_step_size = final_time / n_steps;
      flow_time->minimum_step_size =
        std::getenv("IMMERSX_RUN_MMS_STUDIES") != nullptr ?
          final_time / n_steps :
          final_time * 1.e-6;
      flow_time->maximum_order                 = 1;
      flow_time->maximum_non_linear_iterations = 3;
      const bool study = std::getenv("IMMERSX_RUN_MMS_STUDIES") != nullptr;
      // The registered smoke gate is deliberately inexpensive.  Opt-in
      // studies use solver tolerances suitable for measuring discretization
      // errors; their rates remain diagnostic until the production linear
      // solver scales on the finest coupled levels.
      flow_time->absolute_tolerance = study ? 1.e-8 : 1.e-2;
      flow_time->relative_tolerance = study ? 1.e-8 : 1.e-2;
      flow_time->ls_norm_factor     = 1.e-2;
      flow_time->output_frequency   = 0;

      solid_parameters->initial_refinement =
        solid_level == numbers::invalid_unsigned_int ? flow_level : solid_level;
      solid_parameters->name_of_grid = "subdivided_cylinder";
      solid_parameters->arguments_for_grid =
        "4:" + std::to_string(par.outer_r) + ":" +
        std::to_string(par.length / 2.);
      solid_parameters->density       = par.solid_density;
      solid_parameters->lame_mu       = par.shear_modulus;
      solid_parameters->lame_lambda   = par.lame_lambda;
      solid_parameters->dirichlet_ids = {0};

      const auto constants = function_constants();
      set_parsed_function(solid_parameters->initial_displacement,
                          spatial_case ? spatial_displacement_expression() :
                                         displacement_expression(),
                          constants);
      set_parsed_function(solid_parameters->initial_velocity,
                          "0;0;0",
                          constants);
      set_parsed_function(solid_parameters->displacement_boundary,
                          "0;0;0",
                          constants);
      set_parsed_function(solid_parameters->velocity_boundary,
                          "0;0;0",
                          constants);
      set_parsed_function(solid_parameters->body_force,
                          spatial_case ? spatial_body_force_expression() :
                                         body_force_expression(),
                          constants);
      set_parsed_function(solid_parameters->exact_solution,
                          spatial_case ? spatial_displacement_expression() :
                                         displacement_expression(),
                          constants);

      wall_lift->section.inclusion_degree      = 1;
      wall_lift->section.refinement_level      = 1;
      wall_lift->section.selected_coefficients = {3u, 7u};
      wall_lift->section.n_q_points            = 8;
      wall_lift->representative_n_q_points     = 2;

      solid_problem = std::make_unique<SolidProblem>(*solid_parameters);
      solid_problem->make_grid();
      solid_problem->setup_fe();
      solid_problem->setup_system();
      solid_problem->assemble_operators();
      solid_problem->set_initial_conditions();
      flow_problem->setup();

      adapter = std::make_unique<Adapter>(*flow_time, MPI_COMM_WORLD);
      solid_fields.emplace(adapter->add(*solid_problem, "elastodynamics"));
      flow_fields.emplace(
        adapter->add(metric_flow_x(*flow_problem), "blood-flow"));
      solid_space = std::make_unique<FESpaceView<3, 3>>(
        solid_problem->dof_handler(),
        solid_problem->mapping(),
        solid_problem->constraints(),
        &solid_problem->locally_relevant_dofs());
      solid_field = std::make_unique<SolidField>(
        solid_space->field(solid_fields->fields().displacement,
                           "displacement",
                           FEValuesExtractors::Vector(0)));
      wall_observable =
        std::make_unique<WallObservable>(*flow_problem,
                                         flow_fields->fields().area,
                                         flow_fields->fields().area_components,
                                         *wall_lift);
      interaction =
        std::make_unique<Interaction>(*solid_field, *wall_observable);
      const auto lambda_field = interaction->multiplier_field();
      const auto wall_displacement =
        interaction->radial_displacement(*wall_lift);
      const auto radial_law = wall_observable->radial_law();
      const auto lambda_wall =
        lift(value(lambda_field),
             *wall_lift,
             SourceThicknessEvaluator<3>(
               [radial_law](const dealii::Point<3> &point,
                            const double            time,
                            const std::vector<double> &) {
                 return std::sqrt(radial_law.resting_area(point, time) /
                                  dealii::numbers::PI);
               }));
      const auto kinematic_constraint =
        make_constraint(weak_term(value(*solid_field), lambda_wall) -
                        weak_term(wall_displacement, lambda_field));
      coupling_fields =
        adapter->add(kinematic_constraint, "vessel-wall").fields();
      adapter->add(*interaction,
                   "vessel-wall-pressure",
                   flow_fields->fields().state,
                   coupling_fields.multiplier);
    }

    void
    initialize(GlobalVector &state,
               GlobalVector &state_dot,
               const bool    use_exact_state = true)
    {
      auto displacement = solid_problem->displacement();
      auto velocity     = solid_problem->velocity();
      adapter->field(state, solid_fields->fields().displacement) = displacement;
      adapter->field(state, solid_fields->fields().velocity)     = velocity;
      auto &flow_state = adapter->field(state, flow_fields->fields().state);
      if (!use_exact_state)
        {
          flow_problem->initialize_state(flow_state, 0.);
          flow_problem->initialize_state_derivative(
            adapter->field(state_dot, flow_fields->fields().state), 0.);
          adapter->field(state_dot, solid_fields->fields().displacement) =
            velocity;
          FlowVector acceleration;
          solid_problem->initial_acceleration(acceleration);
          adapter->field(state_dot, solid_fields->fields().velocity) =
            acceleration;
          set_exact_multiplier(state, 0.);
          adapter->field(state_dot, coupling_fields.multiplier) = 0.;
          return;
        }
      FunctionParser<3> exact_solution(4);
      const auto        area_expression =
        spatial_case ? spatial_flow_area_expression() : flow_area_expression();
      const auto velocity_expression =
        spatial_case ? "0" : flow_velocity_expression();
      exact_solution.initialize(FunctionParser<3>::default_variable_names() +
                                  ",t",
                                area_expression + ";" + velocity_expression +
                                  ";" + area_expression + ";" +
                                  velocity_expression,
                                {{"pi", numbers::PI}},
                                true);
      exact_solution.set_time(0.);
      FlowVector exact_fe_state;
      exact_fe_state.reinit(flow_problem->dof_handler().locally_owned_dofs(),
                            MPI_COMM_WORLD);
      VectorTools::interpolate(flow_problem->dof_handler(),
                               exact_solution,
                               exact_fe_state);
      exact_fe_state.compress(VectorOperation::insert);
      flow_state = 0.;
      for (const auto index : flow_problem->dof_handler().locally_owned_dofs())
        flow_state[index] = exact_fe_state[index];
      flow_state.compress(VectorOperation::insert);
      flow_problem->initialize_trace_unknowns(flow_state, 0.);
      flow_problem->initialize_state_derivative(
        adapter->field(state_dot, flow_fields->fields().state), 0.);
      FunctionParser<3> exact_derivative(4);
      exact_derivative.initialize(
        FunctionParser<3>::default_variable_names() + ",t",
        spatial_case ? "0;0;0;0" :
                       "0;" + flow_velocity_time_derivative_expression() +
                         ";0;" + flow_velocity_time_derivative_expression(),
        {{"pi", numbers::PI}},
        true);
      exact_derivative.set_time(0.);
      FlowVector exact_fe_derivative;
      exact_fe_derivative.reinit(
        flow_problem->dof_handler().locally_owned_dofs(), MPI_COMM_WORLD);
      VectorTools::interpolate(flow_problem->dof_handler(),
                               exact_derivative,
                               exact_fe_derivative);
      exact_fe_derivative.compress(VectorOperation::insert);
      auto &flow_state_dot =
        adapter->field(state_dot, flow_fields->fields().state);
      for (const auto index : flow_problem->dof_handler().locally_owned_dofs())
        flow_state_dot[index] = exact_fe_derivative[index];
      flow_state_dot.compress(VectorOperation::insert);
      adapter->field(state_dot, solid_fields->fields().displacement) = velocity;
      FunctionParser<3> exact_acceleration(3);
      std::string       acceleration_formula = acceleration_expression();
      const Parameters  par;
      acceleration_formula =
        replace_symbol(acceleration_formula, "a0", number(reference_area(par)));
      acceleration_formula =
        replace_symbol(acceleration_formula, "r0", number(par.reference_r));
      acceleration_formula =
        replace_symbol(acceleration_formula, "r1", number(par.outer_r));
      acceleration_formula =
        replace_symbol(acceleration_formula, "amp", number(par.area_amplitude));
      acceleration_formula =
        replace_symbol(acceleration_formula, "om", number(par.omega));
      acceleration_formula =
        replace_symbol(acceleration_formula, "k", number(wave_number(par)));
      acceleration_formula =
        replace_symbol(acceleration_formula, "half", number(par.length / 2.));
      acceleration_formula =
        replace_symbol(acceleration_formula, "pi", number(numbers::PI));
      exact_acceleration.initialize(
        FunctionParser<3>::default_variable_names() + ",t",
        spatial_case ? "0;0;0" : acceleration_formula,
        {{"pi", numbers::PI}},
        true);
      exact_acceleration.set_time(0.);
      SolidVector acceleration;
      acceleration.reinit(solid_problem->locally_owned_dofs(), MPI_COMM_WORLD);
      VectorTools::interpolate(solid_problem->dof_handler(),
                               exact_acceleration,
                               acceleration);
      acceleration.compress(VectorOperation::insert);
      adapter->field(state_dot, solid_fields->fields().velocity) = acceleration;
      set_exact_multiplier(state, 0.);
      adapter->field(state_dot, coupling_fields.multiplier) = 0.;
    }

    FlowVector
    exact_flow_fe_state(const double time) const
    {
      FlowVector owned;
      owned.reinit(flow_problem->dof_handler().locally_owned_dofs(),
                   MPI_COMM_WORLD);
      FunctionParser<3> exact(4);
      initialize_exact_flow_function(exact, time, spatial_case);
      VectorTools::interpolate(flow_problem->dof_handler(), exact, owned);
      owned.compress(VectorOperation::insert);
      FlowVector result;
      result.reinit(flow_problem->dof_handler().locally_owned_dofs(),
                    DoFTools::extract_locally_relevant_dofs(
                      flow_problem->dof_handler()),
                    MPI_COMM_WORLD);
      result = owned;
      result.update_ghost_values();
      return result;
    }

    FlowVector
    numerical_flow_fe_state(const GlobalVector &state) const
    {
      FlowVector owned;
      owned.reinit(flow_problem->dof_handler().locally_owned_dofs(),
                   MPI_COMM_WORLD);
      const auto &flow_state =
        adapter->field(state, flow_fields->fields().state);
      for (const auto index : flow_problem->dof_handler().locally_owned_dofs())
        owned[index] = flow_state[index];
      owned.compress(VectorOperation::insert);
      FlowVector result;
      result.reinit(flow_problem->dof_handler().locally_owned_dofs(),
                    DoFTools::extract_locally_relevant_dofs(
                      flow_problem->dof_handler()),
                    MPI_COMM_WORLD);
      result = owned;
      result.update_ghost_values();
      return result;
    }

    double
    flow_error(const GlobalVector &state,
               const double        time,
               const unsigned int  component) const
    {
      auto              numerical = numerical_flow_fe_state(state);
      FunctionParser<3> exact(4);
      initialize_exact_flow_function(exact, time, spatial_case);
      Vector<double> cellwise_error(
        flow_problem->triangulation().n_active_cells());
      ComponentSelectFunction<3> mask(component, 4);
      VectorTools::integrate_difference(StaticMappingQ1<1, 3>::mapping,
                                        flow_problem->dof_handler(),
                                        numerical,
                                        exact,
                                        cellwise_error,
                                        QGauss<1>(4),
                                        VectorTools::L2_norm,
                                        &mask);
      return VectorTools::compute_global_error(flow_problem->triangulation(),
                                               cellwise_error,
                                               VectorTools::L2_norm,
                                               2.);
    }

    double
    solid_error(const GlobalVector &state,
                const double        time,
                const bool          velocity,
                const bool          h1 = false) const
    {
      const auto &owned_numerical =
        adapter->field(state,
                       velocity ? solid_fields->fields().velocity :
                                  solid_fields->fields().displacement);
      SolidVector numerical;
      numerical.reinit(solid_problem->locally_owned_dofs(),
                       solid_problem->locally_relevant_dofs(),
                       MPI_COMM_WORLD);
      SolidVector owned;
      owned.reinit(solid_problem->locally_owned_dofs(), MPI_COMM_WORLD);
      for (const auto index : solid_problem->locally_owned_dofs())
        owned[index] = owned_numerical[index];
      owned.compress(VectorOperation::insert);
      numerical = owned;
      numerical.update_ghost_values();
      auto exact =
        solid_exact_function(Parameters{}, time, spatial_case, velocity);
      Vector<double> cellwise_error(
        solid_problem->triangulation().n_active_cells());
      VectorTools::integrate_difference(solid_problem->mapping(),
                                        solid_problem->dof_handler(),
                                        numerical,
                                        exact,
                                        cellwise_error,
                                        QGauss<3>(4),
                                        h1 ? VectorTools::H1_seminorm :
                                             VectorTools::L2_norm);
      return VectorTools::compute_global_error(solid_problem->triangulation(),
                                               cellwise_error,
                                               h1 ? VectorTools::H1_seminorm :
                                                    VectorTools::L2_norm,
                                               2.);
    }

    static unsigned int
    global_dof_count(const IndexSet &locally_owned)
    {
      return static_cast<unsigned int>(
        Utilities::MPI::sum(locally_owned.n_elements(), MPI_COMM_WORLD));
    }

    ErrorRecord
    errors(const GlobalVector &state,
           const GlobalVector &state_dot,
           const double        time,
           const unsigned int  level) const
    {
      ErrorRecord result;
      result.level  = level;
      result.h_flow = 0.;
      for (const auto &cell :
           flow_problem->triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          result.h_flow = std::max(result.h_flow, cell->diameter());
      result.h_flow  = Utilities::MPI::max(result.h_flow, MPI_COMM_WORLD);
      result.h_solid = 0.;
      for (const auto &cell :
           solid_problem->triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          result.h_solid = std::max(result.h_solid, cell->diameter());
      result.h_solid    = Utilities::MPI::max(result.h_solid, MPI_COMM_WORLD);
      result.flow_dofs  = global_dof_count(flow_problem->locally_owned_dofs());
      result.solid_dofs = global_dof_count(solid_problem->locally_owned_dofs());
      result.multiplier_dofs =
        global_dof_count(interaction->multiplier_locally_owned_dofs());
      result.area_l2           = flow_error(state, time, 0);
      result.velocity_l2       = flow_error(state, time, 1);
      result.displacement_l2   = solid_error(state, time, false);
      result.displacement_h1   = solid_error(state, time, false, true);
      result.velocity_solid_l2 = solid_error(state, time, true);

      auto exact_state = adapter->make_state();
      exact_state      = state;
      set_exact_multiplier(exact_state, time);
      const auto &numerical_multiplier =
        adapter->field(state, coupling_fields.multiplier);
      const auto &exact_multiplier_field =
        adapter->field(exact_state, coupling_fields.multiplier);
      auto multiplier_difference = numerical_multiplier;
      multiplier_difference -= exact_multiplier_field;
      result.multiplier_l2 = multiplier_difference.l2_norm();

      auto residual = adapter->make_state();
      adapter->solver().residual(time, state, state_dot, residual);
      result.constraint =
        adapter->field(residual, coupling_fields.multiplier).l2_norm();
      return result;
    }

    void
    set_exact_multiplier(GlobalVector &state, const double time) const
    {
      auto &multiplier = adapter->field(state, coupling_fields.multiplier);
      multiplier       = 0.;
      const Parameters                            par;
      std::map<types::global_dof_index, Point<3>> support_points;
      MappingQ1<1, 3>                             mapping;
      DoFTools::map_dofs_to_support_points(mapping,
                                           flow_problem->dof_handler(),
                                           support_points);
      for (const auto &point : wall_observable->points())
        for (unsigned int i = 0; i < point.dof_indices.size(); ++i)
          {
            const auto dof = point.dof_indices[i];
            const auto it  = support_points.find(dof);
            if (it != support_points.end() &&
                multiplier.locally_owned_elements().is_element(
                  point.multiplier_dof_indices[i]))
              multiplier[point.multiplier_dof_indices[i]] =
                spatial_case ?
                  spatial_multiplier(par, it->second[0] + par.length / 2.) :
                  exact_multiplier(par, it->second[0] + par.length / 2., time);
          }
      multiplier.compress(VectorOperation::insert);
      multiplier.update_ghost_values();
    }

    std::unique_ptr<FlowProblem>                 flow_problem;
    std::unique_ptr<TimeParameters>              flow_time;
    std::unique_ptr<ElastodynamicsParameters<3>> solid_parameters;
    std::unique_ptr<SolidProblem>                solid_problem;
    std::unique_ptr<WallObservable::Lift>        wall_lift;
    std::unique_ptr<FESpaceView<3, 3>>           solid_space;
    std::unique_ptr<SolidField>                  solid_field;
    std::unique_ptr<WallObservable>              wall_observable;
    std::unique_ptr<Interaction>                 interaction;
    std::unique_ptr<Adapter>                     adapter;
    std::optional<decltype(std::declval<Adapter &>().add(
      std::declval<const SolidProblem &>()))>
      solid_fields;
    std::optional<decltype(std::declval<Adapter &>().add(
      std::declval<const decltype(metric_flow_x(
        std::declval<FlowProblem &>())) &>()))>
                     flow_fields;
    ConstraintFields coupling_fields;
    bool             spatial_case = false;
  };

  ErrorRecord
  run_transient_case(const unsigned int flow_level,
                     const unsigned int solid_level,
                     const unsigned int n_steps,
                     const unsigned int logical_level,
                     const double       final_time = 0.1)
  {
    MMSFixture fixture(flow_level, final_time, false, n_steps, solid_level);
    auto       state     = fixture.adapter->make_state();
    auto       state_dot = fixture.adapter->make_state();
    fixture.initialize(state, state_dot);
    EXPECT_GT(fixture.adapter->solve(state, state_dot), 0u);
    EXPECT_TRUE(std::isfinite(state.l2_norm()));
    auto residual = fixture.adapter->make_state();
    fixture.adapter->solver().residual(final_time, state, state_dot, residual);
    EXPECT_TRUE(std::isfinite(residual.l2_norm()));
    auto result = fixture.errors(state, state_dot, final_time, logical_level);
    result.n_steps = n_steps;
    return result;
  }

  ErrorRecord
  run_stationary_case(const unsigned int flow_level,
                      const unsigned int solid_level,
                      const unsigned int logical_level,
                      const double       final_time = 0.1)
  {
    MMSFixture fixture(flow_level, final_time, true, 1, solid_level);
    auto       state     = fixture.adapter->make_state();
    auto       state_dot = fixture.adapter->make_state();
    fixture.initialize(state, state_dot);
    EXPECT_GT(fixture.adapter->solve(state, state_dot), 0u);
    EXPECT_TRUE(std::isfinite(state.l2_norm()));
    auto residual = fixture.adapter->make_state();
    fixture.adapter->solver().residual(final_time, state, state_dot, residual);
    EXPECT_TRUE(std::isfinite(residual.l2_norm()));
    auto result = fixture.errors(state, state_dot, final_time, logical_level);
    result.n_steps = 1;
    return result;
  }

  void
  write_error_table(const std::string              &name,
                    const std::vector<ErrorRecord> &records)
  {
    const auto rate = [](const double previous,
                         const double current,
                         const double refinement_ratio) {
      return previous > 0. && current > 0. && refinement_ratio > 1. ?
               std::log(previous / current) / std::log(refinement_ratio) :
               std::numeric_limits<double>::quiet_NaN();
    };
    const auto refinement_ratio = [&records, &name](const unsigned int level) {
      if (level == 0)
        return 1.;
      if (name == "temporal-study")
        return static_cast<double>(records[level].n_steps) /
               records[level - 1].n_steps;
      return records[level - 1].h_flow / records[level].h_flow;
    };
    const auto filename = TestPaths::output_path(
      "metric-flow-x-elastodynamics-mms/" + name + ".csv");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(filename.parent_path());
        std::ofstream output(filename);
        output << "level,n_steps,h_flow,h_solid,flow_dofs,solid_dofs,"
                  "multiplier_dofs,area_l2,velocity_l2,displacement_l2,"
                  "displacement_h1,velocity_solid_l2,multiplier_l2,constraint,"
                  "area_rate,velocity_rate,displacement_rate,"
                  "displacement_h1_rate,velocity_solid_rate,multiplier_rate,"
                  "constraint_rate\n";
        for (const auto &record : records)
          {
            const auto level = record.level;
            output << record.level << ',' << record.n_steps << ','
                   << std::setprecision(17) << record.h_flow << ','
                   << record.h_solid << ',' << record.flow_dofs << ','
                   << record.solid_dofs << ',' << record.multiplier_dofs << ','
                   << record.area_l2 << ',' << record.velocity_l2 << ','
                   << record.displacement_l2 << ',' << record.displacement_h1
                   << ',' << record.velocity_solid_l2 << ','
                   << record.multiplier_l2 << ',' << record.constraint;
            if (level > 0)
              {
                const auto ratio = refinement_ratio(level);
                output << ','
                       << rate(records[level - 1].area_l2,
                               record.area_l2,
                               ratio)
                       << ','
                       << rate(records[level - 1].velocity_l2,
                               record.velocity_l2,
                               ratio)
                       << ','
                       << rate(records[level - 1].displacement_l2,
                               record.displacement_l2,
                               ratio)
                       << ','
                       << rate(records[level - 1].displacement_h1,
                               record.displacement_h1,
                               ratio)
                       << ','
                       << rate(records[level - 1].velocity_solid_l2,
                               record.velocity_solid_l2,
                               ratio)
                       << ','
                       << rate(records[level - 1].multiplier_l2,
                               record.multiplier_l2,
                               ratio)
                       << ','
                       << rate(records[level - 1].constraint,
                               record.constraint,
                               ratio);
              }
            else
              output << ",nan,nan,nan,nan,nan,nan,nan";
            output << '\n';
          }

        std::cout
          << "\n"
          << name << "\n"
          << "level  steps       h_flow       area_L2       velocity_L2"
             "       displacement_L2       displacement_H1"
             "       velocity_solid_L2       multiplier_L2       constraint\n";
        for (const auto &record : records)
          {
            std::cout << std::setw(5) << record.level << std::setw(7)
                      << record.n_steps << std::scientific
                      << std::setprecision(6) << std::setw(13) << record.h_flow
                      << std::setw(14) << record.area_l2 << std::setw(15)
                      << record.velocity_l2 << std::setw(22)
                      << record.displacement_l2 << std::setw(22)
                      << record.displacement_h1 << std::setw(24)
                      << record.velocity_solid_l2 << std::setw(18)
                      << record.multiplier_l2 << std::setw(15)
                      << record.constraint;
            if (record.level > 0)
              {
                const auto ratio = refinement_ratio(record.level);
                std::cout << "  rates: A="
                          << rate(records[record.level - 1].area_l2,
                                  record.area_l2,
                                  ratio)
                          << " u="
                          << rate(records[record.level - 1].displacement_l2,
                                  record.displacement_l2,
                                  ratio)
                          << " lambda="
                          << rate(records[record.level - 1].multiplier_l2,
                                  record.multiplier_l2,
                                  ratio)
                          << " G="
                          << rate(records[record.level - 1].constraint,
                                  record.constraint,
                                  ratio);
              }
            std::cout << '\n';
          }
      }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  TEST(OneVesselMMSDriver, MPI_ActualTwoWayTransientSolve)
  {
    MMSFixture fixture(0, 0.1);
    auto       state     = fixture.adapter->make_state();
    auto       state_dot = fixture.adapter->make_state();
    fixture.initialize(state, state_dot, true);
    const auto steps = fixture.adapter->solve(state, state_dot);
    EXPECT_GT(steps, 0u);
    EXPECT_TRUE(std::isfinite(state.l2_norm()));

    auto residual = fixture.adapter->make_state();
    fixture.adapter->solver().residual(fixture.flow_time->final_time,
                                       state,
                                       state_dot,
                                       residual);
    EXPECT_TRUE(std::isfinite(residual.l2_norm()));
    const auto errors = fixture.errors(state, state_dot, 0.1, 0);
    std::cout << "MMS errors: A=" << errors.area_l2
              << " U=" << errors.velocity_l2 << " u=" << errors.displacement_l2
              << " H1=" << errors.displacement_h1
              << " V=" << errors.velocity_solid_l2
              << " lambda=" << errors.multiplier_l2
              << " G=" << errors.constraint << std::endl;
  }

  TEST(OneVesselMMSDriver, MPI_ActualFourLevelSpatialStudy)
  {
    if (std::getenv("IMMERSX_RUN_MMS_STUDIES") == nullptr)
      GTEST_SKIP() << "Set IMMERSX_RUN_MMS_STUDIES=1 for the expensive study.";
    std::vector<ErrorRecord> records;
    for (unsigned int level = 0; level < 4; ++level)
      records.push_back(run_stationary_case(level, level, level));
    write_error_table("spatial-study", records);
    for (unsigned int level = 1; level < records.size(); ++level)
      {
        EXPECT_LT(records[level].area_l2, 1.e-6);
        EXPECT_LT(records[level].velocity_l2, 1.e-4);
        EXPECT_LT(records[level].displacement_l2, 1.e-6);
        EXPECT_LT(records[level].displacement_h1, 1.e-4);
        EXPECT_LT(records[level].velocity_solid_l2, 1.e-4);
        EXPECT_LT(records[level].multiplier_l2, 1.e-3);
        EXPECT_LT(records[level].constraint, 1.e-8);
      }
  }

  TEST(OneVesselMMSDriver, MPI_ActualFourLevelTemporalStudy)
  {
    if (std::getenv("IMMERSX_RUN_MMS_STUDIES") == nullptr)
      GTEST_SKIP() << "Set IMMERSX_RUN_MMS_STUDIES=1 for the expensive study.";
    std::vector<ErrorRecord> records;
    for (unsigned int level = 0; level < 4; ++level)
      records.push_back(run_transient_case(0, 0, 1u << level, level));
    write_error_table("temporal-study", records);
    for (unsigned int level = 1; level < records.size(); ++level)
      {
        EXPECT_LT(records[level].area_l2, 1.e-6);
        EXPECT_LT(records[level].velocity_l2, 1.e-4);
        EXPECT_LT(records[level].displacement_l2, 1.e-6);
        EXPECT_LT(records[level].displacement_h1, 1.e-4);
        EXPECT_LT(records[level].velocity_solid_l2, 1.e-4);
        EXPECT_LT(records[level].multiplier_l2, 1.e-3);
        EXPECT_LT(records[level].constraint, 1.e-8);
      }
  }

  TEST(OneVesselMMSDriver, MPI_ActualFourLevelCombinedStudy)
  {
    if (std::getenv("IMMERSX_RUN_MMS_STUDIES") == nullptr)
      GTEST_SKIP() << "Set IMMERSX_RUN_MMS_STUDIES=1 for the expensive study.";
    std::vector<ErrorRecord> records;
    for (unsigned int level = 0; level < 4; ++level)
      records.push_back(run_transient_case(level, level, 1u << level, level));
    write_error_table("combined-study", records);
    for (unsigned int level = 1; level < records.size(); ++level)
      {
        EXPECT_LT(records[level].area_l2, 1.e-6);
        EXPECT_LT(records[level].velocity_l2, 1.e-4);
        EXPECT_LT(records[level].displacement_l2, 1.e-6);
        EXPECT_LT(records[level].displacement_h1, 1.e-4);
        EXPECT_LT(records[level].velocity_solid_l2, 1.e-4);
        EXPECT_LT(records[level].multiplier_l2, 1.e-3);
        EXPECT_LT(records[level].constraint, 1.e-8);
      }
  }

} // namespace

#else

TEST(OneVesselMMSDriver, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
