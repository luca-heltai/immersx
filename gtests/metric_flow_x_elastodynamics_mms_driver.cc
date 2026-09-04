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
#include <immersx/algebra/vessel_wall_interaction.h>
#include <immersx/core/sundials_ida_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastodynamics.h>
#include <immersx/physics/elastodynamics_semidiscrete.h>
#include <immersx/physics/metric_flow_x.h>
#include <immersx/physics/metric_flow_x_vessel_wall_representation.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
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

  using FlowProblem         = MetricFlowX::BloodFlowSystem<1, 3>;
  using FlowVector          = MetricFlowX::VectorType;
  using GlobalVector        = ImmersXLA::MPI::BlockVector;
  using Adapter             = IDAAdapter<FlowVector, GlobalVector>;
  using SolidProblem        = ElastodynamicsSolver<3>;
  using SolidRepresentation = VectorFiniteElementRepresentation<3, 3>;
  using WallRepresentation  = MetricFlowXAreaRadialDisplacementRepresentation;
  using Interaction =
    VesselWallInteraction<SolidRepresentation, WallRepresentation>;

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
    const std::string axial  = "(-(mu+lam)*" + dss + "*" + div + ")";
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
    const std::string axial  = "(-(mu+lam)*" + dss + "*" + div + ")";
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
    MMSFixture(const unsigned int level,
               const double       final_time,
               const bool         spatial = false)
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
      wall_lift = std::make_unique<WallRepresentation::Lift>(
        "/MetricFlowX vessel wall lift/");
      ParticleCouplingParameters<3> search_parameters;
      reset_parameter_handler_to_root(ParameterAcceptor::prm);

      const auto parameter_file =
        write_mms_parameter_file(level, final_time, spatial_case);
      initialize_parameters(parameter_file);
      flow_problem->initialize_params(parameter_file);

      const Parameters par;
      flow_time->initial_time                  = 0.;
      flow_time->final_time                    = final_time;
      flow_time->time_step                     = final_time;
      flow_time->initial_step_size             = final_time;
      flow_time->minimum_step_size             = final_time * 1.e-6;
      flow_time->maximum_order                 = 1;
      flow_time->maximum_non_linear_iterations = 3;
      flow_time->absolute_tolerance            = 1.e-2;
      flow_time->relative_tolerance            = 1.e-2;
      flow_time->output_frequency              = 0;

      solid_parameters->initial_refinement = level;
      solid_parameters->name_of_grid       = "subdivided_cylinder";
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
      solid_representation = std::make_unique<SolidRepresentation>(
        solid_problem->triangulation(),
        solid_problem->dof_handler(),
        solid_problem->locally_owned_dofs(),
        solid_problem->locally_relevant_dofs(),
        solid_problem->constraints(),
        solid_problem->mapping(),
        FEValuesExtractors::Vector(0));
      wall_representation =
        std::make_unique<WallRepresentation>(*flow_problem,
                                             flow_fields->fields().area,
                                             *wall_lift);
      interaction = std::make_unique<Interaction>(*solid_representation,
                                                  *wall_representation,
                                                  search_parameters);
      interaction->assemble();
      coupling_fields = adapter
                          ->add(*interaction,
                                "vessel-wall",
                                solid_fields->fields().displacement,
                                solid_fields->fields().velocity,
                                flow_fields->fields().state)
                          .fields();
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
                            DoFTools::extract_locally_relevant_dofs(
                              flow_problem->dof_handler()),
                            MPI_COMM_WORLD);
      VectorTools::interpolate(flow_problem->dof_handler(),
                               exact_solution,
                               exact_fe_state);
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
        flow_problem->dof_handler().locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(flow_problem->dof_handler()),
        MPI_COMM_WORLD);
      VectorTools::interpolate(flow_problem->dof_handler(),
                               exact_derivative,
                               exact_fe_derivative);
      auto &flow_state_dot =
        adapter->field(state_dot, flow_fields->fields().state);
      for (const auto index : flow_problem->dof_handler().locally_owned_dofs())
        flow_state_dot[index] = exact_fe_derivative[index];
      flow_state_dot.compress(VectorOperation::insert);
      adapter->field(state_dot, solid_fields->fields().displacement) = velocity;
      FlowVector acceleration;
      solid_problem->initial_acceleration(acceleration);
      adapter->field(state_dot, solid_fields->fields().velocity) = acceleration;
      set_exact_multiplier(state, 0.);
      adapter->field(state_dot, coupling_fields.multiplier) = 0.;
    }

    void
    set_exact_multiplier(GlobalVector &state, const double time)
    {
      auto &multiplier = adapter->field(state, coupling_fields.multiplier);
      multiplier       = 0.;
      const Parameters                            par;
      std::map<types::global_dof_index, Point<3>> support_points;
      MappingQ1<1, 3>                             mapping;
      DoFTools::map_dofs_to_support_points(mapping,
                                           flow_problem->dof_handler(),
                                           support_points);
      for (const auto &point : wall_representation->points())
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
    std::unique_ptr<WallRepresentation::Lift>    wall_lift;
    std::unique_ptr<SolidRepresentation>         solid_representation;
    std::unique_ptr<WallRepresentation>          wall_representation;
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
  }

} // namespace

#else

TEST(OneVesselMMSDriver, FeatureMacroIsEnabled)
{
  GTEST_SKIP() << "MetricFlowX or deal.II SUNDIALS support is unavailable.";
}

#endif
