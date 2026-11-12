// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/algebra/mumps_preconditioner.h>
#  include <immersx/core/constraint.h>
#  include <immersx/core/domain.h>
#  include <immersx/core/observable_lift.h>
#  include <immersx/core/overlap_integration_support.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/core/weak_term.h>
#  include <immersx/io/lifted_particle_output.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastic_static_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_pressure.h>
#  include <immersx/physics/metric_flow_x_wall.h>

#  include <cmath>
#  include <cstdlib>
#  include <filesystem>
#  include <fstream>
#  include <iostream>
#  include <memory>
#  include <optional>
#  include <sstream>
#  include <stdexcept>
#  include <string>
#  include <type_traits>
#  include <utility>
#  include <vector>

#  include "metric_flow_x_elastostatic_checkpoint.h"

namespace
{
  using namespace dealii;
  using namespace ImmersX;
  std::string
  metric_flow_parameter(const std::string &name)
  {
    auto &prm = ParameterAcceptor::prm;
    prm.enter_subsection("MetricFlowSystem<1, 3>");
    const auto result = prm.get(name);
    prm.leave_subsection();
    return result;
  }

  std::string
  parameter_time(const double time)
  {
    std::ostringstream value;
    value.precision(17);
    value << time;
    return value.str();
  }

  void
  set_metric_flow_parameter(const std::string &name, const std::string &value)
  {
    auto &prm = ParameterAcceptor::prm;
    prm.enter_subsection("MetricFlowSystem<1, 3>");
    prm.set(name, value);
    prm.leave_subsection();
  }


  // Regression diagnostics use native Vector output; checkpoints use FE
  // transfer.
  template <typename GlobalVector>
  void
  write_regression_state(const GlobalVector &state, const std::string &filename)
  {
    std::ofstream file;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      file.open(filename);
    bool output_ok =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0 || file.good();
    AssertThrow(Utilities::MPI::min(output_ok ? 1u : 0u, MPI_COMM_WORLD) == 1,
                ExcMessage("Cannot open regression state output."));
    for (unsigned int block = 0; block < state.n_blocks(); ++block)
      {
        std::vector<double> local(state.block(block).size(), 0.),
          global(local.size());
        for (const auto index : state.block(block).locally_owned_elements())
          local[index] = state.block(block)[index];
        Utilities::MPI::sum(local, MPI_COMM_WORLD, global);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            Vector<double> serial(global.begin(), global.end());
            serial.print(file, 17, true, true);
          }
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        file.close();
        output_ok = file.good();
      }
    AssertThrow(Utilities::MPI::min(output_ok ? 1u : 0u, MPI_COMM_WORLD) == 1,
                ExcMessage("Cannot finish regression state output."));
  }

  struct Options
  {
    std::string           parameter_file, restart, checkpoint;
    std::optional<double> final_time;
  };

  template <typename Flow, typename = void>
  struct HasCheckpoint : std::false_type
  {};
  template <typename Flow>
  struct HasCheckpoint<
    Flow,
    std::void_t<decltype(std::declval<const Flow &>().save_checkpoint(
      std::declval<const std::string &>(),
      0.,
      std::declval<const MetricFlowX::VectorType &>(),
      std::declval<const MetricFlowX::VectorType &>(),
      std::declval<std::function<void(const std::string &)>>()))>>
    : std::true_type
  {};

  template <typename FlowProblem = MetricFlowX::BloodFlowSystem<1, 3>>
  void
  run_coupled_problem(const Options &options)
  {
    using SolidProblem   = ElasticStaticProblem<3>;
    namespace Checkpoint = elastostatic_checkpoint_detail;
    using FieldVector    = MetricFlowX::VectorType;
    using GlobalVector   = ImmersXLA::MPI::BlockVector;
    using Adapter        = IDAAdapter<FieldVector, GlobalVector>;
    using WallLift       = TensorProductLift<1, 2, 3, 3>;

    using ReferenceField =
      decltype(std::declval<FiniteElementSpaceView<1, 3>>().field("a0"));
    std::unique_ptr<FiniteElementSpace<1, 3>> lambda_space_storage,
      restored_reference_space;
    std::optional<FiniteElementSpaceView<1, 3>> reference_view;
    std::optional<ReferenceField>               reference_field;
    FieldVector reference_area, restored_lambda, restored_lambda_dot;
    ParameterAcceptor::clear();
    FlowProblem            flow_problem(MPI_COMM_WORLD);
    TimeIntervalParameters flow_time("/", false);
    IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA parameters/",
                                 &flow_time);
    // Preserve native differential data and solve for algebraic states and
    // differential rates in the coupled system, rather than solving K*u=f
    // alone.
    ida_parameters.correction_type_at_initial_time = "use_y_diff";
    ElasticStaticParameters<3> solid_parameters;
    // Keep the bound value and its archived parameter entry in sync.
    solid_parameters.domain_parameters.enter_my_subsection(
      ParameterAcceptor::prm);
    ParameterAcceptor::prm.set("Triangulation type", "fullydistributed");
    solid_parameters.domain_parameters.leave_my_subsection(
      ParameterAcceptor::prm);
    WallLift wall_lift("/MetricFlowX vessel wall lift/");
    FiniteElementSpaceParameters<1, 3> lambda_parameters(
      "/MetricFlowX multiplier space/", "auto");
    // Native MetricFlowX owns the mesh configuration; this observer imports a0.
    DomainParameters<1, 3>         flow_domain_parameters("/", false);
    LiftedParticleOutputParameters particle_output(
      "/MetricFlowX coupling/Particle output/");
    ParameterAcceptor linear_solver_parameters(
      "/MetricFlowSystem<1, 3>/IDA parameters/Linear solver/");
    std::string linear_solver = "iterative";
    linear_solver_parameters.add_parameter("Type",
                                           linear_solver,
                                           "Coupled IDA linear solver.",
                                           ParameterAcceptor::prm,
                                           Patterns::Selection(
                                             "iterative|umfpack|mumps"));
    ParameterAcceptor coupling("/MetricFlowX coupling/");
#  ifdef DEAL_II_WITH_MUMPS
    std::string flow_block_solver = "mumps";
#  else
    std::string flow_block_solver = "ilu";
#  endif
    coupling.add_parameter(
      "Flow block solver",
      flow_block_solver,
      "Native flow block preconditioner in the iterative coupled solve.",
      ParameterAcceptor::prm,
      Patterns::Selection("ilu|mumps"));
    bool pressure_feedback = false;
    coupling.add_parameter(
      "Pressure feedback",
      pressure_feedback,
      "Apply solid traction as native external pressure. False retains native flow wall physics and solves a one-way geometric constraint.");
    double pressure_tolerance = 0.;
    coupling.add_parameter(
      "Wall pressure absolute tolerance",
      pressure_tolerance,
      "Pascals; zero retains the common IDA absolute tolerance.",
      ParameterAcceptor::prm,
      Patterns::Double(0.));
    bool stop_at_output_times = true;
    coupling.add_parameter(
      "Stop at output times",
      stop_at_output_times,
      "Take an internal IDA step at saved times to retain nonlinear wall compatibility.");

    const bool                          restarting = !options.restart.empty();
    std::optional<Checkpoint::Metadata> restored;
    if (restarting)
      {
        AssertThrow(
          HasCheckpoint<FlowProblem>::value,
          ExcMessage(
            "Restart requires a MetricFlowX package with native checkpoint hooks."));
        restored = Checkpoint::read_metadata(options.restart, MPI_COMM_WORLD);
        initialize_parameters_from_string(restored->parameters);
        if constexpr (HasCheckpoint<FlowProblem>::value)
          flow_problem.load_checkpoint(
            (std::filesystem::path(options.restart) / "flow").string(),
            [&](const std::string &) {
              lambda_parameters.finite_element = restored->lambda_fe;
              lambda_space_storage = std::make_unique<FiniteElementSpace<1, 3>>(
                flow_problem.triangulation(), lambda_parameters);
              const auto view = lambda_space_storage->view();
              restored_reference_space =
                std::make_unique<FiniteElementSpace<1, 3>>(
                  flow_problem.triangulation(),
                  FETools::get_fe_by_name<1, 3>(restored->reference_fe));
              reference_view = restored_reference_space->view();
              reference_field =
                reference_view->field("a0",
                                      FEValuesExtractors::Scalar(
                                        restored->reference_component));
              FieldVector owned_reference;
              owned_reference.reinit(reference_view->locally_owned_dofs(),
                                     MPI_COMM_WORLD);
              restored_lambda.reinit(view.locally_owned_dofs(), MPI_COMM_WORLD);
              restored_lambda_dot.reinit(view.locally_owned_dofs(),
                                         MPI_COMM_WORLD);
              Checkpoint::restore_fields(view.dof_handler(),
                                         restored_lambda,
                                         restored_lambda_dot,
                                         reference_view->dof_handler(),
                                         owned_reference);
              reference_area.reinit(reference_view->locally_owned_dofs(),
                                    reference_view->locally_relevant_dofs(),
                                    MPI_COMM_WORLD);
              reference_area = owned_reference;
            });
        flow_time.initial_time                         = restored->time;
        ida_parameters.correction_type_at_initial_time = "none";
      }
    else
      initialize_parameters(options.parameter_file);
    if (options.final_time)
      flow_time.final_time = *options.final_time;
    AssertThrow(
      flow_time.final_time > flow_time.initial_time,
      ExcMessage(
        "Final time must exceed the restored or configured initial time."));
    AssertThrow(options.checkpoint.empty() || stop_at_output_times,
                ExcMessage(
                  "Checkpoint writing requires Stop at output times = true."));
    AssertThrow(
      options.checkpoint.empty() || HasCheckpoint<FlowProblem>::value,
      ExcMessage(
        "Checkpoint writing requires a MetricFlowX package with native checkpoint hooks."));
    AssertThrow(
      options.checkpoint.empty() ||
        solid_parameters.domain_parameters.triangulation_type ==
          "fullydistributed",
      ExcMessage("Checkpoint writing requires a fullydistributed solid mesh."));
    AssertThrow(
      solid_parameters.n_refinement_cycles == 1,
      ExcMessage(
        "The coupled elastostatic application requires one refinement cycle."));
    const auto check_static_material = [](const auto &material) {
      AssertThrow(
        material.rho == 0. && material.neta == 0. &&
          material.rayleigh_alpha == 0. && material.rayleigh_beta == 0.,
        ExcMessage(
          "Elastostatics requires zero solid density, viscosity and Rayleigh damping."));
    };
    check_static_material(solid_parameters.default_material_properties);
    for (const auto &[id, material] :
         solid_parameters.material_properties_by_id)
      check_static_material(*material);
    AssertThrow(
      restarting ||
        ida_parameters.correction_type_at_initial_time == "use_y_diff",
      ExcMessage(
        "Elastostatic coupling requires coupled IDA use_y_diff initial correction."));
    AssertThrow(
      linear_solver != "umfpack" ||
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
      ExcMessage(
        "UMFPACK requires one MPI rank; use MUMPS for an MPI direct solve."));

    if (!restarting)
      {
        const auto initial_output = metric_flow_parameter("Output directory");
        std::filesystem::create_directories(initial_output);
        flow_problem.initialize_params(options.parameter_file,
                                       (std::filesystem::path(initial_output) /
                                        "last_used_parameters.prm")
                                         .string());
      }
    // ParameterHandler::set also updates bound values, keeping the archived
    // resolved configuration consistent with command-line overrides.
    auto &prm = ParameterAcceptor::prm;
    prm.enter_subsection("MetricFlowSystem<1, 3>");
    prm.enter_subsection("IDA parameters");
    if (options.final_time)
      prm.set("Final time", parameter_time(*options.final_time));
    if (restarting)
      {
        prm.set("Initial time", parameter_time(restored->time));
        prm.enter_subsection("Initial condition correction parameters");
        prm.set("Correction type at initial time", "none");
        prm.leave_subsection();
      }
    prm.leave_subsection();
    prm.leave_subsection();
    if (restarting)
      {
        std::string resumed_output;
        Checkpoint::collective_file_operation(MPI_COMM_WORLD, [&]() {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
            return;
          const auto base =
            std::filesystem::absolute(options.restart).string() + "-output";
          for (unsigned int suffix = 0;; ++suffix)
            {
              auto candidate = base + "-" + std::to_string(suffix);
              if (std::filesystem::create_directory(candidate))
                {
                  resumed_output = std::move(candidate);
                  break;
                }
            }
        });
        resumed_output =
          Utilities::MPI::broadcast(MPI_COMM_WORLD, resumed_output, 0);
        set_metric_flow_parameter("Output directory", resumed_output + "/flow");
        prm.enter_subsection("Elastic static");
        prm.set("Output directory", resumed_output + "/solid");
        prm.leave_subsection();
      }
    const auto flow_output_directory =
      metric_flow_parameter("Output directory");
    std::filesystem::create_directories(flow_output_directory);
    // Coupling assembly registers temporary acceptors whose names contain
    // process addresses. Only application configuration belongs in the archive.
    std::ostringstream checkpoint_parameters;
    if (!options.checkpoint.empty())
      ParameterAcceptor::prm.print_parameters(checkpoint_parameters,
                                              ParameterHandler::ShortPRM);
    SolidProblem solid_problem(solid_parameters);
    if (restarting)
      solid_problem.setup_from_checkpoint(
        (std::filesystem::path(options.restart) / "solid_mesh").string(),
        [&](auto &mesh) {
          Checkpoint::restore_solid_manifolds(mesh, restored->solid_manifolds);
        });
    else
      solid_problem.setup();
    solid_problem.update_time(flow_time.initial_time);
    // A complete vector Dirichlet face or a positive Robin spring on a face
    // removes rigid motion on a connected solid. Imported disconnected
    // components and deficient multiplier spaces still need separate checks.
    const auto     &solid_fe = solid_problem.dof_handler().get_fe();
    FEFaceValues<3> anchor_values(
      solid_problem.mapping(),
      solid_fe,
      compatibility::gauss_quadrature<2>(
        solid_fe.reference_cell().face_reference_cell(0), solid_fe.degree + 1),
      update_quadrature_points);
    unsigned int local_anchor = 0;
    for (const auto &cell : solid_problem.dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary())
            {
              const auto id = cell->face(face)->boundary_id();
              if (solid_parameters.dirichlet_ids.count(id) != 0)
                local_anchor = 1;
              else if (solid_parameters.robin_ids.count(id) != 0)
                {
                  anchor_values.reinit(cell, face);
                  const auto &coefficient =
                    solid_parameters.get_robin_coefficient(id);
                  bool positive = true;
                  for (const auto &point :
                       anchor_values.get_quadrature_points())
                    positive = positive &&
                               std::isfinite(coefficient.value(point)) &&
                               coefficient.value(point) > 0.;
                  if (positive)
                    local_anchor = 1;
                }
            }
    AssertThrow(Utilities::MPI::max(local_anchor, MPI_COMM_WORLD) != 0,
                ExcMessage(
                  "Elastostatic coupling needs an actual Dirichlet face or a "
                  "Robin face with positive initial spring coefficient to "
                  "anchor solid rigid motion."));

    std::unique_ptr<Domain<1, 3>> flow_domain;
    if (!restarting)
      {
        flow_domain_parameters.name_of_grid =
          metric_flow_parameter("Vtk file path for mesh input");
        flow_domain_parameters.read_vtk_fields = true;
        flow_domain =
          std::make_unique<Domain<1, 3>>(flow_domain_parameters,
                                         flow_problem.triangulation());
        flow_problem.setup(
          [&](const auto &coarse_mesh, const unsigned int refinement) {
            flow_domain_parameters.initial_refinement = refinement;
            flow_domain->prepare_grid(coarse_mesh);
            flow_domain->finalize_grid();
          });
      }
    else
      lambda_parameters.finite_element = restored->lambda_fe;
    if (lambda_parameters.finite_element == "auto")
      lambda_parameters.finite_element =
        "FE_DGQ(" +
        std::to_string(flow_problem.finite_element().base_element(0).degree) +
        ")";
    if (!lambda_space_storage)
      lambda_space_storage =
        std::make_unique<FiniteElementSpace<1, 3>>(flow_problem.triangulation(),
                                                   lambda_parameters);
    auto       &lambda_space = *lambda_space_storage;
    const auto *area_fe      = dynamic_cast<const FE_DGQ<1, 3> *>(
      &flow_problem.finite_element().base_element(0));
    const auto *lambda_fe =
      dynamic_cast<const FE_DGQ<1, 3> *>(&lambda_space.finite_element());
    AssertThrow(
      area_fe != nullptr && lambda_fe != nullptr &&
        area_fe->degree == lambda_fe->degree,
      ExcMessage(
        "The multiplier must use the same local DGQ basis as native area."));
    const auto lambda_view = lambda_space.view();
    const auto lambda      = lambda_view.field("lambda");
    if (!restarting)
      {
        const auto a0   = flow_domain->field("a0");
        reference_view  = a0.space();
        reference_field = a0.field();
        reference_area.reinit(reference_view->locally_owned_dofs(),
                              reference_view->locally_relevant_dofs(),
                              MPI_COMM_WORLD);
        reference_area = a0.coefficients();
      }
    reference_area.update_ghost_values();
    const auto reference_radius =
      sqrt(frozen(*reference_field, reference_area) / numbers::PI);
    const auto reference_wall =
      lift(value(lambda), wall_lift, reference_radius);
    GridTools::Cache<3> tissue_cache(solid_problem.triangulation(),
                                     solid_problem.mapping());
    const auto          supports =
      make_overlap_supports(IntegrationSupport<1, 3>(
                              flow_problem.triangulation()),
                            lambda.mapping(),
                            reference_wall.representative_quadrature(),
                            reference_wall.lifted_points(),
                            tissue_cache);
    const auto [pressure_projection, multiplier_metric] =
      make_metric_flow_x_pressure_projection(lambda, supports.overlap);

    Adapter adapter(flow_time, ida_parameters, MPI_COMM_WORLD);
    adapter.set_stop_at_output_times(stop_at_output_times);
    adapter.set_schur_primal_coupling();
    if (linear_solver == "iterative")
      adapter.set_linear_solver(Adapter::LinearSolver::iterative);
    else if (linear_solver == "umfpack")
      adapter.set_linear_solver(Adapter::LinearSolver::umfpack);
    else if (linear_solver == "mumps")
      adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    const auto solid_description = elastic_static_semidiscrete(solid_problem);
    const auto solid_fields = adapter.add(solid_description, "elastostatics");
    auto       flow_description = metric_flow_x(flow_problem);
    if (flow_block_solver == "mumps")
      {
#  ifdef DEAL_II_WITH_MUMPS
        flow_description.preconditioner_factory = [](const auto &matrix,
                                                     const auto &prototype) {
          return make_mumps_preconditioner<FieldVector>(matrix,
                                                        prototype,
                                                        MPI_COMM_WORLD);
        };
#  else
        AssertThrow(
          false, ExcMessage("Flow block MUMPS requires deal.II with MUMPS."));
#  endif
      }
    const auto flow_fields     = adapter.add(flow_description, "blood-flow");
    const auto coupling_fields = adapter.add(
      metric_flow_x_wall_constraint(flow_problem,
                                    solid_fields.fields().displacement,
                                    flow_fields.fields(),
                                    lambda,
                                    wall_lift,
                                    reference_radius,
                                    supports,
                                    pressure_projection,
                                    multiplier_metric,
                                    solid_fields.fields().displacement,
                                    {solid_fields.fields().displacement.id(),
                                     flow_fields.fields().state},
                                    pressure_feedback),
      "vessel-wall");
    if (pressure_tolerance > 0.)
      adapter.set_absolute_tolerance(coupling_fields.fields().multiplier,
                                     pressure_tolerance);
    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    state          = 0.;
    state_dot      = 0.;
    if (restarting)
      {
        adapter.field(state, solid_fields.fields().displacement) =
          solid_problem.solution();
        auto &restored_displacement =
          adapter.field(state, solid_fields.fields().displacement);
        for (const auto &line : solid_problem.constraints().get_lines())
          if (!line.entries.empty() &&
              solid_problem.locally_owned_dofs().is_element(line.index))
            restored_displacement[line.index] = line.inhomogeneity;
        restored_displacement.compress(VectorOperation::insert);
        adapter.field(state, coupling_fields.fields().multiplier) =
          restored_lambda;
        adapter.field(state_dot, coupling_fields.fields().multiplier) =
          restored_lambda_dot;
        if constexpr (HasCheckpoint<FlowProblem>::value)
          {
            adapter.field(state, flow_fields.fields().state) =
              flow_problem.state();
            adapter.field(state_dot, flow_fields.fields().state) =
              flow_problem.state_derivative();
          }
      }
    else
      {
        flow_problem.initialize_state(adapter.field(state,
                                                    flow_fields.fields().state),
                                      flow_time.initial_time);
        flow_problem.initialize_state_derivative(
          adapter.field(state_dot, flow_fields.fields().state),
          flow_time.initial_time);
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Coupled block DoFs:";
        for (unsigned int b = 0; b < state.n_blocks(); ++b)
          std::cout << ' ' << state.block(b).size();
        std::cout << "\nCoupled total DoFs: " << state.size() << std::endl;
      }
    auto state_constraints =
      std::make_shared<GlobalVector>(adapter.make_state());
    *state_constraints = 0.;
    auto &flow_constraints =
      adapter.field(*state_constraints, flow_fields.fields().state);
    for (const auto component :
         {FlowProblem::Component::area, FlowProblem::Component::area_trace})
      for (const auto index : flow_problem.component_dofs(component))
        flow_constraints[index] = 2.;
    state_constraints->compress(VectorOperation::insert);

    const auto flow_output_name = metric_flow_parameter("Output filename");
    auto       wall_points      = reference_wall.lifted_points();
    for (auto &point : wall_points)
      point.mode_values = {point.mode_vector[0],
                           point.mode_vector[1],
                           point.mode_vector[2]};
    LiftedParticleLabels particle_labels;
    particle_labels.value_name      = "wall_traction";
    particle_labels.mode_labels     = {"radial"};
    particle_labels.mode_components = 3;

    // Native flow states can also contain trace and terminal unknowns. Map
    // their semantic field back to FE numbering before sampling the curve.
    const auto sample_field = [&](const auto &field, const auto &source) {
      FieldVector owned;
      owned.reinit(source);
      owned = source;
      if (field.constraints().n_constraints() != 0)
        field.constraints().distribute(owned);
      FieldVector execution(source.locally_owned_elements(),
                            field.locally_relevant_dofs(),
                            MPI_COMM_WORLD);
      execution = owned;
      execution.update_ghost_values();
      FieldVector native(field.dof_handler().locally_owned_dofs(),
                         MPI_COMM_WORLD);
      native = 0.;
      for (const auto index : field.dof_handler().locally_owned_dofs())
        if (field.has_execution_index(index))
          native[index] = execution[field.execution_index(index)];
      native.compress(VectorOperation::insert);
      const auto relevant =
        DoFTools::extract_locally_relevant_dofs(field.dof_handler());
      FieldVector ghosted(field.dof_handler().locally_owned_dofs(),
                          relevant,
                          MPI_COMM_WORLD);
      ghosted = native;
      ghosted.update_ghost_values();
      return sample_at_representative_points(field.dof_handler(),
                                             field.mapping(),
                                             ghosted,
                                             wall_points);
    };
    std::vector<std::pair<double, std::string>> flow_records,
      multiplier_records, particle_records;
    adapter.set_output_step([&](const double        time,
                                const GlobalVector &solution,
                                const GlobalVector &solution_dot,
                                const unsigned int  step) {
      if (step == 0)
        if (const char *filename =
              std::getenv("IMMERSX_ELASTOSTATIC_STATE_OUTPUT"))
          write_regression_state(solution, std::string(filename) + ".initial");
      const auto &flow_state =
        adapter.field(solution, flow_fields.fields().state);
      const auto &flow_derivative =
        adapter.field(solution_dot, flow_fields.fields().state);
      typename FlowProblem::ExternalPressureProvider pressure_provider;
      if (pressure_feedback)
        pressure_provider = metric_flow_x_pressure_provider(
          -1. * value(coupling_fields.fields().multiplier),
          adapter.field(solution, coupling_fields.fields().multiplier),
          pressure_projection);
      if (pressure_feedback)
        flow_problem.set_external_pressure_provider(pressure_provider);
      else
        flow_problem.clear_external_pressure_provider();
      auto pressure = flow_problem.make_state();
      flow_problem.compute_pressure(flow_state,
                                    flow_derivative,
                                    pressure,
                                    time);
      flow_problem.output_results(flow_state, pressure, step);
      flow_problem.write_output_step(time, flow_state, &flow_derivative);
      flow_problem.clear_external_pressure_provider();
      solid_problem.update_time(time);
      solid_problem.set_solution(
        adapter.field(solution, solid_fields.fields().displacement));
      solid_problem.output_results(step, time);
      if (particle_output.enabled())
        {
          std::vector<double> coefficients, data;
          if (particle_output.selected("modal_coefficients") ||
              particle_output.selected("lifted_value"))
            coefficients =
              sample_field(coupling_fields.fields().multiplier,
                           adapter.field(solution,
                                         coupling_fields.fields().multiplier));
          if (particle_output.selected("modal_data"))
            {
              const auto &area_field = flow_fields.fields().area;
              const auto  area       = sample_field(area_field, flow_state);
              const auto  reference =
                sample_at_representative_points(reference_view->dof_handler(),
                                                reference_view->mapping(),
                                                reference_area,
                                                wall_points);
              const auto width = area_field.finite_element().n_components();
              const auto reference_width =
                reference_view->finite_element().n_components();
              for (std::size_t q = 0; q < wall_points.size(); ++q)
                {
                  const auto current =
                    area[q * width + area_field.extractor().component];
                  const auto initial =
                    reference[q * reference_width +
                              reference_field->extractor().component];
                  AssertThrow(current > 0. && initial > 0.,
                              ExcMessage(
                                "Wall particle radii require positive area."));
                  data.push_back(std::sqrt(current / numbers::PI) -
                                 std::sqrt(initial / numbers::PI));
                }
            }
          const auto columns = make_lifted_particle_columns(
            particle_output, particle_labels, wall_points, coefficients, data);
          std::optional<FieldVector> displacement;
          if (particle_output.selected("bulk_field"))
            {
              displacement.emplace(solid_problem.locally_owned_dofs(),
                                   solid_problem.locally_relevant_dofs(),
                                   MPI_COMM_WORLD);
              *displacement = solid_problem.solution();
              displacement->update_ghost_values();
            }
          const LiftedBulkField<3, FieldVector> bulk{
            "displacement",
            solid_problem.dof_handler(),
            solid_problem.mapping(),
            displacement ? &*displacement : nullptr,
            0,
            3};
          const auto particle_filename =
            "wall_particles_" + std::to_string(step) + ".vtu";
          write_lifted_particles(particle_output,
                                 columns,
                                 wall_points,
                                 bulk,
                                 (std::filesystem::path(flow_output_directory) /
                                  particle_filename)
                                   .string());
          particle_records.emplace_back(time, particle_filename);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::ofstream pvd(std::filesystem::path(flow_output_directory) /
                                "wall_particles.pvd");
              DataOutBase::write_pvd_record(pvd, particle_records);
            }
        }
      DataOut<1, 3> multiplier_output;
      multiplier_output.attach_dof_handler(lambda_view.dof_handler());
      multiplier_output.add_data_vector(
        adapter.field(solution, coupling_fields.fields().multiplier),
        "pressure_jump",
        DataOut<1, 3>::type_dof_data);
      multiplier_output.build_patches(lambda_view.mapping());
      const auto filename = "pressure_jump_" + std::to_string(step) + ".vtu";
      multiplier_output.write_vtu_in_parallel(
        (std::filesystem::path(flow_output_directory) / filename).string(),
        MPI_COMM_WORLD);
      flow_records.emplace_back(time,
                                flow_output_name + "_" +
                                  Utilities::int_to_string(step, 5) + ".pvtu");
      multiplier_records.emplace_back(time, filename);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ofstream flow_pvd(std::filesystem::path(flow_output_directory) /
                                 (flow_output_name + ".pvd"));
          DataOutBase::write_pvd_record(flow_pvd, flow_records);
          std::ofstream multiplier_pvd(
            std::filesystem::path(flow_output_directory) / "pressure_jump.pvd");
          DataOutBase::write_pvd_record(multiplier_pvd, multiplier_records);
        }
    });
    adapter.solver().get_constraint_vector =
      [state_constraints]() -> GlobalVector & { return *state_constraints; };
    flow_problem.open_output_files();
    const auto accepted_steps = adapter.solve(state, state_dot);
    if (const char *filename = std::getenv("IMMERSX_ELASTOSTATIC_STATE_OUTPUT"))
      write_regression_state(state, filename);
    if (!options.checkpoint.empty())
      {
        if constexpr (HasCheckpoint<FlowProblem>::value)
          {
            Checkpoint::Metadata metadata;
            metadata.ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
            metadata.time  = flow_time.final_time;
            metadata.parameters   = checkpoint_parameters.str();
            metadata.reference_fe = reference_view->finite_element().get_name();
            metadata.reference_component =
              reference_field->extractor().component;
            metadata.lambda_fe = lambda_view.finite_element().get_name();
            solid_problem.update_time(metadata.time);
            solid_problem.set_solution(
              adapter.field(state, solid_fields.fields().displacement));
            Checkpoint::save(
              options.checkpoint,
              metadata,
              flow_problem,
              solid_problem,
              adapter.field(state, flow_fields.fields().state),
              adapter.field(state_dot, flow_fields.fields().state),
              lambda_view.dof_handler(),
              adapter.field(state, coupling_fields.fields().multiplier),
              adapter.field(state_dot, coupling_fields.fields().multiplier),
              reference_view->dof_handler(),
              reference_area,
              MPI_COMM_WORLD);
          }
      }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "MetricFlowX elastostatic accepted IDA steps: "
                << accepted_steps << "\nStatic AMG setups: "
                << solid_description.preconditioner_builds() << std::endl;
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      Options options;
      for (int i = 1; i < argc; ++i)
        {
          const std::string argument(argv[i]);
          const auto        value = [&]() {
            AssertThrow(i + 1 < argc,
                        dealii::ExcMessage("Missing value after " + argument));
            return std::string(argv[++i]);
          };
          if (argument == "--restart")
            options.restart = value();
          else if (argument == "--checkpoint")
            options.checkpoint = value();
          else if (argument == "--final-time")
            {
              const auto  text   = value();
              std::size_t parsed = 0;
              options.final_time = std::stod(text, &parsed);
              AssertThrow(parsed == text.size() &&
                            std::isfinite(*options.final_time),
                          dealii::ExcMessage("Invalid final time."));
            }
          else
            {
              AssertThrow(!argument.empty() && argument.front() != '-' &&
                            options.parameter_file.empty(),
                          dealii::ExcMessage("Unexpected argument: " +
                                             argument));
              options.parameter_file = argument;
            }
        }
      AssertThrow(
        options.restart.empty() != options.parameter_file.empty(),
        dealii::ExcMessage(
          "usage: metric_flow_x_elastostatic PARAMETER_FILE [--checkpoint DIRECTORY] [--final-time T] or --restart DIRECTORY --final-time T [--checkpoint DIRECTORY]"));
      // Keep application options out of PETSc's command-line database.
      int    runtime_argc        = 1;
      char  *runtime_arguments[] = {argv[0], nullptr};
      char **runtime_argv        = runtime_arguments;
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(runtime_argc,
                                                                  runtime_argv,
                                                                  1);
      run_coupled_problem(options);
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
