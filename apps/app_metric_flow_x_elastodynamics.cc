// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/core/constraint.h>
#  include <immersx/core/domain.h>
#  include <immersx/core/observable_lift.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/core/weak_term.h>
#  include <immersx/coupling/immersed_repartitioner.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_pressure.h>
#  include <immersx/vascularization/metric_flow_x.h>

#  include <cmath>
#  include <filesystem>
#  include <fstream>
#  include <iostream>
#  include <memory>
#  include <optional>
#  include <set>
#  include <stdexcept>
#  include <string>
#  include <type_traits>
#  include <vector>

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

  // Print the solid boundary ids and the number of vessel cells with a vertex
  // outside the solid mesh. This only checks the centreline vertices. The
  // lifted wall points lie one radius away from the centreline, so a count of
  // zero does not guarantee that the lifted wall is inside the solid. Lifted
  // points outside the solid stop the assembly with "Some lifted integration
  // points lie outside the background mesh".
  template <typename FlowTriangulation, typename SolidProblem>
  void
  report_vessel_geometry(const FlowTriangulation &flow_triangulation,
                         const SolidProblem      &solid_problem)
  {
    std::vector<Point<3>> points;
    unsigned int          n_local_cells = 0;
    for (const auto &cell : flow_triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          ++n_local_cells;
          for (const auto v : cell->vertex_indices())
            points.push_back(cell->vertex(v));
        }

    Utilities::MPI::RemotePointEvaluation<3, 3> evaluation;
    evaluation.reinit(points,
                      solid_problem.triangulation(),
                      solid_problem.mapping());
    unsigned int n_local_outside = 0;
    for (unsigned int i = 0; i < points.size(); i += 2)
      if (!evaluation.point_found(i) || !evaluation.point_found(i + 1))
        ++n_local_outside;

    const auto n_cells   = Utilities::MPI::sum(n_local_cells, MPI_COMM_WORLD);
    const auto n_outside = Utilities::MPI::sum(n_local_outside, MPI_COMM_WORLD);
    const auto local_ids = solid_problem.triangulation().get_boundary_ids();
    const auto boundary_ids = Utilities::MPI::compute_set_union(
      std::set<types::boundary_id>(local_ids.begin(), local_ids.end()),
      MPI_COMM_WORLD);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Solid boundary ids:";
        for (const auto id : boundary_ids)
          std::cout << ' ' << id;
        std::cout << std::endl
                  << "Vessel cells with a vertex outside the solid mesh: "
                  << n_outside << " of " << n_cells << std::endl;
      }
  }

  template <class Flow>
  Vascularization::MetricFlowXTransfer
  prepare_generated_flow(Flow                                    &flow,
                         const Vascularization::GenerationResult &generation,
                         const double                             young_modulus,
                         const double wall_thickness,
                         const double reference_pressure,
                         const double terminal_resistance,
                         const double outlet_pressure)
  {
    if constexpr (Vascularization::has_metric_flow_x_network_input<Flow>)
      return Vascularization::prepare_metric_flow_x(
        flow,
        generation,
        [&](Vascularization::Id, const Vascularization::Vessel &) {
          typename Flow::VesselProperties properties;
          properties.E      = young_modulus;
          properties.h_wall = wall_thickness;
          properties.p_d    = reference_pressure;
          properties.p0     = 0.;
          return properties;
        },
        [&](Vascularization::Id,
            const Vascularization::Node &,
            const Vascularization::Terminal *) {
          return typename Flow::RCRPhysics{
            0., terminal_resistance, 0., outlet_pressure, "generated"};
        });
    else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Generated flow requires MetricFlowX with in-memory NetworkPhysics input."));
        return {};
      }
  }

  void
  run_coupled_problem(const std::string &parameter_file)
  {
    using SolidProblem = ElastodynamicsSolver<3>;
    using FlowProblem  = MetricFlowX::BloodFlowSystem<1, 3>;
    using FieldVector  = MetricFlowX::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
    using WallLift     = TensorProductLift<1, 2, 3, 3>;

    ParameterAcceptor::clear();
    FlowProblem            flow_problem(MPI_COMM_WORLD);
    TimeIntervalParameters flow_time("/", false);
    FixedStepParameters    fixed_step("/", false);
    IDAParameters ida_parameters("/MetricFlowSystem<1, 3>/IDA parameters/",
                                 &flow_time);
    ElastodynamicsParameters<3> solid_parameters("/Elastodynamics/",
                                                 &flow_time,
                                                 &fixed_step,
                                                 &ida_parameters);
    WallLift                    wall_lift("/MetricFlowX vessel wall lift/");
    FiniteElementSpaceParameters<1, 3> lambda_parameters(
      "/MetricFlowX multiplier space/", "auto");

    ParameterAcceptor coupled_solver(
      "/MetricFlowSystem<1, 3>/IDA parameters/Linear solver/");
    std::string linear_solver = "iterative";
    coupled_solver.add_parameter(
      "Type",
      linear_solver,
      "Coupled IDA linear solver. Direct solvers materialize each Jacobian. UMFPACK requires one rank; MUMPS supports MPI.",
      ParameterAcceptor::prm,
      Patterns::Selection("iterative|umfpack|mumps"));

    ParameterAcceptor coupling("/MetricFlowX coupling/");
    std::string       embedded_partitioning = "default";
    coupling.add_parameter(
      "Embedded partitioning policy",
      embedded_partitioning,
      "default preserves ordinary Domain ownership; immersed follows the solid cell owner at each vessel cell center.",
      ParameterAcceptor::prm,
      Patterns::Selection("default|immersed"));

    Vascularization::Parameters generation_parameters;
    bool                        generate_network = false;
    double vessel_young_modulus = 0., vessel_wall_thickness = 0.,
           vessel_reference_pressure = 0., terminal_resistance = 0.,
           outlet_pressure = 0.;
    coupling.add_parameter(
      "Generate vascular network",
      generate_network,
      "Generate and validate on the final solid background before flow setup.");
    coupling.add_parameter(
      "Generated vessel Young modulus",
      vessel_young_modulus,
      "Explicit synthetic or measured vessel Young modulus in Pa.");
    coupling.add_parameter("Generated vessel wall thickness",
                           vessel_wall_thickness,
                           "Explicit vessel wall thickness in m.");
    coupling.add_parameter(
      "Generated vessel reference pressure",
      vessel_reference_pressure,
      "Vessel reference pressure in Pa, applied once at the rest area.");
    coupling.add_parameter("Generated terminal resistance",
                           terminal_resistance,
                           "Native RCR R2 in Pa s/m^3; R1 and C are zero.");
    coupling.add_parameter("Generated outlet pressure",
                           outlet_pressure,
                           "Native terminal outlet pressure in Pa.");

    initialize_parameters(parameter_file);
    flow_problem.initialize_params(parameter_file);

    AssertThrow(linear_solver != "umfpack" ||
                  Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage(
                  "The coupled UMFPACK solver requires one MPI rank."));
#  ifndef DEAL_II_WITH_UMFPACK
    AssertThrow(
      linear_solver != "umfpack",
      ExcMessage(
        "The coupled UMFPACK solver requires deal.II configured with UMFPACK."));
#  endif
#  ifndef DEAL_II_WITH_MUMPS
    AssertThrow(
      linear_solver != "mumps",
      ExcMessage(
        "The coupled MUMPS solver requires deal.II configured with MUMPS."));
#  endif
    SolidProblem solid_problem(solid_parameters);
    solid_problem.make_grid();
    solid_problem.setup_fe();

    DomainParameters<1, 3> flow_domain_parameters("/MetricFlowX domain/");
    flow_domain_parameters.name_of_grid =
      metric_flow_parameter("Vtk file path for mesh input");
    flow_domain_parameters.read_vtk_fields = true;
    Domain<1, 3>                         flow_domain(flow_domain_parameters,
                             flow_problem.triangulation());
    Vascularization::GenerationResult    generated;
    Vascularization::MetricFlowXTransfer transfer;
    if (generate_network)
      {
        AssertThrow(
          Vascularization::has_metric_flow_x_network_input<FlowProblem>,
          ExcMessage(
            "Generated flow requires MetricFlowX with in-memory NetworkPhysics input."));
        AssertThrow(
          std::isfinite(vessel_young_modulus) && vessel_young_modulus > 0. &&
            std::isfinite(vessel_wall_thickness) &&
            vessel_wall_thickness > 0. && std::isfinite(terminal_resistance) &&
            terminal_resistance > 0. &&
            std::isfinite(vessel_reference_pressure) &&
            std::isfinite(outlet_pressure),
          ExcMessage(
            "Generated flow requires explicit positive wall material and terminal resistance, and finite pressures."));
        generated = Vascularization::generate(solid_problem.triangulation(),
                                              solid_problem.mapping(),
                                              {},
                                              generation_parameters,
                                              MPI_COMM_WORLD);
        AssertThrow(
          generated.validation.valid(),
          ExcMessage(
            "Vascular generation failed; the result cannot be adopted."));
        transfer = prepare_generated_flow(flow_problem,
                                          generated,
                                          vessel_young_modulus,
                                          vessel_wall_thickness,
                                          vessel_reference_pressure,
                                          terminal_resistance,
                                          outlet_pressure);
      }
    else
      {
        flow_problem.setup(
          [&](const auto &coarse_mesh, const unsigned int refinement) {
            flow_domain_parameters.initial_refinement = refinement;
            flow_domain.prepare_grid(coarse_mesh);
            if (embedded_partitioning == "default")
              flow_domain.finalize_grid();
            else if (embedded_partitioning == "immersed")
              {
                const ImmersedRepartitioner<1, 3> policy(
                  solid_problem.triangulation());
                flow_domain.finalize_grid(policy);
              }
          });
      }
    if (lambda_parameters.finite_element == "auto")
      lambda_parameters.finite_element =
        "FE_DGQ(" +
        std::to_string(flow_problem.finite_element().base_element(0).degree) +
        ")";
    FiniteElementSpace<1, 3> lambda_space(flow_problem.triangulation(),
                                          lambda_parameters);
    const auto              *area_fe = dynamic_cast<const FE_DGQ<1, 3> *>(
      &flow_problem.finite_element().base_element(0));
    const auto *lambda_fe =
      dynamic_cast<const FE_DGQ<1, 3> *>(&lambda_space.finite_element());
    AssertThrow(area_fe != nullptr && lambda_fe != nullptr &&
                  area_fe->degree == lambda_fe->degree &&
                  area_fe->n_dofs_per_cell() == lambda_fe->n_dofs_per_cell(),
                ExcMessage("The multiplier FE must have the same local "
                           "basis as MetricFlowX area."));
    const auto lambda_view = lambda_space.view();
    const auto lambda      = lambda_view.field("lambda");
    // A separate quadratic DG field represents pi*r(s)^2 exactly, including
    // a tapered physical vessel. Its storage remains alive throughout IDA.
    std::unique_ptr<FiniteElementSpace<1, 3>>   generated_reference_space;
    std::optional<FiniteElementSpaceView<1, 3>> generated_reference_view;
    FieldVector                                 reference_area;
    const auto                                  reference_field = [&]() {
      if (generate_network)
        {
          generated_reference_space =
            std::make_unique<FiniteElementSpace<1, 3>>(
              flow_problem.triangulation(), std::make_unique<FE_DGQ<1, 3>>(2));
          generated_reference_view.emplace(generated_reference_space->view());
          const auto &reference_view = *generated_reference_view;
          reference_area.reinit(reference_view.locally_owned_dofs(),
                                reference_view.locally_relevant_dofs(),
                                MPI_COMM_WORLD);
          // Trilinos ghosted vectors are read-only. Assemble owned values,
          // then import them into the field used by lifted evaluation.
          FieldVector owned_reference_area;
          owned_reference_area.reinit(reference_view.locally_owned_dofs(),
                                      MPI_COMM_WORLD);
          std::vector<types::global_dof_index> indices(
            reference_view.finite_element().n_dofs_per_cell());
          for (const auto &cell :
               reference_view.dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);
                const auto interval =
                  Vascularization::physical_reference(cell->id(),
                                                      transfer.flow_cells);
                const auto &properties =
                  flow_problem.vessel_properties(cell->material_id());
                for (unsigned int i = 0; i < indices.size(); ++i)
                  {
                    const double xi = cell->get_fe().unit_support_point(i)[0];
                    const double t =
                      interval.begin + xi * (interval.end - interval.begin);
                    const double radius =
                      (1. - t) * properties.r_in + t * properties.r_out;
                    owned_reference_area[indices[i]] =
                      numbers::PI * radius * radius;
                  }
              }
          owned_reference_area.compress(VectorOperation::insert);
          reference_area = owned_reference_area;
          reference_area.update_ghost_values();
          return reference_view.field("a0");
        }
      const auto a0 = flow_domain.field("a0");
      reference_area.reinit(flow_domain.vtk_fields().locally_owned_dofs(),
                            flow_domain.vtk_fields().locally_relevant_dofs(),
                            MPI_COMM_WORLD);
      reference_area = a0.coefficients();
      reference_area.update_ghost_values();
      return a0.field();
    }();
    const auto A0 = frozen(reference_field, reference_area);

    solid_problem.setup_system();
    solid_problem.assemble_operators();
    solid_problem.set_initial_conditions();
    report_vessel_geometry(flow_problem.triangulation(), solid_problem);

    Adapter adapter(flow_time, ida_parameters, MPI_COMM_WORLD);
    adapter.set_schur_primal_coupling();
    if (linear_solver == "iterative")
      adapter.set_linear_solver(Adapter::LinearSolver::iterative);
    else if (linear_solver == "umfpack")
      adapter.set_linear_solver(Adapter::LinearSolver::umfpack);
    else if (linear_solver == "mumps")
      adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    const auto solid_fields = adapter.add(solid_problem, "elastodynamics");
    const auto flow_fields =
      adapter.add(metric_flow_x(flow_problem), "blood-flow");

    const auto coupling_fields = adapter.add(
      [&](auto &builder) {
        const auto id          = builder.algebraic_field(lambda.name(),
                                                lambda.locally_owned_dofs(),
                                                lambda.locally_relevant_dofs());
        const auto multiplier  = lambda.with_id(id);
        const auto radius      = sqrt(A0 / numbers::PI);
        const auto lambda_wall = lift(value(multiplier), wall_lift, radius);
        const auto negative_wall_displacement =
          lift(radius - sqrt(value(flow_fields.fields().area) / numbers::PI),
               wall_lift,
               radius);

        weak_term(value(solid_fields.fields().displacement), test(lambda_wall))
          .add(builder);
        weak_term(negative_wall_displacement, test(multiplier)).add(builder);
        weak_term(lambda_wall, test(solid_fields.fields().velocity))
          .add(builder);

        const auto pressure      = -1. * value(multiplier);
        const auto flow_id       = flow_fields.fields().state;
        auto       pressure_term = builder.term(flow_id, "native-pressure");
        pressure_term.residual([&, pressure, flow_id, id](const auto &context) {
          auto contribution = std::make_shared<MetricFlowX::VectorType>(
            flow_problem.make_state());
          *contribution = 0.;
          flow_problem.add_external_pressure_residual(
            context.time(),
            context.state(flow_id),
            metric_flow_x_pressure_provider(pressure, context.state(id)),
            *contribution);
          dealii::PackagedOperation<MetricFlowX::VectorType> result;
          result.reinit_vector = [&flow_problem](auto &dst, const bool) {
            flow_problem.reinit_state(dst);
          };
          result.apply     = [contribution](auto &dst) { dst = *contribution; };
          result.apply_add = [contribution](auto &dst) {
            dst += *contribution;
          };
          return result;
        });
        typename std::decay_t<decltype(builder)>::Model::MatrixOperatorFactory
          pressure_jacobian = [&, pressure, flow_id](const auto &context) {
            return metric_flow_x_pressure_operator(flow_problem,
                                                   pressure,
                                                   context.state(flow_id),
                                                   context.time());
          };
        pressure_term.state(id, std::move(pressure_jacobian));
        if constexpr (detail::has_external_pressure_state_jacobian<
                        std::decay_t<decltype(flow_problem)>>::value)
          {
            typename std::decay_t<decltype(builder)>::Model::
              MatrixOperatorFactory pressure_state_jacobian =
                [&, pressure, flow_id, id](const auto &context) {
                  return metric_flow_x_pressure_state_operator(
                    flow_problem,
                    pressure,
                    context.state(id),
                    context.state(flow_id),
                    context.time());
                };
            pressure_term.state(flow_id, std::move(pressure_state_jacobian));
          }


        builder.saddle_point(id,
                             {solid_fields.fields().displacement.id(),
                              solid_fields.fields().velocity.id(),
                              flow_fields.fields().state},
                             detail::make_multiplier_metric(builder,
                                                            multiplier));
        return ConstraintFields<std::decay_t<decltype(lambda)>>{multiplier};
      },
      "vessel-wall");

    auto state     = adapter.make_state();
    auto state_dot = adapter.make_state();
    initialize_elastodynamics_adapter_state(
      adapter, solid_fields, solid_problem, state, state_dot);
    auto &flow_state = adapter.field(state, flow_fields.fields().state);
    flow_problem.initialize_state(flow_state, flow_time.initial_time);
    flow_problem.initialize_state_derivative(
      adapter.field(state_dot, flow_fields.fields().state),
      flow_time.initial_time);
    adapter.field(state, coupling_fields.fields().multiplier)     = 0.;
    adapter.field(state_dot, coupling_fields.fields().multiplier) = 0.;

    const auto flow_output_directory =
      metric_flow_parameter("Output directory");
    const auto flow_output_name = metric_flow_parameter("Output filename");
    std::vector<std::pair<double, std::string>> multiplier_output_records;
    std::vector<std::pair<double, std::string>> flow_output_records;
    const auto                                  output = [&solid_problem,
                         &solid_parameters,
                         &flow_problem,
                         &adapter,
                         &lambda_view,
                         &multiplier_output_records,
                         &flow_output_records,
                         coupling_fields,
                         flow_fields,
                         flow_output_name,
                         flow_output_directory](const GlobalVector &solution,
                                                const unsigned int  step,
                                                const double        time) {
      std::filesystem::create_directories(flow_output_directory);
      std::filesystem::create_directories(solid_parameters.output_directory);
      const auto &flow_state =
        adapter.field(solution, flow_fields.fields().state);
      auto pressure = flow_problem.make_state();
      flow_problem.compute_pressure(
        flow_state,
        pressure,
        time,
        metric_flow_x_pressure_provider(
          -1. * value(coupling_fields.fields().multiplier),
          adapter.field(solution, coupling_fields.fields().multiplier)));
      flow_problem.output_results(flow_state, pressure, step);
      // MetricFlowX's native writer uses its runner's private clock. The
      // coupled adapter supplies the accepted times for this collection.
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          flow_output_records.emplace_back(time,
                                           flow_output_name + "_" +
                                             Utilities::int_to_string(step, 5) +
                                             ".pvtu");
          std::ofstream pvd(std::filesystem::path(flow_output_directory) /
                            (flow_output_name + ".pvd"));
          DataOutBase::write_pvd_record(pvd, flow_output_records);
        }
      solid_problem.output_results();
      DataOut<1, 3> multiplier_output;
      multiplier_output.attach_dof_handler(lambda_view.dof_handler());
      multiplier_output.add_data_vector(
        adapter.field(solution, coupling_fields.fields().multiplier),
        "pressure_jump",
        DataOut<1, 3>::type_dof_data);
      multiplier_output.build_patches(lambda_view.mapping());
      const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      const auto filename = "pressure_jump_" + std::to_string(step) + ".vtu";
      multiplier_output.write_vtu_in_parallel(
        (std::filesystem::path(flow_output_directory) / filename).string(),
        MPI_COMM_WORLD);
      multiplier_output_records.emplace_back(time, filename);
      if (rank == 0)
        {
          std::ofstream pvd(std::filesystem::path(flow_output_directory) /
                            "pressure_jump.pvd");
          DataOutBase::write_pvd_record(pvd, multiplier_output_records);
        }
      MPI_Barrier(MPI_COMM_WORLD);
    };

    adapter.set_output_step([&solid_problem, &adapter, output, solid_fields](
                              const double        time,
                              const GlobalVector &solution,
                              const GlobalVector &solution_dot,
                              const unsigned int  step) {
      solid_problem.accept_state(
        adapter.field(solution, solid_fields.fields().displacement),
        adapter.field(solution, solid_fields.fields().velocity),
        time,
        step);
      output(solution, step, time);
      (void)solution_dot;
    });

    auto initial_flow_state = flow_problem.make_state();
    if (generate_network)
      initial_flow_state = adapter.field(state, flow_fields.fields().state);
    const auto accepted_steps = adapter.solve(state, state_dot);
    if (generate_network)
      {
        auto flow_change = flow_problem.make_state();
        flow_change      = adapter.field(state, flow_fields.fields().state);
        flow_change -= initial_flow_state;
        const double flow_norm = flow_change.l2_norm();
        const double solid_norm =
          adapter.field(state, solid_fields.fields().displacement).l2_norm();
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Generated coupled flow change: " << flow_norm
                    << "\nGenerated coupled solid displacement: " << solid_norm
                    << std::endl;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "MetricFlowX elastodynamics accepted IDA steps: "
                << accepted_steps << std::endl;
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
