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
#  include <immersx/core/overlap_integration_support.h>
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/core/weak_term.h>
#  include <immersx/coupling/immersed_repartitioner.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/metric_flow_x.h>
#  include <immersx/physics/metric_flow_x_pressure.h>
#  include <immersx/vascularization/io.h>
#  include <immersx/vascularization/metric_flow_x.h>

#  include <algorithm>
#  include <cmath>
#  include <filesystem>
#  include <fstream>
#  include <iomanip>
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
  auto
  initialize_flow_parameters(Flow              &flow,
                             const std::string &input_filename,
                             const std::string &output_filename,
                             int)
    -> decltype(flow.initialize_params(input_filename, output_filename), void())
  {
    flow.initialize_params(input_filename, output_filename);
  }

  template <class Flow>
  void
  initialize_flow_parameters(Flow              &flow,
                             const std::string &input_filename,
                             const std::string &,
                             long)
  {
    flow.initialize_params(input_filename);
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

  template <class Adapter, class Flow>
  auto
  set_native_flow_tolerances(Adapter      &adapter,
                             Flow         &flow,
                             const FieldId field,
                             int)
    -> decltype(flow.make_absolute_tolerances(), void())
  {
    adapter.set_local_tolerances(field, [&flow](auto &values) {
      values = flow.make_absolute_tolerances();
    });
  }

  template <class Adapter, class Flow>
  void
  set_native_flow_tolerances(Adapter &, Flow &, const FieldId, long)
  {
    AssertThrow(
      false,
      ExcMessage(
        "Native flow tolerances require a compatible MetricFlowX package."));
  }

  struct ArchivedFlowParameters : ParameterAcceptor
  {
    ArchivedFlowParameters()
      : ParameterAcceptor("/MetricFlowX coupling/Archived network/")
    {
      add_parameter(
        "Directory",
        directory,
        "Native physical network archive. Empty uses the ordinary VTK or generation path.");
      add_parameter(
        "Young moduli",
        young_moduli,
        "Wall Young moduli in Pa, in arterial, venous, portal order.");
      add_parameter(
        "Wall thickness ratios",
        thickness_ratios,
        "Wall thickness divided by mean reference radius, in arterial, venous, portal order.");
      add_parameter(
        "Reference pressures",
        reference_pressures,
        "Wall reference pressures in Pa, in arterial, venous, portal order.");
      add_parameter("Portal pressure",
                    portal_pressure,
                    "Prescribed portal root reservoir pressure in Pa.");
      add_parameter("Venous pressure",
                    venous_pressure,
                    "Prescribed venous root reservoir pressure in Pa.");
      add_parameter(
        "Root resistance",
        root_resistance,
        "Positive root reservoir resistance in Pa s/m^3; saved tissue compartments supply their own resistances.");
      add_parameter("Redistribution halo",
                    halo,
                    "Physical archive halo in metres.");
    }

    std::string         directory;
    std::vector<double> young_moduli{0., 0., 0.};
    std::vector<double> thickness_ratios{0., 0., 0.};
    std::vector<double> reference_pressures{0., 0., 0.};
    double portal_pressure = 0., venous_pressure = 0., root_resistance = 0.,
           halo = 0.;
  };

  template <class Flow>
  Vascularization::MetricFlowXTransfer
  prepare_archived_flow(Flow                                    &flow,
                        const Vascularization::GenerationResult &generation,
                        const ArchivedFlowParameters            &parameters)
  {
    namespace V = Vascularization;
    AssertThrow(
      parameters.young_moduli.size() == 3 &&
        parameters.thickness_ratios.size() == 3 &&
        parameters.reference_pressures.size() == 3,
      ExcMessage(
        "Archived wall parameters require three values in arterial, venous, portal order."));
    for (unsigned int i = 0; i < 3; ++i)
      AssertThrow(
        std::isfinite(parameters.young_moduli[i]) &&
          parameters.young_moduli[i] > 0. &&
          std::isfinite(parameters.thickness_ratios[i]) &&
          parameters.thickness_ratios[i] > 0. &&
          std::isfinite(parameters.reference_pressures[i]),
        ExcMessage(
          "Archived wall material must be positive and reference pressure finite."));
    AssertThrow(
      std::isfinite(parameters.portal_pressure) &&
        std::isfinite(parameters.venous_pressure) &&
        std::isfinite(parameters.root_resistance) &&
        parameters.root_resistance > 0.,
      ExcMessage(
        "Archived root pressures must be finite and reservoir resistance positive."));

    // Every tissue leaf must connect through a saved bed. Independent outlet
    // resistances would introduce additional drains beside those compartments.
    std::set<V::Id>    linked;
    const auto         comm = flow.mpi_communicator();
    std::vector<V::Id> owned_linked;
    for (const auto &pair : generation.terminal_pairs)
      owned_linked.insert(owned_linked.end(), {pair.arterial, pair.venous});
    for (const auto &triplet : generation.terminal_triplets)
      owned_linked.insert(owned_linked.end(),
                          {triplet.arterial, triplet.portal, triplet.venous});
    for (const auto &ids : Utilities::MPI::all_gather(comm, owned_linked))
      linked.insert(ids.begin(), ids.end());
    unsigned int unlinked = 0;
    const auto   rank     = Utilities::MPI::this_mpi_process(comm);
    for (const auto &[id, terminal] : generation.network.terminals)
      if (terminal.owner == rank && linked.count(id) == 0)
        ++unlinked;
    AssertThrow(
      Utilities::MPI::sum(unlinked, comm) == 0,
      ExcMessage(
        "The archived network contains tissue leaves without microvascular connections."));

    if constexpr (V::has_metric_flow_x_network_input<Flow>)
      return V::prepare_metric_flow_x(
        flow,
        generation,
        [&](V::Id, const V::Vessel &vessel) {
          unsigned int role = 0;
          if (vessel.system == V::System::venous)
            role = 1;
          else if (vessel.system == V::System::portal)
            role = 2;
          typename Flow::VesselProperties result;
          result.E      = parameters.young_moduli[role];
          result.h_wall = parameters.thickness_ratios[role] * .5 *
                          (vessel.radii[0] + vessel.radii[1]);
          result.p_d = parameters.reference_pressures[role];
          result.p0  = 0.;
          return result;
        },
        [&](V::Id, const V::Node &node, const V::Terminal *terminal) {
          // The native input validates positive R2 even at linked leaves;
          // those leaves use the saved resistor graph instead of this RCR.
          typename Flow::RCRPhysics result;
          result.R2 = parameters.root_resistance;
          if (terminal != nullptr)
            result.label = "microvascular-compartment";
          else if (node.port == V::Port::portal_inlet)
            {
              result.P_out = parameters.portal_pressure;
              result.label = "portal-root";
            }
          else if (node.port == V::Port::venous_outlet)
            {
              result.P_out = parameters.venous_pressure;
              result.label = "venous-root";
            }
          else
            AssertThrow(
              false, ExcMessage("Unsupported real port in archived network."));
          return result;
        });
    else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Archived flow requires MetricFlowX NetworkPhysics input."));
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

    ArchivedFlowParameters      archive_parameters;
    Vascularization::Parameters generation_parameters;
    bool                        use_native_tolerances = false;
    double displacement_tolerance = 0., velocity_tolerance = 0.,
           pressure_tolerance = 0.;
    coupling.add_parameter(
      "Use native flow tolerances",
      use_native_tolerances,
      "Use MetricFlowX area, velocity and capacitor absolute tolerances in the coupled adapter.");
    coupling.add_parameter(
      "Displacement absolute tolerance",
      displacement_tolerance,
      "Metres; zero retains the common IDA absolute tolerance.");
    coupling.add_parameter(
      "Velocity absolute tolerance",
      velocity_tolerance,
      "Metres per second; zero retains the common IDA absolute tolerance.");
    coupling.add_parameter(
      "Wall pressure absolute tolerance",
      pressure_tolerance,
      "Pascals; zero retains the common IDA absolute tolerance.");
    bool   generate_network     = false;
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
    const bool archived_network = !archive_parameters.directory.empty();
    AssertThrow(!archived_network || !generate_network,
                ExcMessage(
                  "Choose either an archived network or vascular generation."));
    const bool physical_network = archived_network || generate_network;
    if (archived_network)
      {
        auto path = std::filesystem::path(archive_parameters.directory);
        if (path.is_relative())
          path = std::filesystem::absolute(parameter_file).parent_path() / path;
        archive_parameters.directory = path.lexically_normal().string();
      }
    const auto flow_output_directory =
      metric_flow_parameter("Output directory");
    initialize_flow_parameters(flow_problem,
                               parameter_file,
                               (std::filesystem::path(flow_output_directory) /
                                "last_used_parameters.prm")
                                 .string(),
                               0);

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
    Domain<1, 3> flow_domain(flow_domain_parameters,
                             flow_problem.triangulation());
    std::optional<Vascularization::GenerationResult>    generated;
    std::optional<Vascularization::MetricFlowXTransfer> transfer;
    if (archived_network)
      {
        auto archive =
          Vascularization::load_network(archive_parameters.directory,
                                        solid_problem.triangulation(),
                                        solid_problem.mapping(),
                                        archive_parameters.halo,
                                        MPI_COMM_WORLD);
        generated.emplace();
        generated->network           = std::move(archive.network);
        generated->territories       = std::move(archive.territories);
        generated->terminal_pairs    = std::move(archive.terminal_pairs);
        generated->terminal_triplets = std::move(archive.terminal_triplets);
        generated->validation        = Vascularization::validate_data(
          generated->network, Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
        generated->discretization =
          Vascularization::discretize(solid_problem.triangulation(),
                                      solid_problem.mapping(),
                                      generated->network,
                                      generation_parameters.discretization,
                                      MPI_COMM_WORLD);
        transfer =
          prepare_archived_flow(flow_problem, *generated, archive_parameters);
      }
    else if (generate_network)
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
          generated->validation.valid(),
          ExcMessage(
            "Vascular generation failed; the result cannot be adopted."));
        transfer = prepare_generated_flow(flow_problem,
                                          *generated,
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
    // Match the native basis to avoid a rounded initial wall displacement.
    // Raise the reference degree when a physical linear radius needs it.
    std::unique_ptr<FiniteElementSpace<1, 3>>   generated_reference_space;
    std::optional<FiniteElementSpaceView<1, 3>> generated_reference_view;
    FieldVector                                 reference_area;
    const auto                                  reference_field = [&]() {
      if (physical_network)
        {
          unsigned int reference_degree = area_fe->degree;
          unsigned int local_taper = 0;
          const auto   rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
          for (const auto &[id, vessel] : generated->network.vessels)
            if (vessel.owner == rank && vessel.radii[0] != vessel.radii[1])
              local_taper = 1;
          if (Utilities::MPI::max(local_taper, MPI_COMM_WORLD) != 0)
            reference_degree = std::max(2u, reference_degree);
          generated_reference_space =
            std::make_unique<FiniteElementSpace<1, 3>>(
              flow_problem.triangulation(),
              std::make_unique<FE_DGQ<1, 3>>(reference_degree));
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
                                                      transfer->flow_cells);
                const auto &properties =
                  flow_problem.vessel_properties(cell->material_id());
                for (unsigned int i = 0; i < indices.size(); ++i)
                  {
                    const double xi = cell->get_fe().unit_support_point(i)[0];
                    const double t =
                      interval.begin + xi * (interval.end - interval.begin);
                    const double radius =
                      properties.r_in == properties.r_out ?
                                                         properties.r_in :
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
    if (physical_network)
      {
        const auto    rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        std::ofstream scales(std::filesystem::path(flow_output_directory) /
                             ("mesh_scale-" + std::to_string(rank) + ".csv"));
        AssertThrow(
          Utilities::MPI::min(scales.good() ? 1u : 0u, MPI_COMM_WORLD) == 1,
          ExcMessage("Cannot write physical-network mesh scale diagnostics."));
        scales.precision(17);
        scales
          << "vessel,begin,end,length_m,h_min_m,h_max_m,inside,short_cell\n";
        for (const auto &cell : generated->discretization.cells)
          scales << cell.reference.vessel << ',' << cell.reference.begin << ','
                 << cell.reference.end << ','
                 << Vascularization::length(generated->network,
                                            cell.reference.vessel) *
                      (cell.reference.end - cell.reference.begin)
                 << ',' << cell.h_min << ',' << cell.h_max << ',' << cell.inside
                 << ',' << cell.short_cell << '\n';
        if (rank == 0)
          std::cout << "Physical network discretization: "
                    << generated->discretization.short_cells << " short cells, "
                    << generated->discretization.exterior_cells
                    << " exterior cells" << std::endl;
      }
    generated.reset();
    transfer.reset();
    const auto A0 = frozen(reference_field, reference_area);

    solid_problem.setup_system();
    solid_problem.assemble_operators();
    solid_problem.set_initial_conditions();
    report_vessel_geometry(flow_problem.triangulation(), solid_problem);

    const auto reference_radius = sqrt(A0 / numbers::PI);
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
        const auto radius      = reference_radius;
        const auto lambda_wall = reference_wall.with_id(id);
        const auto negative_wall_displacement =
          lift(radius - sqrt(value(flow_fields.fields().area) / numbers::PI),
               wall_lift,
               radius);

        weak_term(value(solid_fields.fields().displacement), test(lambda_wall))
          .on(supports.overlap)
          .add(builder);
        weak_term(negative_wall_displacement, test(multiplier))
          .on(supports.overlap)
          .add(builder);
        weak_term(-1. * value(multiplier), test(multiplier))
          .on(supports.outside)
          .add(builder);
        weak_term(lambda_wall, test(solid_fields.fields().velocity))
          .on(supports.overlap)
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
            metric_flow_x_pressure_provider(pressure,
                                            context.state(id),
                                            pressure_projection),
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
                                                   context.time(),
                                                   pressure_projection);
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
                    context.time(),
                    pressure_projection);
                };
            pressure_term.state(flow_id, std::move(pressure_state_jacobian));
          }


        auto metric        = builder.matrix_operator(*multiplier_metric);
        metric.materialize = [multiplier_metric] { return multiplier_metric; };
        metric.materialize_into = [multiplier_metric](auto &destination) {
          destination.copy_from(*multiplier_metric);
        };
        builder.saddle_point(id,
                             {solid_fields.fields().displacement.id(),
                              solid_fields.fields().velocity.id(),
                              flow_fields.fields().state},
                             metric);
        return ConstraintFields<std::decay_t<decltype(lambda)>>{multiplier};
      },
      "vessel-wall");

    AssertThrow(displacement_tolerance >= 0. && velocity_tolerance >= 0. &&
                  pressure_tolerance >= 0.,
                ExcMessage(
                  "Coupled absolute tolerance overrides must be nonnegative."));
    if (displacement_tolerance > 0.)
      adapter.set_absolute_tolerance(solid_fields.fields().displacement,
                                     displacement_tolerance);
    if (velocity_tolerance > 0.)
      adapter.set_absolute_tolerance(solid_fields.fields().velocity,
                                     velocity_tolerance);
    if (pressure_tolerance > 0.)
      adapter.set_absolute_tolerance(coupling_fields.fields().multiplier,
                                     pressure_tolerance);
    if (use_native_tolerances)
      set_native_flow_tolerances(adapter,
                                 flow_problem,
                                 flow_fields.fields().state,
                                 0);

    auto state = adapter.make_state();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Coupled block DoFs:";
        for (unsigned int b = 0; b < state.n_blocks(); ++b)
          std::cout << ' ' << state.block(b).size();
        std::cout << "\nCoupled total DoFs: " << state.size() << std::endl;
      }
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
    if (ida_parameters.correction_type_at_initial_time == "use_y_dot")
      state_dot = 0.;

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
                         pressure_projection,
                         flow_fields,
                         flow_output_name,
                         flow_output_directory](
                          const GlobalVector &solution,
                          const GlobalVector &solution_dot,
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
          adapter.field(solution, coupling_fields.fields().multiplier),
          pressure_projection));
      flow_problem.output_results(flow_state, pressure, step);
      flow_problem.set_external_pressure_provider(
        metric_flow_x_pressure_provider(
          -1. * value(coupling_fields.fields().multiplier),
          adapter.field(solution, coupling_fields.fields().multiplier),
          pressure_projection));
      const auto &flow_derivative =
        adapter.field(solution_dot, flow_fields.fields().state);
      flow_problem.write_output_step(time, flow_state, &flow_derivative);
      flow_problem.clear_external_pressure_provider();
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
      // Raw operators retain physical Dirichlet values and include the Robin
      // spring. Evaluate both quantities collectively before rank-zero I/O.
      solid_problem.update_boundary_operators(time);
      FieldVector applied_load, stiffness_action;
      solid_problem.body_force_at_time(time, applied_load);
      stiffness_action.reinit(solid_problem.displacement());
      solid_problem.stiffness_matrix().vmult(stiffness_action,
                                             solid_problem.displacement());
      const double elastic_energy =
        .5 * (solid_problem.displacement() * stiffness_action);
      const double applied_work = solid_problem.displacement() * applied_load;
      bool         energy_output_ok = true;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ofstream energy(
            std::filesystem::path(solid_parameters.output_directory) /
              "energies.csv",
            step == 0 ? std::ios::out : std::ios::out | std::ios::app);
          if (step == 0)
            energy << "time_s,elastic_and_robin_energy_J,applied_load_work_J\n";
          energy << std::setprecision(17) << time << ',' << elastic_energy
                 << ',' << applied_work << '\n';
          energy.close();
          energy_output_ok = energy.good();
        }
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(
                                        energy_output_ok),
                                      MPI_COMM_WORLD) == 1,
                  ExcMessage("Cannot write the solid energy diagnostics."));

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

    auto initial_flow_state = flow_problem.make_state();
    adapter.set_output_step(
      [&solid_problem,
       &adapter,
       output,
       solid_fields,
       flow_fields,
       physical_network,
       &initial_flow_state](const double        time,
                            const GlobalVector &solution,
                            const GlobalVector &solution_dot,
                            const unsigned int  step) {
        if (physical_network && step == 0)
          initial_flow_state =
            adapter.field(solution, flow_fields.fields().state);
        solid_problem.accept_state(
          adapter.field(solution, solid_fields.fields().displacement),
          adapter.field(solution, solid_fields.fields().velocity),
          time,
          step);
        output(solution, solution_dot, step, time);
      });

    flow_problem.open_output_files();
    const auto accepted_steps = adapter.solve(state, state_dot);
    if (physical_network)
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
