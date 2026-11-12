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

#include <deal.II/dofs/dof_tools.h>

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
#  include <immersx/grid/lift_co_refinement.h>
#  include <immersx/io/lifted_particle_output.h>
#  include <immersx/io/utils.h>
#  include <immersx/physics/elastodynamics.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#  include <immersx/physics/fictitious_interior.h>
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
#  include <limits>
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
  // zero does not guarantee that the lifted wall is inside the solid. The
  // interaction integrates wall terms on the overlap and sets lambda to zero
  // on the exterior support.
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
      add_parameter(
        "Sealed input terminal ids",
        sealed_input_terminals,
        "Physical Terminal ids explicitly closed to flow; every other tissue leaf must belong to a saved microvascular bed.");
      add_parameter("Redistribution halo",
                    halo,
                    "Physical archive halo in metres.");
    }

    std::vector<Vascularization::Id> sealed_input_terminals;
    std::string                      directory;
    std::vector<double>              young_moduli{0., 0., 0.};
    std::vector<double>              thickness_ratios{0., 0., 0.};
    std::vector<double>              reference_pressures{0., 0., 0.};
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

    // Unselected anatomical ends need an explicit zero-flow declaration.
    // Independent outlet resistances would add drains beside the saved beds.
    const std::set<V::Id> sealed(parameters.sealed_input_terminals.begin(),
                                 parameters.sealed_input_terminals.end());
    AssertThrow(sealed.size() == parameters.sealed_input_terminals.size(),
                ExcMessage("A sealed input terminal id is repeated."));
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
      if (terminal.owner == rank &&
          ((linked.count(id) == 0 && sealed.count(id) == 0) ||
           (linked.count(id) != 0 && sealed.count(id) != 0)))
        ++unlinked;
    AssertThrow(
      Utilities::MPI::sum(unlinked, comm) == 0,
      ExcMessage(
        "Every archived tissue leaf must have exactly one microvascular connection or explicit sealed declaration."));

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
        },
        sealed);
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

    LiftedParticleOutputParameters particle_output(
      "/MetricFlowX coupling/Particle output/");

    ParameterAcceptor coupled_solver(
      "/MetricFlowSystem<1, 3>/IDA parameters/Linear solver/");
    std::string linear_solver = "iterative";
    coupled_solver.add_parameter(
      "Type",
      linear_solver,
      "Coupled IDA linear solver. Direct solvers materialize each Jacobian. UMFPACK requires one rank; MUMPS supports MPI.",
      ParameterAcceptor::prm,
      Patterns::Selection("iterative|umfpack|mumps"));

    unsigned int flow_direct_inverse_limit   = 0;
    unsigned int schur_materialization_limit = 0;
    coupled_solver.add_parameter(
      "Maximum replicated flow inverse DoFs",
      flow_direct_inverse_limit,
      "Use replicated dense LU up to this global flow field size; zero retains the default ILU.");
    coupled_solver.add_parameter(
      "Maximum materialized Schur DoFs",
      schur_materialization_limit,
      "Cache Schur columns up to this global multiplier size per Jacobian; zero keeps matrix-free inner actions.");

    RefinementParameters<1> co_refinement(
      "/MetricFlowX coupling/Co-refinement/");
    co_refinement.refinement_strategy = "both";
    bool enable_co_refinement         = false;
    co_refinement.add_parameter(
      "Enabled",
      enable_co_refinement,
      "Co-refine the solid and fluid using physical lift supports before creating application FE spaces.");
    unsigned int maximum_section_level = 10;
    co_refinement.add_parameter(
      "Max section refinement level",
      maximum_section_level,
      "Limit adaptive circumferential refinement independently of the solid/fluid mesh level cap.");
    std::optional<LiftCoRefinementReport> co_refinement_report;

    ParameterAcceptor coupling("/MetricFlowX coupling/");
    bool              remove_fictitious_modes = false;
    coupling.add_parameter(
      "Remove fictitious modal response",
      remove_fictitious_modes,
      "Remove two averaged elastic and viscous strain modes in one straight uniform-radius reference cylinder, retaining all native inertia.");
    std::string embedded_partitioning = "default";
    coupling.add_parameter(
      "Embedded partitioning policy",
      embedded_partitioning,
      "default preserves ordinary Domain ownership; immersed follows the solid cell owner at each vessel cell center.",
      ParameterAcceptor::prm,
      Patterns::Selection("default|immersed"));

    ArchivedFlowParameters      archive_parameters;
    Vascularization::Parameters generation_parameters;
    bool                        use_native_tolerances      = false;
    bool                        ramp_solid_tolerances      = true;
    bool                        stop_at_output_times       = false;
    bool                        setup_only                 = false;
    int                         mumps_workspace_relaxation = -1;
    double displacement_tolerance = 0., velocity_tolerance = 0.,
           pressure_tolerance = 0.;
    coupling.add_parameter(
      "Use native flow tolerances",
      use_native_tolerances,
      "Use MetricFlowX area, velocity and capacitor absolute tolerances in the coupled adapter.");
    coupling.add_parameter(
      "Apply tolerance ramp to solid fields",
      ramp_solid_tolerances,
      "Apply the configured relative and absolute tolerance ramp to solid displacement and velocity; false keeps their configured tolerances throughout initialization and integration.");
    coupling.add_parameter(
      "Stop at output times",
      stop_at_output_times,
      "Take an internal IDA step at every saved time to preserve nonlinear algebraic constraints in output.");
    coupling.add_parameter(
      "Setup only",
      setup_only,
      "Build the coupled spaces and report block DoFs without initializing the state, correcting initial conditions or integrating.");
    coupling.add_parameter(
      "MUMPS workspace relaxation",
      mumps_workspace_relaxation,
      "Extra workspace percentage; -1 retains the MUMPS default.",
      ParameterAcceptor::prm,
      Patterns::Integer(-1));
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

    // Only Grid scale and Triangulation type are read from this section. The
    // mesh file, field import and refinement come from MetricFlowX.
    DomainParameters<1, 3> flow_domain_parameters("/MetricFlowX domain/");

    initialize_parameters(parameter_file);
    const auto setup_policy = static_cast<unsigned int>(setup_only);
    AssertThrow(Utilities::MPI::min(setup_policy, MPI_COMM_WORLD) ==
                  Utilities::MPI::max(setup_policy, MPI_COMM_WORLD),
                ExcMessage(
                  "All ranks must use the same coupled setup-only policy."));
    AssertThrow(
      !remove_fictitious_modes ||
        detail::has_external_pressure_state_jacobian<FlowProblem>::value,
      ExcMessage(
        "Remove fictitious modal response requires MetricFlowX with "
        "add_external_pressure_state_jacobian(). Rebuild against matching "
        "headers and libraries from a compatible MetricFlowX installation."));
    const bool archived_network = !archive_parameters.directory.empty();
    AssertThrow(!archived_network || !generate_network,
                ExcMessage(
                  "Choose either an archived network or vascular generation."));
    const bool physical_network = archived_network || generate_network;
    AssertThrow(
      !enable_co_refinement || !physical_network,
      ExcMessage(
        "Lift co-refinement currently requires native VTK flow input."));
    if (enable_co_refinement && embedded_partitioning != "default")
      {
        AssertThrow(co_refinement.embedded_partitioning_policy == "default" ||
                      co_refinement.embedded_partitioning_policy ==
                        embedded_partitioning,
                    ExcMessage("Conflicting embedded partitioning policies."));
        co_refinement.embedded_partitioning_policy = embedded_partitioning;
      }
    AssertThrow(!physical_network || embedded_partitioning == "default",
                ExcMessage("Embedded partitioning policy = immersed applies "
                           "only to the flow mesh read from the VTK file, not "
                           "to archived or generated networks."));
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
    if (!enable_co_refinement)
      {
        solid_problem.make_grid();
        solid_problem.setup_fe();
      }

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
            if (enable_co_refinement)
              {
                solid_problem.make_grid([&](auto &solid_domain) {
                  solid_domain.prepare_grid();
                  solid_domain.refinement_triangulation().set_mesh_smoothing(
                    Triangulation<3>::limit_level_difference_at_vertices);
                  co_refinement_report = co_refine_lifted_domains(
                    solid_domain,
                    flow_domain,
                    wall_lift,
                    co_refinement,
                    [&](const CellId &cell, const Point<1> &unit) {
                      const double area =
                        flow_domain.prepared_field_value("a0", cell, unit);
                      AssertThrow(std::isfinite(area) && area > 0.,
                                  ExcMessage("Reference a0 must be positive."));
                      return std::sqrt(area / numbers::PI);
                    },
                    maximum_section_level);
                });
              }
            else if (embedded_partitioning == "default")
              flow_domain.finalize_grid();
            else if (embedded_partitioning == "immersed")
              {
                const ImmersedRepartitioner<1, 3> policy(
                  solid_problem.triangulation());
                flow_domain.finalize_grid(policy);
              }
          });
      }
    if (enable_co_refinement)
      solid_problem.setup_fe();

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

    const auto reference_radius = sqrt(A0 / numbers::PI);
    const auto reference_wall =
      lift(value(lambda), wall_lift, reference_radius);
    auto wall_points = reference_wall.lifted_points();
    // This scalar multiplier uses the lift's normalized sum of mode directions.
    // Export that effective radial mode rather than the individual raw modes.
    for (auto &point : wall_points)
      point.mode_values = {point.mode_vector[0],
                           point.mode_vector[1],
                           point.mode_vector[2]};
    solid_problem.setup_system();
    if (enable_co_refinement)
      {
        const auto               &mesh = solid_problem.triangulation();
        std::vector<unsigned int> cells(mesh.n_global_levels(), 0);
        unsigned int              hanging = 0;
        for (const auto &cell : mesh.active_cell_iterators())
          if (cell->is_locally_owned())
            ++cells[cell->level()];
        for (auto &count : cells)
          count = Utilities::MPI::sum(count, MPI_COMM_WORLD);
        for (const auto &line :
             solid_problem.hanging_node_constraints().get_lines())
          if (solid_problem.locally_owned_dofs().is_element(line.index))
            ++hanging;
        hanging            = Utilities::MPI::sum(hanging, MPI_COMM_WORLD);
        const auto    rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        std::ofstream mesh_report;
        if (rank == 0)
          mesh_report.open(std::filesystem::path(flow_output_directory) /
                           "co_refinement.csv");
        AssertThrow(Utilities::MPI::min(rank != 0 || mesh_report.good() ? 1u :
                                                                          0u,
                                        MPI_COMM_WORLD) == 1,
                    ExcMessage("Cannot write co-refinement diagnostics."));
        if (rank == 0)
          {
            const auto &report = co_refinement_report.value();
            mesh_report.precision(17);
            mesh_report
              << "level,active_cells,solid_dofs,hanging_dofs,initial_solid_cells,"
                 "initial_fluid_cells,fluid_cells,solid_passes,fluid_passes,"
                 "section_passes,section_level,point_cell_pairs,outside_points,"
                 "axial_lower_violations,section_lower_violations,"
                 "axial_upper_violations,section_upper_violations,"
                 "axial_min_ratio,axial_max_ratio,section_min_ratio,section_max_ratio,"
                 "lower_factor,upper_factor\n";
            for (unsigned int level = 0; level < cells.size(); ++level)
              mesh_report << level << ',' << cells[level] << ','
                          << solid_problem.n_dofs() << ',' << hanging << ','
                          << report.initial_solid_cells << ','
                          << report.initial_fluid_cells << ','
                          << report.fluid_cells << ',' << report.solid_passes
                          << ',' << report.fluid_passes << ','
                          << report.section_passes << ','
                          << report.section_level << ','
                          << report.point_cell_pairs << ','
                          << report.outside_points << ','
                          << report.lower_violations[0] << ','
                          << report.lower_violations[1] << ','
                          << report.upper_violations[0] << ','
                          << report.upper_violations[1] << ','
                          << report.minimum_ratio[0] << ','
                          << report.maximum_ratio[0] << ','
                          << report.minimum_ratio[1] << ','
                          << report.maximum_ratio[1] << ','
                          << co_refinement.diameter_bounds().first << ','
                          << co_refinement.diameter_bounds().second << '\n';
          }
      }
    solid_problem.assemble_operators();
    solid_problem.set_initial_conditions();
    report_vessel_geometry(flow_problem.triangulation(), solid_problem);

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
    if (mumps_workspace_relaxation >= 0)
      adapter.set_mumps_workspace_relaxation(
        static_cast<unsigned int>(mumps_workspace_relaxation));
    adapter.set_schur_primal_coupling();
    adapter.set_schur_materialization_limit(schur_materialization_limit);
    if (linear_solver == "iterative")
      adapter.set_linear_solver(Adapter::LinearSolver::iterative);
    else if (linear_solver == "umfpack")
      adapter.set_linear_solver(Adapter::LinearSolver::umfpack);
    else if (linear_solver == "mumps")
      adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    const auto solid_fields = adapter.add(solid_problem, "elastodynamics");
    AssertThrow(
      Utilities::MPI::min(flow_direct_inverse_limit, MPI_COMM_WORLD) ==
          Utilities::MPI::max(flow_direct_inverse_limit, MPI_COMM_WORLD) &&
        Utilities::MPI::min(flow_problem.locally_owned_dofs().size(),
                            MPI_COMM_WORLD) ==
          Utilities::MPI::max(flow_problem.locally_owned_dofs().size(),
                              MPI_COMM_WORLD),
      ExcMessage("Flow inverse limits or dimensions differ across ranks."));
    auto flow_description = metric_flow_x(flow_problem);
    if (flow_direct_inverse_limit > 0 &&
        flow_problem.locally_owned_dofs().size() <= flow_direct_inverse_limit)
      flow_description.preconditioner_factory =
        [flow_direct_inverse_limit](const auto &matrix, const auto &prototype) {
          return make_replicated_direct_preconditioner<FieldVector>(
            matrix, prototype, flow_direct_inverse_limit);
        };
    const auto flow_fields = adapter.add(flow_description, "blood-flow");

    using ModalCorrection =
      FictitiousInteriorCorrection<FieldVector,
                                   typename SolidProblem::MatrixType>;
    std::shared_ptr<ModalCorrection> modal_correction;
    if (remove_fictitious_modes)
      {
        std::vector<Point<3>> local_points, points;
        for (const auto &cell :
             flow_problem.triangulation().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              local_points.push_back(cell->vertex(0));
              local_points.push_back(cell->vertex(1));
            }
        for (const auto &part :
             Utilities::MPI::all_gather(MPI_COMM_WORLD, local_points))
          points.insert(points.end(), part.begin(), part.end());
        AssertThrow(points.size() >= 2,
                    ExcMessage("Modal correction requires a cylinder."));
        const auto origin  = points[0];
        auto       tangent = points[1] - origin;
        tangent /= tangent.norm();
        std::vector<std::pair<double, double>> intervals;
        bool                                   straight = true;
        for (unsigned int i = 0; i < points.size(); i += 2)
          {
            const auto a = points[i] - origin, b = points[i + 1] - origin;
            straight = straight &&
                       (a - (a * tangent) * tangent).norm() < 1.e-10 &&
                       (b - (b * tangent) * tangent).norm() < 1.e-10;
            intervals.emplace_back(std::min(a * tangent, b * tangent),
                                   std::max(a * tangent, b * tangent));
          }
        std::sort(intervals.begin(), intervals.end());
        for (unsigned int i = 1; i < intervals.size(); ++i)
          straight = straight && std::abs(intervals[i].first -
                                          intervals[i - 1].second) < 1.e-10;
        double     minimum = std::numeric_limits<double>::max(), maximum = 0.;
        const auto selected_reference_dofs =
          DoFTools::extract_dofs(reference_field.dof_handler(),
                                 reference_field.component_mask());
        for (const auto i : selected_reference_dofs)
          if (reference_field.dof_handler().locally_owned_dofs().is_element(i))
            {
              const double area =
                reference_area[reference_field.execution_index(i)];
              minimum = std::min(minimum, area);
              maximum = std::max(maximum, area);
            }
        minimum      = Utilities::MPI::min(minimum, MPI_COMM_WORLD);
        maximum      = Utilities::MPI::max(maximum, MPI_COMM_WORLD);
        bool covered = true;
        for (const auto &point : reference_wall.lifted_points())
          covered = covered && supports.overlap.includes_lifted_point(point);
        covered = Utilities::MPI::min(covered ? 1u : 0u, MPI_COMM_WORLD) == 1;
        AssertThrow(
          straight,
          ExcMessage(
            "Modal correction requires one straight, connected, nonoverlapping cylinder."));
        AssertThrow(
          minimum > 0 && maximum - minimum < 1.e-12 * maximum,
          ExcMessage(
            "Modal correction requires a positive, uniform reference area."));
        AssertThrow(
          covered,
          ExcMessage(
            "Modal correction requires full reference-wall quadrature coverage."));
        FictitiousInteriorCylinder cylinder{origin +
                                              intervals.front().first * tangent,
                                            origin +
                                              intervals.back().second * tangent,
                                            std::sqrt(minimum / numbers::PI)};
        modal_correction   = std::make_shared<ModalCorrection>(solid_problem,
                                                             solid_parameters,
                                                             cylinder);
        const auto elastic = modal_correction->elastic_operator();
        const auto viscous = modal_correction->viscous_operator();
        const auto displacement_id = solid_fields.fields().displacement.id();
        const auto velocity_id     = solid_fields.fields().velocity.id();
        adapter.add(
          [=, &solid_problem](auto &builder) {
            builder.second_order_preconditioner(
              displacement_id,
              matrix_operator<FieldVector>(solid_problem.mass_matrix()),
              matrix_operator<FieldVector>(solid_problem.stiffness_matrix()),
              matrix_operator<FieldVector>(solid_problem.damping_matrix()));
            auto term = builder.term(velocity_id, "fictitious-interior-modes");
            term.residual([=](const auto &context) {
              return -1. * (elastic.view * context.state(displacement_id) +
                            viscous.view * context.state(velocity_id));
            });
            if (linear_solver == "iterative")
              term.state(displacement_id, -1. * elastic.view)
                .state(velocity_id, -1. * viscous.view);
            else if (linear_solver == "umfpack" || linear_solver == "mumps")
              term.state(displacement_id, -1. * elastic)
                .state(velocity_id, -1. * viscous);
            return velocity_id;
          },
          "fictitious-interior");
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Modal lumen volume: " << std::setprecision(17)
                    << modal_correction->volume() << " (reference "
                    << modal_correction->reference_volume() << ")" << std::endl;
      }

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
          .with_cached_assembly()
          .add_constraint_terms(
            builder, id, 1., 0, solid_fields.fields().velocity);
        weak_term(negative_wall_displacement, test(multiplier))
          .on(supports.overlap)
          .add(builder);
        weak_term(-1. * value(multiplier), test(multiplier))
          .on(supports.outside)
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
    if (!ramp_solid_tolerances)
      {
        adapter.set_tolerance_ramp_enabled(solid_fields.fields().displacement,
                                           false);
        adapter.set_tolerance_ramp_enabled(solid_fields.fields().velocity,
                                           false);
      }
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
    if (setup_only)
      {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout
            << "Coupled setup completed without state initialization or integration."
            << std::endl;
        return;
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
    std::vector<std::pair<double, std::string>> particle_output_records;
    const auto                                  output = [&solid_problem,
                         &solid_parameters,
                         &flow_problem,
                         &adapter,
                         &lambda_view,
                         &multiplier_output_records,
                         &flow_output_records,
                         &particle_output,
                         &particle_output_records,
                         &wall_points,
                         &reference_area,
                         reference_field,
                         coupling_fields,
                         pressure_projection,
                         modal_correction,
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
      if (particle_output.enabled())
        {
          // The native flow state also carries HDG/RCR entries beyond the FE
          // DoFs. Constrain its full layout before extracting FE coefficients.
          const auto ghosted = [](const auto &field, const auto &source) {
            FieldVector owned;
            owned.reinit(source);
            owned = source;
            if (Utilities::MPI::sum(field.constraints().n_constraints(),
                                    MPI_COMM_WORLD) != 0)
              field.constraints().distribute(owned);
            FieldVector execution;
            execution.reinit(source.locally_owned_elements(),
                             field.locally_relevant_dofs(),
                             MPI_COMM_WORLD);
            execution = owned;
            execution.update_ghost_values();
            FieldVector native;
            native.reinit(field.dof_handler().locally_owned_dofs(),
                          MPI_COMM_WORLD);
            native = 0.;
            for (const auto index : field.dof_handler().locally_owned_dofs())
              if (field.has_execution_index(index))
                native[index] = execution[field.execution_index(index)];
            native.compress(VectorOperation::insert);
            FieldVector result;
            result.reinit(field.dof_handler().locally_owned_dofs(),
                          DoFTools::extract_locally_relevant_dofs(
                            field.dof_handler()),
                          MPI_COMM_WORLD);
            result = native;
            result.update_ghost_values();
            return result;
          };
          const auto coefficients =
            ghosted(lambda_view.field("lambda"),
                    adapter.field(solution,
                                  coupling_fields.fields().multiplier));
          const auto samples =
            sample_at_representative_points(lambda_view.dof_handler(),
                                            lambda_view.mapping(),
                                            coefficients,
                                            wall_points);
          std::vector<double> target;
          if (particle_output.selected("modal_data"))
            {
              const auto area        = flow_fields.fields().area;
              const auto flow_values = ghosted(area, flow_state);
              const auto areas       = sample_at_representative_points(
                area.dof_handler(), area.mapping(), flow_values, wall_points);
              const auto reference =
                sample_at_representative_points(reference_field.dof_handler(),
                                                reference_field.mapping(),
                                                reference_area,
                                                wall_points);
              for (std::size_t q = 0; q < wall_points.size(); ++q)
                target.push_back(
                  std::sqrt(areas[q * area.finite_element().n_components() +
                                  area.extractor().component] /
                            numbers::PI) -
                  std::sqrt(reference[q * reference_field.finite_element()
                                            .n_components() +
                                      reference_field.extractor().component] /
                            numbers::PI));
            }
          LiftedParticleLabels labels;
          labels.value_name      = "wall_traction";
          labels.mode_labels     = {"radial"};
          labels.mode_components = 3;
          const auto columns     = make_lifted_particle_columns(
            particle_output, labels, wall_points, samples, target);
          FieldVector displacement;
          if (particle_output.selected("bulk_field"))
            {
              FieldVector owned;
              owned.reinit(solid_problem.locally_owned_dofs(), MPI_COMM_WORLD);
              owned = solid_problem.displacement();
              solid_problem.constraints().distribute(owned);
              displacement.reinit(solid_problem.locally_owned_dofs(),
                                  DoFTools::extract_locally_relevant_dofs(
                                    solid_problem.dof_handler()),
                                  MPI_COMM_WORLD);
              displacement = owned;
              displacement.update_ghost_values();
            }
          const auto filename =
            "wall_particles_" + std::to_string(step) + ".vtu";
          write_lifted_particles(
            particle_output,
            columns,
            wall_points,
            LiftedBulkField<3, FieldVector>{"displacement",
                                            solid_problem.dof_handler(),
                                            solid_problem.mapping(),
                                            &displacement,
                                            0,
                                            3},
            (std::filesystem::path(flow_output_directory) / filename).string());
          particle_output_records.emplace_back(time, filename);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::ofstream pvd(std::filesystem::path(flow_output_directory) /
                                "wall_particles.pvd");
              DataOutBase::write_pvd_record(pvd, particle_output_records);
            }
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
      FieldVector  velocity_action;
      velocity_action.reinit(solid_problem.velocity());
      solid_problem.mass_matrix().vmult(velocity_action,
                                        solid_problem.velocity());
      const double kinetic_energy =
        .5 * (solid_problem.velocity() * velocity_action);
      solid_problem.damping_matrix().vmult(velocity_action,
                                           solid_problem.velocity());
      const double native_dissipation =
        solid_problem.velocity() * velocity_action;
      const double removed_dissipation =
        modal_correction ? modal_correction->removed_viscous_dissipation(
                             solid_problem.velocity()) :
                           0.;
      const double removed_energy = modal_correction ?
                                      modal_correction->removed_elastic_energy(
                                        solid_problem.displacement()) :
                                      0.;
      const auto   modal_strain =
        modal_correction ?
            modal_correction->modes(solid_problem.displacement()) :
            std::array<double, 2>{};
      const auto modal_rate =
        modal_correction ? modal_correction->modes(solid_problem.velocity()) :
                           std::array<double, 2>{};
      bool energy_output_ok = true;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ofstream energy(
            std::filesystem::path(solid_parameters.output_directory) /
              "energies.csv",
            step == 0 ? std::ios::out : std::ios::out | std::ios::app);
          if (step == 0)
            energy
              << "time_s,elastic_and_robin_energy_J,applied_load_work_J,removed_modal_elastic_energy_J,effective_elastic_and_robin_energy_J,modal_g,modal_h,modal_g_dot,modal_h_dot,sampled_lumen_volume_m3,reference_lumen_volume_m3,kinetic_energy_J,native_dissipation_W,removed_modal_dissipation_W,effective_dissipation_W\n";
          energy << std::setprecision(17) << time << ',' << elastic_energy
                 << ',' << applied_work << ',' << removed_energy << ','
                 << elastic_energy - removed_energy << ',' << modal_strain[0]
                 << ',' << modal_strain[1] << ',' << modal_rate[0] << ','
                 << modal_rate[1] << ','
                 << (modal_correction ? modal_correction->volume() : 0.) << ','
                 << (modal_correction ? modal_correction->reference_volume() :
                                        0.)
                 << ',' << kinetic_energy << ',' << native_dissipation << ','
                 << removed_dissipation << ','
                 << native_dissipation - removed_dissipation << '\n';
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

    // IDA must limit Newton trials before either cell or trace areas leave
    // the constitutive domain. Its constraint code 2 means strictly positive.
    auto state_constraints =
      std::make_shared<GlobalVector>(adapter.make_state());
    *state_constraints = 0.;
    auto &flow_constraints =
      adapter.field(*state_constraints, flow_fields.fields().state);
    types::global_dof_index constrained_area_dofs = 0;
    for (const auto component :
         {FlowProblem::Component::area, FlowProblem::Component::area_trace})
      {
        const auto &indices = flow_problem.component_dofs(component);
        constrained_area_dofs += indices.n_elements();
        for (const auto index : indices)
          flow_constraints[index] = 2.;
      }
    state_constraints->compress(VectorOperation::insert);
    constrained_area_dofs =
      Utilities::MPI::sum(constrained_area_dofs, MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "IDA positive cell/trace area DoFs: "
                << constrained_area_dofs << std::endl;
    adapter.solver().get_constraint_vector =
      [state_constraints]() -> GlobalVector & { return *state_constraints; };

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
