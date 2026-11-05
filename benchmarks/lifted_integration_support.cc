// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <immersx/core/contributor.h>
#include <immersx/core/domain.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace dealii;
using namespace ImmersX;

namespace
{
  using Clock = std::chrono::steady_clock;

  unsigned int
  argument(const int          argc,
           char             **argv,
           const std::string &name,
           const unsigned int fallback)
  {
    const std::string prefix = "--" + name + "=";
    for (int i = 1; i < argc; ++i)
      {
        const std::string item(argv[i]);
        if (item.rfind(prefix, 0) == 0)
          return static_cast<unsigned int>(
            std::stoul(item.substr(prefix.size())));
      }
    return fallback;
  }

  double
  milliseconds(const Clock::duration duration)
  {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  std::pair<double, double>
  median_and_mad(std::vector<double> samples)
  {
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    for (auto &sample : samples)
      sample = std::abs(sample - median);
    std::sort(samples.begin(), samples.end());
    return {median, samples[samples.size() / 2]};
  }
} // namespace

int
main(int argc, char **argv)
{
  const unsigned int refinements = argument(argc, argv, "refinements", 4);
  const unsigned int repetitions = argument(argc, argv, "repetitions", 9);
  // Do not pass benchmark options to PETSc: its unused-option diagnostics
  // otherwise contaminate the JSONL stream at MPI finalization.
  std::vector<char *> runtime_arguments;
  for (int i = 0; i < argc; ++i)
    {
      const std::string option(argv[i]);
      if (option.rfind("--refinements=", 0) != 0 &&
          option.rfind("--repetitions=", 0) != 0)
        runtime_arguments.push_back(argv[i]);
    }
  int runtime_argc = static_cast<int>(runtime_arguments.size());
  runtime_arguments.push_back(nullptr);
  char                           **runtime_argv = runtime_arguments.data();
  Utilities::MPI::MPI_InitFinalize mpi(runtime_argc, runtime_argv, 1);
  AssertThrow(repetitions > 0,
              ExcMessage("The benchmark needs at least one repetition."));

  // Keep assembly diagnostics out of the JSONL records on stdout.
  auto *const json_output = std::cout.rdbuf(std::cerr.rdbuf());

  DomainParameters<1, 3> parameters("/Lifted support benchmark domain/");
  parameters.name_of_grid       = "hyper_cube";
  parameters.arguments_for_grid = "-1: 1: false";
  parameters.initial_refinement = refinements;
  Domain<1, 3> domain(parameters);

  TensorProductLift<1, 2, 3> lift("/Lifted support benchmark geometry/");
  lift.thickness                 = "0.05";
  lift.representative_n_q_points = 3;
  lift.section.inclusion_type    = "hyper_cube";
  lift.section.inclusion_degree  = 0;
  lift.section.n_q_points        = 2;

  MPI_Barrier(domain.communicator());
  const auto setup_start = Clock::now();
  domain.make_grid();
  const double local_setup_ms = milliseconds(Clock::now() - setup_start);
  const double setup_ms =
    Utilities::MPI::max(local_setup_ms, domain.communicator());

  MPI_Barrier(domain.communicator());
  const auto   cold_start    = Clock::now();
  auto         support       = domain.lift(lift);
  const double local_cold_ms = milliseconds(Clock::now() - cold_start);
  const double cold_build_ms =
    Utilities::MPI::max(local_cold_ms, domain.communicator());

  double local_measure = 0.;
  for (const auto &point : support.locally_owned_quadrature_points())
    local_measure += point.weight;
  const double measure =
    Utilities::MPI::sum(local_measure, domain.communicator());
  const auto n_points =
    Utilities::MPI::sum(static_cast<unsigned int>(
                          support.n_locally_owned_quadrature_points()),
                        domain.communicator());
  AssertThrow(std::isfinite(measure) && measure > 0.,
              ExcMessage("Lifted support produced an invalid measure."));
  AssertThrow(std::abs(measure - 0.2) < 1.e-10,
              ExcMessage("Lifted support measure differs from the expected "
                         "straight-strip measure."));

  support.rebuild(); // Untimed warmup.
  std::vector<double> rebuild_samples;
  rebuild_samples.reserve(repetitions);
  for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
    {
      MPI_Barrier(domain.communicator());
      const auto start = Clock::now();
      support.rebuild();
      const double local_ms = milliseconds(Clock::now() - start);
      rebuild_samples.push_back(
        Utilities::MPI::max(local_ms, domain.communicator()));
    }
  const auto [rebuild_median_ms, rebuild_mad_ms] =
    median_and_mad(rebuild_samples);

  MPI_Barrier(domain.communicator());
  const auto                              background_setup_start = Clock::now();
  parallel::distributed::Triangulation<3> background(domain.communicator());
  GridGenerator::hyper_cube(background, -1.1, 1.1);
  background.refine_global(refinements);
  FE_Q<3>       finite_element(1);
  DoFHandler<3> dof_handler(background);
  dof_handler.distribute_dofs(finite_element);
  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
  const auto relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  const auto space =
    fe_space(dof_handler, StaticMappingQ1<3>::mapping, constraints, relevant);
  StateLayout layout;
  const auto  source = space.field(layout, "bulk-source");
  const auto  target = space.field(layout, "bulk-target");
  using Vector       = ImmersXLA::MPI::Vector;
  using Matrix       = ImmersXLA::MPI::SparseMatrix;
  using Model        = SemiDiscreteModel<Vector, Matrix>;

  const double background_setup_ms =
    Utilities::MPI::max(milliseconds(Clock::now() - background_setup_start),
                        domain.communicator());
  std::shared_ptr<Model>  model;
  std::shared_ptr<Matrix> assembled_matrix;
  const auto              assemble_once = [&]() {
    MPI_Barrier(domain.communicator());
    const auto start = Clock::now();
    model            = std::make_shared<Model>();
    SemidiscreteBuilder<Vector, Matrix> builder(layout, *model);
    weak_term(value(source), test(target), support).add(builder);
    const StateView<Vector>         state_view(layout, 0.);
    const EvaluationContext<Vector> context(0., state_view);
    const auto matrix = model->state_matrix_operator(target.field_id(),
                                                     source.field_id(),
                                                     context);
    AssertThrow(matrix.has_value(),
                ExcMessage("The lifted weak term did not expose a matrix."));
    assembled_matrix = matrix->matrix();
    AssertThrow(assembled_matrix != nullptr,
                ExcMessage(
                  "The lifted weak-term matrix was not materialized."));
    return Utilities::MPI::max(milliseconds(Clock::now() - start),
                               domain.communicator());
  };
  const double first_assembly_ms = assemble_once();
  Vector       ones, action;
  ones.reinit(source.locally_owned_dofs(), domain.communicator());
  action.reinit(target.locally_owned_dofs(), domain.communicator());
  ones = 1.;
  ones.compress(VectorOperation::insert);
  const auto apply_once = [&]() {
    MPI_Barrier(domain.communicator());
    const auto start = Clock::now();
    assembled_matrix->vmult(action, ones);
    return Utilities::MPI::max(milliseconds(Clock::now() - start),
                               domain.communicator());
  };
  const double first_action_ms = apply_once();
  const double action_error    = std::abs(ones * action - measure);
  AssertThrow(action_error < 1.e-10,
              ExcMessage(
                "Lifted mass action differs from the support measure."));
  assemble_once();
  apply_once();
  std::vector<double> assembly_samples, action_samples;
  assembly_samples.reserve(repetitions);
  for (unsigned int repetition = 0; repetition < repetitions; ++repetition)
    {
      assembly_samples.push_back(assemble_once());
      action_samples.push_back(apply_once());
    }
  const auto [assembly_median_ms, assembly_mad_ms] =
    median_and_mad(assembly_samples);

  const auto [action_median_ms, action_mad_ms] = median_and_mad(action_samples);

  std::cout.rdbuf(json_output);
  if (Utilities::MPI::this_mpi_process(domain.communicator()) == 0)
    std::cout << "{\"case\":\"lifted_geometry_rebuild\","
              << "\"deal_ii_version\":\"" << DEAL_II_PACKAGE_VERSION << "\","
              << "\"dimension\":2,\"representative_dimension\":1,"
              << "\"spacedimension\":3,\"ranks\":"
              << Utilities::MPI::n_mpi_processes(domain.communicator())
              << ",\"refinements\":" << refinements
              << ",\"quadrature_points\":" << n_points
              << ",\"measure\":" << measure << ",\"setup_ms\":" << setup_ms
              << ",\"cold_support_build_ms\":" << cold_build_ms
              << ",\"warm_rebuild_median_ms\":" << rebuild_median_ms
              << ",\"warm_rebuild_mad_ms\":" << rebuild_mad_ms
              << ",\"first_bulk_assembly_ms\":" << first_assembly_ms
              << ",\"warm_bulk_assembly_median_ms\":" << assembly_median_ms
              << ",\"warm_bulk_assembly_mad_ms\":" << assembly_mad_ms
              << ",\"background_setup_ms\":" << background_setup_ms
              << ",\"background_dofs\":" << dof_handler.n_dofs()
              << ",\"cold_operator_action_ms\":" << first_action_ms
              << ",\"operator_action_median_ms\":" << action_median_ms
              << ",\"operator_action_mad_ms\":" << action_mad_ms
              << ",\"analytical_action_error\":" << action_error
              << ",\"timing_scope\":\"barrier_then_max_rank_wall_time\""
              << ",\"repetitions\":" << repetitions
              << ",\"native_reference\":\"unavailable\","
              << "\"immersx_over_dealii_ratio\":\"N/A\"}" << std::endl;
}
