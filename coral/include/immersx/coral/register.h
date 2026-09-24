// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coral_register_h
#define immersx_coral_register_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <coral.h>
#include <coral_log.h>
#include <coral_network.h>
#include <coral_plugin.h>
#include <immersx/physics/poisson.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ImmersX::Coral
{
  using json = nlohmann::json;

  inline std::string
  dimensions(const int dim, const int spacedim)
  {
    return std::to_string(dim) + "," + std::to_string(spacedim);
  }

  inline void
  register_common_types()
  {
    coral::detail::set_type_alias<unsigned int>("unsigned int");
    coral::detail::set_type_alias<std::string>("std::string");

    coral::NodeObject::register_elementary_type<std::string>();
    coral::NodeObject::register_elementary_type<bool>();
    coral::NodeObject::register_elementary_type<int>();
    coral::NodeObject::register_elementary_type<unsigned int>();
    coral::NodeObject::register_elementary_type<double>();
    coral::Network::register_node();
  }

  template <int dim, int spacedim>
  inline std::string
  poisson_parameters_name()
  {
    return "ImmersX::PoissonParameters<" + dimensions(dim, spacedim) + ">";
  }

  template <int dim, int spacedim>
  inline std::string
  poisson_name()
  {
    return "ImmersX::Poisson<" + dimensions(dim, spacedim) + ">";
  }

  template <int dim, int spacedim>
  void
  load_poisson_parameters(ImmersX::PoissonParameters<dim, spacedim> &parameters,
                          const std::string                         &file_name)
  {
    (void)parameters;
    std::ifstream input(file_name);
    AssertThrow(input.good(),
                dealii::ExcMessage("Could not open Coral parameter file '" +
                                   file_name + "'."));

    // ParameterAcceptor owns the declaration and parsing callbacks used by
    // the existing Problem parameter classes. A graph should keep one
    // parameter bundle alive and load it before constructing its Problem.
    dealii::ParameterAcceptor::initialize(input);
  }

  template <int dim, int spacedim>
  double
  poisson_solution_l2_norm(const ImmersX::PoissonSolver<dim, spacedim> &problem)
  {
    return problem.solution_l2_norm();
  }

  template <int dim, int spacedim>
  bool
  poisson_solution_is_finite(
    const ImmersX::PoissonSolver<dim, spacedim> &problem)
  {
    return problem.solution_is_finite();
  }

  template <int dim, int spacedim>
  void
  register_poisson_types()
  {
    using Parameters = ImmersX::PoissonParameters<dim, spacedim>;
    using Problem    = ImmersX::PoissonSolver<dim, spacedim>;

    coral::detail::set_type_alias<Parameters>(
      poisson_parameters_name<dim, spacedim>());
    coral::detail::set_type_alias<Problem>(poisson_name<dim, spacedim>());

    coral::NodeObject::register_type<Parameters, const std::string &>(
      "subsection");
    coral::NodeObject::register_function(
      std::function<void(Parameters &, const std::string &)>(
        &load_poisson_parameters<dim, spacedim>),
      {"ImmersX::LoadPoissonParameters<" + dimensions(dim, spacedim) + ">",
       "parameters",
       "parameter_file"});

    coral::NodeObject::register_type<Problem, const Parameters &>("parameters");

    const auto name = poisson_name<dim, spacedim>();
    coral::NodeObject::register_method<Problem, void>(&Problem::make_grid,
                                                      {name + "::make_grid",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::setup_fe,
                                                      {name + "::setup_fe",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::setup_system,
                                                      {name + "::setup_system",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::assemble_system, {name + "::assemble_system", "problem"});
    coral::NodeObject::register_method<Problem, void>(&Problem::solve,
                                                      {name + "::solve",
                                                       "problem"});
    coral::NodeObject::register_method<Problem, void>(
      &Problem::output_results, {name + "::output_results", "problem"});
    coral::NodeObject::register_function(
      std::function<double(const Problem &)>(
        &poisson_solution_l2_norm<dim, spacedim>),
      {name + "::solution_l2_norm", "problem", "norm"});
    coral::NodeObject::register_function(
      std::function<bool(const Problem &)>(
        &poisson_solution_is_finite<dim, spacedim>),
      {name + "::solution_is_finite", "problem", "is_finite"});
  }

  template <int spacedim>
  void
  register_immersx_types()
  {
    register_common_types();
    register_poisson_types<1, spacedim>();
    if constexpr (spacedim >= 2)
      register_poisson_types<2, spacedim>();
    if constexpr (spacedim >= 3)
      register_poisson_types<3, spacedim>();
  }

  template <int spacedim>
  struct PluginState
  {
    static inline std::unique_ptr<dealii::Utilities::MPI::MPI_InitFinalize>
      mpi_session;
  };

  template <int spacedim>
  inline std::string
  plugin_name()
  {
    return "immersx-coral-" + std::to_string(spacedim) + "d";
  }

  template <int spacedim>
  int
  load_plugin(const char *subjson, const CoralLogger *logger)
  {
    coral_active_logger      = logger;
    coral_active_plugin_name = coral_plugin_name();

    bool         mpi_enabled     = false;
    unsigned int max_num_threads = dealii::numbers::invalid_unsigned_int;
    std::vector<std::string> args;

    if (subjson != nullptr)
      {
        try
          {
            const auto init = json::parse(subjson);
            if (init.contains("MPI"))
              {
                const auto &mpi = init.at("MPI");
                mpi_enabled     = mpi.value("enabled", mpi_enabled);
                max_num_threads = mpi.value("max_num_threads", max_num_threads);
                args            = mpi.value("args", args);
              }
          }
        catch (const std::exception &exception)
          {
            coral_log_error("Invalid ImmersX plugin initialization JSON: %s",
                            exception.what());
            return 1;
          }
      }

    if (mpi_enabled)
      {
        std::vector<char *> argv;
        argv.reserve(args.size());
        for (auto &argument : args)
          argv.push_back(argument.data());
        int    argc     = static_cast<int>(argv.size());
        char **argv_ptr = argv.data();
        PluginState<spacedim>::mpi_session =
          std::make_unique<dealii::Utilities::MPI::MPI_InitFinalize>(
            argc, argv_ptr, max_num_threads);
      }

    register_immersx_types<spacedim>();
    return 0;
  }

  template <int spacedim>
  void
  unload_plugin()
  {
    PluginState<spacedim>::mpi_session.reset();
  }
} // namespace ImmersX::Coral

#endif // immersx_coral_register_h
