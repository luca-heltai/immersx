// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/coral/coupled_poisson.h>
#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

namespace ImmersX::Coral
{
  namespace
  {
    class ApplicationParameters : public dealii::ParameterAcceptor
    {
    public:
      ApplicationParameters()
        : ParameterAcceptor("/Coupled Poisson/")
      {
        add_parameter("Output directory", output_directory);
        add_parameter("Multiplier output name", multiplier_output_name);
      }

      std::string output_directory       = ".";
      std::string multiplier_output_name = "multiplier";
    };
  } // namespace

  class CoupledPoisson2D::Impl
  {
  public:
    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    explicit Impl(const std::string &parameter_file)
      : application_parameters()
      , bulk_parameters()
      , embedded_parameters("/Embedded Poisson/")
      , adapter_parameters()
      , bulk_problem(bulk_parameters)
      , embedded_problem(embedded_parameters)
      , adapter(adapter_parameters, MPI_COMM_WORLD)
    {
      dealii::ParameterAcceptor::initialize(parameter_file);

      bulk_parameters.output_directory =
        application_parameters.output_directory;
      embedded_parameters.output_directory =
        application_parameters.output_directory;
    }

    void
    run()
    {
      const auto initialize_problem = [](auto &problem) {
        problem.make_grid();
        problem.setup_fe();
        problem.setup_system();
        problem.assemble_system();
      };
      initialize_problem(bulk_problem);
      initialize_problem(embedded_problem);

      const auto bulk     = adapter.add(bulk_problem, "bulk");
      const auto embedded = adapter.add(embedded_problem, "embedded");

      const auto bulk_view = fe_space(bulk_problem.dof_handler(),
                                      dealii::StaticMappingQ1<2>::mapping,
                                      bulk_problem.constraints(),
                                      bulk_problem.locally_relevant_dofs());
      const auto embedded_view =
        fe_space(embedded_problem.dof_handler(),
                 dealii::StaticMappingQ1<1, 2>::mapping,
                 embedded_problem.constraints(),
                 embedded_problem.locally_relevant_dofs());

      dealii::FE_DGQ<1, 2>     multiplier_fe(0);
      dealii::DoFHandler<1, 2> multiplier_dh(embedded_problem.triangulation());
      multiplier_dh.distribute_dofs(multiplier_fe);
      const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
      const auto multiplier_relevant =
        DoFTools::extract_locally_relevant_dofs(multiplier_dh);
      dealii::AffineConstraints<double> multiplier_constraints;
      multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
      multiplier_constraints.close();

      const auto multiplier_view =
        fe_space(multiplier_dh,
                 dealii::StaticMappingQ1<1, 2>::mapping,
                 multiplier_constraints,
                 multiplier_relevant);
      const auto bulk_field =
        bulk_view.field(bulk.fields().solution, "bulk_solution");
      const auto embedded_field =
        embedded_view.field(embedded.fields().solution, "embedded_solution");
      const auto multiplier = multiplier_view.field("lambda");
      const auto constraint =
        make_constraint(weak_term(value(bulk_field), test(multiplier)) -
                        weak_term(value(embedded_field), test(multiplier)));
      const auto coupling = adapter.add(constraint, "continuity");

      auto state = adapter.make_state();
      adapter.solve(state);

      bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
      embedded_problem.set_solution(
        adapter.field(state, embedded.fields().solution));
      bulk_problem.output_results();
      embedded_problem.output_results();

      write_multiplier_output(multiplier_dh,
                              adapter.field(state,
                                            coupling.fields().multiplier));

      auto residual = adapter.make_state();
      adapter.evaluate_residual(state, residual);
      residual_norm_storage = residual.l2_norm();
      AssertThrow(
        std::isfinite(residual_norm_storage) && residual_norm_storage < 1.e-7,
        dealii::ExcMessage("The Coral coupled Poisson residual is too large."));
    }

    double
    residual_norm() const
    {
      return residual_norm_storage;
    }

  private:
    void
    write_multiplier_output(const dealii::DoFHandler<1, 2> &multiplier_dh,
                            const ImmersXLA::MPI::Vector   &multiplier) const
    {
      std::filesystem::create_directories(
        application_parameters.output_directory);
      dealii::DataOut<1, 2> multiplier_output;
      multiplier_output.attach_dof_handler(multiplier_dh);
      multiplier_output.add_data_vector(multiplier,
                                        "lagrange_multiplier",
                                        dealii::DataOut<1, 2>::type_dof_data);
      multiplier_output.build_patches();
      const auto rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const auto multiplier_filename =
        std::filesystem::path(application_parameters.output_directory) /
        (application_parameters.multiplier_output_name + "-0." +
         std::to_string(rank) + ".vtu");
      std::ofstream multiplier_file(multiplier_filename);
      multiplier_output.write_vtu(multiplier_file);
      if (rank == 0)
        {
          std::ofstream multiplier_pvd(
            std::filesystem::path(application_parameters.output_directory) /
            (application_parameters.multiplier_output_name + ".pvd"));
          dealii::DataOutBase::write_pvd_record(
            multiplier_pvd,
            {{0., application_parameters.multiplier_output_name + "-0.0.vtu"}});
        }
    }

    ApplicationParameters   application_parameters;
    PoissonParameters<2>    bulk_parameters;
    PoissonParameters<1, 2> embedded_parameters;
    LinearSolverParameters  adapter_parameters;
    PoissonSolver<2>        bulk_problem;
    PoissonSolver<1, 2>     embedded_problem;
    Adapter                 adapter;
    double residual_norm_storage = std::numeric_limits<double>::quiet_NaN();
  };

  CoupledPoisson2D::CoupledPoisson2D(const std::string &parameter_file)
    : implementation(std::make_unique<Impl>(parameter_file))
  {}

  CoupledPoisson2D::~CoupledPoisson2D() = default;

  void
  CoupledPoisson2D::run()
  {
    implementation->run();
  }

  double
  CoupledPoisson2D::residual_norm() const
  {
    return implementation->residual_norm();
  }
} // namespace ImmersX::Coral
