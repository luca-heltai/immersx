// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <immersx/coral/coupled_poisson_elasticity.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/observable.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/poisson.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

#include "coupled_poisson_elasticity.h"

namespace ImmersX::Coral
{
  class CoupledPoissonElasticity3D::Impl
  {
  public:
    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    explicit Impl(const std::string &parameter_file)
      : poisson_parameters()
      , elasticity_parameters()
      , pressure_lift("/Pressure lift/")
      , surface()
      , adapter_parameters()
      , poisson_problem(poisson_parameters)
      , elasticity_problem(elasticity_parameters)
      , adapter(adapter_parameters, MPI_COMM_WORLD)
    {
      dealii::ParameterAcceptor::initialize(parameter_file);
    }

    void
    run()
    {
      AssertThrow(!has_run,
                  dealii::ExcMessage(
                    "The coupled Poisson-elasticity workflow can run only "
                    "once."));

      poisson_problem.make_grid();
      poisson_problem.setup_fe();
      poisson_problem.setup_system();
      poisson_problem.assemble_system();

      elasticity_problem.setup();

      const auto poisson_fields = adapter.add(poisson_problem, "pressure");
      const auto elastic_fields = adapter.add(elasticity_problem, "elastic");

      const auto elastic_space =
        fe_space(elasticity_problem.dof_handler(),
                 dealii::StaticMappingQ1<3, 3>::mapping,
                 elasticity_problem.constraints(),
                 elasticity_problem.locally_relevant_dofs());
      const auto displacement =
        elastic_space.field(elastic_fields.fields().displacement,
                            "displacement",
                            dealii::FEValuesExtractors::Vector(0));

      const auto poisson_space =
        fe_space(poisson_problem.dof_handler(),
                 dealii::StaticMappingQ1<1, 3>::mapping,
                 poisson_problem.constraints(),
                 poisson_problem.locally_relevant_dofs());
      const auto poisson_field =
        poisson_space.field(poisson_fields.fields().solution,
                            "pressure",
                            dealii::FEValuesExtractors::Scalar(0));

      const auto pressure =
        ImmersX::make_lift(CoupledPoissonElasticity::Pressure{}.factor *
                             ImmersX::value(poisson_field),
                           pressure_lift);
      const auto traction = pressure * ImmersX::normal(surface);
      adapter.add(ImmersX::weak_term(traction, ImmersX::test(displacement)),
                  "pressure-traction");

      auto state = adapter.make_state();
      adapter.solve(state);

      const auto &poisson_state =
        adapter.field(state, poisson_fields.fields().solution);
      const auto &elastic_state =
        adapter.field(state, elastic_fields.fields().displacement);
      poisson_problem.set_solution(poisson_state);
      elasticity_problem.set_solution(elastic_state);
      poisson_problem.output_results();
      elasticity_problem.output_results(0);

      auto residual = adapter.make_state();
      adapter.evaluate_residual(state, residual);
      residual_norm_storage = residual.l2_norm();
      pressure_scale_error_storage =
        std::abs(CoupledPoissonElasticity::Pressure{}.factor - 2.);
      traction_balance_error_storage =
        CoupledPoissonElasticity::traction_balance(pressure,
                                                   surface,
                                                   displacement,
                                                   elasticity_problem,
                                                   poisson_state,
                                                   elastic_state);

      std::filesystem::create_directories(poisson_parameters.output_directory);
      std::ofstream diagnostics(
        std::filesystem::path(poisson_parameters.output_directory) /
        "coupled_poisson_elasticity_diagnostics.txt");
      diagnostics << "coupled_residual = " << residual_norm_storage << '\n'
                  << "pressure_scale_error = " << pressure_scale_error_storage
                  << '\n'
                  << "traction_balance_error = "
                  << traction_balance_error_storage << '\n';

      AssertThrow(std::isfinite(residual_norm_storage) &&
                    residual_norm_storage < 1.e-9,
                  dealii::ExcMessage("Coupled solve residual is too large."));
      AssertThrow(pressure_scale_error_storage < 1.e-14,
                  dealii::ExcMessage("Pressure scaling check failed."));
      AssertThrow(traction_balance_error_storage < 1.e-9,
                  dealii::ExcMessage("Traction balance check failed."));
      has_run = true;
    }

    double
    residual_norm() const
    {
      return residual_norm_storage;
    }

    double
    pressure_scale_error() const
    {
      return pressure_scale_error_storage;
    }

    double
    traction_balance_error() const
    {
      return traction_balance_error_storage;
    }

  private:
    PoissonParameters<1, 3>                   poisson_parameters;
    ElasticStaticParameters<3, 3>             elasticity_parameters;
    CoupledPoissonElasticity::PressureLift    pressure_lift;
    CoupledPoissonElasticity::CylinderSurface surface;
    LinearSolverParameters                    adapter_parameters;
    PoissonSolver<1, 3>                       poisson_problem;
    ElasticStaticProblem<3, 3>                elasticity_problem;
    Adapter                                   adapter;
    double residual_norm_storage = std::numeric_limits<double>::quiet_NaN();
    double pressure_scale_error_storage =
      std::numeric_limits<double>::quiet_NaN();
    double traction_balance_error_storage =
      std::numeric_limits<double>::quiet_NaN();
    bool has_run = false;
  };

  CoupledPoissonElasticity3D::CoupledPoissonElasticity3D(
    const std::string &parameter_file)
    : implementation(std::make_unique<Impl>(parameter_file))
  {}

  CoupledPoissonElasticity3D::~CoupledPoissonElasticity3D() = default;

  void
  CoupledPoissonElasticity3D::run()
  {
    implementation->run();
  }

  double
  CoupledPoissonElasticity3D::residual_norm() const
  {
    return implementation->residual_norm();
  }

  double
  CoupledPoissonElasticity3D::pressure_scale_error() const
  {
    return implementation->pressure_scale_error();
  }

  double
  CoupledPoissonElasticity3D::traction_balance_error() const
  {
    return implementation->traction_balance_error();
  }
} // namespace ImmersX::Coral
