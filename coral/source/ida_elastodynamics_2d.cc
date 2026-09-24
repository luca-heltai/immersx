// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>

#include <immersx/coral/ida_elastodynamics.h>
#include <immersx/physics/elastodynamics.h>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <immersx/core/sundials_ida_adapter.h>
#  include <immersx/physics/elastodynamics_semidiscrete.h>
#endif

namespace ImmersX::Coral
{
  class IDAElastodynamics2D::Impl
  {
  public:
    explicit Impl(const std::string &parameter_file)
      : parameters()
      , problem(parameters)
#ifdef DEAL_II_WITH_SUNDIALS
      , adapter(parameters.time_parameters,
                parameters.ida_parameters,
                MPI_COMM_WORLD)
#endif
    {
      dealii::ParameterAcceptor::initialize(parameter_file);
    }

    void
    run()
    {
#ifdef DEAL_II_WITH_SUNDIALS
      AssertThrow(!has_run,
                  dealii::ExcMessage(
                    "The IDA elastodynamics workflow can run only once."));
      problem.make_grid();
      problem.setup_fe();
      problem.setup_system();
      problem.assemble_operators();
      problem.set_initial_conditions();

      const auto fields = adapter.add(problem, "elastodynamics");
      adapter.set_output_step([this, fields](const double        time,
                                             const GlobalVector &state,
                                             const GlobalVector &state_dot,
                                             const unsigned int  step) {
        problem.accept_state(
          this->adapter.field(state, fields.fields().displacement),
          this->adapter.field(state, fields.fields().velocity),
          time,
          step);
        problem.output_results();
        (void)state_dot;
      });

      auto state     = adapter.make_state();
      auto state_dot = adapter.make_state();
      initialize_elastodynamics_adapter_state(
        adapter, fields, problem, state, state_dot);
      adapter.solve(state, state_dot);
      problem.accept_state(adapter.field(state, fields.fields().displacement),
                           adapter.field(state, fields.fields().velocity),
                           parameters.time_parameters.final_time,
                           parameters.fixed_step_parameters.number_of_steps);
      problem.output_results();
      has_run = true;
#else
      (void)parameter_file;
      AssertThrow(false,
                  dealii::ExcMessage(
                    "IDAElastodynamics2D requires deal.II with SUNDIALS "
                    "support."));
#endif
    }

    bool
    state_is_finite() const
    {
      return problem.state_is_finite();
    }

    double
    current_time() const
    {
      return problem.current_time();
    }

  private:
    using Problem = ElastodynamicsSolver<2, 2>;
#ifdef DEAL_II_WITH_SUNDIALS
    using FieldVector  = typename Problem::VectorType;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = IDAAdapter<FieldVector, GlobalVector>;
#endif

    ElastodynamicsParameters<2, 2> parameters;
    Problem                        problem;
#ifdef DEAL_II_WITH_SUNDIALS
    Adapter adapter;
#endif
    bool has_run = false;
  };

  IDAElastodynamics2D::IDAElastodynamics2D(const std::string &parameter_file)
    : implementation(std::make_unique<Impl>(parameter_file))
  {}

  IDAElastodynamics2D::~IDAElastodynamics2D() = default;

  void
  IDAElastodynamics2D::run()
  {
    implementation->run();
  }

  bool
  IDAElastodynamics2D::state_is_finite() const
  {
    return implementation->state_is_finite();
  }

  double
  IDAElastodynamics2D::current_time() const
  {
    return implementation->current_time();
  }
} // namespace ImmersX::Coral
