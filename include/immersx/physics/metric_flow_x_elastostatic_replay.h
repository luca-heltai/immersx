// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_metric_flow_x_elastostatic_replay_h
#define immersx_metric_flow_x_elastostatic_replay_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <deal.II/base/conditional_ostream.h>
#  include <deal.II/base/mpi.h>
#  include <deal.II/base/parameter_acceptor.h>

#  include <deal.II/fe/fe_dgq.h>
#  include <deal.II/fe/fe_tools.h>
#  include <deal.II/fe/mapping_q1.h>

#  include <deal.II/grid/grid_tools_cache.h>

#  include <deal.II/numerics/data_out.h>

#  include <immersx/core/fe_space.h>
#  include <immersx/core/linear_adapter.h>
#  include <immersx/core/observable_lift.h>
#  include <immersx/core/overlap_integration_support.h>
#  include <immersx/coupling/tensor_product_lift.h>
#  include <immersx/physics/elastic_static.h>
#  include <immersx/physics/metric_flow_x_pressure.h>
#  include <immersx/physics/metric_flow_x_wall.h>
#  include <metric_flow_x/blood_flow_system.h>

#  include <algorithm>
#  include <cmath>
#  include <cstdlib>
#  include <filesystem>
#  include <fstream>
#  include <memory>
#  include <optional>
#  include <string>
#  include <tuple>
#  include <type_traits>
#  include <utility>
#  include <vector>

namespace ImmersX
{
  /** Parameters of a static solid driven by a saved MetricFlowX history.
   *
   * The solid, wall lift and multiplier sections use the same names as the
   * metric_flow_x_elastostatic application, so both can share them.
   */
  struct MetricFlowXElastostaticReplayParameters
    : public dealii::ParameterAcceptor
  {
    MetricFlowXElastostaticReplayParameters()
      : dealii::ParameterAcceptor("/MetricFlowX replay/")
      , solid("/Elastic static/")
      , wall_lift("/MetricFlowX vessel wall lift/")
      , multiplier_space("/MetricFlowX multiplier space/", "auto")
      , execution("/MetricFlowX replay/Linear adapter/")
    {
      add_parameter(
        "Checkpoint history",
        checkpoint_history,
        "Directory written by blood_flow --checkpoint-directory. Every "
        "committed checkpoint in it is replayed in time order.");
    }

    ElasticStaticParameters<3>         solid;
    TensorProductLift<1, 2, 3, 3>      wall_lift;
    FiniteElementSpaceParameters<1, 3> multiplier_space;
    LinearAdapterParameters            execution;
    std::string                        checkpoint_history;
  };

  /** A saved native flow state: its physical time and checkpoint directory. */
  struct MetricFlowXCheckpoint
  {
    double      time = 0.;
    std::string directory;
  };

  /** Committed native checkpoints in a blood_flow history, sorted by time.
   *
   * MetricFlowX names each checkpoint `checkpoint-<hexadecimal time>`.
   * Staging directories ending in `.incomplete` are skipped. Collective.
   */
  inline std::vector<MetricFlowXCheckpoint>
  metric_flow_x_checkpoint_history(const std::string &history,
                                   const MPI_Comm     communicator)
  {
    using namespace dealii;
    std::vector<std::pair<double, std::string>> entries;
    std::string                                 error;
    if (Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        try
          {
            const std::string prefix = "checkpoint-";
            for (const auto &entry :
                 std::filesystem::directory_iterator(history))
              {
                const auto name = entry.path().filename().string();
                if (!entry.is_directory() || name.rfind(prefix, 0) != 0 ||
                    entry.path().extension() == ".incomplete")
                  continue;
                const auto text = name.substr(prefix.size());
                char      *end  = nullptr;
                const auto time = std::strtod(text.c_str(), &end);
                AssertThrow(end == text.c_str() + text.size() &&
                              std::isfinite(time),
                            ExcMessage("Cannot read the time of checkpoint " +
                                       name + "."));
                entries.emplace_back(
                  time, std::filesystem::absolute(entry.path()).string());
              }
            std::sort(entries.begin(), entries.end());
            AssertThrow(!entries.empty(),
                        ExcMessage("No MetricFlowX checkpoints in " + history +
                                   "."));
          }
        catch (const std::exception &exception)
          {
            error = exception.what();
          }
      }
    error = Utilities::MPI::broadcast(communicator, error, 0);
    AssertThrow(error.empty(), ExcMessage(error));
    entries = Utilities::MPI::broadcast(communicator, entries, 0);
    std::vector<MetricFlowXCheckpoint> result;
    for (const auto &[time, directory] : entries)
      result.push_back({time, directory});
    return result;
  }

  /**
   * One-way coupling of a saved MetricFlowX history to static elasticity.
   *
   * For each checkpoint time t_n, the solid solves
   *
   *   K u + B lambda = f(t_n),    B^T u = P_h(A_n),
   *
   * with the native flow area A_n read from the checkpoint and
   * P_h(A) = (sqrt(A/pi) - sqrt(a0/pi), mu) on the lifted wall overlap.
   * These are the solid and wall rows of metric_flow_x_elastostatic without
   * pressure feedback. The reference area a0 is the area of the first
   * checkpoint in the history.
   *
   * The ElasticStaticProblem owns the solid; one LinearAdapter owns the
   * saddle-point system. Stiffness, wall pairing and multiplier metric are
   * assembled once; each step updates the loads, the Dirichlet data and the
   * shared area vector. A direct or MUMPS solver is factored once.
   *
   * Native MetricFlowX checkpoints load into a fresh BloodFlowSystem and
   * reparse the global ParameterAcceptor tree. setup() and solve_step()
   * therefore call ParameterAcceptor::clear() before each load: parse all
   * parameters before setup(). All checkpoints must come from one history,
   * with the MPI size used to write them.
   */
  class MetricFlowXElastostaticReplay
  {
  public:
    using FlowProblem  = MetricFlowX::BloodFlowSystem<1, 3>;
    using Vector       = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<Vector, GlobalVector>;
    using SolidFields  = ElasticStaticFields<3>;
    using Multiplier   = Field<1, 3, dealii::FEValuesExtractors::Scalar>;

    static_assert(
      std::is_same_v<Vector, MetricFlowX::VectorType>,
      "MetricFlowX and ImmersX must use the same linear algebra backend.");

    explicit MetricFlowXElastostaticReplay(
      const MetricFlowXElastostaticReplayParameters &parameters,
      const MPI_Comm communicator = MPI_COMM_WORLD)
      : parameters_(parameters)
      , communicator_(communicator)
      , pcout_(std::cout,
               dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      , solid_(parameters.solid, communicator)
    {}

    /** Load the first checkpoint, build the solid and the wall coupling. */
    void
    setup()
    {
      using namespace dealii;
      AssertThrow(!adapter_, ExcMessage("The replay may only be set up once."));
      AssertThrow(!parameters_.checkpoint_history.empty(),
                  ExcMessage("Set MetricFlowX replay/Checkpoint history."));
      history_ =
        metric_flow_x_checkpoint_history(parameters_.checkpoint_history,
                                         communicator_);
      pcout_ << "MetricFlowX replay: " << history_.size()
             << " checkpoint(s) from " << history_.front().time << " to "
             << history_.back().time << std::endl;

      // The first checkpoint provides the flow mesh for the multiplier and
      // the reference area a0.
      ParameterAcceptor::clear();
      flow_ = std::make_unique<FlowProblem>(communicator_);
      flow_->load_checkpoint(history_.front().directory);
      loaded_step_ = 0;

      flow_space_.emplace(flow_->dof_handler(),
                          StaticMappingQ1<1, 3>::mapping,
                          flow_->constraints(),
                          &flow_->locally_relevant_dofs());
      const auto area_field =
        flow_space_->field("area", flow_->area_extractor());
      // The wall constraint shares this vector. Overwriting it with the
      // state of another checkpoint changes only P_h(A) in the next solve.
      area_  = std::make_shared<Vector>(flow_->locally_owned_dofs(),
                                       flow_->locally_relevant_dofs(),
                                       communicator_);
      *area_ = flow_->state();
      area_->update_ghost_values();
      const Vector reference_area(*area_);
      const auto   reference_radius =
        sqrt(frozen(area_field, reference_area) / numbers::PI);

      const auto *area_fe = dynamic_cast<const FE_DGQ<1, 3> *>(
        &flow_->finite_element().base_element(0));
      AssertThrow(area_fe != nullptr,
                  ExcMessage("The native flow area must use FE_DGQ."));
      const auto &name  = parameters_.multiplier_space.finite_element;
      multiplier_space_ = std::make_unique<FiniteElementSpace<1, 3>>(
        flow_->triangulation(),
        name == "auto" ? std::unique_ptr<FiniteElement<1, 3>>(
                           std::make_unique<FE_DGQ<1, 3>>(area_fe->degree)) :
                         FETools::get_fe_by_name<1, 3>(name));
      const auto *lambda_fe = dynamic_cast<const FE_DGQ<1, 3> *>(
        &multiplier_space_->finite_element());
      AssertThrow(lambda_fe != nullptr && area_fe->degree == lambda_fe->degree,
                  ExcMessage("The multiplier must use the same local DGQ "
                             "basis as native area."));
      multiplier_view_.emplace(multiplier_space_->view());
      const auto lambda = multiplier_view_->field("lambda");

      solid_.use_fixed_stiffness();
      solid_.setup();
      solid_.update_time(history_.front().time);

      const auto reference_wall =
        lift(value(lambda), parameters_.wall_lift, reference_radius);
      GridTools::Cache<3> solid_cache(solid_.triangulation(), solid_.mapping());
      const auto          supports =
        make_overlap_supports(IntegrationSupport<1, 3>(flow_->triangulation()),
                              lambda.mapping(),
                              reference_wall.representative_quadrature(),
                              reference_wall.lifted_points(),
                              solid_cache);
      const auto metric = std::get<1>(
        make_metric_flow_x_pressure_projection(lambda, supports.overlap));

      adapter_ =
        std::make_unique<Adapter>(parameters_.execution, communicator_);
      solid_fields_.emplace(adapter_->add(solid_, "elastostatics").fields());
      multiplier_.emplace(adapter_
                            ->add(metric_flow_x_prescribed_wall_constraint(
                                    solid_fields_->displacement,
                                    frozen(area_field, area_),
                                    lambda,
                                    parameters_.wall_lift,
                                    reference_radius,
                                    supports,
                                    metric),
                                  "vessel-wall")
                            .fields()
                            .multiplier);
      state_    = adapter_->make_state();
      residual_ = adapter_->make_state();
    }

    /** Load the flow area of one checkpoint and solve the static system. */
    void
    solve_step(const unsigned int step)
    {
      using namespace dealii;
      AssertThrow(adapter_, ExcMessage("Call setup() before solve_step()."));
      AssertIndexRange(step, history_.size());
      if (step != loaded_step_)
        {
          // Each native load needs a fresh flow object and parses the global
          // parameter tree. Clearing it first keeps the add_parameter actions
          // of destroyed flows from running.
          ParameterAcceptor::clear();
          FlowProblem scratch(communicator_);
          scratch.load_checkpoint(history_[step].directory);
          assert_same_discretization(scratch);
          *area_ = scratch.state();
          area_->update_ghost_values();
          loaded_step_ = step;
        }

      solid_.update_time(history_[step].time);
      const bool direct =
        parameters_.execution.solver == LinearSolver::direct ||
        parameters_.execution.solver == LinearSolver::mumps;
      if (direct)
        {
          if (!factorized_)
            adapter_->setup_direct(state_);
          factorized_ = true;
          adapter_->solve_with_current_direct(state_);
        }
      else
        adapter_->solve(state_);

      adapter_->evaluate_residual(state_, residual_);
      residual_norm_ = residual_.l2_norm();
      AssertThrow(std::isfinite(residual_norm_),
                  ExcMessage("Nonfinite replay residual."));
      solid_.set_solution(displacement());
      pcout_ << "Replay step " << step << ", time " << history_[step].time
             << ", residual " << residual_norm_ << std::endl;
    }

    /** Write solid displacement and wall multiplier for one step. */
    void
    output_step(const unsigned int step)
    {
      using namespace dealii;
      solid_.output_results(step, history_[step].time);
      const std::filesystem::path directory(parameters_.solid.output_directory);
      DataOut<1, 3>               output;
      output.attach_dof_handler(multiplier_view_->dof_handler());
      output.add_data_vector(multiplier(),
                             "pressure_jump",
                             DataOut<1, 3>::type_dof_data);
      output.build_patches(multiplier_view_->mapping());
      const auto filename = "pressure_jump_" + std::to_string(step) + ".vtu";
      output.write_vtu_in_parallel((directory / filename).string(),
                                   communicator_);
      multiplier_records_.emplace_back(history_[step].time, filename);
      if (Utilities::MPI::this_mpi_process(communicator_) == 0)
        {
          std::ofstream pvd(directory / "pressure_jump.pvd");
          DataOutBase::write_pvd_record(pvd, multiplier_records_);
        }
    }

    /** Replay every checkpoint of the history. */
    void
    run()
    {
      setup();
      for (unsigned int step = 0; step < n_steps(); ++step)
        {
          solve_step(step);
          output_step(step);
        }
      pcout_ << "MetricFlowX replay steps: " << n_steps()
             << "\nStatic stiffness assemblies: "
             << solid_.stiffness_assemblies() << std::endl;
    }

    unsigned int
    n_steps() const
    {
      return history_.size();
    }

    double
    time(const unsigned int step) const
    {
      AssertIndexRange(step, history_.size());
      return history_[step].time;
    }

    const Vector &
    displacement() const
    {
      return adapter_->field(state_, solid_fields_->displacement);
    }

    const Vector &
    multiplier() const
    {
      return adapter_->field(state_, *multiplier_);
    }

    /** Native flow area coefficients of the last loaded checkpoint. */
    const Vector &
    area() const
    {
      return *area_;
    }

    double
    residual_norm() const
    {
      return residual_norm_;
    }

    const ElasticStaticProblem<3> &
    solid() const
    {
      return solid_;
    }

    const FlowProblem &
    flow() const
    {
      return *flow_;
    }

  private:
    void
    assert_same_discretization(const FlowProblem &loaded) const
    {
      using namespace dealii;
      const auto  &reference = *flow_;
      unsigned int same =
        reference.dof_handler().n_dofs() == loaded.dof_handler().n_dofs() &&
        reference.locally_owned_dofs() == loaded.locally_owned_dofs() &&
        reference.triangulation().n_global_active_cells() ==
          loaded.triangulation().n_global_active_cells();
      if (same)
        {
          std::vector<types::global_dof_index> a, b;
          auto first  = reference.dof_handler().begin_active();
          auto second = loaded.dof_handler().begin_active();
          for (; same && first != reference.dof_handler().end() &&
                 second != loaded.dof_handler().end();
               ++first, ++second)
            {
              // Artificial cells of a fully distributed mesh have no CellId.
              if (first->is_artificial() != second->is_artificial())
                same = 0;
              else if (first->is_artificial())
                continue;
              else if (first->id() != second->id() ||
                       first->is_locally_owned() != second->is_locally_owned())
                same = 0;
              else if (first->is_locally_owned())
                {
                  a.resize(first->get_fe().n_dofs_per_cell());
                  b.resize(second->get_fe().n_dofs_per_cell());
                  first->get_dof_indices(a);
                  second->get_dof_indices(b);
                  same =
                    first->active_fe_index() == second->active_fe_index() &&
                    a == b &&
                    first->center().distance(second->center()) <=
                      1.e-12 * first->diameter();
                }
            }
          same = same && first == reference.dof_handler().end() &&
                 second == loaded.dof_handler().end();
        }
      AssertThrow(Utilities::MPI::min(same, communicator_) == 1,
                  ExcMessage("All replayed checkpoints must share the mesh "
                             "and DoF numbering of the first one."));
    }

    const MetricFlowXElastostaticReplayParameters &parameters_;
    MPI_Comm                                       communicator_;
    dealii::ConditionalOStream                     pcout_;
    std::vector<MetricFlowXCheckpoint>             history_;

    std::unique_ptr<FlowProblem>                flow_;
    std::optional<FiniteElementSpaceView<1, 3>> flow_space_;
    std::shared_ptr<Vector>                     area_;
    std::unique_ptr<FiniteElementSpace<1, 3>>   multiplier_space_;
    std::optional<FiniteElementSpaceView<1, 3>> multiplier_view_;
    ElasticStaticProblem<3>                     solid_;
    std::unique_ptr<Adapter>                    adapter_;
    std::optional<SolidFields>                  solid_fields_;
    std::optional<Multiplier>                   multiplier_;
    GlobalVector                                state_, residual_;
    unsigned int                                loaded_step_   = 0;
    bool                                        factorized_    = false;
    double                                      residual_norm_ = 0.;
    std::vector<std::pair<double, std::string>> multiplier_records_;
  };
} // namespace ImmersX

#endif // IMMERSX_WITH_METRIC_FLOW_X
#endif // immersx_metric_flow_x_elastostatic_replay_h
