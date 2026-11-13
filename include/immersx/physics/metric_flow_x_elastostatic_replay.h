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
#  include <immersx/io/lifted_particle_output.h>
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
      , particle_output("/MetricFlowX replay/Particle output/")
    {
      add_parameter(
        "Flow history",
        flow_history,
        "Directory written by blood_flow --coupling-output-directory "
        "(a layout checkpoint and fields-<time> directories) or by blood_flow "
        "--checkpoint-directory (checkpoint-<time> directories). Every saved "
        "time in it is replayed in time order.");
    }

    ElasticStaticParameters<3>         solid;
    TensorProductLift<1, 2, 3, 3>      wall_lift;
    FiniteElementSpaceParameters<1, 3> multiplier_space;
    LinearAdapterParameters            execution;
    LiftedParticleOutputParameters     particle_output;
    std::string                        flow_history;
  };

  /** A saved native flow time: its physical time and directory. */
  struct MetricFlowXCheckpoint
  {
    double      time = 0.;
    std::string directory;
  };

  /** The saved times of a MetricFlowX history, sorted by time. */
  struct MetricFlowXFlowHistory
  {
    /** Native checkpoint that provides the mesh, physics and DoF layout. */
    std::string layout;
    /** True when steps hold only coupling fields (fields-<time>), false when
     * they are full checkpoints (checkpoint-<time>). */
    bool coupling_fields = false;
    std::vector<MetricFlowXCheckpoint> steps;
  };

  /** List the saved times of a blood_flow history. Collective.
   *
   * A directory with a `layout` checkpoint is a coupling output: its
   * `fields-<hexadecimal time>` directories are the steps. Otherwise the
   * steps are the `checkpoint-<hexadecimal time>` directories, and the first
   * one is the layout. Staging directories (`.incomplete`) and names whose
   * suffix is not a time, such as `<checkpoint>-coupling`, are skipped.
   */
  inline MetricFlowXFlowHistory
  metric_flow_x_flow_history(const std::string &history,
                             const MPI_Comm     communicator)
  {
    using namespace dealii;
    std::vector<std::pair<double, std::string>> entries;
    std::string                                 layout, error;
    bool                                        coupling_fields = false;
    if (Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        try
          {
            const std::filesystem::path root(history);
            coupling_fields = std::filesystem::is_directory(root / "layout");
            const std::string prefix =
              coupling_fields ? "fields-" : "checkpoint-";
            for (const auto &entry : std::filesystem::directory_iterator(root))
              {
                const auto name = entry.path().filename().string();
                if (!entry.is_directory() || name.rfind(prefix, 0) != 0 ||
                    entry.path().extension() == ".incomplete")
                  continue;
                const auto text = name.substr(prefix.size());
                char      *end  = nullptr;
                const auto time = std::strtod(text.c_str(), &end);
                if (text.empty() || end != text.c_str() + text.size() ||
                    !std::isfinite(time))
                  continue;
                entries.emplace_back(
                  time, std::filesystem::absolute(entry.path()).string());
              }
            std::sort(entries.begin(), entries.end());
            AssertThrow(!entries.empty(),
                        ExcMessage("No saved MetricFlowX " + prefix +
                                   "<time> directories in " + history + "."));
            layout = coupling_fields ?
                       std::filesystem::absolute(root / "layout").string() :
                       entries.front().second;
          }
        catch (const std::exception &exception)
          {
            error = exception.what();
          }
      }
    error = Utilities::MPI::broadcast(communicator, error, 0);
    AssertThrow(error.empty(), ExcMessage(error));
    MetricFlowXFlowHistory result;
    result.layout = Utilities::MPI::broadcast(communicator, layout, 0);
    result.coupling_fields =
      Utilities::MPI::broadcast(communicator, coupling_fields, 0);
    entries = Utilities::MPI::broadcast(communicator, entries, 0);
    for (const auto &[time, directory] : entries)
      result.steps.push_back({time, directory});
    return result;
  }

  /**
   * One-way coupling of a saved MetricFlowX history to static elasticity.
   *
   * For each saved time t_n, the solid solves
   *
   *   K u + B lambda = f(t_n),    B^T u = P_h(A_n),
   *
   * with the native flow area A_n read from the history and
   * P_h(A) = (sqrt(A/pi) - sqrt(a_d/pi), mu) on the lifted wall overlap.
   * These are the solid and wall rows of metric_flow_x_elastostatic without
   * pressure feedback. The reference area a_d is the tapered diastolic area of
   * BloodFlowSystem::reference_area(), relative to which MetricFlowX evaluates
   * its wall laws.
   *
   * The ElasticStaticProblem owns the solid; one LinearAdapter owns the
   * saddle-point system. Stiffness, wall pairing and multiplier metric are
   * assembled once; each step updates the loads, the Dirichlet data and the
   * shared area vector. A direct or MUMPS solver is factored once.
   *
   * setup() loads the layout checkpoint once into a BloodFlowSystem. That
   * load parses the archived parameter tree, so setup() first calls
   * ParameterAcceptor::clear(): parse all parameters before setup(). Each step
   * then reads only its state (load_checkpoint_state()) or its area field
   * (load_coupling_field()) into the same object, which checks that it was
   * written on the same mesh and partition. Run with the MPI size used to
   * write the history.
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

    /** Load the flow layout, build the solid and the wall coupling. */
    void
    setup()
    {
      using namespace dealii;
      AssertThrow(!adapter_, ExcMessage("The replay may only be set up once."));
      AssertThrow(!parameters_.flow_history.empty(),
                  ExcMessage("Set MetricFlowX replay/Flow history."));
      const auto history =
        metric_flow_x_flow_history(parameters_.flow_history, communicator_);
      history_ = history.steps;
      coupling_fields_ = history.coupling_fields;
      pcout_ << "MetricFlowX replay: " << history_.size()
             << (coupling_fields_ ? " field" : " checkpoint")
             << " time(s) from " << history_.front().time << " to "
             << history_.back().time << std::endl;

      // The layout provides the flow mesh for the multiplier and the
      // reference area. Its archived parameter text is parsed into the global
      // tree; clearing the registry first keeps it from overwriting the
      // replay's own acceptors.
      ParameterAcceptor::clear();
      flow_ = std::make_unique<FlowProblem>(communicator_);
      flow_->load_checkpoint(history.layout);
      if (!coupling_fields_)
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
      auto owned_reference = flow_->make_state();
      flow_->interpolate_reference_area(owned_reference);
      reference_area_.reinit(flow_->locally_owned_dofs(),
                             flow_->locally_relevant_dofs(),
                             communicator_);
      reference_area_ = owned_reference;
      reference_area_.update_ghost_values();
      const auto reference_radius =
        sqrt(frozen(area_field, reference_area_) / numbers::PI);

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
      // Wall particles are the lifted quadrature points, labelled as in the
      // coupled elastostatic application.
      wall_points_ = reference_wall.lifted_points();
      for (auto &point : wall_points_)
        point.mode_values = {point.mode_vector[0],
                             point.mode_vector[1],
                             point.mode_vector[2]};

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

    /** Read the flow area of one saved time and solve the static system. */
    void
    solve_step(const unsigned int step)
    {
      using namespace dealii;
      AssertThrow(adapter_, ExcMessage("Call setup() before solve_step()."));
      AssertIndexRange(step, history_.size());
      if (loaded_step_ != step)
        {
          const auto &saved = history_[step];
          if (coupling_fields_)
            {
              auto       values = flow_->make_state();
              const auto t =
                flow_->load_coupling_field(saved.directory, "area", values);
              AssertThrow(t == saved.time,
                          ExcMessage("The time stored in " + saved.directory +
                                     " does not match its name."));
              *area_ = values;
            }
          else
            {
              flow_->load_checkpoint_state(saved.directory);
              *area_ = flow_->state();
            }
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
      output_wall_particles(step, directory);
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

    /** Native flow area coefficients of the last loaded time. */
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
    /** Write the lifted wall points with the selected particle properties.
     *
     * wall_traction is the multiplier, mode_radial its unit radial
     * direction, and data_radial the radius change sqrt(A/pi) - sqrt(a_d/pi)
     * imposed at this step.
     */
    void
    output_wall_particles(const unsigned int           step,
                          const std::filesystem::path &directory)
    {
      using namespace dealii;
      const auto &options = parameters_.particle_output;
      if (!options.enabled())
        return;
      LiftedParticleLabels labels;
      labels.value_name      = "wall_traction";
      labels.mode_labels     = {"radial"};
      labels.mode_components = 3;

      std::vector<double> coefficients, data;
      if (options.selected("modal_coefficients") ||
          options.selected("lifted_value"))
        coefficients = sample_field_at_representative_points(*multiplier_,
                                                             multiplier(),
                                                             wall_points_);
      if (options.selected("modal_data"))
        {
          const auto &dofs      = flow_->dof_handler();
          const auto  width     = flow_->finite_element().n_components();
          const auto  component = flow_->area_extractor().component;
          const auto  current   = sample_at_representative_points(
            dofs, StaticMappingQ1<1, 3>::mapping, *area_, wall_points_);
          const auto reference =
            sample_at_representative_points(dofs,
                                            StaticMappingQ1<1, 3>::mapping,
                                            reference_area_,
                                            wall_points_);
          for (std::size_t q = 0; q < wall_points_.size(); ++q)
            {
              const auto a  = current[q * width + component];
              const auto a_d = reference[q * width + component];
              AssertThrow(a > 0. && a_d > 0.,
                          ExcMessage(
                            "Wall particle radii require positive area."));
              data.push_back(std::sqrt(a / numbers::PI) -
                             std::sqrt(a_d / numbers::PI));
            }
        }
      const auto columns = make_lifted_particle_columns(
        options, labels, wall_points_, coefficients, data);

      std::optional<Vector> displacement;
      if (options.selected("bulk_field"))
        {
          displacement.emplace(solid_.locally_owned_dofs(),
                               solid_.locally_relevant_dofs(),
                               communicator_);
          *displacement = solid_.solution();
          displacement->update_ghost_values();
        }
      const LiftedBulkField<3, Vector> bulk{"displacement",
                                            solid_.dof_handler(),
                                            solid_.mapping(),
                                            displacement ? &*displacement :
                                                           nullptr,
                                            0,
                                            3};
      const auto filename = "wall_particles_" + std::to_string(step) + ".vtu";
      write_lifted_particles(
        options, columns, wall_points_, bulk, (directory / filename).string());
      particle_records_.emplace_back(history_[step].time, filename);
      if (Utilities::MPI::this_mpi_process(communicator_) == 0)
        {
          std::ofstream pvd(directory / "wall_particles.pvd");
          DataOutBase::write_pvd_record(pvd, particle_records_);
        }
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
    bool                                        coupling_fields_ = false;
    std::optional<unsigned int>                 loaded_step_;
    bool                                        factorized_    = false;
    double                                      residual_norm_ = 0.;
    std::vector<std::pair<double, std::string>> multiplier_records_;
    Vector                                      reference_area_;
    std::vector<LiftedWallPoint<3>>             wall_points_;
    std::vector<std::pair<double, std::string>> particle_records_;
  };
} // namespace ImmersX

#endif // IMMERSX_WITH_METRIC_FLOW_X
#endif // immersx_metric_flow_x_elastostatic_replay_h
