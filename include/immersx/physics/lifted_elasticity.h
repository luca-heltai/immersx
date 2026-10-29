// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#ifndef immersx_lifted_elasticity_h
#define immersx_lifted_elasticity_h

#include <deal.II/base/function_parser.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <immersx/core/boundary_conditions.h>
#include <immersx/core/constraint.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/input_field_selector.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/symbolic_field_evaluator.h>
#include <immersx/coupling/lifted_modal_load.h>
#include <immersx/grid/local_co_refinement.h>
#include <immersx/io/output_handler.h>
#include <immersx/physics/elastic_static.h>
#include <immersx/physics/static_elasticity_term.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ImmersX
{
  /** Static 2D/3D elasticity with optional 3D curve-to-surface coupling.
   */
  template <int dim = 3>
  struct LiftedElasticityParameters : public ElasticStaticParameters<dim>
  {
    LiftedElasticityParameters()
      : ElasticStaticParameters<dim>("/Lifted elasticity/")
      , immersed_domain("/Lifted elasticity/Immersed domain/")
      , multiplier_space("/Lifted elasticity/Multiplier space/")
      , geometry("/Lifted elasticity/Lift/")
      , co_refinement("/Lifted elasticity/Co-refinement/")
      , execution("/Lifted elasticity/Linear adapter/")
    {
      this->add_parameter("Coupling",
                          coupling,
                          "none, multiplier, or pressure",
                          this->prm,
                          dealii::Patterns::Selection(
                            "none|multiplier|pressure"));
      this->add_parameter("Modal data expressions", modal_data);
      this->add_parameter("Modal data fields", modal_data_fields);
      this->add_parameter("Radius field", radius_field);
      this->add_parameter("Weak Dirichlet boundary ids", weak_dirichlet_ids);
      this->add_parameter("Normal flux boundary ids", normal_flux_ids);
      this->add_parameter("Weak Dirichlet penalty coefficient", penalty_term);
      this->add_parameter("Refinement strategy",
                          refinement_strategy,
                          "",
                          this->prm,
                          dealii::Patterns::Selection(
                            "global|fixed_fraction|fixed_number"));
      this->add_parameter("Refinement fraction", refinement_fraction);
      this->add_parameter("Coarsening fraction", coarsening_fraction);
      this->domain_parameters.initial_refinement = 2;
      this->domain_parameters.arguments_for_grid = "-1: 1: false";
      immersed_domain.arguments_for_grid         = "-0.5: 0.5: false";
      immersed_domain.triangulation_type         = "fullydistributed";
      immersed_domain.initial_refinement         = 1;
      geometry.section.selected_coefficients     = {0, 1, 2};
      geometry.representative_n_q_points         = 2;
      this->output_name                          = "lifted_elasticity";
      execution.solver                           = LinearSolver::automatic;
    }
    DomainParameters<1, dim>             immersed_domain;
    FiniteElementSpaceParameters<1, dim> multiplier_space;
    TensorProductLift<1, 2, dim, dim>    geometry;
    RefinementParameters<1>              co_refinement;
    LinearAdapterParameters              execution;
    std::string              coupling = dim == 3 ? "multiplier" : "none";
    std::vector<std::string> modal_data{"0", "0", "0"};
    std::string              radius_field;
    std::string              modal_data_fields;
    std::set<dealii::types::boundary_id> weak_dirichlet_ids, normal_flux_ids;
    double                               penalty_term        = 100.;
    std::string                          refinement_strategy = "global";
    double refinement_fraction = 0.3, coarsening_fraction = 0.;
  };

  /** Application composition; legacy Elasticity and ElasticStatic are not used
   * to assemble or solve this system. All execution storage belongs to adapter.
   */
  template <int dim = 3>
  class LiftedElasticity
  {
  public:
    using Vector       = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<Vector, GlobalVector>;
    using Displacement = Field<dim, dim, dealii::FEValuesExtractors::Vector>;
    using Multiplier   = Field<1, dim, dealii::FEValuesExtractors::Scalar>;

    explicit LiftedElasticity(const LiftedElasticityParameters<dim> &parameters,
                              const MPI_Comm communicator = MPI_COMM_WORLD)
      : parameters_(parameters)
      , communicator_(communicator)
      , background_(parameters.domain_parameters, communicator)
      , immersed_(parameters.immersed_domain, communicator)
    {}

    void
    setup(const std::function<void(dealii::Triangulation<dim> &)>
            &configure_background = {})
    {
      AssertThrow(!adapter_,
                  dealii::ExcMessage(
                    "LiftedElasticity::setup() may only be called once."));
      validate();
      if (parameters_.coupling == "none")
        background_.make_grid();
      else if (parameters_.coupling == "multiplier" ||
               parameters_.coupling == "pressure")
        {
          co_refinement_report_ =
            LocalCoRefinement<dim, 1, dim>(parameters_.co_refinement)
              .run(background_, immersed_);
          if (dealii::Utilities::MPI::this_mpi_process(communicator_) == 0)
            std::cout << "Co-refinement: "
                      << co_refinement_report_->intersecting_pairs
                      << " intersecting pairs, "
                      << co_refinement_report_->lower_violations
                      << " lower and "
                      << co_refinement_report_->upper_violations
                      << " upper violations." << std::endl;
        }
      if (configure_background)
        configure_background(background_.triangulation());
      assemble();
    }

    void
    solve()
    {
      AssertThrow(adapter_ != nullptr,
                  dealii::ExcMessage("Call setup() before solve()."));
      adapter_->solve(state_);
      auto residual = adapter_->make_state();
      adapter_->evaluate_residual(state_, residual);
      residual_norm_ = residual.l2_norm();
      AssertThrow(std::isfinite(residual_norm_),
                  dealii::ExcMessage("Nonfinite static residual."));
      displacement_->constraints().distribute(
        adapter_->field(state_, *displacement_));
      for (const auto &field : multipliers_)
        field.constraints().distribute(adapter_->field(state_, field));
    }

    void
    run()
    {
      setup();
      for (unsigned int cycle = 0; cycle < parameters_.n_refinement_cycles;
           ++cycle)
        {
          solve();
          if (dealii::Utilities::MPI::this_mpi_process(communicator_) == 0)
            std::cout << "Static cycle " << cycle << ", residual "
                      << residual_norm_ << std::endl;
          output(cycle);
          Vector ghosted(displacement_->locally_owned_dofs(),
                         displacement_->locally_relevant_dofs(),
                         communicator_);
          ghosted = solution();
          ghosted.update_ghost_values();
          if (displacement_->finite_element().reference_cell().is_hyper_cube())
            parameters_.convergence_table.error_from_exact(
              displacement_->dof_handler(),
              ghosted,
              parameters_.exact_solution);
          if (cycle + 1 < parameters_.n_refinement_cycles)
            refine(ghosted);
        }
      if (dealii::Utilities::MPI::this_mpi_process(communicator_) == 0 &&
          displacement_->finite_element().reference_cell().is_hyper_cube())
        parameters_.convergence_table.output_table(std::cout);
    }

    const Vector &
    solution() const
    {
      return adapter_->field(state_, *displacement_);
    }
    const std::vector<Multiplier> &
    multipliers() const
    {
      return multipliers_;
    }
    /** Native coefficient vector containing all modal FE components. */
    const Vector &
    modal_coefficients() const
    {
      AssertThrow(!multipliers_.empty(),
                  dealii::ExcMessage(
                    "This problem has no modal coefficients."));
      return adapter_->field(state_, multipliers_.front());
    }
    const Displacement &
    displacement() const
    {
      return *displacement_;
    }
    const Domain<dim> &
    background() const
    {
      return background_;
    }
    const Domain<1, dim> &
    immersed() const
    {
      return immersed_;
    }
    const std::optional<LocalCoRefinementReport> &
    initial_co_refinement_report() const
    {
      return co_refinement_report_;
    }
    double
    residual_norm() const
    {
      return residual_norm_;
    }

  private:
    void
    validate() const
    {
      AssertThrow(parameters_.n_refinement_cycles > 0,
                  dealii::ExcMessage("At least one static cycle is required."));
      AssertThrow(parameters_.coupling == "none" ||
                    parameters_.coupling == "multiplier" ||
                    parameters_.coupling == "pressure",
                  dealii::ExcMessage("Unknown coupling mode."));
      AssertThrow(
        parameters_.domain_parameters.triangulation_type == "distributed" ||
          parameters_.n_refinement_cycles == 1,
        dealii::ExcMessage(
          "Multiple static cycles require a distributed background."));
      AssertThrow(parameters_.refinement_strategy == "global" ||
                    parameters_.refinement_strategy == "fixed_fraction" ||
                    parameters_.refinement_strategy == "fixed_number",
                  dealii::ExcMessage("Unknown static refinement strategy."));
      AssertThrow(
        dim == 3 || parameters_.coupling == "none",
        dealii::ExcMessage(
          "Codimension-two coupling in 2D needs a 0D Domain and FE space; use legacy point inclusions. Uncoupled 2D static elasticity is supported."));
      for (const auto id : parameters_.dirichlet_ids)
        AssertThrow(!parameters_.weak_dirichlet_ids.count(id) &&
                      !parameters_.neumann_ids.count(id) &&
                      !parameters_.normal_flux_ids.count(id),
                    dealii::ExcMessage(
                      "Boundary condition id sets must be disjoint."));
      for (const auto id : parameters_.weak_dirichlet_ids)
        AssertThrow(!parameters_.neumann_ids.count(id) &&
                      !parameters_.normal_flux_ids.count(id),
                    dealii::ExcMessage(
                      "Boundary condition id sets must be disjoint."));
      for (const auto id : parameters_.neumann_ids)
        AssertThrow(!parameters_.normal_flux_ids.count(id),
                    dealii::ExcMessage(
                      "Boundary condition id sets must be disjoint."));
    }

    void
    assemble()
    {
      using namespace dealii;
      adapter_ =
        std::make_unique<Adapter>(parameters_.execution, communicator_);
      const auto reference =
        detail::uniform_reference_cell(background_.triangulation());
      AssertThrow(reference.has_value(),
                  ExcMessage("The elasticity mesh has no cells."));
      std::unique_ptr<FiniteElement<dim>> element;
      if (parameters_.space_parameters.finite_element != "auto")
        element = FETools::get_fe_by_name<dim>(
          parameters_.space_parameters.finite_element);
      else if (reference->is_hyper_cube())
        element =
          std::make_unique<FESystem<dim>>(FE_Q<dim>(parameters_.fe_degree),
                                          dim);
      else if (reference->is_simplex())
        element = std::make_unique<FESystem<dim>>(
          FE_SimplexP<dim>(parameters_.fe_degree), dim);
      displacement_space_ =
        std::make_unique<FiniteElementSpace<dim>>(background_.triangulation(),
                                                  std::move(element));
      AssertThrow(
        displacement_space_->finite_element().n_components() == dim &&
          displacement_space_->finite_element().is_primitive(),
        ExcMessage(
          "The displacement space must be primitive with dim components."));
      BoundaryConditions<dim> conditions(*displacement_space_);
      for (const auto id : parameters_.dirichlet_ids)
        conditions.add_dirichlet(id, parameters_.get_dirichlet_bc(id));
      for (const auto id : parameters_.normal_flux_ids)
        conditions.add_nonzero_normal_flux(id, parameters_.get_neumann_bc(id));
      conditions.update(0.);
      displacement_view_ = std::make_unique<FiniteElementSpaceView<dim>>(
        displacement_space_->view());
      displacement_ = adapter_
                        ->add(algebraic_field(vector_field(*displacement_view_,
                                                           "displacement")),
                              "solid")
                        .fields();
      adapter_->add(static_elasticity_term(*displacement_, parameters_),
                    "elasticity");
      adapter_->add_preconditioner(displacement_->id(),
                                   [](const auto &matrix, const auto &reinit) {
                                     return make_amg_preconditioner(matrix,
                                                                    reinit);
                                   });
      if (!parameters_.weak_dirichlet_ids.empty())
        adapter_->add(static_elasticity_nitsche(*displacement_, parameters_),
                      "boundary");
      multipliers_.clear();
      if (parameters_.coupling == "multiplier" ||
          parameters_.coupling == "pressure")
        assemble_coupling();
      state_ = adapter_->make_state();
    }

    void
    assemble_coupling()
    {
      if constexpr (dim == 3)
        {
          using namespace dealii;
          TensorProductLiftSupport<1, 2, 3, 3> support(
            parameters_.geometry.parameters());
          const auto n_modes =
            support.reference_cross_section().n_selected_basis();
          AssertThrow(
            parameters_.modal_data.size() == n_modes,
            ExcMessage(
              "Provide one Modal data expression per selected cross-section mode."));
          auto finite_element = FETools::get_fe_by_name<1, 3>(
            parameters_.multiplier_space.finite_element);
          if (finite_element->n_components() == 1 && n_modes > 1)
            finite_element =
              std::make_unique<FESystem<1, 3>>(*finite_element, n_modes);
          AssertThrow(
            finite_element->n_components() == n_modes &&
              finite_element->is_primitive(),
            ExcMessage(
              "The modal FE space needs one scalar component per selected mode."));
          multiplier_space_ = std::make_unique<FiniteElementSpace<1, 3>>(
            immersed_.triangulation(), std::move(finite_element));
          BoundaryConditions<1, 3> hanging_nodes(*multiplier_space_);
          hanging_nodes.update(0.);
          multiplier_view_ = std::make_unique<FiniteElementSpaceView<1, 3>>(
            multiplier_space_->view());
          Vector                         imported_values;
          std::vector<InputFieldBinding> bindings;
          if (!parameters_.radius_field.empty() ||
              !parameters_.modal_data_fields.empty())
            {
              const auto &properties = immersed_.vtk_fields();
              imported_values.reinit(properties.locally_owned_dofs(),
                                     properties.locally_relevant_dofs(),
                                     communicator_);
              imported_values = properties.coefficients();
              imported_values.update_ghost_values();
              bindings =
                InputFieldSelector::resolve(parameters_.modal_data_fields,
                                            properties.catalog());
            }
          FunctionParser<3>      data(n_modes);
          SymbolicFieldEvaluator symbolic_data;
          if (bindings.empty())
            data.initialize(FunctionParser<3>::default_variable_names() + ",t",
                            parameters_.modal_data,
                            {{"pi", numbers::PI}, {"E", numbers::E}},
                            true);
          else
            {
              std::vector<std::string> symbols;
              for (const auto &binding : bindings)
                symbols.push_back(binding.symbol_name);
              symbolic_data.initialize(parameters_.modal_data,
                                       symbols,
                                       {{"pi", numbers::PI},
                                        {"E", numbers::E}});
            }
          using Lifted =
            decltype(lift(value(multiplier_view_->field("multiplier")),
                          parameters_.geometry));
          using Weak = decltype(weak_term(value(*displacement_),
                                          test(std::declval<Lifted>())));
          std::map<std::pair<CellId, unsigned int>, std::vector<double>>
                     data_cache;
          const auto data_at = [&](const typename Lifted::Point &point,
                                   const unsigned int            component) {
            if (bindings.empty())
              return data.value(point.representative_point, component);
            const auto key =
              std::make_pair(point.source_cell_id, point.representative_qpoint);
            auto found = data_cache.find(key);
            if (found == data_cache.end())
              {
                const auto &descriptor =
                  immersed_.vtk_fields()
                    .catalog()[bindings.front().field_index];
                const auto source =
                  immersed_
                    .field(descriptor.name, bindings.front().field_component)
                    .field();
                const auto cell =
                  source.triangulation()
                    .create_cell_iterator(point.source_cell_id)
                    ->as_dof_handler_iterator(source.dof_handler());
                const auto reference =
                  source.mapping().transform_real_to_unit_cell(
                    cell, point.representative_point);
                FEValues<1, 3> values(source.mapping(),
                                      source.finite_element(),
                                      Quadrature<1>(
                                        std::vector<Point<1>>{reference}),
                                      update_values);
                values.reinit(cell);
                std::vector<dealii::Vector<double>> samples(
                  1,
                  dealii::Vector<double>(
                    source.finite_element().n_components()));
                values.get_function_values(imported_values, samples);
                std::vector<double> inputs;
                for (const auto &binding : bindings)
                  inputs.push_back(samples[0][binding.fe_component]);
                found = data_cache
                          .emplace(key,
                                   symbolic_data.evaluate(
                                     point.representative_point, 0., inputs))
                          .first;
              }
            return found->second.at(component);
          };
          ConstraintSum<Weak>                           terms;
          std::vector<Lifted>                           lifts;
          std::shared_ptr<ImmersXLA::MPI::SparseMatrix> metric;
          std::shared_ptr<Vector>                       rhs;
          for (unsigned int slot = 0; slot < n_modes; ++slot)
            {
              const auto field =
                multiplier_view_->field("multiplier",
                                        FEValuesExtractors::Scalar(slot));
              const auto make_lift = [&]() {
                if (parameters_.radius_field.empty())
                  return lift(value(field), parameters_.geometry).mode(slot);
                const auto radius = immersed_.field(parameters_.radius_field);
                return lift(value(field),
                            parameters_.geometry,
                            frozen(radius.field(), imported_values))
                  .mode(slot);
              };
              auto lifted = make_lift();
              const auto [mass, load] =
                lifted_modal_load(lifted, data_at, slot);
              if (!metric)
                {
                  metric = mass;
                  rhs    = load;
                }
              else
                {
                  metric->add(1., *mass);
                  *rhs += *load;
                }
              terms.add(weak_term(value(*displacement_), test(lifted)));
              lifts.push_back(std::move(lifted));
            }
          FieldId multiplier_id;
          if (parameters_.coupling == "multiplier")
            {
              multiplier_id =
                adapter_
                  ->add(lifted_modal_constraint(std::move(terms), metric, *rhs),
                        "immersed modal constraint")
                  .fields()
                  .multiplier.id();
            }
          else if (parameters_.coupling == "pressure")
            {
              const auto registered =
                adapter_
                  ->add(algebraic_field(multiplier_view_->field("multiplier")),
                        "pressure")
                  .fields();
              multiplier_id = registered.id();
              adapter_->add(affine_field_term(registered, metric, rhs),
                            "pressure projection");
              adapter_->add_preconditioner(
                multiplier_id, [](const auto &matrix, const auto &reinit) {
                  return make_amg_preconditioner(matrix, reinit);
                });
              for (const auto &lifted : lifts)
                adapter_->add(weak_term(lifted.with_id(multiplier_id),
                                        test(*displacement_).scaled(-1.)),
                              "pressure load");
            }
          for (unsigned int slot = 0; slot < n_modes; ++slot)
            multipliers_.push_back(
              multiplier_view_
                ->field("multiplier_" + std::to_string(slot),
                        FEValuesExtractors::Scalar(slot))
                .with_id(multiplier_id));
        }
    }

    void
    output(const unsigned int cycle)
    {
      const std::filesystem::path     directory(parameters_.output_directory);
      OutputHandler<dim, dim, Vector> output(*displacement_view_,
                                             directory,
                                             parameters_.output_name + "_" +
                                               std::to_string(cycle));
      output.add_field(*displacement_);
      adapter_->write_output(output, state_, cycle);
      if (!multipliers_.empty())
        {
          OutputHandler<1, dim, Vector> modal_output(*multiplier_view_,
                                                     directory,
                                                     parameters_.output_name +
                                                       "_modes_" +
                                                       std::to_string(cycle));
          for (const auto &field : multipliers_)
            modal_output.add_field(field);
          adapter_->write_output(modal_output, state_, cycle);
        }
    }

    void
    refine(const Vector &ghosted)
    {
      using namespace dealii;
      auto &tria = dynamic_cast<parallel::distributed::Triangulation<dim> &>(
        background_.triangulation());
      if (parameters_.refinement_strategy == "global")
        {
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->is_locally_owned())
              cell->set_refine_flag();
        }
      else if (parameters_.refinement_strategy == "fixed_fraction" ||
               parameters_.refinement_strategy == "fixed_number")
        {
          dealii::Vector<float> errors(tria.n_active_cells());
          KellyErrorEstimator<dim>::estimate(
            displacement_->dof_handler(),
            QGauss<dim - 1>(displacement_space_->finite_element().degree + 1),
            {},
            ghosted,
            errors);
          if (parameters_.refinement_strategy == "fixed_fraction")
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_fraction(
                tria,
                errors,
                parameters_.refinement_fraction,
                parameters_.coarsening_fraction);
          else if (parameters_.refinement_strategy == "fixed_number")
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_number(tria,
                                              errors,
                                              parameters_.refinement_fraction,
                                              parameters_.coarsening_fraction);
        }
      adapter_.reset();
      displacement_.reset();
      multipliers_.clear();
      displacement_view_.reset();
      displacement_space_.reset();
      multiplier_view_.reset();
      multiplier_space_.reset();
      tria.execute_coarsening_and_refinement();
      assemble();
    }

    const LiftedElasticityParameters<dim>          &parameters_;
    MPI_Comm                                        communicator_;
    Domain<dim>                                     background_;
    Domain<1, dim>                                  immersed_;
    std::unique_ptr<FiniteElementSpace<dim>>        displacement_space_;
    std::unique_ptr<FiniteElementSpace<1, dim>>     multiplier_space_;
    std::unique_ptr<FiniteElementSpaceView<dim>>    displacement_view_;
    std::unique_ptr<FiniteElementSpaceView<1, dim>> multiplier_view_;
    std::optional<Displacement>                     displacement_;
    std::vector<Multiplier>                         multipliers_;
    std::unique_ptr<Adapter>                        adapter_;
    GlobalVector                                    state_;
    std::optional<LocalCoRefinementReport>          co_refinement_report_;
    double                                          residual_norm_ = 0.;
  };
} // namespace ImmersX
#endif
