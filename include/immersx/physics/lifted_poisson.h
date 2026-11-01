// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_lifted_poisson_h
#define immersx_lifted_poisson_h

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <immersx/core/boundary_conditions.h>
#include <immersx/core/field_contributor.h>
#include <immersx/core/input_field_selector.h>
#include <immersx/core/known_term.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/core/modal_lift.h>
#include <immersx/core/symbolic_field_evaluator.h>
#include <immersx/grid/local_co_refinement.h>
#include <immersx/io/output_handler.h>

#include <fstream>
#include <limits>
#include <map>

namespace ImmersX
{
  struct LiftedPoissonParameters : public dealii::ParameterAcceptor
  {
    LiftedPoissonParameters()
      : ParameterAcceptor("/Lifted Poisson/")
      , bulk("/Lifted Poisson/Background/")
      , representative("/Lifted Poisson/Representative/")
      , bulk_space("/Lifted Poisson/Background space/")
      , modal_space("/Lifted Poisson/Modal space/", "FE_Q<1,3>(1)")
      , lift("/Lifted Poisson/Lift/")
      , co_refinement("/Lifted Poisson/Co-refinement/")
      , solver("/Lifted Poisson/Solver/")
      , output("/Lifted Poisson/Output/")
      , forcing("/Lifted Poisson/Forcing/")
      , boundary("/Lifted Poisson/Boundary/")
      , exact_solution("/Lifted Poisson/Exact solution/")
    {
      add_parameter("Modal data", modal_data);
      add_parameter("Modal data fields", modal_data_fields);
      add_parameter("Radius field", radius_field);
      add_parameter("Dirichlet boundary ids", dirichlet_ids);
      add_parameter("Refinement cycles",
                    refinement_cycles,
                    "",
                    prm,
                    dealii::Patterns::Integer(1));
      add_parameter("Refinement strategy",
                    refinement_strategy,
                    "",
                    prm,
                    dealii::Patterns::Selection(
                      "global|fixed_fraction|fixed_number"));
      add_parameter("Refinement fraction",
                    refinement_fraction,
                    "",
                    prm,
                    dealii::Patterns::Double(0., 1.));
      add_parameter("Coarsening fraction",
                    coarsening_fraction,
                    "",
                    prm,
                    dealii::Patterns::Double(0., 1.));
      add_parameter(
        "Maximum cells", maximum_cells, "", prm, dealii::Patterns::Integer(1));
      add_parameter("Compute errors", compute_errors);
    }
    DomainParameters<3>                bulk;
    DomainParameters<1, 3>             representative;
    FiniteElementSpaceParameters<3>    bulk_space;
    FiniteElementSpaceParameters<1, 3> modal_space;
    TensorProductLift<1, 2, 3>         lift;
    RefinementParameters<1>            co_refinement;
    LinearAdapterParameters            solver;
    OutputHandlerParameters            output;
    dealii::ParameterAcceptorProxy<dealii::Functions::ParsedFunction<3>>
                                         forcing, boundary, exact_solution;
    bool                                 compute_errors = false;
    std::string                          modal_data     = "1";
    std::string                          radius_field;
    std::string                          modal_data_fields;
    std::set<dealii::types::boundary_id> dirichlet_ids{0};
    unsigned int                         refinement_cycles   = 1;
    std::string                          refinement_strategy = "fixed_fraction";
    double       refinement_fraction = .3, coarsening_fraction = 0.;
    unsigned int maximum_cells = 20000;
  };

  using LiftedPoissonVector = ImmersXLA::MPI::Vector;
  using LiftedPoissonState  = ImmersXLA::MPI::BlockVector;
  using LiftedPoissonAdapter =
    LinearAdapter<LiftedPoissonVector, LiftedPoissonState>;

  /** Read a fixed scalar radius from an imported FE observable. Retain a
   * relevant read-only vector: representative cells may need off-rank DoFs.
   */
  template <typename ObservableType>
  auto
  modal_radius(const ObservableType &observable)
  {
    const auto field        = observable.source();
    auto       coefficients = std::make_shared<LiftedPoissonVector>();
    coefficients->reinit(field.locally_owned_dofs(),
                         field.locally_relevant_dofs(),
                         field.space().mpi_communicator());
    *coefficients = observable.template frozen_values<LiftedPoissonVector>();
    coefficients->update_ghost_values();
    return [field, coefficients](const dealii::CellId   &id,
                                 const dealii::Point<3> &point) {
      const auto tria_cell = field.triangulation().create_cell_iterator(id);
      const auto cell = tria_cell->as_dof_handler_iterator(field.dof_handler());
      dealii::Quadrature<1> quadrature(
        {field.mapping().transform_real_to_unit_cell(cell, point)});
      dealii::FEValues<1, 3> values(field.mapping(),
                                    field.finite_element(),
                                    quadrature,
                                    dealii::update_values);
      values.reinit(cell);
      std::vector<double> radius(1);
      values[field.extractor()].get_function_values(*coefficients, radius);
      return radius[0];
    };
  }

  /** Resolve coefficient expressions against the Domain-owned input catalogue.
   * Imported values are evaluated on the current representative cell, not
   * interpolated as nodal expression values or searched on a third FE space.
   */
  inline ModalDataEvaluator
  modal_data_evaluator(const Domain<1, 3> &representative,
                       const std::string  &expressions,
                       unsigned int        n_modes,
                       const std::string  &selector = {})
  {
    const auto components =
      dealii::Utilities::split_string_list(expressions, ';');
    AssertThrow(components.size() == n_modes,
                dealii::ExcMessage(
                  "Provide one Modal data expression per selected mode."));
    const std::map<std::string, double> constants{{"pi", dealii::numbers::PI},
                                                  {"E", dealii::numbers::E}};
    std::vector<InputFieldBinding>      bindings;
    if (!selector.empty())
      bindings =
        InputFieldSelector::resolve(selector,
                                    representative.vtk_fields().catalog());
    if (bindings.empty())
      {
        auto function = std::make_shared<dealii::FunctionParser<3>>(n_modes);
        function->initialize("x,y,z,t", components, constants, true);
        return [function, n_modes](const auto &, const auto &point) {
          std::vector<double> values(n_modes);
          for (unsigned int c = 0; c < n_modes; ++c)
            values[c] = function->value(point, c);
          return values;
        };
      }
    const auto &properties   = representative.vtk_fields();
    auto        coefficients = std::make_shared<LiftedPoissonVector>();
    coefficients->reinit(properties.locally_owned_dofs(),
                         properties.locally_relevant_dofs(),
                         representative.communicator());
    *coefficients = properties.coefficients();
    coefficients->update_ghost_values();
    auto symbolic = std::make_shared<SymbolicFieldEvaluator>();
    std::vector<std::string> symbols;
    for (const auto &binding : bindings)
      symbols.push_back(binding.symbol_name);
    symbolic->initialize(components, symbols, constants);
    return [space = properties.space(),
            coefficients,
            symbolic,
            bindings](const auto &id, const auto &point) {
      const auto cell = space.dof_handler()
                          .get_triangulation()
                          .create_cell_iterator(id)
                          ->as_dof_handler_iterator(space.dof_handler());
      const dealii::Quadrature<1> quadrature(
        {space.mapping().transform_real_to_unit_cell(cell, point)});
      dealii::FEValues<1, 3> values(space.mapping(),
                                    space.finite_element(),
                                    quadrature,
                                    dealii::update_values);
      values.reinit(cell);
      std::vector<dealii::Vector<double>> samples(
        1, dealii::Vector<double>(space.finite_element().n_components()));
      values.get_function_values(*coefficients, samples);
      std::vector<double> inputs;
      for (const auto &binding : bindings)
        inputs.push_back(samples[0][binding.fe_component]);
      return symbolic->evaluate(point, 0., inputs);
    };
  }

  class LiftedPoissonOutputState : public StateAccessor<LiftedPoissonVector>
  {
  public:
    LiftedPoissonOutputState(const LiftedPoissonAdapter &adapter,
                             const LiftedPoissonState   &state)
      : adapter_(adapter)
      , state_(state)
    {}
    const LiftedPoissonVector &
    field(FieldId id, double) const override
    {
      return adapter_.field(state_, id);
    }

  private:
    const LiftedPoissonAdapter &adapter_;
    const LiftedPoissonState   &state_;
  };

  /** Application composition, independent of ReducedPoisson/ReducedCoupling. */
  inline void
  run_lifted_poisson(const LiftedPoissonParameters &parameters)
  {
    using namespace dealii;
    Domain<3>    bulk(parameters.bulk);
    Domain<1, 3> representative(parameters.representative);
    LocalCoRefinement<3, 1, 3>(parameters.co_refinement)
      .run(bulk, representative);
    const TensorProductLiftSupport<1, 2, 3> resolved_lift(
      parameters.lift.parameters());
    const unsigned int modes =
      resolved_lift.reference_cross_section().n_selected_basis();
    const auto               data = modal_data_evaluator(representative,
                                           parameters.modal_data,
                                           modes,
                                           parameters.modal_data_fields);
    FiniteElementSpace<3>    space(bulk.triangulation(), parameters.bulk_space);
    FiniteElementSpace<1, 3> modal_space(representative.triangulation(),
                                         parameters.modal_space);
    for (unsigned int cycle = 0; cycle < parameters.refinement_cycles; ++cycle)
      {
        // Every cycle creates fresh contributors, stencils and execution state.
        // The steady solve has no accepted transient history to transfer.
        if (cycle > 0)
          space.distribute_dofs();
        BoundaryConditions<3> boundary(space);
        for (const auto id : parameters.dirichlet_ids)
          boundary.add_dirichlet(id, parameters.boundary);
        boundary.update(0.);
        auto                 view       = space.view();
        auto                 modal_view = modal_space.view();
        LiftedPoissonAdapter adapter(parameters.solver, bulk.communicator());
        const auto           u =
          adapter.add(algebraic_field(view.field("u")), "background").fields();
        adapter.add_preconditioner(u.id(),
                                   [](const auto &matrix, const auto &reinit) {
                                     return make_amg_preconditioner(matrix,
                                                                    reinit);
                                   });
        adapter.add(weak_term(gradient(u), gradient(test(u))), "Poisson");
        const KnownSource<double, 3> negative_forcing(
          [&parameters](const auto &context) {
            return -parameters.forcing.value(context.point);
          });
        adapter.add(known_term(negative_forcing, test(u)), "forcing");
        using ModalField = decltype(modal_view.field("lambda"));
        typename ModalLiftedObservable<ModalField>::Radius radius;
        if (!parameters.radius_field.empty())
          {
            const auto input = representative.field(parameters.radius_field);
            radius = modal_radius(frozen(input.field(), input.coefficients()));
          }
        auto geometry =
          modal_lift(modal_view.field("lambda"), parameters.lift, radius);
        const auto lambda =
          adapter
            .add(LiftedMultiplierInteraction(u, geometry, data),
                 "lifted multiplier")
            .fields()
            .multiplier;
        auto state = adapter.make_state();
        adapter.solve(state);
        auto residual = adapter.make_state();
        adapter.evaluate_residual(state, residual);
        const double residual_norm = residual.l2_norm();
        if (Utilities::MPI::this_mpi_process(bulk.communicator()) == 0)
          std::cout << "LiftedPoisson cycle " << cycle
                    << ": bulk dofs=" << space.dof_handler().n_dofs()
                    << ", modal dofs=" << modal_space.dof_handler().n_dofs()
                    << ", residual=" << residual_norm << std::endl;
        OutputHandler<3, 3, LiftedPoissonVector> output(
          view, parameters.output, "bulk_" + std::to_string(cycle));
        output.add_field(u);
        LiftedPoissonOutputState accessor(adapter, state);
        output.write(accessor, cycle);
        OutputHandler<1, 3, LiftedPoissonVector> modal_output(
          modal_view, parameters.output, "modes_" + std::to_string(cycle));
        for (unsigned int c = 0; c < modes; ++c)
          modal_output.add_field(
            modal_view
              .field("lambda_" +
                       std::to_string(
                         parameters.lift.section.selected_coefficients[c]),
                     FEValuesExtractors::Scalar(c))
              .with_id(lambda.id()));
        modal_output.write(accessor, cycle);
        geometry.write_lifted_output(u,
                                     adapter.field(state, lambda),
                                     parameters.output.output_directory +
                                       "/lifted_" + std::to_string(cycle) +
                                       ".vtu");
        auto physical_solution = adapter.field(state, u);
        view.constraints().distribute(physical_solution);
        LiftedPoissonVector ghosted_solution;
        ghosted_solution.reinit(space.locally_owned_dofs(),
                                space.locally_relevant_dofs(),
                                bulk.communicator());
        ghosted_solution = physical_solution;
        ghosted_solution.update_ghost_values();
        double l2_error = std::numeric_limits<double>::quiet_NaN(),
               h1_error = l2_error;
        if (parameters.compute_errors)
          {
            Vector<float> errors(bulk.triangulation().n_active_cells());
            VectorTools::integrate_difference(
              view.mapping(),
              space.dof_handler(),
              ghosted_solution,
              parameters.exact_solution,
              errors,
              compatibility::gauss_quadrature<3>(
                space.finite_element().reference_cell(),
                space.finite_element().degree + 2),
              VectorTools::L2_norm);
            l2_error = VectorTools::compute_global_error(bulk.triangulation(),
                                                         errors,
                                                         VectorTools::L2_norm);
            VectorTools::integrate_difference(
              view.mapping(),
              space.dof_handler(),
              ghosted_solution,
              parameters.exact_solution,
              errors,
              compatibility::gauss_quadrature<3>(
                space.finite_element().reference_cell(),
                space.finite_element().degree + 2),
              VectorTools::H1_norm);
            h1_error = VectorTools::compute_global_error(bulk.triangulation(),
                                                         errors,
                                                         VectorTools::H1_norm);
          }
        const double bulk_norm  = physical_solution.l2_norm();
        const double modal_norm = adapter.field(state, lambda).l2_norm();
        if (Utilities::MPI::this_mpi_process(bulk.communicator()) == 0)
          {
            const auto    mode = cycle == 0 ? std::ios::out : std::ios::app;
            std::ofstream diagnostics(parameters.output.output_directory +
                                        "/diagnostics.csv",
                                      mode);
            AssertThrow(diagnostics.good(), ExcIO());
            diagnostics.precision(17);
            if (cycle == 0)
              diagnostics
                << "cycle,cells,bulk_dofs,modal_dofs,residual,bulk_norm,modal_norm,L2_error,H1_error\n";
            diagnostics << cycle << ','
                        << bulk.triangulation().n_global_active_cells() << ','
                        << space.dof_handler().n_dofs() << ','
                        << modal_space.dof_handler().n_dofs() << ','
                        << residual_norm << ',' << bulk_norm << ','
                        << modal_norm << ',' << l2_error << ',' << h1_error
                        << '\n';
          }
        if (cycle + 1 < parameters.refinement_cycles)
          {
            auto *tria =
              dynamic_cast<parallel::distributed::Triangulation<3> *>(
                &bulk.triangulation());
            AssertThrow(
              tria != nullptr,
              ExcMessage(
                "Adaptive cycles require a distributed background Domain."));
            Vector<float> errors(tria->n_active_cells());
            KellyErrorEstimator<3>::estimate(
              space.dof_handler(),
              QGauss<2>(space.finite_element().degree + 1),
              {},
              ghosted_solution,
              errors);
            if (parameters.refinement_strategy == "global")
              tria->refine_global(1);
            else if (parameters.refinement_strategy == "fixed_fraction")
              {
                parallel::distributed::GridRefinement::
                  refine_and_coarsen_fixed_fraction(
                    *tria,
                    errors,
                    parameters.refinement_fraction,
                    parameters.coarsening_fraction);
                tria->execute_coarsening_and_refinement();
              }
            else if (parameters.refinement_strategy == "fixed_number")
              {
                parallel::distributed::GridRefinement::
                  refine_and_coarsen_fixed_number(
                    *tria,
                    errors,
                    parameters.refinement_fraction,
                    parameters.coarsening_fraction,
                    parameters.maximum_cells);
                tria->execute_coarsening_and_refinement();
              }
          }
      }
  }
} // namespace ImmersX
#endif
