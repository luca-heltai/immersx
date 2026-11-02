// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#include <deal.II/lac/lapack_full_matrix.h>

#include <gtest/gtest.h>
#include <immersx/physics/elasticity.h>
#include <immersx/physics/lifted_elasticity.h>

#include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  template <int dim>
  void
  compare_uncoupled(const bool weak,
                    const bool heterogeneous = false,
                    const bool refined       = false,
                    const bool mixed         = false)
  {
    ParameterAcceptor::clear();
    ElasticityProblemParameters<dim> native;
    LiftedElasticityParameters<dim>  modern;
    ElasticityProblem<dim>           reference(native);
    initialize_parameters();
    const std::string force = dim == 2 ? "1; -2" : "1; -2; 0.5";
    ParameterAcceptor::prm.parse_input_from_string(
      "subsection Functions\n subsection Right hand side\n set Function expression = " +
      force +
      "\n end\n end\n"
      "subsection Lifted elasticity\n subsection Functions\n subsection Right hand side\n set Function expression = " +
      force + "\n end\n end\n end\n");
    ParameterAcceptor::parse_all_parameters();
    native.initial_refinement = modern.domain_parameters.initial_refinement = 1;
    native.arguments_for_grid = "-1: 1: false";
    native.default_material_properties.Lame_mu =
      modern.default_material_properties.Lame_mu = 1.;
    native.default_material_properties.Lame_lambda =
      modern.default_material_properties.Lame_lambda = 2.;
    modern.coupling                                  = "none";
    if (mixed)
      {
        native.arguments_for_grid =
          modern.domain_parameters.arguments_for_grid = "-1: 1: true";
        native.dirichlet_ids = modern.dirichlet_ids = {0};
        native.neumann_ids = modern.neumann_ids = {1};
        native.weak_dirichlet_ids = modern.weak_dirichlet_ids = {2};
        native.normal_flux_ids = modern.normal_flux_ids = {3};
        native.penalty_term = modern.penalty_term = 100.;
        ParameterAcceptor::prm.parse_input_from_string(
          "subsection Functions\n subsection Neumann boundary conditions\n set Function expression = " +
          force +
          "\n end\n end\n"
          "subsection Lifted elasticity\n subsection Functions\n subsection Neumann boundary conditions\n set Function expression = " +
          force + "\n end\n end\n end\n");
        ParameterAcceptor::parse_all_parameters();
      }
    if (weak)
      {
        native.dirichlet_ids.clear();
        modern.dirichlet_ids.clear();
        native.weak_dirichlet_ids = {0};
        modern.weak_dirichlet_ids = {0};
        native.penalty_term = modern.penalty_term = 100.;
      }
    if (heterogeneous)
      {
        native.material_properties_by_id[1] =
          std::make_unique<MaterialProperties>("stiff",
                                               "/Native test material/");
        modern.material_properties_by_id[1] =
          std::make_unique<MaterialProperties>("stiff",
                                               "/Modern test material/");
        for (auto *material : {native.material_properties_by_id[1].get(),
                               modern.material_properties_by_id[1].get()})
          {
            material->Lame_mu     = 4.;
            material->Lame_lambda = 7.;
          }
      }
    const auto configure = [&](Triangulation<dim> &tria) {
      if (refined)
        {
          unsigned int marked = 0;
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                if (cell->center()[0] < 0.)
                  {
                    cell->set_refine_flag();
                    ++marked;
                  }
              }
          EXPECT_GT(Utilities::MPI::sum(marked, MPI_COMM_WORLD), 0u);
          const auto before = tria.n_global_active_cells();
          tria.execute_coarsening_and_refinement();
          EXPECT_GT(tria.n_global_active_cells(), before);
        }
      if (heterogeneous)
        for (const auto &cell : tria.active_cell_iterators())
          if (!cell->is_artificial())
            cell->set_material_id(cell->center()[0] > 0. ? 1 : 0);
    };
    reference.make_grid();
    configure(*reference.tria);
    reference.setup_fe();
    reference.setup_dofs();
    reference.assemble_elasticity_system();
    reference.solve_static();
    LiftedElasticity<dim> problem(modern);
    problem.setup(configure);
    problem.solve();
    if (refined)
      {
        AffineConstraints<double> hanging;
        DoFTools::make_hanging_node_constraints(
          problem.displacement().dof_handler(), hanging);
        hanging.close();
        EXPECT_GT(Utilities::MPI::sum(hanging.n_constraints(), MPI_COMM_WORLD),
                  0u);
      }
    auto difference = problem.solution();
    difference -= reference.solution.block(0);
    EXPECT_GT(reference.solution.block(0).l2_norm(), 1.e-6);
    EXPECT_LT(difference.l2_norm(), 1.e-8);
    EXPECT_LT(problem.residual_norm(), 1.e-9);
    LinearAdapterParameters diagnostic_parameters("/Bulk diagnostics/");
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
               diagnostic(diagnostic_parameters, MPI_COMM_WORLD);
    const auto field =
      diagnostic.add(algebraic_field(problem.displacement()), "solid").fields();
    diagnostic.add(static_elasticity_term(field, modern), "elasticity");
    if (!modern.weak_dirichlet_ids.empty())
      diagnostic.add(static_elasticity_nitsche(field, modern), "nitsche");
    auto state = diagnostic.make_state(), residual = diagnostic.make_state();
    diagnostic.evaluate_residual(state, residual);
    auto rhs = residual.block(0);
    rhs += reference.system_rhs.block(0);
    // Nitsche and volume terms may choose different artificial diagonals on
    // constrained rows. Compare their physical free-row equations instead.
    if (mixed)
      for (const auto index : rhs.locally_owned_elements())
        if (field.constraints().is_constrained(index))
          rhs[index] = 0.;
    EXPECT_LT(rhs.l2_norm(), 1.e-11);
    auto probe = reference.system_rhs.block(0), expected = probe,
         actual = probe;
    for (const auto index : probe.locally_owned_elements())
      probe[index] = std::sin(double(index) + 0.2);
    reference.stiffness_matrix.vmult(expected, probe);
    const auto matrix = diagnostic.block_matrix(state);
    matrix.block(0, 0).vmult(actual, probe);
    actual -= expected;
    if (mixed)
      for (const auto index : actual.locally_owned_elements())
        if (field.constraints().is_constrained(index))
          actual[index] = 0.;
    EXPECT_LT(actual.l2_norm(), 1.e-10);
  }

  void
  compare_coupled(
    const bool                 pressure,
    const bool                 variable_radius = false,
    const bool                 higher_modes    = false,
    const LinearPreconditioner preconditioner = LinearPreconditioner::automatic,
    const LinearSolver         solver         = LinearSolver::automatic,
    const bool                 oblique        = false)
  {
    if (variable_radius && !SymbolicFieldEvaluator::available())
      GTEST_SKIP() << "The native variable-radius oracle requires SymEngine.";
    ParameterAcceptor::clear();
    ElasticityProblemParameters<3> native;
    LiftedElasticityParameters<3>  modern;
    ElasticityProblem<3>           reference(native);
    initialize_parameters();
    ParameterAcceptor::prm.parse_input_from_string(R"(
subsection Functions
 subsection Right hand side
  set Function expression = 1; -2; 0.5
 end
end
subsection Lifted elasticity
 subsection Functions
  subsection Right hand side
   set Function expression = 1; -2; 0.5
  end
 end
end
)");
    ParameterAcceptor::parse_all_parameters();
    native.initial_refinement = modern.domain_parameters.initial_refinement = 0;
    native.name_of_grid = modern.domain_parameters.name_of_grid =
      pressure ? "hyper_cube" : "subdivided_hyper_rectangle";
    native.arguments_for_grid = modern.domain_parameters.arguments_for_grid =
      pressure     ? "-1: 1: false" :
      higher_modes ? "4, 4, 3: -1, -1, -1: 1, 1, 1: false" :
                     "3, 3, 3: -1, -1, -1: 1, 1, 1: false";
    if (pressure)
      native.initial_refinement = modern.domain_parameters.initial_refinement =
        1;
    native.default_material_properties.Lame_mu =
      modern.default_material_properties.Lame_mu = 1.;
    native.default_material_properties.Lame_lambda =
      modern.default_material_properties.Lame_lambda = 2.;
    native.coupling_type     = CouplingType::TensorProduct;
    native.pressure_coupling = pressure;
    auto &space =
      native.tensor_product_coupling_parameters.tensor_product_space_parameters;
    // Higher angular modes need a resolved cross-section, rather than a thin
    // cylinder conditioning stress, for the native/modern parity oracle.
    space.reduced_grid_name = TestPaths::data_filename(
      oblique         ? "tests/elasticity_modal_curve.vtk" :
      higher_modes    ? "tests/elasticity_modal_wide_cylinder.vtk" :
      variable_radius ? "tests/one_cylinder_properties.vtk" :
                        "tests/one_cylinder.vtk");
    space.thickness                     = "0.1";
    space.section.selected_coefficients = {0, 1, 2};
    space.n_q_points                    = 3;
    native.tensor_product_coupling_parameters.refinement_parameters
      .lower_factor = 0.0001;
    native.tensor_product_coupling_parameters.refinement_parameters
      .upper_factor                                                    = 1000.;
    native.tensor_product_coupling_parameters.coupling_rhs_expressions = {
      "0.01", "-0.02", "0.03"};
    modern.coupling                     = pressure ? "pressure" : "multiplier";
    modern.execution.preconditioner     = preconditioner;
    modern.execution.solver             = solver;
    modern.immersed_domain.name_of_grid = space.reduced_grid_name;
    modern.immersed_domain.initial_refinement = 0;
    modern.geometry.thickness                 = "0.1";
    modern.geometry.representative_n_q_points = 3;
    modern.modal_data                         = {"0.01", "-0.02", "0.03"};
    modern.co_refinement.lower_factor         = 0.0001;
    modern.co_refinement.upper_factor         = 1000.;
    if (variable_radius)
      {
        space.thickness                        = "r";
        space.input_file_fields                = "r=point:radius";
        modern.immersed_domain.read_vtk_fields = true;
        modern.radius_field                    = "radius";
        modern.modal_data_fields               = "r=point:radius";
        modern.modal_data =
          native.tensor_product_coupling_parameters.coupling_rhs_expressions = {
            "0.1*r", "-0.2*r", "0.05*r"};
      }
    if (higher_modes)
      {
        space.section.inclusion_degree =
          modern.geometry.section.inclusion_degree    = 1;
        space.section.selected_coefficients           = {1, 4, 7};
        modern.geometry.section.selected_coefficients = {1, 4, 7};
      }
    reference.make_grid();
    reference.setup_fe();
    reference.setup_dofs();
    reference.assemble_elasticity_system();
    reference.assemble_coupling();
    if (!pressure)
      {
        // The regression needs independent modal rows, not a mesh-size study.
        // Check the free-row coupling rank before solving the saddle system.
        const auto         &coupling = reference.coupling_matrix;
        std::vector<double> local(coupling.m() * coupling.n(), 0.);
        for (const auto row : reference.owned_dofs[0])
          if (!reference.constraints.is_constrained(row))
            for (auto entry = coupling.begin(row); entry != coupling.end(row);
                 ++entry)
              local[row * coupling.n() + entry->column()] = entry->value();
        std::vector<double> global(local.size());
        Utilities::MPI::sum(local, MPI_COMM_WORLD, global);
        LAPACKFullMatrix<double> singular_values(coupling.m(), coupling.n());
        for (unsigned int i = 0; i < coupling.m(); ++i)
          for (unsigned int j = 0; j < coupling.n(); ++j)
            singular_values(i, j) = global[i * coupling.n() + j];
        singular_values.compute_svd();
        ASSERT_GT(singular_values.singular_value(coupling.n() - 1),
                  1.e-10 * singular_values.singular_value(0));
      }
    reference.solve_static();
    LiftedElasticity<3> problem(modern);
    problem.setup();
    problem.solve();
    ASSERT_TRUE(problem.initial_co_refinement_report().has_value());
    EXPECT_TRUE(problem.initial_co_refinement_report()->satisfied());
    EXPECT_GT(problem.initial_co_refinement_report()->intersecting_pairs, 0u);
    auto difference = problem.solution();
    difference -= reference.solution.block(0);
    EXPECT_GT(reference.solution.block(0).l2_norm(), 1.e-6);
    EXPECT_LT(difference.l2_norm(), 2.e-7);
    EXPECT_LT(problem.residual_norm(), 1.e-9);
    EXPECT_EQ(problem.multipliers().size(), 3u);
    auto modal_difference = problem.modal_coefficients();
    modal_difference -= reference.solution.block(1);
    EXPECT_LT(modal_difference.l2_norm(), 2.e-6);

    // The default modal case checks these solver-independent operators.
    // Explicit solver-policy variants retain the solution and residual oracle.
    if (preconditioner == LinearPreconditioner::augmented_lagrangian ||
        solver == LinearSolver::direct)
      return;

    // Independent production oracles for the physical metric, modal RHS, and
    // coupling action. This checks ordering and frames, not just solution
    // norms.
    LinearAdapterParameters execution_parameters("/Parity diagnostics/");
    LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
               execution(execution_parameters, MPI_COMM_WORLD);
    const auto u =
      execution.add(algebraic_field(problem.displacement()), "u").fields();
    const auto       &view = problem.multipliers().front().space();
    FunctionParser<3> data(3);
    auto              diagnostic_data = modern.modal_data;
    if (variable_radius)
      {
        const std::string r =
          oblique ?
            "(z<0.1 ? 0.08+(0.04/0.3)*(z+0.2) : 0.12-(0.03/0.2)*(z-0.1))" :
          higher_modes ? "(0.25-0.125*z)" :
                         "(0.1-0.05*z)";
        diagnostic_data = {"0.1*" + r, "-0.2*" + r, "0.05*" + r};
      }
    data.initialize(FunctionParser<3>::default_variable_names(),
                    diagnostic_data,
                    {});
    ImmersXLA::MPI::Vector imported_values;
    if (variable_radius)
      {
        const auto &properties = problem.immersed().vtk_fields();
        imported_values.reinit(properties.locally_owned_dofs(),
                               properties.locally_relevant_dofs(),
                               MPI_COMM_WORLD);
        imported_values = properties.coefficients();
        imported_values.update_ghost_values();
      }
    using Lifted =
      decltype(lift(value(view.field("multiplier")), modern.geometry));
    using Weak = decltype(weak_term(value(u), test(std::declval<Lifted>())));
    ConstraintSum<Weak>                           terms;
    std::shared_ptr<ImmersXLA::MPI::SparseMatrix> metric;
    std::shared_ptr<ImmersXLA::MPI::Vector>       load;
    for (unsigned int slot = 0; slot < 3; ++slot)
      {
        const auto field =
          view.field("multiplier", FEValuesExtractors::Scalar(slot));
        const auto lifted = [&] {
          if (!variable_radius)
            return lift(value(field), modern.geometry).mode(slot);
          const auto radius = problem.immersed().field("radius");
          return lift(value(field),
                      modern.geometry,
                      frozen(radius.field(), imported_values))
            .mode(slot);
        }();
        const auto [mass, rhs] = lifted_modal_load(lifted, data, slot);
        if (!metric)
          {
            metric = mass;
            load   = rhs;
          }
        else
          {
            metric->add(1., *mass);
            *load += *rhs;
          }
        terms.add(weak_term(value(u), test(lifted)));
      }
    auto rhs_difference = *load;
    rhs_difference -= reference.system_rhs.block(1);
    EXPECT_LT(rhs_difference.l2_norm(), 1.e-12);
    auto coefficients = *load;
    for (const auto index : coefficients.locally_owned_elements())
      coefficients[index] = std::sin(double(index) + 0.3);
    auto expected = *load, actual = *load;
    reference.inclusion_matrix.vmult(expected, coefficients);
    metric->vmult(actual, coefficients);
    actual -= expected;
    EXPECT_LT(actual.l2_norm(), 1.e-12);
    execution.add(lifted_modal_constraint(std::move(terms), metric, *load),
                  "modal");
    auto       state  = execution.make_state();
    const auto matrix = execution.block_matrix(state);
    auto       bulk   = reference.system_rhs.block(0);
    for (const auto index : bulk.locally_owned_elements())
      bulk[index] = std::cos(double(index) + 0.1);
    reference.coupling_matrix.Tvmult(expected, bulk);
    matrix.block(1, 0).vmult(actual, bulk);
    actual -= expected;
    EXPECT_LT(actual.l2_norm(), 2.e-12);
  }
} // namespace

TEST(LiftedElasticity, BOTH_StrongDirichlet2D)
{
  compare_uncoupled<2>(false);
}
TEST(LiftedElasticity, BOTH_StrongDirichlet3D)
{
  compare_uncoupled<3>(false);
}
TEST(LiftedElasticity, BOTH_Nitsche2D)
{
  compare_uncoupled<2>(true);
}
TEST(LiftedElasticity, BOTH_Nitsche3D)
{
  compare_uncoupled<3>(true);
}
TEST(LiftedElasticity, BOTH_MixedBoundaryConditions2D)
{
  compare_uncoupled<2>(false, false, false, true);
}
TEST(LiftedElasticity, BOTH_MixedBoundaryConditions3D)
{
  compare_uncoupled<3>(false, false, false, true);
}
TEST(LiftedElasticity, BOTH_ModalConstraint)
{
  compare_coupled(false);
}
TEST(LiftedElasticity, BOTH_ModalPressure)
{
  compare_coupled(true);
}
TEST(LiftedElasticity, BOTH_ModalAugmentedLagrangian)
{
  compare_coupled(false,
                  false,
                  false,
                  LinearPreconditioner::augmented_lagrangian);
}
TEST(LiftedElasticity, BOTH_ModalDirect)
{
  compare_coupled(
    false, false, false, LinearPreconditioner::automatic, LinearSolver::direct);
}
TEST(LiftedElasticity, BOTH_ObliqueVariableRadiusConstraint)
{
  compare_coupled(false,
                  true,
                  false,
                  LinearPreconditioner::augmented_lagrangian,
                  LinearSolver::iterative,
                  true);
}
TEST(LiftedElasticity, BOTH_ObliqueVariableRadiusPressure)
{
  compare_coupled(true,
                  true,
                  false,
                  LinearPreconditioner::automatic,
                  LinearSolver::iterative,
                  true);
}
TEST(LiftedElasticity, BOTH_HeterogeneousHangingNodes2D)
{
  compare_uncoupled<2>(false, true, true);
}
TEST(LiftedElasticity, BOTH_HeterogeneousHangingNodes3D)
{
  compare_uncoupled<3>(false, true, true);
}
TEST(LiftedElasticity, BOTH_VariableRadiusModalConstraint)
{
  compare_coupled(false, true);
}
TEST(LiftedElasticity, BOTH_VariableRadiusModalPressure)
{
  compare_coupled(true, true);
}
TEST(LiftedElasticity, BOTH_NonstandardModalPressure)
{
  compare_coupled(true, true, true);
}
TEST(LiftedElasticity, BOTH_NonstandardModalConstraint)
{
  compare_coupled(false, true, true);
}

TEST(LiftedElasticity, BOTH_NonzeroBoundaryLiftManufacturedSolution)
{
  ParameterAcceptor::clear();
  LiftedElasticityParameters<3> parameters;
  initialize_parameters();
  ParameterAcceptor::prm.parse_input_from_string(R"(
subsection Lifted elasticity
 subsection Functions
  subsection Dirichlet boundary conditions
   set Function expression = 0.1; -0.2; 0.3
  end
 end
end
)");
  ParameterAcceptor::parse_all_parameters();
  parameters.domain_parameters.initial_refinement = 1;
  parameters.immersed_domain.initial_refinement   = 0;

  parameters.geometry.thickness         = "0.1";
  parameters.co_refinement.lower_factor = 0.0001;
  parameters.co_refinement.upper_factor = 1000.;
  // For the default x-directed curve, determine the physical constant-mode
  // coefficients independently from the cross-section rotation.
  ReferenceCrossSection<1, 3, 3> section(parameters.geometry.section);
  Tensor<1, 3>                   tangent;
  tangent[0]       = 1.;
  const auto modes = section.get_transformed_mode_values(0, tangent);
  const std::array<double, 3> constant{{0.1, -0.2, 0.3}};
  for (unsigned int slot = 0; slot < 3; ++slot)
    {
      double coefficient = 0.;
      for (unsigned int d = 0; d < 3; ++d)
        coefficient += constant[d] * modes[3 * slot + d];
      parameters.modal_data[slot] = std::to_string(coefficient);
    }
  LiftedElasticity<3> problem(parameters);
  problem.setup([](Triangulation<3> &tria) {
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned() && cell->center()[0] < 0.)
        cell->set_refine_flag();
    tria.execute_coarsening_and_refinement();
  });
  AffineConstraints<double> hanging;
  DoFTools::make_hanging_node_constraints(problem.displacement().dof_handler(),
                                          hanging);
  hanging.close();
  EXPECT_GT(Utilities::MPI::sum(hanging.n_constraints(), MPI_COMM_WORLD), 0u);
  problem.solve();
  ImmersXLA::MPI::Vector         expected(problem.solution());
  Functions::ConstantFunction<3> exact(std::vector<double>{0.1, -0.2, 0.3});
  VectorTools::interpolate(problem.displacement().dof_handler(),
                           exact,
                           expected);
  expected -= problem.solution();
  EXPECT_LT(expected.l2_norm(), 1.e-10);
  EXPECT_LT(problem.modal_coefficients().l2_norm(), 1.e-10);
  EXPECT_LT(problem.residual_norm(), 1.e-10);
}

TEST(LiftedElasticity, ObservableRegistrationUpdatesDependencies)
{
  ParameterAcceptor::clear();
  DomainParameters<1, 3> parameters;
  Domain<1, 3>           domain(parameters);
  domain.make_grid();
  FiniteElementSpaceParameters<1, 3> finite_element;
  FiniteElementSpace<1, 3> space(domain.triangulation(), finite_element);
  const auto               view = space.view();
  const auto expression = value(view.field("lambda")).with_id(FieldId(7));
  ASSERT_EQ(expression.dependencies().size(), 1u);
  EXPECT_EQ(expression.source_field(), FieldId(7));
  EXPECT_EQ(expression.source().id(), FieldId(7));
}
