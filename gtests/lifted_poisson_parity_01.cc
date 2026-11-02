// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#include <deal.II/base/config.h>

#include <gtest/gtest.h>
#ifdef DEAL_II_WITH_VTK
#  include <immersx/physics/lifted_poisson.h>
#  include <immersx/physics/reduced_poisson.h>

#  include <fstream>

#  include "test_paths.h"

using namespace ImmersX;
using namespace dealii;

namespace
{
  std::string
  input_mesh(bool network)
  {
    const auto path = TestPaths::output_path(
      network ? "lifted-parity/network.vtk" : "lifted-parity/cylinder.vtk");
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path);
        out
          << "# vtk DataFile Version 3.0\nModal parity\nASCII\nDATASET UNSTRUCTURED_GRID\n";
        out
          << (network ?
                "POINTS 4 double\n0 0 -.6\n0 0 0\n-.4 .2 .6\n.4 -.2 .6\nCELLS 3 9\n2 0 1\n2 1 2\n2 1 3\nCELL_TYPES 3\n3 3 3\nPOINT_DATA 4\nSCALARS radius double 1\nLOOKUP_TABLE default\n.132 .15 .168 .168\n" :
                "POINTS 2 double\n0 0 -.6\n0 0 .6\nCELLS 1 3\n2 0 1\nCELL_TYPES 1\n3\nPOINT_DATA 2\nSCALARS radius double 1\nLOOKUP_TABLE default\n.132 .168\n");
      }
    MPI_Barrier(MPI_COMM_WORLD);
    return path.string();
  }

  template <int dim>
  auto
  dof_keys(const DoFHandler<dim, 3> &dh)
  {
    using Key = std::pair<std::array<long long, 3>, unsigned int>;
    const auto points =
      DoFTools::map_dofs_to_support_points(StaticMappingQ1<dim, 3>::mapping,
                                           dh);
    std::map<types::global_dof_index, Key> local;
    std::vector<types::global_dof_index> indices(dh.get_fe().n_dofs_per_cell());
    for (const auto &cell : dh.active_cell_iterators())
      if (!cell->is_artificial())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (dh.locally_owned_dofs().is_element(indices[i]))
              {
                const auto p      = points.at(indices[i]);
                local[indices[i]] = {
                  {{std::llround(1.e12 * p[0]),
                    std::llround(1.e12 * p[1]),
                    std::llround(1.e12 * p[2])}},
                  dh.get_fe().system_to_component_index(i).first};
              }
        }
    std::map<types::global_dof_index, Key> result;
    for (const auto &batch : Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
      result.insert(batch.begin(), batch.end());
    return result;
  }

  template <int dim>
  auto
  permutation(const DoFHandler<dim, 3> &modern, const DoFHandler<dim, 3> &old)
  {
    const auto old_keys = dof_keys(old);
    using Key           = typename decltype(old_keys)::mapped_type;
    std::map<Key, types::global_dof_index> old_indices;
    for (const auto &[i, key] : old_keys)
      old_indices.emplace(key, i);
    std::vector<types::global_dof_index> result(modern.n_dofs());
    for (const auto &[i, key] : dof_keys(modern))
      result[i] = old_indices.at(key);
    return result;
  }

  auto
  global_values(const ImmersXLA::MPI::Vector &vector)
  {
    std::map<types::global_dof_index, double> local, result;
    for (const auto i : vector.locally_owned_elements())
      local[i] = vector[i];
    for (const auto &batch : Utilities::MPI::all_gather(MPI_COMM_WORLD, local))
      result.insert(batch.begin(), batch.end());
    return result;
  }

  void
  parity(bool multiple,
         bool variable,
         bool network,
         bool imported_data      = false,
         bool immersed_partition = false)
  {
    if (variable && !SymbolicFieldEvaluator::available())
      GTEST_SKIP() << "The native variable-radius oracle requires SymEngine.";
    ParameterAcceptor::clear();
    const auto                  mesh = input_mesh(network);
    ReducedPoissonParameters<3> legacy_parameters;
    auto &coupling_parameters = legacy_parameters.reduced_coupling_parameters;
    auto &tensor = coupling_parameters.tensor_product_space_parameters;
    tensor.reduced_grid_name = mesh;
    tensor.thickness         = variable ? "radius" : "0.15";
    tensor.input_file_fields =
      imported_data ? "r=point:radius,radius" : (variable ? "radius" : "");
    tensor.section.inclusion_degree = multiple ? 1 : 0;
    tensor.section.selected_coefficients =
      multiple ? std::vector<unsigned int>{2, 0} : std::vector<unsigned int>{0};
    coupling_parameters.coupling_rhs_expressions =
      multiple ? std::vector<std::string>{"0.05*z", "0.5+0.1*z"} :
                 std::vector<std::string>{"0.5+0.1*z"};
    if (imported_data)
      coupling_parameters.coupling_rhs_expressions = {"0.1*r*r+0.01*pi+t",
                                                      "2*r+0.001*E+0.1*z"};
    auto &ref                       = coupling_parameters.refinement_parameters;
    ref.space_pre_refinement_cycles = 2;
    ref.embedded_pre_refinement_cycles = 0;
    ref.max_refinement_level           = 0;
    legacy_parameters.solver_name      = "Schur";
    legacy_parameters.output_directory =
      TestPaths::output_directory("lifted-parity/legacy");
    ParameterAcceptor::initialize("", "");
    ParameterAcceptor::prm.parse_input_from_string(
      "subsection Reduced Poisson\nsubsection Right hand side\nset Function expression = 1+x\nend\nend\n");
    ParameterAcceptor::parse_all_parameters();
    legacy_parameters.inner_control.set_reduction(1.e-12);
    legacy_parameters.inner_control.set_tolerance(1.e-14);
    legacy_parameters.inner_control.set_max_steps(2000);
    legacy_parameters.outer_control.set_reduction(1.e-12);
    legacy_parameters.outer_control.set_tolerance(1.e-14);
    legacy_parameters.outer_control.set_max_steps(2000);
    ReducedPoisson<3> legacy(legacy_parameters);
    legacy.make_grid();
    legacy.setup_fe();
    legacy.setup_dofs();
    legacy.assemble_poisson_system();
    legacy.assemble_coupling_system();
    legacy.solve();

    DomainParameters<3> dp("/Parity/Bulk/");
    dp.arguments_for_grid = "-1:1:false";
    DomainParameters<1, 3> rp("/Parity/Representative/");
    rp.name_of_grid    = mesh;
    rp.read_vtk_fields = variable;
    Domain<3>    bulk(dp);
    Domain<1, 3> representative(rp);
    if (immersed_partition)
      ref.embedded_partitioning_policy = "immersed";
    LocalCoRefinement<3, 1, 3>(ref).run(bulk, representative);
    FiniteElementSpaceParameters<3>    fp("/Parity/Background space/");
    FiniteElementSpaceParameters<1, 3> mp("/Parity/Modal space/",
                                          multiple ?
                                            "FESystem<1,3>[FE_Q<1,3>(1)^2]" :
                                            "FESystem<1,3>[FE_Q<1,3>(1)]");
    FiniteElementSpace<3>              space(bulk.triangulation(), fp);
    FiniteElementSpace<1, 3>   modal_space(representative.triangulation(), mp);
    BoundaryConditions<3>      bc(space);
    Functions::ZeroFunction<3> zero;
    bc.add_dirichlet(0, zero);
    bc.update(0.);
    auto view       = space.view();
    auto modal_view = modal_space.view();
    ASSERT_EQ(space.dof_handler().n_dofs(), legacy.bulk_dof_handler().n_dofs());
    const auto bulk_permutation =
      permutation(space.dof_handler(), legacy.bulk_dof_handler());
    const auto modal_permutation =
      permutation(modal_space.dof_handler(),
                  legacy.representative_dof_handler());
    const auto legacy_values           = global_values(legacy.bulk_values());
    const auto legacy_modal_values     = global_values(legacy.modal_values());
    const auto legacy_load             = global_values(legacy.modal_rhs());
    const bool compare_preconditioners = !variable && !network;
    if (compare_preconditioners)
      {
        legacy_parameters.solver_name = "AL";
        legacy.solve();
        auto old_al_difference = legacy.bulk_values();
        for (const auto i : old_al_difference.locally_owned_elements())
          old_al_difference[i] -= legacy_values.at(i);
        EXPECT_LT(old_al_difference.l2_norm(), 1.e-7);
        auto old_al_modal_difference = legacy.modal_values();
        for (const auto i : old_al_modal_difference.locally_owned_elements())
          old_al_modal_difference[i] -= legacy_modal_values.at(i);
        EXPECT_LT(old_al_modal_difference.l2_norm(), 1.e-6);
      }
    std::map<std::pair<types::global_dof_index, types::global_dof_index>,
             double>
                local_entries, old_entries;
    const auto &old_matrix = legacy.assembled_coupling();
    for (const auto i : legacy.bulk_values().locally_owned_elements())
      for (auto entry = old_matrix.begin(i); entry != old_matrix.end(i);
           ++entry)
        local_entries[{i, entry->column()}] = entry->value();
    for (const auto &batch :
         Utilities::MPI::all_gather(MPI_COMM_WORLD, local_entries))
      old_entries.insert(batch.begin(), batch.end());
    TensorProductLift<1, 2, 3> descriptor("/Parity/Lift/");
    descriptor.thickness                = tensor.thickness;
    descriptor.section.inclusion_degree = tensor.section.inclusion_degree;
    descriptor.section.selected_coefficients =
      tensor.section.selected_coefficients;
    descriptor.representative_n_q_points = 3;
    using ModalField                     = decltype(modal_view.field("lambda"));
    typename ModalLiftedObservable<ModalField>::Radius radius;
    if (variable)
      {
        const auto input = representative.field("radius");
        radius = modal_radius(frozen(input.field(), input.coefficients()));
      }
    auto geometry = modal_lift(modal_view.field("lambda"), descriptor, radius);
    std::string expressions;
    for (const auto &component : coupling_parameters.coupling_rhs_expressions)
      {
        if (!expressions.empty())
          expressions += ';';
        expressions += component;
      }
    const auto data =
      modal_data_evaluator(representative,
                           expressions,
                           multiple ? 2 : 1,
                           imported_data ? "r=point:radius" : "");
    const std::vector<LinearPreconditioner> policies =
      compare_preconditioners ?
        std::vector<LinearPreconditioner>{
          LinearPreconditioner::schur,
          LinearPreconditioner::augmented_lagrangian} :
        std::vector<LinearPreconditioner>{LinearPreconditioner::schur};
    for (const auto policy : policies)
      {
        SCOPED_TRACE(policy == LinearPreconditioner::schur ? "Schur" : "AL");
        LinearAdapterParameters options("/Parity/Solver/");
        options.solver             = LinearSolver::iterative;
        options.preconditioner     = policy;
        options.maximum_iterations = 2000;
        options.tolerance          = 1.e-13;
        LiftedPoissonAdapter adapter(options, MPI_COMM_WORLD);
        const auto u = adapter.add(algebraic_field(view.field("u"))).fields();
        adapter.add_preconditioner(u.id(),
                                   [](const auto &matrix, const auto &reinit) {
                                     return make_amg_preconditioner(matrix,
                                                                    reinit);
                                   });
        adapter.add(weak_term(gradient(u), gradient(test(u))));
        adapter.add(known_term(KnownSource<double, 3>([](const auto &context) {
                                 return -(1. + context.point[0]);
                               }),
                               test(u)));
        const auto lambda =
          adapter.add(LiftedMultiplierInteraction(u, geometry, data))
            .fields()
            .multiplier;
        auto        state  = adapter.make_state();
        const auto  blocks = adapter.block_matrix(state);
        const auto &modern = blocks.block(0, 1);
        const auto &old    = legacy.assembled_coupling();
        ASSERT_EQ(modern.m(), old.m());
        ASSERT_EQ(modern.n(), old.n());
        double local_error = 0.;
        for (const auto i : space.locally_owned_dofs())
          for (unsigned int j = 0; j < old.n(); ++j)
            local_error = std::max(
              local_error,
              std::abs(
                modern.el(i, j) -
                old_entries[{bulk_permutation[i], modal_permutation[j]}]));
        EXPECT_LT(Utilities::MPI::max(local_error, MPI_COMM_WORLD), 1.e-11);
        auto residual = adapter.make_state();
        adapter.evaluate_residual(state, residual);
        auto load = adapter.field(residual, lambda);
        load *= -1.;
        for (const auto i : load.locally_owned_elements())
          load[i] -= legacy_load.at(modal_permutation[i]);
        EXPECT_LT(load.l2_norm(), 1.e-11);
        adapter.solve(state);
        adapter.evaluate_residual(state, residual);
        EXPECT_LT(residual.l2_norm(), 1.e-8);
        auto difference = adapter.field(state, u);
        for (const auto i : difference.locally_owned_elements())
          difference[i] -= legacy_values.at(bulk_permutation[i]);
        EXPECT_LT(difference.l2_norm(), 1.e-7);
        auto modal_difference = adapter.field(state, lambda);
        for (const auto i : modal_difference.locally_owned_elements())
          modal_difference[i] -= legacy_modal_values.at(modal_permutation[i]);
        EXPECT_LT(modal_difference.l2_norm(), 1.e-6);
        EXPECT_GT(adapter.field(state, u).l2_norm(), 1.e-3);
      }
  }
} // namespace
TEST(LiftedPoisson, BOTH_LegacySingleCylinderParity)
{
  parity(false, false, false);
}
TEST(LiftedPoisson, BOTH_LegacyMultipleModesParity)
{
  parity(true, false, false);
}
TEST(LiftedPoisson, BOTH_LegacyVariableRadiusParity)
{
  parity(true, true, false);
}
TEST(LiftedPoisson, BOTH_LegacyNetworkParity)
{
  parity(true, true, true);
}
TEST(LiftedPoisson, BOTH_LegacyImmersedPartitionParity)
{
  parity(true, true, true, false, true);
}
TEST(LiftedPoisson, BOTH_LegacyImportedModalDataParity)
{
  parity(true, true, true, true);
}
#endif
