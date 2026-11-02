// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/matrix_creator.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/weak_term.h>

#include <cmath>

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct SumWithFrozenInput
  {
    template <typename Value>
    Value
    evaluate(const Value &active, const Value &offset) const
    {
      return active + offset;
    }

    template <typename Value>
    Value
    linearize(const Value &,
              const Value &,
              const Value &direction,
              const Value &frozen_direction) const
    {
      return direction + frozen_direction;
    }
  };

  struct PhysicalSupport
  {
    static constexpr int intrinsic_dimension = 1;
    static constexpr int ambient_dimension   = 2;

    struct PointData
    {
      Point<2>      point;
      double        weight;
      std::uint64_t stable_id;
      unsigned int  owner_rank;
    };

    std::vector<PointData> points;

    const auto &
    locally_owned_quadrature_points() const
    {
      return points;
    }

    MPI_Comm
    mpi_communicator() const
    {
      return MPI_COMM_WORLD;
    }
  };

  template <unsigned int components, typename VectorType, typename MatrixType>
  void
  check_nonlinear_owned_storage()
  {
    parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(mesh);
    mesh.refine_global(1);
    FESystem<2>   fe(FE_Q<2>(1), components);
    DoFHandler<2> dh(mesh);
    dh.distribute_dofs(fe);
    AffineConstraints<double> constraints;
    constraints.close();
    const auto relevant = DoFTools::extract_locally_relevant_dofs(dh);
    const auto space =
      fe_space(dh, StaticMappingQ1<2>::mapping, constraints, relevant);
    StateLayout layout;
    const auto  u = [&]() {
      if constexpr (components == 1)
        return space.field(layout, "u");
      else
        return space.field(layout, "u", FEValuesExtractors::Vector(0));
    }();
    const auto offset = u.with_id(FieldId{});
    VectorType owned, direction, dual, action, transpose, residual, perturbed;
    for (auto *vector : {&owned,
                         &direction,
                         &dual,
                         &action,
                         &transpose,
                         &residual,
                         &perturbed})
      detail::initialize_weak_vector(*vector,
                                     dh.locally_owned_dofs(),
                                     MPI_COMM_WORLD);
    for (const auto i : dh.locally_owned_dofs())
      {
        owned[i]     = 1. + std::sin(0.7 * i);
        direction[i] = std::cos(0.3 * i);
        dual[i]      = 0.2 + std::sin(0.9 * i);
      }
    owned.compress(VectorOperation::insert);
    direction.compress(VectorOperation::insert);
    dual.compress(VectorOperation::insert);
    VectorType state, frozen_coefficients;
    if constexpr (detail::has_index_set_vector_reinit<VectorType>::value)
      {
        state.reinit(dh.locally_owned_dofs(), relevant, MPI_COMM_WORLD);
        frozen_coefficients.reinit(dh.locally_owned_dofs(),
                                   relevant,
                                   MPI_COMM_WORLD);
      }
    else
      {
        state.reinit(dh.n_dofs());
        frozen_coefficients.reinit(dh.n_dofs());
      }
    state = owned;
    VectorType frozen_owned(owned);
    frozen_owned             = 0.7;
    frozen_coefficients      = frozen_owned;
    const auto observable    = transform(value(u),
                                      frozen(offset, frozen_coefficients),
                                      SumWithFrozenInput{});
    const auto check_support = [&](const auto &support) {
      SemiDiscreteModel<VectorType, MatrixType>   model;
      SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
      weak_term(observable, test(u), support).add(builder);
      StateView<VectorType> state_view(layout, 0.);
      state_view.bind(u.field_id(), state);
      const EvaluationContext<VectorType> context(0., state_view);
      residual = 0.;
      model.evaluate_row(u.field_id(), context, residual);
      const auto jacobian =
        model.state_matrix_operator(u.field_id(), u.field_id(), context);
      ASSERT_TRUE(jacobian.has_value());
      jacobian->view.vmult(action, direction);

      // Keep the linear assembly oracle separate from nonlinear assembly.
      SemiDiscreteModel<VectorType, MatrixType>   linear_model;
      SemidiscreteBuilder<VectorType, MatrixType> linear_builder(layout,
                                                                 linear_model);
      weak_term(value(u), test(u), support).add(linear_builder);
      const auto linear =
        linear_model.state_matrix_operator(u.field_id(), u.field_id(), context);
      ASSERT_TRUE(linear.has_value());
      linear->view.vmult(perturbed, direction);
      perturbed -= action;
      EXPECT_LT(perturbed.l2_norm(), 1.e-12);
      jacobian->view.Tvmult(transpose, dual);
      EXPECT_NEAR(action * dual, direction * transpose, 1.e-12);
      owned.add(1.e-6, direction);
      state     = owned;
      perturbed = 0.;
      model.evaluate_row(u.field_id(), context, perturbed);
      perturbed -= residual;
      perturbed *= 1.e6;
      perturbed -= action;
      EXPECT_LT(perturbed.l2_norm(), 1.e-8);
      owned.add(-1.e-6, direction);
      state = owned;
    };
    check_support(IntegrationSupport<2>(mesh));

    DynamicSparsityPattern dsp(dh.n_dofs(), dh.n_dofs(), relevant);
    DoFTools::make_sparsity_pattern(dh, dsp, constraints, true);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dh.locally_owned_dofs(),
                                               MPI_COMM_WORLD,
                                               relevant);
    MatrixType native;
    const auto native_sparsity =
      detail::initialize_weak_matrix(native,
                                     dh.locally_owned_dofs(),
                                     dh.locally_owned_dofs(),
                                     dsp,
                                     MPI_COMM_WORLD);
    MatrixCreator::create_mass_matrix(StaticMappingQ1<2>::mapping,
                                      dh,
                                      QGauss<2>(2),
                                      native,
                                      static_cast<const Function<2> *>(nullptr),
                                      constraints);
    native.vmult(perturbed, direction);
    perturbed -= action;
    EXPECT_LT(perturbed.l2_norm(), 1.e-12);

    Triangulation<1, 2> support_mesh;
    CellData<1>         cell;
    cell.vertices = {{0, 1}};
    support_mesh.create_triangulation(
      {Point<2>(0.1, 0.11), Point<2>(0.9, 0.89)}, {cell}, SubCellData());
    support_mesh.refine_global(2);
    check_support(IntegrationSupport<1, 2>(support_mesh));

    // Rank zero supplies the points; ParticleHandler must route them to
    // both background partitions before either assembly path uses them.
    PhysicalSupport physical_support;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      for (unsigned int q = 0; q < 8; ++q)
        {
          const double t = (q + 0.5) / 8.;
          physical_support.points.push_back(
            {Point<2>(0.1 + 0.8 * t, 0.11 + 0.78 * t),
             std::hypot(0.8, 0.78) / 8.,
             q,
             0});
        }
    check_support(physical_support);
  }

  template <unsigned int components, typename VectorType, typename MatrixType>
  void
  check_frozen_background_rejection()
  {
    parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD),
      other_mesh(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(mesh);
    GridGenerator::hyper_cube(other_mesh);
    mesh.refine_global(2);
    other_mesh.refine_global(2);
    FESystem<2>   fe(FE_Q<2>(1), components);
    DoFHandler<2> dh(mesh), other_dh(other_mesh);
    dh.distribute_dofs(fe);
    other_dh.distribute_dofs(fe);
    AffineConstraints<double> constraints;
    constraints.close();
    const auto relevant = DoFTools::extract_locally_relevant_dofs(dh);
    const auto other_relevant =
      DoFTools::extract_locally_relevant_dofs(other_dh);
    MappingQ1<2> other_mapping;
    const auto   source_space =
      fe_space(dh, StaticMappingQ1<2>::mapping, constraints, relevant);
    StateLayout layout;
    const auto  u = [&]() {
      if constexpr (components == 1)
        return source_space.field(layout, "u");
      else
        return source_space.field(layout, "u", FEValuesExtractors::Vector(0));
    }();
    VectorType state;
    if constexpr (detail::has_index_set_vector_reinit<VectorType>::value)
      state.reinit(dh.locally_owned_dofs(), relevant, MPI_COMM_WORLD);
    else
      state.reinit(dh.n_dofs());
    VectorType owned;
    detail::initialize_weak_vector(owned,
                                   dh.locally_owned_dofs(),
                                   MPI_COMM_WORLD);
    owned = 2.;
    state = owned;
    StateView<VectorType> state_view(layout, 0.);
    state_view.bind(u.field_id(), state);
    const EvaluationContext<VectorType> context(0., state_view);
    for (const bool different_mapping : {false, true})
      {
        SCOPED_TRACE(different_mapping ? "mapping" : "mesh");
        const auto input_space =
          different_mapping ?
            fe_space(dh, other_mapping, constraints, relevant) :
            fe_space(other_dh,
                     StaticMappingQ1<2>::mapping,
                     constraints,
                     other_relevant);
        const auto offset = [&]() {
          if constexpr (components == 1)
            return input_space.field("offset");
          else
            return input_space.field("offset", FEValuesExtractors::Vector(0));
        }();
        VectorType coefficients, frozen_owned;
        detail::initialize_weak_vector(frozen_owned,
                                       offset.locally_owned_dofs(),
                                       MPI_COMM_WORLD);
        frozen_owned = 1.;
        if constexpr (detail::has_index_set_vector_reinit<VectorType>::value)
          coefficients.reinit(offset.locally_owned_dofs(),
                              offset.locally_relevant_dofs(),
                              MPI_COMM_WORLD);
        else
          coefficients.reinit(offset.dof_handler().n_dofs());
        coefficients             = frozen_owned;
        const auto observable    = transform(value(u),
                                          frozen(offset, coefficients),
                                          SumWithFrozenInput{});
        const auto check_support = [&](const auto &support) {
          SemiDiscreteModel<VectorType, MatrixType>   model;
          SemidiscreteBuilder<VectorType, MatrixType> builder(layout, model);
          weak_term(observable, test(u), support).add(builder);
          VectorType residual;
          detail::initialize_weak_vector(residual,
                                         dh.locally_owned_dofs(),
                                         MPI_COMM_WORLD);
          residual = 0.;
          EXPECT_THROW(model.evaluate_row(u.field_id(), context, residual),
                       ExceptionBase);
          EXPECT_THROW(model.state_matrix_operator(u.field_id(),
                                                   u.field_id(),
                                                   context),
                       ExceptionBase);
        };
        check_support(IntegrationSupport<2>(mesh));
        Triangulation<1, 2> support_mesh;
        CellData<1>         cell;
        cell.vertices = {{0, 1}};
        support_mesh.create_triangulation(
          {Point<2>(0.1, 0.11), Point<2>(0.9, 0.89)}, {cell}, SubCellData());
        support_mesh.refine_global(4);
        check_support(IntegrationSupport<1, 2>(support_mesh));
      }
  }
} // namespace

TEST(WeakTerm, NonlinearSupportJacobiansSerialScalar)
{
  check_nonlinear_owned_storage<1, Vector<double>, SparseMatrix<double>>();
}

TEST(WeakTerm, NonlinearSupportJacobiansSerialVector)
{
  check_nonlinear_owned_storage<2, Vector<double>, SparseMatrix<double>>();
}

TEST(WeakTerm, BOTH_NonlinearSupportJacobiansAcrossPartitions)
{
  check_nonlinear_owned_storage<1,
                                ImmersXLA::MPI::Vector,
                                ImmersXLA::MPI::SparseMatrix>();
}

TEST(WeakTerm, BOTH_NonlinearVectorSupportJacobiansAcrossPartitions)
{
  check_nonlinear_owned_storage<2,
                                ImmersXLA::MPI::Vector,
                                ImmersXLA::MPI::SparseMatrix>();
}

TEST(WeakTerm, FrozenInputOnDifferentMeshOrMappingIsRejected)
{
  check_frozen_background_rejection<1, Vector<double>, SparseMatrix<double>>();
  check_frozen_background_rejection<2, Vector<double>, SparseMatrix<double>>();
}

TEST(WeakTerm, BOTH_FrozenInputOnDifferentMeshOrMappingIsRejected)
{
  check_frozen_background_rejection<1,
                                    ImmersXLA::MPI::Vector,
                                    ImmersXLA::MPI::SparseMatrix>();
  check_frozen_background_rejection<2,
                                    ImmersXLA::MPI::Vector,
                                    ImmersXLA::MPI::SparseMatrix>();
}
