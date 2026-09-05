// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_weak_term_h
#define immersx_weak_term_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <boost/signals2/connection.hpp>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/contributor.h>
#include <immersx/core/observable.h>
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  namespace detail
  {
#ifdef IMMERSX_WEAK_TERM_TESTING
    inline std::atomic<unsigned int> weak_term_nonmatching_preparations = 0;
#endif

    template <typename MatrixType, typename = void>
    struct has_distributed_matrix_reinit : std::false_type
    {};

    template <typename MatrixType>
    struct has_distributed_matrix_reinit<
      MatrixType,
      std::void_t<decltype(std::declval<MatrixType &>().reinit(
        std::declval<const dealii::IndexSet &>(),
        std::declval<const dealii::IndexSet &>(),
        std::declval<const dealii::DynamicSparsityPattern &>(),
        std::declval<MPI_Comm>(),
        false))>> : std::true_type
    {};

    template <typename MatrixType, typename = void>
    struct has_matrix_compress : std::false_type
    {};

    template <typename MatrixType>
    struct has_matrix_compress<
      MatrixType,
      std::void_t<decltype(std::declval<MatrixType &>().compress(
        dealii::VectorOperation::add))>> : std::true_type
    {};

    template <typename MatrixType>
    std::shared_ptr<dealii::SparsityPattern>
    initialize_weak_matrix(MatrixType             &matrix,
                           const dealii::IndexSet &row_partition,
                           const dealii::IndexSet &column_partition,
                           const dealii::DynamicSparsityPattern &sparsity,
                           const MPI_Comm                        communicator)
    {
      if constexpr (has_distributed_matrix_reinit<MatrixType>::value)
        {
          matrix.reinit(
            row_partition, column_partition, sparsity, communicator, false);
          return {};
        }
      else
        {
          auto serial_sparsity = std::make_shared<dealii::SparsityPattern>();
          serial_sparsity->copy_from(sparsity);
          matrix.reinit(*serial_sparsity);
          return serial_sparsity;
        }
    }

    template <typename MatrixType>
    void
    compress_weak_matrix(MatrixType &matrix)
    {
      if constexpr (has_matrix_compress<MatrixType>::value)
        matrix.compress(dealii::VectorOperation::add);
    }

    template <typename Type, typename = void>
    struct has_raw_tensor_access : std::false_type
    {};

    template <typename Type>
    struct has_raw_tensor_access<
      Type,
      std::void_t<decltype(std::declval<Type &>().access_raw_entry(0u))>>
      : std::true_type
    {};

    template <typename Type>
    struct is_tensor : std::false_type
    {};

    template <int rank, int dim, typename Number>
    struct is_tensor<dealii::Tensor<rank, dim, Number>> : std::true_type
    {};

    template <typename Type>
    inline constexpr bool is_supported_transport_value =
      std::is_arithmetic_v<std::decay_t<Type>> ||
      is_tensor<std::decay_t<Type>>::value ||
      has_raw_tensor_access<std::decay_t<Type>>::value;

    template <typename Type>
    inline constexpr bool dependent_false = false;

    template <typename Type>
    constexpr unsigned int
    n_stored_components()
    {
      using ValueType = std::decay_t<Type>;
      static_assert(is_supported_transport_value<ValueType>,
                    "Weak-term values must be arithmetic, Tensor, or "
                    "SymmetricTensor types.");
      if constexpr (std::is_arithmetic_v<ValueType>)
        return 1;
      else
        return ValueType::n_independent_components;
    }

    template <typename Type>
    constexpr bool is_scalar_value =
      std::is_arithmetic_v<Type> || std::is_convertible_v<Type, double>;

    template <typename Type>
    void
    flatten_value(const Type &value, std::vector<double> &result)
    {
      using ValueType = std::decay_t<Type>;
      if constexpr (is_scalar_value<ValueType>)
        result.push_back(static_cast<double>(value));
      else if constexpr (is_tensor<ValueType>::value)
        for (unsigned int i = 0; i < n_stored_components<ValueType>(); ++i)
          result.push_back(value[ValueType::unrolled_to_component_indices(i)]);
      else if constexpr (has_raw_tensor_access<ValueType>::value)
        for (unsigned int i = 0; i < n_stored_components<ValueType>(); ++i)
          result.push_back(value.access_raw_entry(i));
      else
        static_assert(dependent_false<ValueType>,
                      "Weak-term values must be arithmetic, Tensor, or "
                      "SymmetricTensor types.");
    }

    template <typename Type, typename Container>
    Type
    unflatten_value(const Container &values, unsigned int &offset)
    {
      Type result;
      using ValueType = std::decay_t<Type>;
      if constexpr (is_scalar_value<ValueType>)
        result = values[offset++];
      else if constexpr (is_tensor<ValueType>::value)
        for (unsigned int i = 0; i < n_stored_components<ValueType>(); ++i)
          result[ValueType::unrolled_to_component_indices(i)] =
            values[offset++];
      else if constexpr (has_raw_tensor_access<ValueType>::value)
        for (unsigned int i = 0; i < n_stored_components<ValueType>(); ++i)
          result.access_raw_entry(i) = values[offset++];
      else
        static_assert(dependent_false<ValueType>,
                      "Weak-term values must be arithmetic, Tensor, or "
                      "SymmetricTensor types.");
      return result;
    }

    /** \cond */
    template <typename Left, typename Right, typename = void>
    struct has_scalar_product : std::false_type
    {};

    template <typename Left, typename Right>
    struct has_scalar_product<Left,
                              Right,
                              std::void_t<decltype(dealii::scalar_product(
                                std::declval<const Left &>(),
                                std::declval<const Right &>()))>>
      : std::true_type
    {};

    template <typename Left, typename Right>
    decltype(auto)
    natural_pairing(const Left &left, const Right &right)
    {
      using LeftType  = std::decay_t<Left>;
      using RightType = std::decay_t<Right>;
      if constexpr (std::is_arithmetic_v<LeftType> &&
                    std::is_arithmetic_v<RightType>)
        return left * right;
      else
        {
          static_assert(
            has_scalar_product<LeftType, RightType>::value,
            "Weak-term expression values must be arithmetic or support "
            "dealii::scalar_product.");
          return dealii::scalar_product(left, right);
        }
    }
    /** \endcond */

    template <typename ObservableType, typename TargetExpression>
    struct WeakAssembly
    {
      static constexpr int dim =
        TargetExpression::source_field_type::dimension();
      static constexpr int spacedim =
        TargetExpression::source_field_type::spacedimension();

      using SourceField = typename ObservableType::source_field_type;

      template <typename MatrixType>
      struct MatrixStorage
      {
        std::shared_ptr<MatrixType>              matrix;
        std::shared_ptr<dealii::SparsityPattern> sparsity;
      };

      static_assert(!std::is_same_v<SourceField, void>,
                    "weak_term requires an observable produced from a Field.");

      template <typename VectorType, typename MatrixType>
      struct PreparedMatrix
      {
        using MatrixOperator =
          typename SemiDiscreteModel<VectorType, MatrixType>::MatrixOperator;

        MatrixStorage<MatrixType>                  storage;
        MatrixOperator                             operator_with_matrix;
        std::function<MatrixStorage<MatrixType>()> rebuild;
        std::function<bool()>                      structure_is_current;
        boost::signals2::scoped_connection         source_connection;
        boost::signals2::scoped_connection         target_connection;
        boost::signals2::scoped_connection         source_mesh_connection;
        boost::signals2::scoped_connection         target_mesh_connection;
        boost::signals2::scoped_connection         source_partition_connection;
        boost::signals2::scoped_connection         target_partition_connection;
        bool                                       valid = true;

        void
        invalidate()
        {
          valid = false;
        }

        void
        ensure()
        {
          if (valid && structure_is_current())
            return;

          storage = rebuild();
          operator_with_matrix =
            ImmersX::matrix_operator<VectorType, MatrixType>(*storage.matrix);
          valid = true;
        }
      };

      template <typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble(const ObservableType &observable, const TargetExpression &target)
      {
        const auto &target_field = target.source();
        AssertThrow(target_field.field_id().is_valid(),
                    dealii::ExcMessage(
                      "A weak term target field must be registered."));
        AssertThrow(observable.is_frozen() ||
                      observable.source_field().is_valid(),
                    dealii::ExcMessage(
                      "A weak term active source field must be registered."));
        AssertThrow(SourceField::spacedimension() == spacedim &&
                      SourceField::dimension() >= dim,
                    dealii::ExcMessage(
                      "Weak-term source and target must share an embedding "
                      "dimension, with the source support no lower than the "
                      "target support."));

        return assemble_from_source<SourceField, VectorType, MatrixType>(
          observable, observable.source(), target);
      }

      static bool
      geometry_is_nonmatching(const ObservableType   &observable,
                              const TargetExpression &target)
      {
        return static_cast<const void *>(
                 &observable.source().dof_handler().get_triangulation()) !=
               static_cast<const void *>(
                 &target.source().dof_handler().get_triangulation());
      }

      template <typename VectorType, typename MatrixType>
      static std::shared_ptr<PreparedMatrix<VectorType, MatrixType>>
      prepare(const ObservableType &observable, const TargetExpression &target)
      {
        return prepare_from_source<SourceField, VectorType, MatrixType>(
          observable, observable.source(), target);
      }

      template <typename VectorType, typename MatrixType>
      static typename SemiDiscreteModel<VectorType, MatrixType>::MatrixOperator
      dynamic_matrix_operator(
        const std::shared_ptr<PreparedMatrix<VectorType, MatrixType>> &prepared)
      {
        using MatrixOperator =
          typename SemiDiscreteModel<VectorType, MatrixType>::MatrixOperator;
        MatrixOperator result;
        result.view.vmult = [prepared](VectorType       &destination,
                                       const VectorType &source) {
          prepared->ensure();
          prepared->operator_with_matrix.view.vmult(destination, source);
        };
        result.view.vmult_add = [prepared](VectorType       &destination,
                                           const VectorType &source) {
          prepared->ensure();
          prepared->operator_with_matrix.view.vmult_add(destination, source);
        };
        result.view.Tvmult = [prepared](VectorType       &destination,
                                        const VectorType &source) {
          prepared->ensure();
          prepared->operator_with_matrix.view.Tvmult(destination, source);
        };
        result.view.Tvmult_add = [prepared](VectorType       &destination,
                                            const VectorType &source) {
          prepared->ensure();
          prepared->operator_with_matrix.view.Tvmult_add(destination, source);
        };
        result.view.reinit_range_vector =
          [prepared](VectorType &destination, const bool omit_zeroing_entries) {
            prepared->ensure();
            prepared->operator_with_matrix.view.reinit_range_vector(
              destination, omit_zeroing_entries);
          };
        result.view.reinit_domain_vector =
          [prepared](VectorType &destination, const bool omit_zeroing_entries) {
            prepared->ensure();
            prepared->operator_with_matrix.view.reinit_domain_vector(
              destination, omit_zeroing_entries);
          };
        result.materialize = [prepared] {
          prepared->ensure();
          return prepared->operator_with_matrix.matrix();
        };
        result.materialize_into = [prepared](MatrixType &destination) {
          prepared->ensure();
          prepared->operator_with_matrix.materialize_into_matrix(destination);
        };
        return result;
      }

    private:
      template <typename FieldType>
      static std::vector<dealii::types::global_dof_index>
      execution_indices(
        const FieldType                                    &field,
        const std::vector<dealii::types::global_dof_index> &native_indices)
      {
        std::vector<dealii::types::global_dof_index> result;
        result.reserve(native_indices.size());
        for (const auto native : native_indices)
          if (field.has_execution_index(native))
            result.push_back(field.execution_index(native));
        return result;
      }

      template <typename SourceField, typename VectorType, typename MatrixType>
      static std::shared_ptr<PreparedMatrix<VectorType, MatrixType>>
      prepare_from_source(const ObservableType   &observable,
                          const SourceField      &source,
                          const TargetExpression &target)
      {
        const auto target_field = target.source();
        const auto degree =
          std::max(source.space().finite_element().degree,
                   target_field.space().finite_element().degree);
        const dealii::QGauss<dim> quadrature(degree + 1);

        auto result =
          std::make_shared<PreparedMatrix<VectorType, MatrixType>>();
        result->rebuild = [observable, source, target, quadrature] {
          return assemble_nonmatching<SourceField, VectorType, MatrixType>(
            observable, source, target, quadrature);
        };
        result->structure_is_current =
          [source,
           target_field,
           source_tria   = &source.dof_handler().get_triangulation(),
           target_tria   = &target_field.dof_handler().get_triangulation(),
           source_n_dofs = source.dof_handler().n_dofs(),
           target_n_dofs = target_field.dof_handler().n_dofs(),
           source_fe     = &source.space().finite_element(),
           target_fe     = &target_field.space().finite_element()] {
            return source_tria == &source.dof_handler().get_triangulation() &&
                   target_tria ==
                     &target_field.dof_handler().get_triangulation() &&
                   source.dof_handler().n_dofs() == source_n_dofs &&
                   target_field.dof_handler().n_dofs() == target_n_dofs &&
                   source_fe == &source.space().finite_element() &&
                   target_fe == &target_field.space().finite_element();
          };
        result->storage = result->rebuild();
        result->operator_with_matrix =
          ImmersX::matrix_operator<VectorType, MatrixType>(
            *result->storage.matrix);

        const auto invalidate = [weak_result = std::weak_ptr(result)] {
          if (const auto prepared = weak_result.lock())
            prepared->invalidate();
        };
        result->source_connection =
          source.dof_handler().get_triangulation().signals.any_change.connect(
            invalidate);
        result->target_connection = target_field.dof_handler()
                                      .get_triangulation()
                                      .signals.any_change.connect(invalidate);
        result->source_mesh_connection =
          source.dof_handler()
            .get_triangulation()
            .signals.mesh_movement.connect(invalidate);
        result->target_mesh_connection =
          target_field.dof_handler()
            .get_triangulation()
            .signals.mesh_movement.connect(invalidate);
        result->source_partition_connection =
          source.dof_handler()
            .get_triangulation()
            .signals.pre_partition.connect(invalidate);
        result->target_partition_connection =
          target_field.dof_handler()
            .get_triangulation()
            .signals.pre_partition.connect(invalidate);
        return result;
      }

      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_from_source(const ObservableType   &observable,
                           const SourceField      &source,
                           const TargetExpression &target)
      {
        const auto target_field = target.source();
        if constexpr (SourceField::dimension() != dim)
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "A mixed-dimensional weak term must use the "
                          "nonmatching assembly backend."));
            return {};
          }
        else
          {
            const auto degree =
              std::max(source.space().finite_element().degree,
                       target_field.space().finite_element().degree);
            const dealii::QGauss<dim> quadrature(degree + 1);
            dealii::UpdateFlags       update_flags = dealii::update_JxW_values |
                                               observable.update_flags() |
                                               target.update_flags();

            if (&source.space().dof_handler().get_triangulation() !=
                &target_field.space().dof_handler().get_triangulation())
              return assemble_nonmatching<SourceField, VectorType, MatrixType>(
                observable, source, target, quadrature);

            AssertThrow(&source.mapping() == &target_field.mapping(),
                        dealii::ExcMessage(
                          "A weak term requires compatible source and target "
                          "mappings."));

            dealii::DynamicSparsityPattern sparsity(
              target_field.locally_owned_dofs().size(),
              source.locally_owned_dofs().size(),
              target_field.locally_owned_dofs());

            if (&source.dof_handler() == &target_field.dof_handler())
              {
                for (const auto &cell :
                     target_field.dof_handler().active_cell_iterators())
                  if (cell->is_locally_owned())
                    {
                      std::vector<dealii::types::global_dof_index> indices(
                        cell->get_fe().n_dofs_per_cell());
                      cell->get_dof_indices(indices);
                      const auto target_indices =
                        execution_indices(target_field, indices);
                      const auto source_indices =
                        execution_indices(source, indices);
                      target_field.constraints().add_entries_local_to_global(
                        target_indices,
                        source.constraints(),
                        source_indices,
                        sparsity,
                        false);
                    }
              }
            else
              for (const auto &source_cell :
                   source.dof_handler().active_cell_iterators())
                {
                  const auto target_cell = source_cell->as_dof_handler_iterator(
                    target_field.dof_handler());
                  if (!target_cell->is_locally_owned())
                    continue;
                  std::vector<dealii::types::global_dof_index> source_indices(
                    source_cell->get_fe().n_dofs_per_cell());
                  std::vector<dealii::types::global_dof_index> target_indices(
                    target_cell->get_fe().n_dofs_per_cell());
                  source_cell->get_dof_indices(source_indices);
                  target_cell->get_dof_indices(target_indices);
                  const auto execution_source_indices =
                    execution_indices(source, source_indices);
                  const auto execution_target_indices =
                    execution_indices(target_field, target_indices);
                  target_field.constraints().add_entries_local_to_global(
                    execution_target_indices,
                    source.constraints(),
                    execution_source_indices,
                    sparsity,
                    false);
                }

            auto matrix = std::make_shared<MatrixType>();
            auto matrix_sparsity =
              initialize_weak_matrix(*matrix,
                                     target_field.locally_owned_dofs(),
                                     source.locally_owned_dofs(),
                                     sparsity,
                                     target_field.space().mpi_communicator());

            if (&source.dof_handler() == &target_field.dof_handler())
              {
                dealii::FEValues<dim, spacedim> values(
                  target_field.mapping(),
                  target_field.space().finite_element(),
                  quadrature,
                  update_flags);
                for (const auto &cell :
                     target_field.dof_handler().active_cell_iterators())
                  {
                    if (!cell->is_locally_owned())
                      continue;
                    values.reinit(cell);
                    assemble_cell(observable,
                                  source,
                                  values,
                                  target,
                                  values,
                                  cell,
                                  cell,
                                  quadrature,
                                  *matrix);
                  }
              }
            else
              {
                dealii::FEValues<dim, spacedim> source_values(
                  source.mapping(),
                  source.space().finite_element(),
                  quadrature,
                  update_flags);
                dealii::FEValues<dim, spacedim> target_values(
                  target_field.mapping(),
                  target_field.space().finite_element(),
                  quadrature,
                  update_flags);
                for (const auto &source_cell :
                     source.dof_handler().active_cell_iterators())
                  {
                    const auto target_cell =
                      source_cell->as_dof_handler_iterator(
                        target_field.dof_handler());
                    if (!target_cell->is_locally_owned())
                      continue;
                    source_values.reinit(source_cell);
                    target_values.reinit(target_cell);
                    assemble_cell(observable,
                                  source,
                                  source_values,
                                  target,
                                  target_values,
                                  source_cell,
                                  target_cell,
                                  quadrature,
                                  *matrix);
                  }
              }
            compress_weak_matrix(*matrix);
            return {std::move(matrix), std::move(matrix_sparsity)};
          }
      }

      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_nonmatching(const ObservableType          &observable,
                           const SourceField             &source,
                           const TargetExpression        &target,
                           const dealii::Quadrature<dim> &quadrature)
      {
#ifdef IMMERSX_WEAK_TERM_TESTING
        ++weak_term_nonmatching_preparations;
#endif
        if constexpr (SourceField::dimension() == spacedim)
          return assemble_nonmatching_reverse<SourceField,
                                              VectorType,
                                              MatrixType>(observable,
                                                          source,
                                                          target,
                                                          quadrature);
        else
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "This weak-term geometry combination is not "
                          "implemented."));
            return {};
          }
      }

      /** Assemble a nonmatching pairing by transposing the natural source-row
       * matrix. Target quadrature points are searched in the full-dimensional
       * source mesh, so rows remain owned by the source background ranks and no
       * distributed matrix contribution exchange is needed before the explicit
       * transpose. */
      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_nonmatching_reverse(const ObservableType          &observable,
                                   const SourceField             &source,
                                   const TargetExpression        &target,
                                   const dealii::Quadrature<dim> &quadrature)
      {
        static_assert(SourceField::dimension() == spacedim,
                      "Nonmatching weak terms require a full-dimensional "
                      "source background.");

        const auto  target_field = target.source();
        const auto *source_tria =
          dynamic_cast<const dealii::parallel::TriangulationBase<spacedim> *>(
            &source.space().dof_handler().get_triangulation());
        AssertThrow(source_tria != nullptr,
                    dealii::ExcMessage(
                      "Mixed-dimensional weak terms require a distributed "
                      "source background triangulation."));

        using Point = detail::CouplingPoint<spacedim, double>;
        const unsigned int n_target_dofs = [&target_field] {
          for (const auto &cell :
               target_field.dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                std::vector<dealii::types::global_dof_index> indices(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(indices);
                unsigned int count = 0;
                for (const auto index : indices)
                  if (target_field.has_execution_index(index))
                    ++count;
                return count;
              }
          return 0u;
        }();
        AssertThrow(n_target_dofs > 0,
                    dealii::ExcMessage(
                      "A nonmatching weak term target has no execution DoFs."));
        const unsigned int target_components =
          n_stored_components<typename TargetExpression::value_type>();
        const unsigned int n_properties =
          1 + n_target_dofs + n_target_dofs * target_components;

        std::vector<Point>              target_points;
        dealii::FEValues<dim, spacedim> target_values(
          target_field.mapping(),
          target_field.space().finite_element(),
          quadrature,
          dealii::update_JxW_values | dealii::update_quadrature_points |
            target.update_flags());
        std::vector<dealii::types::global_dof_index> target_indices(
          n_target_dofs);
        for (const auto &cell :
             target_field.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              target_values.reinit(cell);
              cell->get_dof_indices(target_indices);
              for (const auto q : target_values.quadrature_point_indices())
                {
                  Point point;
                  point.point = target_values.quadrature_point(q);
                  point.representative_point = point.point;
                  point.weight               = target_values.JxW(q);
                  point.source_entity_id     = cell->global_active_cell_index();
                  point.representative_qpoint = q;
                  point.stable_id             = static_cast<std::uint64_t>(
                                      cell->global_active_cell_index()) *
                                      quadrature.size() +
                                    q;
                  point.basis_values.reserve(n_target_dofs * target_components);
                  for (unsigned int i = 0; i < target_indices.size(); ++i)
                    if (target_field.has_execution_index(target_indices[i]))
                      {
                        point.dof_indices.push_back(
                          target_field.execution_index(target_indices[i]));
                        flatten_value(target.operation()(
                                        target_values[target_field.extractor()],
                                        i,
                                        q),
                                      point.basis_values);
                      }
                  AssertDimension(point.dof_indices.size(), n_target_dofs);
                  target_points.emplace_back(std::move(point));
                }
            }

        ParticleCouplingParameters<spacedim> particle_parameters(
          "/ImmersX/weak term/reverse/" +
          std::to_string(reinterpret_cast<std::uintptr_t>(&source)));
        ParticleCoupling<spacedim> distribution(particle_parameters);
        distribution.initialize_particle_handler(*source_tria,
                                                 source.mapping(),
                                                 n_properties);
        std::vector<dealii::Point<spacedim>> points;
        std::vector<std::vector<double>>     properties;
        points.reserve(target_points.size());
        properties.reserve(target_points.size());
        for (const auto &point : target_points)
          {
            points.push_back(point.point);
            std::vector<double> point_properties;
            point_properties.reserve(n_properties);
            point_properties.push_back(point.weight);
            for (const auto dof : point.dof_indices)
              {
                const double encoded = static_cast<double>(dof);
                AssertThrow(
                  static_cast<dealii::types::global_dof_index>(encoded) == dof,
                  dealii::ExcMessage(
                    "A target DoF index cannot be transported exactly "
                    "through ParticleHandler properties."));
                point_properties.push_back(encoded);
              }
            point_properties.insert(point_properties.end(),
                                    point.basis_values.begin(),
                                    point.basis_values.end());
            properties.emplace_back(std::move(point_properties));
          }
        distribution.insert_points(points, properties);

        const unsigned int n_source_dofs =
          source.space().finite_element().n_dofs_per_cell();
        dealii::DynamicSparsityPattern sparsity(
          source.dof_handler().n_dofs(),
          target_field.locally_owned_dofs().size(),
          source.locally_relevant_dofs());
        std::vector<dealii::types::global_dof_index> source_indices(
          n_source_dofs);
        for (const auto &particle : distribution.get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename SourceField::space_type::DoFHandlerType::
                cell_iterator source_cell(*particle.get_surrounding_cell(),
                                          &source.dof_handler());
              source_cell->get_dof_indices(source_indices);
              const auto &particle_properties = particle.get_properties();
              for (unsigned int i = 0; i < n_target_dofs; ++i)
                {
                  const auto target_dof =
                    static_cast<dealii::types::global_dof_index>(
                      particle_properties[1 + i]);
                  for (unsigned int j = 0; j < n_source_dofs; ++j)
                    sparsity.add(source_indices[j], target_dof);
                }
            }
        dealii::SparsityTools::distribute_sparsity_pattern(
          sparsity,
          source.locally_owned_dofs(),
          source.space().mpi_communicator(),
          source.locally_relevant_dofs());

        auto matrix = std::make_shared<MatrixType>();
        auto matrix_sparsity =
          initialize_weak_matrix(*matrix,
                                 source.locally_owned_dofs(),
                                 target_field.locally_owned_dofs(),
                                 sparsity,
                                 source.space().mpi_communicator());

        const auto                 source_flags = observable.update_flags();
        const auto                 scale = observable.scale() * target.scale();
        dealii::FullMatrix<double> local;
        for (const auto &particle : distribution.get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename SourceField::space_type::DoFHandlerType::
                cell_iterator source_cell(*particle.get_surrounding_cell(),
                                          &source.dof_handler());
              source_cell->get_dof_indices(source_indices);
              const dealii::Quadrature<spacedim> source_point_quadrature(
                std::vector<dealii::Point<spacedim>>{
                  particle.get_reference_location()});
              dealii::FEValues<spacedim, spacedim> source_point_values(
                source.mapping(),
                source.space().finite_element(),
                source_point_quadrature,
                source_flags);
              source_point_values.reinit(source_cell);
              const auto &particle_properties = particle.get_properties();
              std::vector<dealii::types::global_dof_index> target_dofs(
                n_target_dofs);
              for (unsigned int i = 0; i < n_target_dofs; ++i)
                target_dofs[i] = static_cast<dealii::types::global_dof_index>(
                  particle_properties[1 + i]);
              local.reinit(n_source_dofs, n_target_dofs);
              for (unsigned int i = 0; i < n_source_dofs; ++i)
                for (unsigned int j = 0; j < n_target_dofs; ++j)
                  {
                    const auto source_value = observable.operation()(
                      source_point_values[source.extractor()], i, 0);
                    unsigned int target_offset =
                      1 + n_target_dofs + j * target_components;
                    const auto target_value =
                      unflatten_value<typename TargetExpression::value_type>(
                        particle_properties, target_offset);
                    local(i, j) += scale *
                                   natural_pairing(source_value, target_value) *
                                   particle_properties[0];
                  }
              source.constraints().distribute_local_to_global(
                local,
                source_indices,
                target_field.constraints(),
                target_dofs,
                *matrix);
            }
        matrix->compress(dealii::VectorOperation::add);
        return {ImmersX::detail::transpose_matrix(matrix), nullptr};
      }

      template <typename SourceField,
                typename SourceValues,
                typename TargetValues,
                typename SourceCell,
                typename TargetCell,
                typename MatrixType>
      static void
      assemble_cell(const ObservableType          &observable,
                    const SourceField             &source,
                    const SourceValues            &source_values,
                    const TargetExpression        &target,
                    const TargetValues            &target_values,
                    const SourceCell              &source_cell,
                    const TargetCell              &target_cell,
                    const dealii::Quadrature<dim> &quadrature,
                    MatrixType                    &matrix)
      {
        const auto &target_field = target.source();
        std::vector<dealii::types::global_dof_index> source_indices(
          source_cell->get_fe().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> target_indices(
          target_cell->get_fe().n_dofs_per_cell());
        source_cell->get_dof_indices(source_indices);
        target_cell->get_dof_indices(target_indices);

        std::vector<unsigned int>                    source_positions;
        std::vector<unsigned int>                    target_positions;
        std::vector<dealii::types::global_dof_index> source_execution_indices;
        std::vector<dealii::types::global_dof_index> target_execution_indices;
        for (unsigned int j = 0; j < source_indices.size(); ++j)
          if (source.has_execution_index(source_indices[j]))
            {
              source_positions.push_back(j);
              source_execution_indices.push_back(
                source.execution_index(source_indices[j]));
            }
        for (unsigned int i = 0; i < target_indices.size(); ++i)
          if (target_field.has_execution_index(target_indices[i]))
            {
              target_positions.push_back(i);
              target_execution_indices.push_back(
                target_field.execution_index(target_indices[i]));
            }

        const auto &trial_view = source_values[source.extractor()];
        const auto &test_view  = target_values[target_field.extractor()];
        dealii::FullMatrix<double> local(target_positions.size(),
                                         source_positions.size());
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < target_positions.size(); ++i)
            for (unsigned int j = 0; j < source_positions.size(); ++j)
              local(i, j) +=
                observable.scale() * target.scale() *
                detail::natural_pairing(
                  observable.operation()(trial_view, source_positions[j], q),
                  target.operation()(test_view, target_positions[i], q)) *
                target_values.JxW(q);

        target_field.constraints().distribute_local_to_global(
          local,
          target_execution_indices,
          source.constraints(),
          source_execution_indices,
          matrix);
      }
    };

    /** Pointwise assembly for a state-dependent Observable. */
    template <typename ObservableType, typename TargetExpression>
    struct NonlinearWeakAssembly
    {
      static constexpr int dim =
        TargetExpression::source_field_type::dimension();
      static constexpr int spacedim =
        TargetExpression::source_field_type::spacedimension();

      using SourceField = typename ObservableType::source_field_type;

      template <typename MatrixType>
      struct MatrixStorage
      {
        std::shared_ptr<MatrixType>              matrix;
        std::shared_ptr<dealii::SparsityPattern> sparsity;
      };

      template <typename VectorType, typename FieldType, typename = void>
      struct has_distributed_reinit : std::false_type
      {};

      template <typename VectorType, typename FieldType>
      struct has_distributed_reinit<
        VectorType,
        FieldType,
        std::void_t<decltype(std::declval<VectorType &>().reinit(
          std::declval<const dealii::IndexSet &>(),
          std::declval<const dealii::IndexSet &>(),
          std::declval<MPI_Comm>()))>> : std::true_type
      {};

      template <typename VectorType, typename FieldType>
      static void
      reinit_result(VectorType &vector, const FieldType &field)
      {
        if constexpr (has_distributed_reinit<VectorType, FieldType>::value)
          vector.reinit(field.locally_owned_dofs(),
                        field.locally_relevant_dofs(),
                        field.space().mpi_communicator());
        else
          vector.reinit(field.dof_handler().n_dofs());
      }

      template <typename VectorType,
                typename ObservableType_,
                typename Context,
                typename Cell,
                typename SourceValues>
      static typename std::decay_t<ObservableType_>::value_type
      primitive_value(const ObservableType_ &observable,
                      const Context         &context,
                      const Cell            &source_cell,
                      const SourceValues    &source_values,
                      const unsigned int     q)
      {
        using Observable = std::decay_t<decltype(observable)>;
        using Value      = typename Observable::value_type;

        const auto input_cell = source_cell->as_dof_handler_iterator(
          observable.source().dof_handler());
        std::vector<dealii::types::global_dof_index> indices(
          input_cell->get_fe().n_dofs_per_cell());
        input_cell->get_dof_indices(indices);
        const auto &coefficients =
          observable.is_frozen() ?
            observable.template frozen_values<VectorType>() :
            context.state(observable.source_field());
        std::vector<dealii::types::global_dof_index> execution_indices;
        execution_indices.reserve(indices.size());
        std::vector<unsigned int> positions;
        positions.reserve(indices.size());
        for (unsigned int i = 0; i < indices.size(); ++i)
          if (observable.source().has_execution_index(indices[i]))
            {
              positions.push_back(i);
              execution_indices.push_back(
                observable.source().execution_index(indices[i]));
            }
        std::vector<double> execution_values(execution_indices.size());
        observable.source().constraints().get_dof_values(
          coefficients,
          execution_indices.begin(),
          execution_values.begin(),
          execution_values.end());

        const auto &view   = source_values[observable.source().extractor()];
        Value       result = Value{};
        for (unsigned int i = 0; i < positions.size(); ++i)
          result +=
            execution_values[i] * observable.operation()(view, positions[i], q);
        return observable.scale() * result;
      }

      template <typename ObservableType_, typename Cell, typename SourceValues>
      static typename std::decay_t<ObservableType_>::value_type
      primitive_derivative(const ObservableType_ &observable,
                           const FieldId          field,
                           const Cell            &source_cell,
                           const SourceValues    &source_values,
                           const unsigned int     q,
                           const unsigned int     basis)
      {
        using Observable = std::decay_t<decltype(observable)>;
        using Value      = typename Observable::value_type;
        if (observable.is_frozen() || observable.source_field() != field)
          return Value{};
        const auto input_cell = source_cell->as_dof_handler_iterator(
          observable.source().dof_handler());
        AssertIndexRange(basis, input_cell->get_fe().n_dofs_per_cell());
        const auto &view = source_values[observable.source().extractor()];
        return observable.scale() * observable.operation()(view, basis, q);
      }

      template <typename VectorType,
                typename Context,
                typename Cell,
                typename SourceValues>
      static typename std::decay_t<ObservableType>::value_type
      value_at(const ObservableType &observable,
               const Context        &context,
               const Cell           &source_cell,
               const SourceValues   &source_values,
               const unsigned int    q,
               const double          time)
      {
        const auto evaluator =
          [&](const auto &input, const auto &, const double) {
            return primitive_value<VectorType>(
              input, context, source_cell, source_values, q);
          };
        return evaluate_observable_input(observable,
                                         source_values.quadrature_point(q),
                                         time,
                                         evaluator);
      }

      template <typename VectorType,
                typename Context,
                typename Cell,
                typename SourceValues>
      static typename std::decay_t<ObservableType>::value_type
      derivative_at(const ObservableType &observable,
                    const Context        &context,
                    const FieldId         field,
                    const Cell           &source_cell,
                    const SourceValues   &source_values,
                    const unsigned int    q,
                    const unsigned int    basis,
                    const double          time)
      {
        const auto evaluator =
          [&](const auto &input, const auto &, const double) {
            return primitive_value<VectorType>(
              input, context, source_cell, source_values, q);
          };
        const auto derivative_evaluator = [&](const auto   &input,
                                              const FieldId input_field,
                                              const auto &,
                                              const double) {
          return primitive_derivative(
            input, input_field, source_cell, source_values, q, basis);
        };
        return linearize_observable_input(observable,
                                          field,
                                          source_values.quadrature_point(q),
                                          time,
                                          evaluator,
                                          derivative_evaluator);
      }

      template <typename VectorType, typename MatrixType, typename Context>
      static std::shared_ptr<VectorType>
      residual(const ObservableType   &observable,
               const TargetExpression &target,
               const Context          &context)
      {
        const auto &source       = observable.source();
        const auto  target_field = target.source();
        AssertThrow(SourceField::dimension() == dim &&
                      &source.dof_handler().get_triangulation() ==
                        &target_field.dof_handler().get_triangulation(),
                    dealii::ExcMessage(
                      "Nonlinear weak terms currently require matching "
                      "source and target meshes."));

        const auto degree =
          std::max(source.space().finite_element().degree,
                   target_field.space().finite_element().degree);
        const dealii::QGauss<dim> quadrature(degree + 1);
        const auto                flags = dealii::update_JxW_values |
                           observable.update_flags() | target.update_flags();
        dealii::FEValues<dim, spacedim> source_values(
          source.mapping(), source.space().finite_element(), quadrature, flags);
        dealii::FEValues<dim, spacedim> target_values(
          target_field.mapping(),
          target_field.space().finite_element(),
          quadrature,
          flags);

        auto result = std::make_shared<VectorType>();
        reinit_result(*result, target_field);
        *result = 0.;
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned())
            {
              const auto target_cell = source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
              if (!target_cell->is_locally_owned())
                continue;
              source_values.reinit(source_cell);
              target_values.reinit(target_cell);
              target_cell->get_dof_indices(target_indices);
              const auto &target_view = target_values[target_field.extractor()];
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                {
                  const auto value = value_at<VectorType>(observable,
                                                          context,
                                                          source_cell,
                                                          source_values,
                                                          q,
                                                          context.time());
                  for (unsigned int i = 0; i < target_indices.size(); ++i)
                    if (target_field.has_execution_index(target_indices[i]))
                      (*result)(
                        target_field.execution_index(target_indices[i])) +=
                        target.scale() *
                        natural_pairing(value,
                                        target.operation()(target_view, i, q)) *
                        target_values.JxW(q);
                }
            }
        result->compress(dealii::VectorOperation::add);
        return result;
      }

      template <typename VectorType, typename MatrixType, typename Context>
      static MatrixStorage<MatrixType>
      jacobian(const ObservableType   &observable,
               const TargetExpression &target,
               const FieldId           field,
               const Context          &context)
      {
        const auto &source        = observable.source();
        const auto &column_source = observable.source_for(field);
        const auto  target_field  = target.source();
        AssertThrow(SourceField::dimension() == dim &&
                      &source.dof_handler().get_triangulation() ==
                        &target_field.dof_handler().get_triangulation() &&
                      &column_source.dof_handler().get_triangulation() ==
                        &source.dof_handler().get_triangulation(),
                    dealii::ExcMessage(
                      "Nonlinear weak terms currently require matching "
                      "source meshes."));

        const auto degree =
          std::max(source.space().finite_element().degree,
                   target_field.space().finite_element().degree);
        const dealii::QGauss<dim> quadrature(degree + 1);
        const auto                flags = dealii::update_JxW_values |
                           observable.update_flags() | target.update_flags();

        dealii::DynamicSparsityPattern sparsity(
          target_field.locally_owned_dofs().size(),
          column_source.locally_owned_dofs().size(),
          target_field.locally_owned_dofs());
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> column_indices(
          column_source.space().finite_element().n_dofs_per_cell());
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned())
            {
              const auto target_cell = source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
              if (!target_cell->is_locally_owned())
                continue;
              const auto column_cell = source_cell->as_dof_handler_iterator(
                column_source.dof_handler());
              target_cell->get_dof_indices(target_indices);
              column_cell->get_dof_indices(column_indices);
              std::vector<dealii::types::global_dof_index>
                target_execution_indices;
              std::vector<dealii::types::global_dof_index>
                column_execution_indices;
              for (const auto index : target_indices)
                if (target_field.has_execution_index(index))
                  target_execution_indices.push_back(
                    target_field.execution_index(index));
              for (const auto index : column_indices)
                if (column_source.has_execution_index(index))
                  column_execution_indices.push_back(
                    column_source.execution_index(index));
              target_field.constraints().add_entries_local_to_global(
                target_execution_indices,
                column_source.constraints(),
                column_execution_indices,
                sparsity,
                false);
            }

        MatrixStorage<MatrixType> result;
        result.matrix = std::make_shared<MatrixType>();
        result.sparsity =
          initialize_weak_matrix(*result.matrix,
                                 target_field.locally_owned_dofs(),
                                 column_source.locally_owned_dofs(),
                                 sparsity,
                                 target_field.space().mpi_communicator());

        dealii::FEValues<dim, spacedim> source_values(
          source.mapping(), source.space().finite_element(), quadrature, flags);
        dealii::FEValues<dim, spacedim> target_values(
          target_field.mapping(),
          target_field.space().finite_element(),
          quadrature,
          flags);
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned())
            {
              const auto target_cell = source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
              if (!target_cell->is_locally_owned())
                continue;
              const auto column_cell = source_cell->as_dof_handler_iterator(
                column_source.dof_handler());
              source_values.reinit(source_cell);
              target_values.reinit(target_cell);
              target_cell->get_dof_indices(target_indices);
              column_cell->get_dof_indices(column_indices);
              const auto &target_view = target_values[target_field.extractor()];
              std::vector<unsigned int> target_positions;
              std::vector<unsigned int> column_positions;
              std::vector<dealii::types::global_dof_index>
                target_execution_indices;
              std::vector<dealii::types::global_dof_index>
                column_execution_indices;
              for (unsigned int i = 0; i < target_indices.size(); ++i)
                if (target_field.has_execution_index(target_indices[i]))
                  {
                    target_positions.push_back(i);
                    target_execution_indices.push_back(
                      target_field.execution_index(target_indices[i]));
                  }
              for (unsigned int j = 0; j < column_indices.size(); ++j)
                if (column_source.has_execution_index(column_indices[j]))
                  {
                    column_positions.push_back(j);
                    column_execution_indices.push_back(
                      column_source.execution_index(column_indices[j]));
                  }
              dealii::FullMatrix<double> local(target_positions.size(),
                                               column_positions.size());
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                for (unsigned int i = 0; i < target_positions.size(); ++i)
                  for (unsigned int j = 0; j < column_positions.size(); ++j)
                    local(i, j) +=
                      target.scale() *
                      natural_pairing(
                        derivative_at<VectorType>(observable,
                                                  context,
                                                  field,
                                                  source_cell,
                                                  source_values,
                                                  q,
                                                  column_positions[j],
                                                  context.time()),
                        target.operation()(target_view,
                                           target_positions[i],
                                           q)) *
                      target_values.JxW(q);
              target_field.constraints().distribute_local_to_global(
                local,
                target_execution_indices,
                column_source.constraints(),
                column_execution_indices,
                *result.matrix);
            }
        result.matrix->compress(dealii::VectorOperation::add);
        return result;
      }

      template <typename VectorType, typename MatrixType, typename Context>
      static dealii::LinearOperator<VectorType, VectorType>
      linearize(const ObservableType   &observable,
                const TargetExpression &target,
                const FieldId           field,
                const Context          &context)
      {
        const auto storage =
          jacobian<VectorType, MatrixType>(observable, target, field, context);
        const auto matrix   = storage.matrix;
        const auto sparsity = storage.sparsity;
        const auto matrix_view =
          ImmersX::matrix_operator<VectorType, MatrixType>(*matrix).view;
        const auto range  = context.state(target.source().field_id());
        const auto domain = context.state(field);

        dealii::LinearOperator<VectorType, VectorType> result;
        result.reinit_range_vector = [range](VectorType &vector,
                                             const bool  omit) {
          vector.reinit(range, omit);
        };
        result.reinit_domain_vector = [domain](VectorType &vector,
                                               const bool  omit) {
          vector.reinit(domain, omit);
        };
        result.vmult =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.vmult(destination, source);
          };
        result.vmult_add =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.vmult_add(destination, source);
          };
        result.Tvmult =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.Tvmult(destination, source);
          };
        result.Tvmult_add =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.Tvmult_add(destination, source);
          };
        return result;
      }
    };

    /** Assemble a lifted Observable against its representative FE field. */
    template <typename ObservableType, typename TargetExpression>
    struct LiftedWeakAssembly
    {
      using TargetField = typename TargetExpression::source_field_type;
      using Point       = typename ObservableType::Point;
      template <typename MatrixType>
      using MatrixStorage =
        typename NonlinearWeakAssembly<ObservableType, TargetExpression>::
          template MatrixStorage<MatrixType>;

      static_assert(std::is_arithmetic_v<typename TargetExpression::value_type>,
                    "A lifted weak term currently needs a scalar target.");

      template <typename Type>
      static double
      radial_value(const Type &value, const Point &point)
      {
        if constexpr (std::is_arithmetic_v<std::decay_t<Type>>)
          return static_cast<double>(value);
        else
          return point.mode_vector * value;
      }

      template <typename VectorType>
      static void
      reinit_result(VectorType &vector, const TargetField &field)
      {
        if constexpr (NonlinearWeakAssembly<ObservableType, TargetExpression>::
                        template has_distributed_reinit<VectorType,
                                                        TargetField>::value)
          vector.reinit(field.locally_owned_dofs(),
                        field.locally_relevant_dofs(),
                        field.space().mpi_communicator());
        else
          vector.reinit(field.locally_owned_dofs().size());
      }

      template <typename Cell>
      static std::vector<double>
      target_basis(const TargetExpression &target,
                   const Cell             &cell,
                   const Point            &point)
      {
        const auto &target_field = target.source();
        const auto  reference =
          target_field.mapping().transform_real_to_unit_cell(
            cell, point.representative_point);
        const dealii::Quadrature<TargetField::dimension()> quadrature(
          std::vector<dealii::Point<TargetField::dimension()>>{reference});
        dealii::FEValues<TargetField::dimension(),
                         TargetField::spacedimension()>
          values(target_field.mapping(),
                 target_field.space().finite_element(),
                 quadrature,
                 target.update_flags());
        values.reinit(cell);
        const auto         &view = values[target.source().extractor()];
        std::vector<double> result(cell->get_fe().n_dofs_per_cell());
        for (unsigned int i = 0; i < result.size(); ++i)
          result[i] = target.operation()(view, i, 0);
        return result;
      }

      template <typename VectorType, typename MatrixType, typename Context>
      static std::shared_ptr<VectorType>
      residual(const ObservableType   &observable,
               const TargetExpression &target,
               const Context          &context)
      {
        const auto &source       = observable.source();
        const auto  target_field = target.source();
        AssertThrow(
          &source.dof_handler().get_triangulation() ==
            &target_field.dof_handler().get_triangulation(),
          dealii::ExcMessage(
            "A lifted weak term requires a representative target mesh."));

        auto result = std::make_shared<VectorType>();
        reinit_result(*result, target_field);
        *result = 0.;

        const auto points = observable.lifted_points();
        for (const auto &cell :
             target_field.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              std::vector<dealii::types::global_dof_index> indices(
                cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(indices);
              for (std::size_t q = 0; q < points.size(); ++q)
                if (points[q].source_entity_id ==
                    cell->global_active_cell_index())
                  {
                    const auto   basis = target_basis(target, cell, points[q]);
                    const double value =
                      radial_value(observable.evaluate_point(context, q),
                                   points[q]);
                    for (unsigned int i = 0; i < indices.size(); ++i)
                      {
                        const auto row =
                          target_field.execution_index(indices[i]);
                        if (row != std::numeric_limits<
                                     dealii::types::global_dof_index>::max())
                          (*result)[row] += target.scale() * value * basis[i] *
                                            points[q].weight;
                      }
                  }
            }
        result->compress(dealii::VectorOperation::add);
        return result;
      }

      template <typename VectorType, typename MatrixType, typename Context>
      static MatrixStorage<MatrixType>
      jacobian(const ObservableType   &observable,
               const TargetExpression &target,
               const FieldId           field,
               const Context          &context)
      {
        const auto &source        = observable.source();
        const auto &column_source = observable.source_for(field);
        const auto  target_field  = target.source();
        AssertThrow(
          &source.dof_handler().get_triangulation() ==
              &target_field.dof_handler().get_triangulation() &&
            &column_source.dof_handler().get_triangulation() ==
              &source.dof_handler().get_triangulation(),
          dealii::ExcMessage(
            "A lifted weak-term derivative requires matching representative "
            "meshes."));

        dealii::DynamicSparsityPattern sparsity(
          target_field.locally_owned_dofs().size(),
          column_source.locally_owned_dofs().size(),
          target_field.locally_owned_dofs());
        const auto         points = observable.lifted_points();
        const unsigned int n_columns =
          column_source.space().finite_element().n_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> column_indices(n_columns);
        for (const auto &cell :
             target_field.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(target_indices);
              const auto source_cell =
                cell->as_dof_handler_iterator(source.dof_handler());
              source_cell->get_dof_indices(column_indices);
              std::vector<dealii::types::global_dof_index> rows;
              rows.reserve(target_indices.size());
              for (const auto index : target_indices)
                {
                  const auto row = target_field.execution_index(index);
                  if (row != std::numeric_limits<
                               dealii::types::global_dof_index>::max())
                    rows.push_back(row);
                }
              target_field.constraints().add_entries_local_to_global(
                rows,
                column_source.constraints(),
                column_indices,
                sparsity,
                false);
            }

        MatrixStorage<MatrixType> result;
        result.matrix = std::make_shared<MatrixType>();
        result.sparsity =
          initialize_weak_matrix(*result.matrix,
                                 target_field.locally_owned_dofs(),
                                 column_source.locally_owned_dofs(),
                                 sparsity,
                                 target_field.space().mpi_communicator());

        for (const auto &cell :
             target_field.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(target_indices);
              const auto source_cell =
                cell->as_dof_handler_iterator(source.dof_handler());
              source_cell->get_dof_indices(column_indices);
              std::vector<unsigned int>                    active_rows;
              std::vector<dealii::types::global_dof_index> rows;
              for (unsigned int i = 0; i < target_indices.size(); ++i)
                if (target_field.execution_index(target_indices[i]) !=
                    std::numeric_limits<dealii::types::global_dof_index>::max())
                  {
                    active_rows.push_back(i);
                    rows.push_back(
                      target_field.execution_index(target_indices[i]));
                  }
              dealii::FullMatrix<double> local(active_rows.size(),
                                               column_indices.size());
              for (std::size_t q = 0; q < points.size(); ++q)
                if (points[q].source_entity_id ==
                    cell->global_active_cell_index())
                  {
                    const auto basis = target_basis(target, cell, points[q]);
                    for (unsigned int row = 0; row < active_rows.size(); ++row)
                      for (unsigned int j = 0; j < column_indices.size(); ++j)
                        local(row, j) +=
                          target.scale() *
                          radial_value(
                            observable.linearize_point(context, field, q, j),
                            points[q]) *
                          basis[active_rows[row]] * points[q].weight;
                  }
              target_field.constraints().distribute_local_to_global(
                local,
                rows,
                column_source.constraints(),
                column_indices,
                *result.matrix);
            }
        compress_weak_matrix(*result.matrix);
        return result;
      }

      template <typename VectorType, typename MatrixType, typename Context>
      static dealii::LinearOperator<VectorType, VectorType>
      linearize(const ObservableType   &observable,
                const TargetExpression &target,
                const FieldId           field,
                const Context          &context)
      {
        const auto storage =
          jacobian<VectorType, MatrixType>(observable, target, field, context);
        const auto matrix_view =
          ImmersX::matrix_operator<VectorType, MatrixType>(*storage.matrix)
            .view;
        const auto range  = context.state(target.source().field_id());
        const auto domain = context.state(field);
        dealii::LinearOperator<VectorType, VectorType> result;
        result.reinit_range_vector = [range](VectorType &vector,
                                             const bool  omit) {
          vector.reinit(range, omit);
        };
        result.reinit_domain_vector = [domain](VectorType &vector,
                                               const bool  omit) {
          vector.reinit(domain, omit);
        };
        result.vmult = [matrix_view](VectorType       &destination,
                                     const VectorType &source) {
          matrix_view.vmult(destination, source);
        };
        result.vmult_add = [matrix_view](VectorType       &destination,
                                         const VectorType &source) {
          matrix_view.vmult_add(destination, source);
        };
        result.Tvmult = [matrix_view](VectorType       &destination,
                                      const VectorType &source) {
          matrix_view.Tvmult(destination, source);
        };
        result.Tvmult_add = [matrix_view](VectorType       &destination,
                                          const VectorType &source) {
          matrix_view.Tvmult_add(destination, source);
        };
        return result;
      }
    };

    /** Assemble a full-dimensional FE quantity against a lifted test. */
    template <typename ObservableType, typename TargetLift>
    struct LiftedTargetWeakAssembly
    {
      using TargetField = typename TargetLift::source_field_type;
      using Point       = typename TargetLift::Point;

      template <typename MatrixType>
      struct MatrixStorage
      {
        std::shared_ptr<MatrixType>              matrix;
        std::shared_ptr<dealii::SparsityPattern> sparsity;
      };

      template <typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble(const ObservableType &observable, const TargetLift &target)
      {
        using SourceField = typename ObservableType::source_field_type;
        static_assert(SourceField::dimension() == SourceField::spacedimension(),
                      "A lifted test needs a full-dimensional trial field.");
        const auto &source       = observable.source();
        const auto &target_field = target.source();
        const auto *source_tria =
          dynamic_cast<const dealii::parallel::TriangulationBase<
            SourceField::spacedimension()> *>(
            &source.dof_handler().get_triangulation());
        AssertThrow(source_tria != nullptr,
                    dealii::ExcMessage(
                      "A lifted weak term needs a distributed source mesh."));

        const auto points = target.lifted_points();
        AssertThrow(!points.empty(),
                    dealii::ExcMessage(
                      "A lifted test cannot assemble without points."));
        const unsigned int n_target_dofs = points.front().dof_indices.size();
        for (const auto &point : points)
          AssertDimension(point.dof_indices.size(), n_target_dofs);
        const unsigned int n_properties =
          1 + 2 * n_target_dofs + SourceField::spacedimension();

        ParticleCouplingParameters<SourceField::spacedimension()>
          particle_parameters(
            "/ImmersX/weak term/lifted-target/" +
            std::to_string(reinterpret_cast<std::uintptr_t>(&source)));
        ParticleCoupling<SourceField::spacedimension()> distribution(
          particle_parameters);
        distribution.initialize_particle_handler(*source_tria,
                                                 source.mapping(),
                                                 n_properties);

        std::vector<dealii::Point<SourceField::spacedimension()>>
                                         physical_points;
        std::vector<std::vector<double>> properties;
        physical_points.reserve(points.size());
        properties.reserve(points.size());
        for (const auto &point : points)
          {
            physical_points.push_back(point.point);
            std::vector<double> data;
            data.reserve(n_properties);
            data.push_back(point.weight);
            for (const auto native : point.dof_indices)
              {
                const auto execution = target_field.execution_index(native);
                AssertThrow(
                  execution !=
                    std::numeric_limits<dealii::types::global_dof_index>::max(),
                  dealii::ExcMessage(
                    "A lifted test contains an unmapped target DoF."));
                data.push_back(static_cast<double>(execution));
              }
            data.insert(data.end(),
                        point.source_basis_values.begin(),
                        point.source_basis_values.end());
            for (unsigned int component = 0;
                 component < SourceField::spacedimension();
                 ++component)
              data.push_back(point.mode_vector[component]);
            properties.emplace_back(std::move(data));
          }
        distribution.insert_points(physical_points, properties);

        const unsigned int n_source_dofs =
          source.space().finite_element().n_dofs_per_cell();
        dealii::DynamicSparsityPattern sparsity(
          source.dof_handler().n_dofs(),
          target_field.locally_owned_dofs().size(),
          source.locally_relevant_dofs());
        std::vector<dealii::types::global_dof_index> source_indices(
          n_source_dofs);
        for (const auto &particle : distribution.get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename SourceField::space_type::DoFHandlerType::
                cell_iterator source_cell(*particle.get_surrounding_cell(),
                                          &source.dof_handler());
              source_cell->get_dof_indices(source_indices);
              const auto &data = particle.get_properties();
              for (unsigned int i = 0; i < n_target_dofs; ++i)
                for (unsigned int j = 0; j < n_source_dofs; ++j)
                  sparsity.add(source_indices[j],
                               static_cast<dealii::types::global_dof_index>(
                                 data[1 + i]));
            }
        dealii::SparsityTools::distribute_sparsity_pattern(
          sparsity,
          source.locally_owned_dofs(),
          source.space().mpi_communicator(),
          source.locally_relevant_dofs());

        MatrixStorage<MatrixType> result;
        result.matrix = std::make_shared<MatrixType>();
        result.sparsity =
          initialize_weak_matrix(*result.matrix,
                                 source.locally_owned_dofs(),
                                 target_field.locally_owned_dofs(),
                                 sparsity,
                                 source.space().mpi_communicator());

        std::vector<dealii::types::global_dof_index> target_indices(
          n_target_dofs);
        for (const auto &particle : distribution.get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename SourceField::space_type::DoFHandlerType::
                cell_iterator source_cell(*particle.get_surrounding_cell(),
                                          &source.dof_handler());
              source_cell->get_dof_indices(source_indices);
              const dealii::Quadrature<SourceField::spacedimension()>
                quadrature(
                  std::vector<dealii::Point<SourceField::spacedimension()>>{
                    particle.get_reference_location()});
              dealii::FEValues<SourceField::spacedimension(),
                               SourceField::spacedimension()>
                source_values(source.mapping(),
                              source.space().finite_element(),
                              quadrature,
                              observable.update_flags());
              source_values.reinit(source_cell);
              const auto &data = particle.get_properties();
              dealii::Tensor<1, SourceField::spacedimension()> normal;
              for (unsigned int component = 0;
                   component < SourceField::spacedimension();
                   ++component)
                normal[component] = data[1 + 2 * n_target_dofs + component];
              const auto &view = source_values[source.extractor()];
              dealii::FullMatrix<double> local(n_source_dofs, n_target_dofs);
              for (unsigned int i = 0; i < n_source_dofs; ++i)
                for (unsigned int j = 0; j < n_target_dofs; ++j)
                  local(i, j) = data[0] * data[1 + n_target_dofs + j] *
                                (observable.operation()(view, i, 0) * normal);
              for (unsigned int j = 0; j < n_target_dofs; ++j)
                target_indices[j] =
                  static_cast<dealii::types::global_dof_index>(data[1 + j]);
              source.constraints().distribute_local_to_global(
                local,
                source_indices,
                target_field.constraints(),
                target_indices,
                *result.matrix);
            }
        compress_weak_matrix(*result.matrix);
        result.matrix = ImmersX::detail::transpose_matrix(result.matrix);
        return result;
      }
    };

  } // namespace detail

  /** A geometry value that supplies normals for a point-supported quantity. */
  template <typename Surface>
  class NormalGeometry
  {
  public:
    explicit NormalGeometry(const Surface &surface)
      : surface_(&surface)
    {}

    const Surface &
    surface() const
    {
      return *surface_;
    }

    const auto &
    particle_coupling_parameters() const
    {
      return surface_->particle_coupling_parameters();
    }

    template <typename Point>
    auto
    normal(const Point &point) const
    {
      return surface_->normal(point);
    }

  private:
    const Surface *surface_;
  };

  /** Attach a normal geometry observable to a point-supported quantity. */
  template <typename Surface>
  NormalGeometry<Surface>
  normal(const Surface &surface)
  {
    return NormalGeometry<Surface>(surface);
  }

  template <typename Quantity, typename Surface>
  class NormalQuantity
  {
  public:
    static constexpr unsigned int support_dimension =
      Quantity::support_dimension;
    static constexpr unsigned int ambient_dimension =
      Quantity::ambient_dimension;
    using value_type = typename Quantity::value_type;

    NormalQuantity(Quantity quantity, NormalGeometry<Surface> geometry)
      : quantity_(std::move(quantity))
      , geometry_(std::move(geometry))
    {}

    const Quantity &
    quantity() const
    {
      return quantity_;
    }

    const NormalGeometry<Surface> &
    geometry() const
    {
      return geometry_;
    }

    const std::vector<FieldId> &
    dependencies() const
    {
      return quantity_.dependencies();
    }

    template <typename Context>
    auto
    evaluate(const Context &context) const
    {
      return quantity_.evaluate(context);
    }

    template <typename Context>
    auto
    linearize(const Context &context, const FieldId field) const
    {
      return quantity_.linearize(context, field);
    }

    template <typename Context>
    auto
    linearize(const Context &context) const
    {
      return quantity_.linearize(context);
    }

    template <typename Quadrature>
    auto
    locally_owned_quadrature_points(const Quadrature &quadrature) const
    {
      return quantity_.locally_owned_quadrature_points(quadrature);
    }

    const auto &
    locally_owned_points() const
    {
      return quantity_.locally_owned_points();
    }

    const auto &
    locally_relevant_points() const
    {
      return quantity_.locally_relevant_points();
    }

    auto
    point_index(const std::size_t point) const
    {
      return quantity_.point_index(point);
    }

    auto
    mpi_communicator() const
    {
      return quantity_.mpi_communicator();
    }

  private:
    Quantity                quantity_;
    NormalGeometry<Surface> geometry_;
  };

  template <typename Quantity, typename Surface>
  NormalQuantity<Quantity, Surface>
  operator*(Quantity quantity, NormalGeometry<Surface> geometry)
  {
    return NormalQuantity<Quantity, Surface>(std::move(quantity),
                                             std::move(geometry));
  }

  namespace detail
  {
    /** Assemble the point-supported normal weak form once for fixed geometry.
     */
    template <typename Quantity,
              typename Surface,
              typename TargetField,
              typename VectorType>
    dealii::LinearOperator<VectorType, typename Quantity::value_type>
    make_normal_load_operator(const NormalQuantity<Quantity, Surface> &load,
                              const TargetField                       &target)
    {
      static_assert(TargetField::dimension() == TargetField::spacedimension(),
                    "Point-supported normal loads require a full-dimensional "
                    "target mesh.");
      using QuantityVector = typename Quantity::value_type;
      using Operator       = dealii::LinearOperator<VectorType, QuantityVector>;
      const auto source_points = load.locally_owned_quadrature_points(
        dealii::Quadrature<Quantity::support_dimension>());
      const auto point_indices = [&load, n = source_points.size()] {
        std::vector<dealii::types::global_dof_index> result;
        result.reserve(n);
        for (std::size_t q = 0; q < n; ++q)
          result.push_back(load.point_index(q));
        return result;
      }();

      const auto &target_triangulation =
        target.dof_handler().get_triangulation();
      const auto *distributed_target =
        dynamic_cast<const dealii::parallel::TriangulationBase<
          TargetField::spacedimension()> *>(&target_triangulation);
      AssertThrow(distributed_target != nullptr,
                  dealii::ExcMessage(
                    "Point-supported weak terms require a distributed target "
                    "triangulation."));
      const auto &target_mapping = target.mapping();
      auto        distribution   = std::make_shared<
        DistributedLiftedQuadrature<TargetField::spacedimension()>>(
        load.geometry().particle_coupling_parameters());
      distribution->initialize(*distributed_target,
                               target_mapping,
                               source_points);

      using Entry  = std::pair<dealii::types::global_dof_index, double>;
      auto entries = std::make_shared<
        std::map<dealii::types::particle_index, std::vector<Entry>>>();
      for (const auto &particle :
           distribution->particle_coupling().get_particles())
        {
          const auto &cell = particle.get_surrounding_cell();
          const typename dealii::DoFHandler<TargetField::dimension(),
                                            TargetField::spacedimension()>::
            cell_iterator target_cell(*cell, &target.dof_handler());
          std::vector<dealii::types::global_dof_index> dof_indices(
            target.space().finite_element().n_dofs_per_cell());
          target_cell->get_dof_indices(dof_indices);
          const dealii::Quadrature<TargetField::dimension()> point_quadrature(
            std::vector<dealii::Point<TargetField::dimension()>>{
              particle.get_reference_location()});
          dealii::FEValues<TargetField::dimension(),
                           TargetField::spacedimension()>
            target_values(target_mapping,
                          target.space().finite_element(),
                          point_quadrature,
                          dealii::update_values);
          target_values.reinit(target_cell);

          const auto &stencil = distribution->stencil(particle.get_id());
          const auto  normal  = load.geometry().normal(particle.get_location());
          for (unsigned int i = 0; i < dof_indices.size(); ++i)
            if (target.locally_owned_dofs().is_element(dof_indices[i]))
              {
                const auto &target_view = target_values[target.extractor()];
                (*entries)[particle.get_id()].emplace_back(
                  dof_indices[i],
                  stencil.physical_weight * (normal * target_view.value(i, 0)));
              }
        }

      const auto owned        = load.locally_owned_points();
      const auto relevant     = load.locally_relevant_points();
      const auto communicator = load.mpi_communicator();
      Operator   result;
      result.reinit_range_vector = [owned    = target.locally_owned_dofs(),
                                    relevant = target.locally_relevant_dofs(),
                                    communicator](VectorType &vector,
                                                  const bool  omit) {
        vector.reinit(owned, relevant, communicator);
        if (!omit)
          vector = 0.;
      };
      result.reinit_domain_vector =
        [owned, relevant, communicator](QuantityVector &vector,
                                        const bool      omit) {
          vector.reinit(owned, relevant, communicator);
          if (!omit)
            vector = 0.;
        };

      const auto values_on_target =
        [distribution, point_indices](const QuantityVector &source) {
          dealii::Vector<double> local_values(point_indices.size());
          for (std::size_t q = 0; q < point_indices.size(); ++q)
            local_values[q] = source[point_indices[q]];
          return distribution->values_on_target(local_values);
        };
      result.vmult = [entries, values_on_target](VectorType &destination,
                                                 const QuantityVector &source) {
        const auto values = values_on_target(source);
        destination       = 0.;
        for (const auto &[id, point_entries] : *entries)
          for (const auto &[row, value] : point_entries)
            destination[row] += value * values.at(id);
      };
      result.vmult_add = [entries,
                          values_on_target](VectorType           &destination,
                                            const QuantityVector &source) {
        const auto values = values_on_target(source);
        for (const auto &[id, point_entries] : *entries)
          for (const auto &[row, value] : point_entries)
            destination[row] += value * values.at(id);
      };
      result.Tvmult =
        [entries, distribution, point_indices](QuantityVector   &destination,
                                               const VectorType &source) {
          std::map<dealii::types::particle_index, double> values;
          for (const auto &[id, point_entries] : *entries)
            for (const auto &[row, value] : point_entries)
              values[id] += value * source[row];
          dealii::Vector<double> local_values(point_indices.size());
          local_values = 0.;
          distribution->add_transpose_to_source(values, local_values);
          destination = 0.;
          for (std::size_t q = 0; q < point_indices.size(); ++q)
            destination[point_indices[q]] += local_values[q];
        };
      result.Tvmult_add =
        [entries, distribution, point_indices](QuantityVector   &destination,
                                               const VectorType &source) {
          std::map<dealii::types::particle_index, double> values;
          for (const auto &[id, point_entries] : *entries)
            for (const auto &[row, value] : point_entries)
              values[id] += value * source[row];
          dealii::Vector<double> local_values(point_indices.size());
          local_values = 0.;
          distribution->add_transpose_to_source(values, local_values);
          for (std::size_t q = 0; q < point_indices.size(); ++q)
            destination[point_indices[q]] += local_values[q];
        };
      return result;
    }
  } // namespace detail

  /** A weak term for a quantity paired with a geometry-provided normal. */
  template <typename Quantity, typename Surface, typename TargetField>
  class NormalWeakTerm
  {
  public:
    NormalWeakTerm(NormalQuantity<Quantity, Surface> load, TargetField target)
      : load_(std::move(load))
      , target_(std::move(target))
    {}

    template <typename VectorType, typename MatrixType>
    FieldId
    add(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      const auto target_id     = target_.field_id();
      const auto operator_view = detail::
        make_normal_load_operator<Quantity, Surface, TargetField, VectorType>(
          load_, target_);
      auto term = builder.term(target_id, "normal-weak-term");
      term.residual([operator_view, load_ = load_](const auto &context) {
        const auto value = load_.evaluate(context);
        typename SemiDiscreteModel<VectorType, MatrixType>::Operation result;
        result.reinit_vector = operator_view.reinit_range_vector;
        result.apply         = [operator_view, value](VectorType &destination) {
          operator_view.vmult(destination, value);
        };
        result.apply_add = [operator_view, value](VectorType &destination) {
          operator_view.vmult_add(destination, value);
        };
        return result;
      });
      using OperatorFactory =
        typename SemiDiscreteModel<VectorType, MatrixType>::OperatorFactory;
      for (const auto field : load_.dependencies())
        term.state(field,
                   OperatorFactory([operator_view, load_ = load_, field](
                                     const auto &context) {
                     return operator_view * load_.linearize(context, field);
                   }));
      return target_id;
    }

    template <typename VectorType, typename MatrixType>
    FieldId
    operator()(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      return add(builder);
    }

    const NormalQuantity<Quantity, Surface> &
    load() const
    {
      return load_;
    }

    const TargetField &
    target() const
    {
      return target_;
    }

  private:
    NormalQuantity<Quantity, Surface> load_;
    TargetField                       target_;
  };

  /** A solver-neutral FE weak term contributed to a residual row. */
  template <typename TrialExpression, typename TestExpression>
  class WeakTerm
  {
  public:
    using target_type = TestExpression;

    WeakTerm(TrialExpression observable, TestExpression target)
      : observable_(std::move(observable))
      , target_(std::move(target))
    {}

    template <typename VectorType, typename MatrixType>
    FieldId
    add(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      using Model          = SemiDiscreteModel<VectorType, MatrixType>;
      const auto target_id = target_.source().field_id();
      auto       term      = builder.term(target_id, "weak_term");
      if constexpr (!detail::is_linear_observable<TrialExpression>::value)
        {
          using Assembly = std::conditional_t<
            detail::is_lifted_observable<TrialExpression>::value,
            detail::LiftedWeakAssembly<TrialExpression, TestExpression>,
            detail::NonlinearWeakAssembly<TrialExpression, TestExpression>>;
          term.residual(
            [observable = observable_, target = target_](const auto &context) {
              const auto values =
                Assembly::template residual<VectorType, MatrixType>(observable,
                                                                    target,
                                                                    context);
              typename Model::Operation result;
              result.reinit_vector = [values](VectorType &vector,
                                              const bool  omit) {
                vector.reinit(*values, omit);
              };
              result.apply = [values](VectorType &destination) {
                destination = *values;
              };
              result.apply_add = [values](VectorType &destination) {
                destination += *values;
              };
              return result;
            });
          using OperatorFactory = typename Model::OperatorFactory;
          for (const auto field : observable_.dependencies())
            term.state(
              field,
              OperatorFactory([observable = observable_,
                               target     = target_,
                               field](const auto &context) {
                return Assembly::template linearize<VectorType, MatrixType>(
                  observable, target, field, context);
              }));
        }
      else if (observable_.is_frozen())
        {
          const auto pairing = make_pairing(builder, target_);
          const auto frozen  = observable_.template frozen_values<VectorType>();
          term.residual([pairing, frozen](const auto &) {
            typename Model::Operation result;
            result.reinit_vector =
              pairing.operator_with_matrix.view.reinit_range_vector;
            result.apply = [pairing, frozen](VectorType &destination) {
              pairing.operator_with_matrix.view.vmult(destination, frozen);
            };
            result.apply_add = [pairing, frozen](VectorType &destination) {
              pairing.operator_with_matrix.view.vmult_add(destination, frozen);
            };
            return result;
          });
        }
      else
        {
          const auto pairing   = make_pairing(builder, target_);
          const auto source_id = pairing.source_id;
          typename Model::MatrixOperatorFactory state_factory =
            [pairing](const typename Model::Context &) {
              (void)pairing.matrix;
              (void)pairing.sparsity;
              return pairing.operator_with_matrix;
            };
          term
            .residual([pairing, source_id](const auto &ctx) {
              return pairing.operator_with_matrix.view * ctx.state(source_id);
            })
            .state(source_id, std::move(state_factory));
        }
      return target_id;
    }

    /** Add both rows of this term to a multiplier constraint. */
    template <typename VectorType, typename MatrixType>
    void
    add_constraint_terms(SemidiscreteBuilder<VectorType, MatrixType> &builder,
                         const FieldId     multiplier,
                         const double      sign,
                         const std::size_t index) const
    {
      const auto multiplier_field = target_.source().with_id(multiplier);
      const auto make_multiplier_expression = [&]() {
        if constexpr (detail::is_lifted_observable<TestExpression>::value)
          return target_.with_id(multiplier);
        else
          return test(multiplier_field);
      };
      const auto multiplier_expression = make_multiplier_expression();
      const auto suffix                = "constraint." + std::to_string(index);
      using Model = SemiDiscreteModel<VectorType, MatrixType>;
      if constexpr (!detail::is_linear_observable<TrialExpression>::value)
        {
          using Assembly = std::conditional_t<
            detail::is_lifted_observable<TrialExpression>::value,
            detail::LiftedWeakAssembly<TrialExpression,
                                       decltype(multiplier_expression)>,
            detail::NonlinearWeakAssembly<TrialExpression,
                                          decltype(multiplier_expression)>>;
          auto multiplier_term =
            builder.term(multiplier, suffix + ".multiplier");
          multiplier_term.residual([observable = observable_,
                                    multiplier_expression,
                                    sign](const auto &context) {
            const auto values =
              Assembly::template residual<VectorType, MatrixType>(
                observable, multiplier_expression, context);
            typename Model::Operation result;
            result.reinit_vector = [values](VectorType &vector,
                                            const bool  omit) {
              vector.reinit(*values, omit);
            };
            result.apply = [values, sign](VectorType &destination) {
              destination = *values;
              destination *= sign;
            };
            result.apply_add = [values, sign](VectorType &destination) {
              destination.add(sign, *values);
            };
            return result;
          });
          using OperatorFactory = typename Model::OperatorFactory;
          for (const auto field : observable_.dependencies())
            multiplier_term.state(
              field,
              OperatorFactory(
                [observable = observable_, multiplier_expression, field, sign](
                  const auto &context) {
                  return sign *
                         Assembly::template linearize<VectorType, MatrixType>(
                           observable, multiplier_expression, field, context);
                }));

          for (const auto field : observable_.dependencies())
            {
              auto participant_term =
                builder.term(field, suffix + ".participant");
              participant_term.residual([observable = observable_,
                                         multiplier_expression,
                                         field,
                                         multiplier,
                                         sign](const auto &context) {
                const auto jacobian =
                  Assembly::template linearize<VectorType, MatrixType>(
                    observable, multiplier_expression, field, context);
                const auto  reaction = dealii::transpose_operator(jacobian);
                const auto *multiplier_state = &context.state(multiplier);
                typename Model::Operation result;
                result.reinit_vector = reaction.reinit_range_vector;
                result.apply =
                  [reaction, multiplier_state, sign](VectorType &destination) {
                    reaction.vmult(destination, *multiplier_state);
                    destination *= sign;
                  };
                result.apply_add =
                  [reaction, multiplier_state, sign](VectorType &destination) {
                    VectorType contribution;
                    reaction.reinit_range_vector(contribution, false);
                    reaction.vmult(contribution, *multiplier_state);
                    destination.add(sign, contribution);
                  };
                return result;
              });
              participant_term.state(
                multiplier,
                OperatorFactory([observable = observable_,
                                 multiplier_expression,
                                 field,
                                 sign](const auto &context) {
                  const auto jacobian =
                    Assembly::template linearize<VectorType, MatrixType>(
                      observable, multiplier_expression, field, context);
                  return sign * dealii::transpose_operator(jacobian);
                }));
            }
        }
      else
        {
          const auto pairing = make_pairing(builder, multiplier_expression);
          const auto reaction =
            ImmersX::transpose_operator(pairing.operator_with_matrix);
          const auto source_id = pairing.source_id;
          builder.term(multiplier, suffix + ".multiplier")
            .residual([pairing, source_id, sign](const auto &context) {
              return sign * (pairing.operator_with_matrix.view *
                             context.state(source_id));
            })
            .state(source_id, sign * pairing.operator_with_matrix);

          builder.term(source_id, suffix + ".participant")
            .residual([reaction, multiplier, sign](const auto &context) {
              return sign * (reaction.view * context.state(multiplier));
            })
            .state(multiplier, sign * reaction);
        }
    }

    template <typename VectorType, typename MatrixType>
    FieldId
    operator()(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      return add(builder);
    }

    const TrialExpression &
    observable() const
    {
      return observable_;
    }

    const TestExpression &
    target() const
    {
      return target_;
    }

  private:
    template <typename VectorType, typename MatrixType>
    struct Pairing
    {
      using Model = SemiDiscreteModel<VectorType, MatrixType>;

      std::shared_ptr<MatrixType>              matrix;
      std::shared_ptr<dealii::SparsityPattern> sparsity;
      typename Model::MatrixOperator           operator_with_matrix;
      FieldId                                  source_id;
      FieldId                                  target_id;
    };

    template <typename VectorType, typename MatrixType, typename Target>
    Pairing<VectorType, MatrixType>
    make_pairing(SemidiscreteBuilder<VectorType, MatrixType> &builder,
                 const Target                                &target) const
    {
      Pairing<VectorType, MatrixType> result;
      if constexpr (detail::is_lifted_observable<std::decay_t<Target>>::value)
        {
          using Assembly =
            detail::LiftedTargetWeakAssembly<TrialExpression, Target>;
          const auto storage =
            Assembly::template assemble<VectorType, MatrixType>(observable_,
                                                                target);
          result.matrix               = storage.matrix;
          result.sparsity             = storage.sparsity;
          result.operator_with_matrix = builder.matrix_operator(*result.matrix);
        }
      else
        {
          using Assembly = detail::WeakAssembly<TrialExpression, Target>;
          const bool nonmatching_geometry =
            Assembly::geometry_is_nonmatching(observable_, target);
          if (nonmatching_geometry)
            {
              const auto prepared =
                Assembly::template prepare<VectorType, MatrixType>(observable_,
                                                                   target);
              result.matrix   = prepared->storage.matrix;
              result.sparsity = prepared->storage.sparsity;
              result.operator_with_matrix =
                Assembly::template dynamic_matrix_operator<VectorType,
                                                           MatrixType>(
                  prepared);
            }
          else
            {
              const auto storage =
                Assembly::template assemble<VectorType, MatrixType>(observable_,
                                                                    target);
              result.matrix   = storage.matrix;
              result.sparsity = storage.sparsity;
              result.operator_with_matrix =
                builder.matrix_operator(*result.matrix);
            }
        }
      result.source_id = observable_.source_field();
      result.target_id = target.source().field_id();
      return result;
    }

    TrialExpression observable_;
    TestExpression  target_;
  };

  /** \cond deduction_guide */
  template <typename TrialExpression, typename TestExpression>
  WeakTerm(TrialExpression, TestExpression)
    -> WeakTerm<TrialExpression, TestExpression>;
  /** \endcond */

  /** Create a residual term representing the FE duality pairing. */
  template <typename Trial, typename Test>
  auto
  weak_term(Trial trial, Test test)
  {
    static_assert(detail::is_test_expression<std::decay_t<Test>>::value,
                  "weak_term requires an explicit test expression; use "
                  "test(field), gradient(test(field)), or another test-side "
                  "FE operation.");
    auto trial_expression = as_fe_expression(trial);
    auto test_expression  = test;
    return WeakTerm<std::decay_t<decltype(trial_expression)>,
                    std::decay_t<decltype(test_expression)>>(
      std::move(trial_expression), std::move(test_expression));
  }

  template <typename Quantity, typename Surface, typename TargetField>
  auto
  weak_term(NormalQuantity<Quantity, Surface> load, TargetField target)
  {
    return NormalWeakTerm<Quantity, Surface, TargetField>(std::move(load),
                                                          std::move(target));
  }

  template <typename Quantity,
            typename Surface,
            typename SourceFieldType,
            typename Operation>
  auto
  weak_term(NormalQuantity<Quantity, Surface>          load,
            TestExpression<SourceFieldType, Operation> target)
  {
    return NormalWeakTerm<Quantity, Surface, SourceFieldType>(std::move(load),
                                                              target.source());
  }
} // namespace ImmersX

#endif // immersx_weak_term_h
