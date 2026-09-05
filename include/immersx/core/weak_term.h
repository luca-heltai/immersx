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

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

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

    template <int dim, int spacedim>
    double
    scalar_shape_value(const dealii::FEValuesBase<dim, spacedim> &fe_values,
                       const dealii::FEValuesExtractors::Scalar  &extractor,
                       const unsigned int                         i,
                       const unsigned int                         q)
    {
      return fe_values.shape_value_component(i, q, extractor.component);
    }

    template <int dim, int spacedim>
    dealii::Tensor<1, spacedim>
    vector_shape_value(const dealii::FEValuesBase<dim, spacedim> &fe_values,
                       const dealii::FEValuesExtractors::Vector  &extractor,
                       const unsigned int                         i,
                       const unsigned int                         q)
    {
      dealii::Tensor<1, spacedim> value;
      for (unsigned int component = 0; component < spacedim; ++component)
        value[component] = fe_values.shape_value_component(
          i, q, extractor.first_vector_component + component);
      return value;
    }

    template <int dim, int spacedim>
    dealii::Tensor<1, spacedim>
    scalar_shape_gradient(const dealii::FEValuesBase<dim, spacedim> &fe_values,
                          const dealii::FEValuesExtractors::Scalar  &extractor,
                          const unsigned int                         i,
                          const unsigned int                         q)
    {
      return fe_values.shape_grad_component(i, q, extractor.component);
    }

    template <typename ObservableType, typename TargetField>
    struct WeakAssembly
    {
      static constexpr int dim      = TargetField::dimension();
      static constexpr int spacedim = TargetField::spacedimension();

      using SourceField = typename ObservableType::source_field_type;

      static_assert(!std::is_same_v<SourceField, void>,
                    "weak_term requires an observable produced from a Field.");

      template <typename MatrixType>
      struct MatrixStorage
      {
        std::shared_ptr<MatrixType>              matrix;
        std::shared_ptr<dealii::SparsityPattern> sparsity;
      };

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
      assemble(const ObservableType &observable, const TargetField &target)
      {
        AssertThrow(target.field_id().is_valid(),
                    dealii::ExcMessage(
                      "A weak term target field must be registered."));
        AssertThrow(observable.is_frozen() ||
                      observable.source_field().is_valid(),
                    dealii::ExcMessage(
                      "A weak term active source field must be registered."));
        AssertThrow(observable.spacedimension() == spacedim &&
                      SourceField::spacedimension() == spacedim &&
                      SourceField::dimension() >= dim,
                    dealii::ExcMessage(
                      "Weak-term source and target must share an embedding "
                      "dimension, with the source support no lower than the "
                      "target support."));

        if (observable.operation() == ObservableOperation::value)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return assemble_from_source<SourceField, VectorType, MatrixType>(
                observable, observable.template source<SourceField>(), target);
            else if constexpr (std::is_same_v<
                                 typename ObservableType::value_type,
                                 dealii::Tensor<1, spacedim>>)
              return assemble_from_source<SourceField, VectorType, MatrixType>(
                observable, observable.template source<SourceField>(), target);
          }
        else if (observable.operation() == ObservableOperation::gradient)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              return assemble_from_source<SourceField, VectorType, MatrixType>(
                observable, observable.template source<SourceField>(), target);
          }

        AssertThrow(false,
                    dealii::ExcMessage(
                      "Unsupported weak-term observable/source pairing."));
        return {};
      }

      static bool
      geometry_is_nonmatching(const ObservableType &observable,
                              const TargetField    &target)
      {
        return static_cast<const void *>(
                 &observable.template source<SourceField>()
                    .dof_handler()
                    .get_triangulation()) !=
               static_cast<const void *>(
                 &target.dof_handler().get_triangulation());
      }

      template <typename VectorType, typename MatrixType>
      static std::shared_ptr<PreparedMatrix<VectorType, MatrixType>>
      prepare(const ObservableType &observable, const TargetField &target)
      {
        if (observable.operation() == ObservableOperation::value)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return prepare_from_source<SourceField, VectorType, MatrixType>(
                observable, observable.template source<SourceField>(), target);
            else if constexpr (std::is_same_v<
                                 typename ObservableType::value_type,
                                 dealii::Tensor<1, spacedim>>)
              return prepare_from_source<SourceField, VectorType, MatrixType>(
                observable, observable.template source<SourceField>(), target);
          }
        else if (observable.operation() == ObservableOperation::gradient)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              return prepare_from_source<SourceField, VectorType, MatrixType>(
                observable, observable.template source<SourceField>(), target);
          }

        AssertThrow(false,
                    dealii::ExcMessage(
                      "Unsupported weak-term observable/source pairing."));
        return {};
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
      template <typename SourceField, typename VectorType, typename MatrixType>
      static std::shared_ptr<PreparedMatrix<VectorType, MatrixType>>
      prepare_from_source(const ObservableType &observable,
                          const SourceField    &source,
                          const TargetField    &target)
      {
        const auto degree = std::max(source.space().finite_element().degree,
                                     target.space().finite_element().degree);
        const dealii::QGauss<dim> quadrature(degree + 1);
        dealii::UpdateFlags       update_flags =
          dealii::update_values | dealii::update_JxW_values;
        if (observable.operation() == ObservableOperation::gradient)
          update_flags |= dealii::update_gradients;

        auto result =
          std::make_shared<PreparedMatrix<VectorType, MatrixType>>();
        result->rebuild =
          [observable, source, target, quadrature, update_flags] {
            return assemble_nonmatching<SourceField, VectorType, MatrixType>(
              observable, source, target, quadrature, update_flags);
          };
        result->structure_is_current =
          [source,
           target,
           source_tria   = &source.dof_handler().get_triangulation(),
           target_tria   = &target.dof_handler().get_triangulation(),
           source_n_dofs = source.dof_handler().n_dofs(),
           target_n_dofs = target.dof_handler().n_dofs(),
           source_fe     = &source.space().finite_element(),
           target_fe     = &target.space().finite_element()] {
            return source_tria == &source.dof_handler().get_triangulation() &&
                   target_tria == &target.dof_handler().get_triangulation() &&
                   source.dof_handler().n_dofs() == source_n_dofs &&
                   target.dof_handler().n_dofs() == target_n_dofs &&
                   source_fe == &source.space().finite_element() &&
                   target_fe == &target.space().finite_element();
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
        result->target_connection =
          target.dof_handler().get_triangulation().signals.any_change.connect(
            invalidate);
        result->source_mesh_connection =
          source.dof_handler()
            .get_triangulation()
            .signals.mesh_movement.connect(invalidate);
        result->target_mesh_connection =
          target.dof_handler()
            .get_triangulation()
            .signals.mesh_movement.connect(invalidate);
        result->source_partition_connection =
          source.dof_handler()
            .get_triangulation()
            .signals.pre_partition.connect(invalidate);
        result->target_partition_connection =
          target.dof_handler()
            .get_triangulation()
            .signals.pre_partition.connect(invalidate);
        return result;
      }

      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_from_source(const ObservableType &observable,
                           const SourceField    &source,
                           const TargetField    &target)
      {
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
                       target.space().finite_element().degree);
            const dealii::QGauss<dim> quadrature(degree + 1);
            dealii::UpdateFlags       update_flags =
              dealii::update_values | dealii::update_JxW_values;
            if (observable.operation() == ObservableOperation::gradient)
              update_flags |= dealii::update_gradients;

            if (&source.space().dof_handler().get_triangulation() !=
                &target.space().dof_handler().get_triangulation())
              return assemble_nonmatching<SourceField, VectorType, MatrixType>(
                observable, source, target, quadrature, update_flags);

            AssertThrow(&source.mapping() == &target.mapping(),
                        dealii::ExcMessage(
                          "A weak term requires compatible source and target "
                          "mappings."));

            dealii::DynamicSparsityPattern sparsity(
              target.dof_handler().n_dofs(),
              source.dof_handler().n_dofs(),
              target.locally_owned_dofs());

            if (&source.dof_handler() == &target.dof_handler())
              for_each_same_cell(target, [&](const auto &cell) {
                if (!cell->is_locally_owned())
                  return;
                std::vector<dealii::types::global_dof_index> indices(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(indices);
                target.constraints().add_entries_local_to_global(
                  indices, source.constraints(), indices, sparsity, false);
              });
            else
              for_each_cell(
                source,
                target,
                [&](const auto &source_cell, const auto &target_cell) {
                  if (!target_cell->is_locally_owned())
                    return;
                  std::vector<dealii::types::global_dof_index> source_indices(
                    source_cell->get_fe().n_dofs_per_cell());
                  std::vector<dealii::types::global_dof_index> target_indices(
                    target_cell->get_fe().n_dofs_per_cell());
                  source_cell->get_dof_indices(source_indices);
                  target_cell->get_dof_indices(target_indices);
                  target.constraints().add_entries_local_to_global(
                    target_indices,
                    source.constraints(),
                    source_indices,
                    sparsity,
                    false);
                });

            auto matrix = std::make_shared<MatrixType>();
            auto matrix_sparsity =
              initialize_weak_matrix(*matrix,
                                     target.locally_owned_dofs(),
                                     source.locally_owned_dofs(),
                                     sparsity,
                                     target.space().mpi_communicator());

            if (&source.dof_handler() == &target.dof_handler())
              {
                dealii::FEValues<dim, spacedim> values(
                  target.mapping(),
                  target.space().finite_element(),
                  quadrature,
                  update_flags);
                for_each_same_cell(target, [&](const auto &cell) {
                  if (!cell->is_locally_owned())
                    return;
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
                });
              }
            else
              {
                dealii::FEValues<dim, spacedim> source_values(
                  source.mapping(),
                  source.space().finite_element(),
                  quadrature,
                  update_flags);
                dealii::FEValues<dim, spacedim> target_values(
                  target.mapping(),
                  target.space().finite_element(),
                  quadrature,
                  update_flags);
                for_each_cell(source,
                              target,
                              [&](const auto &source_cell,
                                  const auto &target_cell) {
                                if (!target_cell->is_locally_owned())
                                  return;
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
                              });
              }
            compress_weak_matrix(*matrix);
            return {std::move(matrix), std::move(matrix_sparsity)};
          }
      }

      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_nonmatching(const ObservableType          &observable,
                           const SourceField             &source,
                           const TargetField             &target,
                           const dealii::Quadrature<dim> &quadrature,
                           const dealii::UpdateFlags      update_flags)
      {
#ifdef IMMERSX_WEAK_TERM_TESTING
        ++weak_term_nonmatching_preparations;
#endif
        if constexpr (SourceField::dimension() == dim && dim == spacedim)
          return assemble_nonmatching_reverse<SourceField,
                                              VectorType,
                                              MatrixType>(
            observable, source, target, quadrature, update_flags);
        else if constexpr (SourceField::dimension() > dim)
          return assemble_nonmatching_reverse<SourceField,
                                              VectorType,
                                              MatrixType>(
            observable, source, target, quadrature, update_flags);
        else
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "This weak-term geometry combination is not "
                          "implemented."));
            return {};
          }
      }

      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_nonmatching_same_dimension(
        const ObservableType          &observable,
        const SourceField             &source,
        const TargetField             &target,
        const dealii::Quadrature<dim> &quadrature,
        const dealii::UpdateFlags      update_flags)
      {
        static_assert(dim == spacedim,
                      "Nonmatching weak terms require full-dimensional "
                      "background meshes.");

        const auto *target_tria =
          dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
            &target.space().dof_handler().get_triangulation());
        AssertThrow(target_tria != nullptr,
                    dealii::ExcMessage(
                      "Nonmatching weak terms require a distributed target "
                      "triangulation."));

        using Point = RepresentationQuadraturePoint<spacedim, double>;
        std::vector<Point> source_points;
        source_points.reserve(
          source.space().dof_handler().get_triangulation().n_active_cells() *
          quadrature.size());

        dealii::FEValues<dim, spacedim> source_values(
          source.mapping(),
          source.space().finite_element(),
          quadrature,
          update_flags | dealii::update_quadrature_points);
        std::vector<dealii::types::global_dof_index> source_indices(
          source.space().finite_element().n_dofs_per_cell());
        const unsigned int n_components =
          std::is_same_v<typename SourceField::value_type, double> ?
            (observable.operation() == ObservableOperation::gradient ?
               spacedim :
               1) :
            spacedim;

        for (const auto &cell : source.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              source_values.reinit(cell);
              cell->get_dof_indices(source_indices);
              for (const auto q : source_values.quadrature_point_indices())
                {
                  Point point;
                  point.point = source_values.quadrature_point(q);
                  point.representative_point = point.point;
                  point.source_entity_id     = cell->global_active_cell_index();
                  point.representative_qpoint = q;
                  point.weight                = source_values.JxW(q);
                  point.stable_id             = static_cast<std::uint64_t>(
                                      cell->global_active_cell_index()) *
                                      quadrature.size() +
                                    q;
                  point.dof_indices = source_indices;
                  point.basis_values.resize(source_indices.size() *
                                            n_components);

                  for (unsigned int i = 0; i < source_indices.size(); ++i)
                    if constexpr (std::is_same_v<
                                    typename SourceField::value_type,
                                    double>)
                      {
                        if (observable.operation() ==
                            ObservableOperation::value)
                          point.basis_values[i] = scalar_shape_value(
                            source_values, source.extractor(), i, q);
                        else
                          {
                            const auto gradient = scalar_shape_gradient(
                              source_values, source.extractor(), i, q);
                            for (unsigned int d = 0; d < spacedim; ++d)
                              point.basis_values[i * spacedim + d] =
                                gradient[d];
                          }
                      }
                    else
                      {
                        const auto value = vector_shape_value(
                          source_values, source.extractor(), i, q);
                        for (unsigned int d = 0; d < spacedim; ++d)
                          point.basis_values[i * spacedim + d] = value[d];
                      }
                  source_points.emplace_back(std::move(point));
                }
            }

        ParticleCouplingParameters<dim> particle_parameters(
          "/ImmersX/weak term/nonmatching/" +
          std::to_string(reinterpret_cast<std::uintptr_t>(&source)));
        DistributedLiftedQuadrature<dim> distribution(particle_parameters);
        distribution.initialize(*target_tria, target.mapping(), source_points);

        struct TargetPoint
        {
          dealii::types::particle_index id;
          dealii::Point<dim>            reference;
        };
        std::map<dealii::CellId, std::vector<TargetPoint>> points_by_cell;
        for (const auto &particle :
             distribution.particle_coupling().get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            points_by_cell[particle.get_surrounding_cell()->id()].push_back(
              {particle.get_id(), particle.get_reference_location()});

        dealii::DynamicSparsityPattern sparsity(target.dof_handler().n_dofs(),
                                                source.dof_handler().n_dofs(),
                                                target.locally_owned_dofs());
        for (const auto &cell : target.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            if (const auto points = points_by_cell.find(cell->id());
                points != points_by_cell.end())
              {
                std::vector<dealii::types::global_dof_index> target_indices(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(target_indices);
                for (const auto &point : points->second)
                  target.constraints().add_entries_local_to_global(
                    target_indices,
                    source.constraints(),
                    distribution.stencil(point.id).source_dof_indices,
                    sparsity,
                    false);
              }

        auto matrix = std::make_shared<MatrixType>();
        auto matrix_sparsity =
          initialize_weak_matrix(*matrix,
                                 target.locally_owned_dofs(),
                                 source.locally_owned_dofs(),
                                 sparsity,
                                 target.space().mpi_communicator());

        for (const auto &cell : target.dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            if (const auto points = points_by_cell.find(cell->id());
                points != points_by_cell.end())
              {
                std::vector<dealii::Point<dim>> reference_points;
                reference_points.reserve(points->second.size());
                for (const auto &point : points->second)
                  reference_points.push_back(point.reference);
                const dealii::Quadrature<dim> point_quadrature(
                  std::move(reference_points));
                dealii::FEValues<dim, spacedim> target_values(
                  target.mapping(),
                  target.space().finite_element(),
                  point_quadrature,
                  dealii::update_values);
                target_values.reinit(cell);

                std::vector<dealii::types::global_dof_index> target_indices(
                  cell->get_fe().n_dofs_per_cell());
                cell->get_dof_indices(target_indices);
                dealii::FullMatrix<double> local;
                for (unsigned int q = 0; q < points->second.size(); ++q)
                  {
                    const auto &stencil =
                      distribution.stencil(points->second[q].id);
                    local.reinit(target_indices.size(),
                                 stencil.source_dof_indices.size());
                    for (unsigned int i = 0; i < target_indices.size(); ++i)
                      for (unsigned int j = 0;
                           j < stencil.source_dof_indices.size();
                           ++j)
                        local(i, j) += nonmatching_contract(observable,
                                                            source,
                                                            target,
                                                            target_values,
                                                            i,
                                                            j,
                                                            q,
                                                            stencil) *
                                       stencil.physical_weight;
                    target.constraints().distribute_local_to_global(
                      local,
                      target_indices,
                      source.constraints(),
                      stencil.source_dof_indices,
                      *matrix);
                  }
              }
        compress_weak_matrix(*matrix);
        return {std::move(matrix), std::move(matrix_sparsity)};
      }

      /** Assemble a full-dimensional nonmatching pairing over the target
       * support.  This is the natural domain for a constraint multiplier: a
       * background participant is evaluated at quadrature points owned by the
       * multiplier mesh. */
      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_nonmatching_target_domain(
        const ObservableType          &observable,
        const SourceField             &source,
        const TargetField             &target,
        const dealii::Quadrature<dim> &quadrature,
        const dealii::UpdateFlags      update_flags)
      {
        const auto *source_tria =
          dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
            &source.space().dof_handler().get_triangulation());
        AssertThrow(source_tria != nullptr,
                    dealii::ExcMessage(
                      "Nonmatching weak terms require a distributed source "
                      "triangulation."));

        using Point = RepresentationQuadraturePoint<dim, double>;
        const bool target_is_vector =
          std::is_same_v<typename TargetField::value_type,
                         dealii::Tensor<1, spacedim>>;
        const unsigned int target_components = target_is_vector ? spacedim : 1;
        const unsigned int n_target_dofs =
          target.space().finite_element().n_dofs_per_cell();

        std::vector<Point>              target_points;
        dealii::FEValues<dim, spacedim> target_values(
          target.mapping(),
          target.space().finite_element(),
          quadrature,
          dealii::update_values | dealii::update_JxW_values |
            dealii::update_quadrature_points);
        std::vector<dealii::types::global_dof_index> target_indices(
          n_target_dofs);
        for (const auto &cell : target.dof_handler().active_cell_iterators())
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
                  point.dof_indices = target_indices;
                  point.basis_values.resize(n_target_dofs * target_components);
                  for (unsigned int i = 0; i < n_target_dofs; ++i)
                    if constexpr (std::is_same_v<
                                    typename TargetField::value_type,
                                    dealii::Tensor<1, spacedim>>)
                      {
                        const auto value = vector_shape_value(
                          target_values, target.extractor(), i, q);
                        for (unsigned int d = 0; d < spacedim; ++d)
                          point.basis_values[i * spacedim + d] = value[d];
                      }
                    else
                      point.basis_values[i] = scalar_shape_value(
                        target_values, target.extractor(), i, q);
                  target_points.emplace_back(std::move(point));
                }
            }

        ParticleCouplingParameters<dim> particle_parameters(
          "/ImmersX/weak term/target-domain/" +
          std::to_string(reinterpret_cast<std::uintptr_t>(&source)));
        DistributedLiftedQuadrature<dim> distribution(particle_parameters);
        distribution.initialize(*source_tria, source.mapping(), target_points);

        const unsigned int n_source_dofs =
          source.space().finite_element().n_dofs_per_cell();
        dealii::DynamicSparsityPattern sparsity(target.dof_handler().n_dofs(),
                                                source.dof_handler().n_dofs(),
                                                target.locally_relevant_dofs());
        std::vector<dealii::types::global_dof_index> source_indices(
          n_source_dofs);
        for (const auto &particle :
             distribution.particle_coupling().get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename SourceField::space_type::DoFHandlerType::
                cell_iterator source_cell(*particle.get_surrounding_cell(),
                                          &source.dof_handler());
              source_cell->get_dof_indices(source_indices);
              const auto &stencil = distribution.stencil(particle.get_id());
              for (const auto target_dof : stencil.source_dof_indices)
                for (const auto source_dof : source_indices)
                  sparsity.add(target_dof, source_dof);
            }
        dealii::SparsityTools::distribute_sparsity_pattern(
          sparsity,
          target.locally_owned_dofs(),
          target.space().mpi_communicator(),
          target.locally_relevant_dofs());

        auto matrix = std::make_shared<MatrixType>();
        auto matrix_sparsity =
          initialize_weak_matrix(*matrix,
                                 target.locally_owned_dofs(),
                                 source.locally_owned_dofs(),
                                 sparsity,
                                 target.space().mpi_communicator());

        const auto source_flags = dealii::UpdateFlags(dealii::update_values) |
                                  (update_flags & dealii::update_gradients);
        for (const auto &particle :
             distribution.particle_coupling().get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename SourceField::space_type::DoFHandlerType::
                cell_iterator source_cell(*particle.get_surrounding_cell(),
                                          &source.dof_handler());
              source_cell->get_dof_indices(source_indices);
              const dealii::Quadrature<dim> source_point_quadrature(
                std::vector<dealii::Point<dim>>{
                  particle.get_reference_location()});
              dealii::FEValues<dim, spacedim> source_values(
                source.mapping(),
                source.space().finite_element(),
                source_point_quadrature,
                source_flags);
              source_values.reinit(source_cell);

              const auto &stencil     = distribution.stencil(particle.get_id());
              const auto &target_dofs = stencil.source_dof_indices;
              dealii::FullMatrix<double> local(target_dofs.size(),
                                               source_indices.size());
              for (unsigned int i = 0; i < target_dofs.size(); ++i)
                for (unsigned int j = 0; j < source_indices.size(); ++j)
                  {
                    double contribution = 0.;
                    if constexpr (std::is_same_v<
                                    typename SourceField::value_type,
                                    double>)
                      {
                        if constexpr (std::is_same_v<
                                        typename ObservableType::value_type,
                                        double>)
                          contribution = scalar_shape_value(source_values,
                                                            source.extractor(),
                                                            j,
                                                            0) *
                                         stencil.source_basis_values[i];
                        else
                          {
                            const auto gradient = scalar_shape_gradient(
                              source_values, source.extractor(), j, 0);
                            dealii::Tensor<1, spacedim> test;
                            for (unsigned int d = 0; d < spacedim; ++d)
                              test[d] =
                                stencil.source_basis_values[i * spacedim + d];
                            contribution = gradient * test;
                          }
                      }
                    else
                      {
                        const auto value = vector_shape_value(
                          source_values, source.extractor(), j, 0);
                        dealii::Tensor<1, spacedim> test;
                        for (unsigned int d = 0; d < spacedim; ++d)
                          test[d] =
                            stencil.source_basis_values[i * spacedim + d];
                        contribution = value * test;
                      }
                    local(i, j) = observable.scale() * contribution *
                                  stencil.physical_weight;
                  }
              target.constraints().distribute_local_to_global(
                local,
                target_dofs,
                source.constraints(),
                source_indices,
                *matrix);
            }
        compress_weak_matrix(*matrix);
        return {std::move(matrix), std::move(matrix_sparsity)};
      }

      /** Assemble a mixed-dimensional pairing by transposing the natural
       * source-row matrix.  Target quadrature points are searched in the
       * full-dimensional source mesh, so rows remain owned by the source
       * background ranks and no distributed matrix contribution exchange is
       * needed before the explicit transpose. */
      template <typename SourceField, typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_nonmatching_reverse(const ObservableType          &observable,
                                   const SourceField             &source,
                                   const TargetField             &target,
                                   const dealii::Quadrature<dim> &quadrature,
                                   const dealii::UpdateFlags      update_flags)
      {
        static_assert(SourceField::dimension() == spacedim,
                      "Mixed-dimensional reverse weak terms require a "
                      "full-dimensional source background.");
        static_assert(SourceField::dimension() >= dim,
                      "Reverse weak-term assembly requires a full-dimensional "
                      "source background.");

        const auto *source_tria =
          dynamic_cast<const dealii::parallel::TriangulationBase<spacedim> *>(
            &source.space().dof_handler().get_triangulation());
        AssertThrow(source_tria != nullptr,
                    dealii::ExcMessage(
                      "Mixed-dimensional weak terms require a distributed "
                      "source background triangulation."));

        using Point = RepresentationQuadraturePoint<spacedim, double>;
        const unsigned int n_target_dofs =
          target.space().finite_element().n_dofs_per_cell();
        const bool target_is_vector =
          std::is_same_v<typename TargetField::value_type,
                         dealii::Tensor<1, spacedim>>;
        const unsigned int target_components = target_is_vector ? spacedim : 1;
        const unsigned int n_properties =
          1 + n_target_dofs + n_target_dofs * target_components;

        std::vector<Point>              target_points;
        dealii::FEValues<dim, spacedim> target_values(
          target.mapping(),
          target.space().finite_element(),
          quadrature,
          dealii::update_values | dealii::update_JxW_values |
            dealii::update_quadrature_points);
        std::vector<dealii::types::global_dof_index> target_indices(
          n_target_dofs);
        for (const auto &cell : target.dof_handler().active_cell_iterators())
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
                  point.dof_indices = target_indices;
                  point.basis_values.resize(n_target_dofs * target_components);
                  for (unsigned int i = 0; i < n_target_dofs; ++i)
                    if constexpr (std::is_same_v<
                                    typename TargetField::value_type,
                                    dealii::Tensor<1, spacedim>>)
                      {
                        const auto value = vector_shape_value(
                          target_values, target.extractor(), i, q);
                        for (unsigned int d = 0; d < spacedim; ++d)
                          point.basis_values[i * spacedim + d] = value[d];
                      }
                    else
                      point.basis_values[i] = scalar_shape_value(
                        target_values, target.extractor(), i, q);
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
        dealii::DynamicSparsityPattern sparsity(source.dof_handler().n_dofs(),
                                                target.dof_handler().n_dofs(),
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
                                 target.locally_owned_dofs(),
                                 sparsity,
                                 source.space().mpi_communicator());

        const auto source_flags = dealii::UpdateFlags(dealii::update_values) |
                                  (update_flags & dealii::update_gradients);
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
                    double source_value = 0.;
                    if constexpr (std::is_same_v<
                                    typename SourceField::value_type,
                                    double>)
                      {
                        if (observable.operation() ==
                            ObservableOperation::value)
                          source_value = scalar_shape_value(source_point_values,
                                                            source.extractor(),
                                                            i,
                                                            0);
                        else
                          {
                            const auto gradient = scalar_shape_gradient(
                              source_point_values, source.extractor(), i, 0);
                            if (target_is_vector)
                              for (unsigned int d = 0; d < spacedim; ++d)
                                local(i, j) +=
                                  gradient[d] *
                                  particle_properties[1 + n_target_dofs +
                                                      j * spacedim + d] *
                                  particle_properties[0];
                            else
                              AssertThrow(
                                false,
                                dealii::ExcMessage(
                                  "A scalar gradient weak term requires a "
                                  "vector target field."));
                            continue;
                          }
                      }
                    else
                      {
                        const auto value = vector_shape_value(
                          source_point_values, source.extractor(), i, 0);
                        if (target_is_vector)
                          for (unsigned int d = 0; d < spacedim; ++d)
                            local(i, j) +=
                              value[d] *
                              particle_properties[1 + n_target_dofs +
                                                  j * spacedim + d] *
                              particle_properties[0];
                        else
                          AssertThrow(
                            false,
                            dealii::ExcMessage(
                              "A vector-source weak term requires a vector "
                              "target field."));
                        continue;
                      }
                    local(i, j) =
                      source_value *
                      particle_properties[1 + n_target_dofs +
                                          (target_is_vector ? j * spacedim :
                                                              j)] *
                      particle_properties[0];
                  }
              source.constraints().distribute_local_to_global(
                local,
                source_indices,
                target.constraints(),
                target_dofs,
                *matrix);
            }
        matrix->compress(dealii::VectorOperation::add);
        return {ImmersX::detail::transpose_matrix(matrix), nullptr};
      }

      template <typename SourceField,
                typename TargetFieldType,
                typename TargetValues>
      static double
      nonmatching_contract(
        const ObservableType  &observable,
        const SourceField     &source,
        const TargetFieldType &target,
        const TargetValues    &target_values,
        const unsigned int     i,
        const unsigned int     j,
        const unsigned int     q,
        const typename DistributedLiftedQuadrature<spacedim>::Stencil &stencil)
      {
        (void)observable;
        (void)source;
        if constexpr (std::is_same_v<typename SourceField::value_type, double>)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return scalar_shape_value(target_values,
                                        target.extractor(),
                                        i,
                                        q) *
                     stencil.source_basis_values[j];
            else if constexpr (std::is_same_v<
                                 typename TargetFieldType::value_type,
                                 dealii::Tensor<1, spacedim>>)
              {
                dealii::Tensor<1, spacedim> gradient;
                for (unsigned int d = 0; d < spacedim; ++d)
                  gradient[d] = stencil.source_basis_values[j * spacedim + d];
                return gradient * vector_shape_value(target_values,
                                                     target.extractor(),
                                                     i,
                                                     q);
              }
            else
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "This scalar-source weak-term pairing is not "
                              "implemented."));
                return 0.;
              }
          }
        else
          {
            if constexpr (std::is_same_v<typename TargetFieldType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              {
                const auto source_value = [&stencil, j] {
                  dealii::Tensor<1, spacedim> value;
                  for (unsigned int d = 0; d < spacedim; ++d)
                    value[d] = stencil.source_basis_values[j * spacedim + d];
                  return value;
                }();
                return source_value * vector_shape_value(target_values,
                                                         target.extractor(),
                                                         i,
                                                         q);
              }
            else
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "This vector-source weak-term pairing is not "
                              "implemented."));
                return 0.;
              }
          }
      }

      template <typename Source, typename Target>
      static void
      assert_cell_pair(const Source &source_cell, const Target &target_cell)
      {
        AssertThrow(source_cell->id() == target_cell->id(),
                    dealii::ExcMessage(
                      "Source and target active-cell traversals disagree."));
      }

      template <typename Source, typename Target, typename Callback>
      static void
      for_each_cell(const Source &source, const Target &target, Callback &&cb)
      {
        auto source_cell = source.dof_handler().begin_active();
        auto source_end  = source.dof_handler().end();
        auto target_cell = target.dof_handler().begin_active();
        auto target_end  = target.dof_handler().end();
        for (; source_cell != source_end && target_cell != target_end;
             ++source_cell, ++target_cell)
          {
            assert_cell_pair(source_cell, target_cell);
            cb(source_cell, target_cell);
          }
        AssertThrow(source_cell == source_end && target_cell == target_end,
                    dealii::ExcMessage(
                      "Source and target spaces have different active-cell "
                      "counts."));
      }

      template <typename Target, typename Callback>
      static void
      for_each_same_cell(const Target &target, Callback &&cb)
      {
        for (auto cell = target.dof_handler().begin_active();
             cell != target.dof_handler().end();
             ++cell)
          cb(cell);
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
                    const TargetField             &target,
                    const TargetValues            &target_values,
                    const SourceCell              &source_cell,
                    const TargetCell              &target_cell,
                    const dealii::Quadrature<dim> &quadrature,
                    MatrixType                    &matrix)
      {
        std::vector<dealii::types::global_dof_index> source_indices(
          source_cell->get_fe().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> target_indices(
          target_cell->get_fe().n_dofs_per_cell());
        source_cell->get_dof_indices(source_indices);
        target_cell->get_dof_indices(target_indices);

        dealii::FullMatrix<double> local(target_indices.size(),
                                         source_indices.size());
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          for (unsigned int i = 0; i < target_indices.size(); ++i)
            for (unsigned int j = 0; j < source_indices.size(); ++j)
              local(i, j) += contract(observable,
                                      source,
                                      source_values,
                                      target,
                                      target_values,
                                      i,
                                      j,
                                      q) *
                             target_values.JxW(q);

        target.constraints().distribute_local_to_global(
          local, target_indices, source.constraints(), source_indices, matrix);
      }

      template <typename SourceField,
                typename TargetFieldType,
                typename SourceValues,
                typename TargetValues>
      static double
      contract(const ObservableType  &observable,
               const SourceField     &source,
               const SourceValues    &source_values,
               const TargetFieldType &target,
               const TargetValues    &target_values,
               const unsigned int     i,
               const unsigned int     j,
               const unsigned int     q)
      {
        const auto scale = observable.scale();
        if constexpr (std::is_same_v<typename SourceField::value_type, double>)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              {
                const auto test =
                  scalar_shape_value(target_values, target.extractor(), i, q);
                const auto trial =
                  scalar_shape_value(source_values, source.extractor(), j, q);
                return scale * test * trial;
              }
            else if constexpr (std::is_same_v<
                                 typename ObservableType::value_type,
                                 dealii::Tensor<1, spacedim>>)
              {
                const auto trial = scalar_shape_gradient(source_values,
                                                         source.extractor(),
                                                         j,
                                                         q);
                const auto test =
                  vector_shape_value(target_values, target.extractor(), i, q);
                return scale * (trial * test);
              }
            else
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "This scalar-source weak-term pairing is not "
                              "implemented."));
                return 0.;
              }
          }
        else
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              {
                const auto trial =
                  vector_shape_value(source_values, source.extractor(), j, q);
                const auto test =
                  vector_shape_value(target_values, target.extractor(), i, q);
                return scale * (trial * test);
              }
            else
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "This vector-source weak-term pairing is not "
                              "implemented."));
                return 0.;
              }
          }
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
                const auto component = target.space()
                                         .finite_element()
                                         .system_to_component_index(i)
                                         .first;
                (*entries)[particle.get_id()].emplace_back(
                  dof_indices[i],
                  stencil.physical_weight *
                    (component < TargetField::spacedimension() ?
                       (normal * detail::vector_shape_value(
                                   target_values, target.extractor(), i, 0)) :
                       0.));
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
  template <typename ObservableType, typename TargetField>
  class WeakTerm
  {
  public:
    using target_type = TargetField;

    WeakTerm(ObservableType observable, TargetField target)
      : observable_(std::move(observable))
      , target_(std::move(target))
    {}

    template <typename VectorType, typename MatrixType>
    FieldId
    add(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      const auto pairing   = make_pairing(builder, target_);
      const auto source_id = pairing.source_id;
      const auto target_id = pairing.target_id;

      using Model = SemiDiscreteModel<VectorType, MatrixType>;
      auto term   = builder.term(target_id, "weak_term");
      if (observable_.is_frozen())
        {
          const auto frozen = observable_.template frozen_values<VectorType>();
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
      const auto multiplier_field = target_.with_id(multiplier);
      const auto pairing          = make_pairing(builder, multiplier_field);
      const auto reaction =
        ImmersX::transpose_operator(pairing.operator_with_matrix);
      const auto source_id = pairing.source_id;

      const auto suffix = "constraint." + std::to_string(index);
      builder.term(multiplier, suffix + ".multiplier")
        .residual([pairing, source_id, sign](const auto &context) {
          return sign *
                 (pairing.operator_with_matrix.view * context.state(source_id));
        })
        .state(source_id, sign * pairing.operator_with_matrix);

      builder.term(source_id, suffix + ".participant")
        .residual([reaction, multiplier, sign](const auto &context) {
          return sign * (reaction.view * context.state(multiplier));
        })
        .state(multiplier, sign * reaction);
    }

    template <typename VectorType, typename MatrixType>
    FieldId
    operator()(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      return add(builder);
    }

    const ObservableType &
    observable() const
    {
      return observable_;
    }

    const TargetField &
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
      using Assembly = detail::WeakAssembly<ObservableType, Target>;
      Pairing<VectorType, MatrixType> result;
      const bool                      nonmatching_geometry =
        Assembly::geometry_is_nonmatching(observable_, target);
      if (nonmatching_geometry)
        {
          const auto prepared =
            Assembly::template prepare<VectorType, MatrixType>(observable_,
                                                               target);
          result.matrix   = prepared->storage.matrix;
          result.sparsity = prepared->storage.sparsity;
          result.operator_with_matrix =
            Assembly::template dynamic_matrix_operator<VectorType, MatrixType>(
              prepared);
        }
      else
        {
          const auto storage =
            Assembly::template assemble<VectorType, MatrixType>(observable_,
                                                                target);
          result.matrix               = storage.matrix;
          result.sparsity             = storage.sparsity;
          result.operator_with_matrix = builder.matrix_operator(*result.matrix);
        }
      result.source_id = observable_.source_field();
      result.target_id = target.field_id();
      return result;
    }

    ObservableType observable_;
    TargetField    target_;
  };

  /** \cond deduction_guide */
  template <typename ObservableType, typename TargetField>
  WeakTerm(ObservableType, TargetField)
    -> WeakTerm<ObservableType, TargetField>;
  /** \endcond */

  /** Create a residual term representing the FE duality pairing. */
  template <typename ObservableType, typename TargetField>
  auto
  weak_term(ObservableType observable, TargetField target)
  {
    return WeakTerm<ObservableType, TargetField>(std::move(observable),
                                                 std::move(target));
  }

  template <typename Quantity, typename Surface, typename TargetField>
  auto
  weak_term(NormalQuantity<Quantity, Surface> load, TargetField target)
  {
    return NormalWeakTerm<Quantity, Surface, TargetField>(std::move(load),
                                                          std::move(target));
  }
} // namespace ImmersX

#endif // immersx_weak_term_h
