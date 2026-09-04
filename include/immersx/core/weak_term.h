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

      using ScalarField =
        Field<dim, spacedim, double, dealii::FEValuesExtractors::Scalar>;
      using VectorField = Field<dim,
                                spacedim,
                                dealii::Tensor<1, spacedim>,
                                dealii::FEValuesExtractors::Vector>;

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
        AssertThrow(observable.source_field().is_valid(),
                    dealii::ExcMessage(
                      "A weak term source field must be registered."));
        AssertThrow(observable.dimension() == dim &&
                      observable.spacedimension() == spacedim,
                    dealii::ExcMessage(
                      "Weak-term source and target dimensions do not "
                      "match."));

        if (observable.operation() == ObservableOperation::value)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return assemble_from_source<ScalarField, VectorType, MatrixType>(
                observable, observable.template source<ScalarField>(), target);
            else
              return assemble_from_source<VectorField, VectorType, MatrixType>(
                observable, observable.template source<VectorField>(), target);
          }
        else
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              return assemble_from_source<ScalarField, VectorType, MatrixType>(
                observable, observable.template source<ScalarField>(), target);
            else
              return assemble_from_source<VectorField, VectorType, MatrixType>(
                observable, observable.template source<VectorField>(), target);
          }
      }

      static bool
      geometry_is_nonmatching(const ObservableType &observable,
                              const TargetField    &target)
      {
        if (observable.operation() == ObservableOperation::value)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return &observable.template source<ScalarField>()
                        .dof_handler()
                        .get_triangulation() !=
                     &target.dof_handler().get_triangulation();
            else
              return &observable.template source<VectorField>()
                        .dof_handler()
                        .get_triangulation() !=
                     &target.dof_handler().get_triangulation();
          }
        else
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              return &observable.template source<ScalarField>()
                        .dof_handler()
                        .get_triangulation() !=
                     &target.dof_handler().get_triangulation();
            else
              return &observable.template source<VectorField>()
                        .dof_handler()
                        .get_triangulation() !=
                     &target.dof_handler().get_triangulation();
          }
      }

      template <typename VectorType, typename MatrixType>
      static std::shared_ptr<PreparedMatrix<VectorType, MatrixType>>
      prepare(const ObservableType &observable, const TargetField &target)
      {
        if (observable.operation() == ObservableOperation::value)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return prepare_from_source<ScalarField, VectorType, MatrixType>(
                observable, observable.template source<ScalarField>(), target);
            else
              return prepare_from_source<VectorField, VectorType, MatrixType>(
                observable, observable.template source<VectorField>(), target);
          }
        else
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         dealii::Tensor<1, spacedim>>)
              return prepare_from_source<ScalarField, VectorType, MatrixType>(
                observable, observable.template source<ScalarField>(), target);
            else
              return prepare_from_source<VectorField, VectorType, MatrixType>(
                observable, observable.template source<VectorField>(), target);
          }
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
        const auto degree = std::max(source.space().finite_element().degree,
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

        dealii::DynamicSparsityPattern sparsity(target.dof_handler().n_dofs(),
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
          std::is_same_v<SourceField, ScalarField> ?
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
                    if constexpr (std::is_same_v<SourceField, ScalarField>)
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
        if constexpr (std::is_same_v<SourceField, ScalarField>)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              return scalar_shape_value(target_values,
                                        target.extractor(),
                                        i,
                                        q) *
                     stencil.source_basis_values[j];
            else if constexpr (std::is_same_v<TargetFieldType, VectorField>)
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
            if constexpr (std::is_same_v<TargetFieldType, VectorField>)
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
        (void)observable;
        if constexpr (std::is_same_v<SourceField, ScalarField>)
          {
            if constexpr (std::is_same_v<typename ObservableType::value_type,
                                         double>)
              {
                const auto test =
                  scalar_shape_value(target_values, target.extractor(), i, q);
                const auto trial =
                  scalar_shape_value(source_values, source.extractor(), j, q);
                return test * trial;
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
                return trial * test;
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
                return trial * test;
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

  /** A solver-neutral FE weak term contributed to a residual row. */
  template <typename ObservableType, typename TargetField>
  class WeakTerm
  {
  public:
    WeakTerm(ObservableType observable, TargetField target)
      : observable_(std::move(observable))
      , target_(std::move(target))
    {}

    template <typename VectorType, typename MatrixType>
    FieldId
    add(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      using Assembly = detail::WeakAssembly<ObservableType, TargetField>;
      const bool nonmatching_geometry =
        Assembly::geometry_is_nonmatching(observable_, target_);
      std::shared_ptr<
        typename Assembly::template PreparedMatrix<VectorType, MatrixType>>
                                                            prepared;
      typename Assembly::template MatrixStorage<MatrixType> matrix_storage;
      typename SemiDiscreteModel<VectorType, MatrixType>::MatrixOperator
        operator_with_matrix;
      if (nonmatching_geometry)
        {
          prepared =
            Assembly::template prepare<VectorType, MatrixType>(observable_,
                                                               target_);
          matrix_storage.matrix   = prepared->storage.matrix;
          matrix_storage.sparsity = prepared->storage.sparsity;
          operator_with_matrix =
            Assembly::template dynamic_matrix_operator<VectorType, MatrixType>(
              prepared);
        }
      else
        {
          matrix_storage =
            Assembly::template assemble<VectorType, MatrixType>(observable_,
                                                                target_);
          operator_with_matrix =
            builder.matrix_operator(*matrix_storage.matrix);
        }
      const auto matrix    = matrix_storage.matrix;
      const auto source_id = observable_.source_field();
      const auto target_id = target_.field_id();

      using Model = SemiDiscreteModel<VectorType, MatrixType>;
      typename Model::MatrixOperatorFactory state_factory =
        [matrix,
         matrix_sparsity = matrix_storage.sparsity,
         operator_with_matrix](const typename Model::Context &) {
          (void)matrix;
          (void)matrix_sparsity;
          return operator_with_matrix;
        };
      builder.term(target_id, "weak_term")
        .residual([matrix,
                   matrix_sparsity = matrix_storage.sparsity,
                   operator_with_matrix,
                   source_id](const auto &ctx) {
          (void)matrix_sparsity;
          return operator_with_matrix.view * ctx.state(source_id);
        })
        .state(source_id, std::move(state_factory));
      return target_id;
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
} // namespace ImmersX

#endif // immersx_weak_term_h
