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

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

// ScratchData includes Sacado, whose Handle type conflicts with OpenCASCADE's
// macro when a geometry header has already been included.
#pragma push_macro("Handle")
#undef Handle
#include <deal.II/meshworker/scratch_data.h>
#pragma pop_macro("Handle")

#include <deal.II/non_matching/coupling.h>

#include <boost/signals2/connection.hpp>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/contributor.h>
#include <immersx/core/integration_support.h>
#include <immersx/core/observable.h>
#include <immersx/coupling/particle_coupling.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  template <typename SourceObservable,
            int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components>
  class LiftedObservable;

  namespace detail
  {
    template <typename Trial>
    struct has_linear_cached_trial : is_linear_observable<Trial>
    {};

    template <typename Source,
              int dim,
              int surface_dim,
              int spacedim,
              int components>
    struct has_linear_cached_trial<
      LiftedObservable<Source, dim, surface_dim, spacedim, components>>
      : is_linear_observable<Source>
    {};

    inline void
    report_retained_quadrature_points_once(const std::uint64_t retained,
                                           const std::uint64_t total,
                                           const MPI_Comm      communicator)
    {
      // Only rank zero prints. No MPI collective is performed here.
      if (dealii::Utilities::MPI::this_mpi_process(communicator) != 0)
        return;

      // Ignore empty supports without consuming the one-time report.
      if (total == 0)
        return;

      // Shared across calls, including different weak-term template types.
      static std::atomic<bool> reported{false};

      if (reported.exchange(true))
        return;

      std::clog << "Quadrature points retained: " << retained << " / " << total
                << " (" << total - retained << " skipped)." << std::endl;
    }
    template <int dim, typename Support>
    dealii::Quadrature<dim>
    support_quadrature(const Support                 &support,
                       const dealii::Quadrature<dim> &fallback)
    {
      if constexpr (is_integration_support<Support>::value)
        {
          if constexpr (Support::dimension() == dim)
            return support.quadrature(fallback);
        }
      return fallback;
    }

    template <typename Support, typename Cell>
    double
    support_point_weight(const Support     &support,
                         const Cell        &cell,
                         const unsigned int q)
    {
      if constexpr (is_integration_support<Support>::value)
        return support.point_weight(cell, q);
      return 1.;
    }

    template <typename Support, typename Field>
    void
    assert_lifted_support_matches(const Support &support, const Field &field)
    {
      if constexpr (is_integration_support<Support>::value)
        support.assert_matches(field.dof_handler().get_triangulation());
      else
        AssertThrow(
          (std::is_same_v<Support, InferredIntegrationSupport>),
          dealii::ExcMessage(
            "A lifted weak term requires its representative cell support."));
    }

    template <typename Support, typename Point>
    bool
    support_lifted_point(const Support &support, const Point &point)
    {
      if constexpr (is_integration_support<Support>::value)
        return support.includes_lifted_point(point);
      return true;
    }

#ifdef IMMERSX_WEAK_TERM_TESTING
    inline std::atomic<unsigned int> weak_term_nonmatching_preparations = 0;
    inline std::atomic<unsigned int> weak_term_matrix_initializations   = 0;
    inline std::atomic<unsigned int> weak_term_affine_cells             = 0;
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
#ifdef IMMERSX_WEAK_TERM_TESTING
      ++weak_term_matrix_initializations;
#endif
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

    template <typename Support, typename = void>
    struct has_support_triangulation : std::false_type
    {};

    template <typename Support>
    struct has_support_triangulation<
      Support,
      std::void_t<decltype(std::declval<const Support &>().triangulation())>>
      : std::true_type
    {};

    template <typename Support, typename = void>
    struct support_triangulation_dimension
      : std::integral_constant<int, Support::intrinsic_dimension>
    {};

    template <typename Support>
    struct support_triangulation_dimension<
      Support,
      std::void_t<decltype(Support::representative_dimension)>>
      : std::integral_constant<int, Support::representative_dimension>
    {};

    template <typename VectorType, typename = void>
    struct has_index_set_vector_reinit : std::false_type
    {};

    template <typename VectorType>
    struct has_index_set_vector_reinit<
      VectorType,
      std::void_t<decltype(std::declval<VectorType &>().reinit(
        std::declval<const dealii::IndexSet &>(),
        std::declval<MPI_Comm>()))>> : std::true_type
    {};

    template <typename VectorType>
    bool
    weak_vector_is_zero(const VectorType &vector, const MPI_Comm communicator)
    {
      const auto norm = vector.linfty_norm();
      if constexpr (has_index_set_vector_reinit<VectorType>::value)
        return norm == 0.;
      else
        return dealii::Utilities::MPI::max(norm, communicator) == 0.;
    }

    template <typename VectorType>
    void
    initialize_weak_vector(VectorType             &vector,
                           const dealii::IndexSet &locally_owned,
                           const MPI_Comm          communicator)
    {
      if constexpr (has_index_set_vector_reinit<VectorType>::value)
        vector.reinit(locally_owned, communicator);
      else
        vector.reinit(locally_owned.size());
    }

    template <typename VectorType, typename = void>
    struct has_vector_compress : std::false_type
    {};

    template <typename VectorType>
    struct has_vector_compress<
      VectorType,
      std::void_t<decltype(std::declval<VectorType &>().compress(
        dealii::VectorOperation::add))>> : std::true_type
    {};

    template <typename VectorType>
    void
    compress_weak_vector(VectorType &vector)
    {
      if constexpr (has_vector_compress<VectorType>::value)
        vector.compress(dealii::VectorOperation::add);
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

    template <typename FieldType>
    void
    assert_homogeneous_weak_constraints(const FieldType   &field,
                                        const std::string &message)
    {
      const auto inhomogeneous = dealii::Utilities::MPI::max(
        static_cast<unsigned int>(field.constraints().has_inhomogeneities()),
        field.space().mpi_communicator());
      AssertThrow(inhomogeneous == 0, dealii::ExcMessage(message));
    }

    template <typename VectorType, typename MatrixType>
    struct WeakAffineAssembly
    {
      const dealii::AffineConstraints<double> *source_constraints = nullptr;
      const dealii::AffineConstraints<double> *target_constraints = nullptr;
      dealii::IndexSet                         locally_owned;
      MPI_Comm                                 communicator;
      VectorType                               values;

      WeakAffineAssembly(const dealii::AffineConstraints<double> &source,
                         const dealii::AffineConstraints<double> &target,
                         const dealii::IndexSet                  &owned,
                         const MPI_Comm                           communicator)
        : source_constraints(&source)
        , target_constraints(&target)
        , locally_owned(owned)
        , communicator(communicator)
      {
        // Initialize collectively before local cell/particle loops. Ranks
        // without contributions must enter collectives in the same order.
        initialize_weak_vector(values, locally_owned, communicator);
        values = 0.;
      }

      void
      add(const dealii::FullMatrix<double>                   &local,
          const std::vector<dealii::types::global_dof_index> &source_indices,
          const std::vector<dealii::types::global_dof_index> &target_indices,
          const bool                                          square = false)
      {
        auto                  &rhs = values;
        dealii::Vector<double> local_rhs(local.m());
        if (square && local.m() != 0)
          {
            source_constraints->distribute_local_to_global(local_rhs,
                                                           target_indices,
                                                           rhs,
                                                           local);
            // Match deal.II's constrained diagonal equations without building
            // a second global matrix just to assemble their right-hand side.
            double average = 0.;
            for (unsigned int i = 0; i < local.m(); ++i)
              average += std::abs(local(i, i));
            average /= local.m();
            if (average == 0.)
              average = local.l1_norm() / local.m();
            if (average == 0.)
              average = 1.;
            for (unsigned int i = 0; i < target_indices.size(); ++i)
              if (source_constraints->is_constrained(target_indices[i]))
                rhs[target_indices[i]] +=
                  (local(i, i) != 0. ? local(i, i) : average) *
                  source_constraints->get_inhomogeneity(target_indices[i]);
          }
        else
          {
            for (unsigned int i = 0; i < local.m(); ++i)
              for (unsigned int j = 0; j < local.n(); ++j)
                local_rhs[i] -=
                  local(i, j) *
                  source_constraints->get_inhomogeneity(source_indices[j]);
            target_constraints->distribute_local_to_global(local_rhs,
                                                           target_indices,
                                                           rhs);
          }
      }

      void
      finish(VectorType &rhs)
      {
        compress_weak_vector(values);
        initialize_weak_vector(rhs, locally_owned, communicator);
        rhs = values;
      }
    };

    template <typename ObservableType, typename TargetExpression>
    struct WeakAssembly
    {
      static constexpr int dim =
        TargetExpression::source_field_type::dimension();
      static constexpr int spacedim =
        TargetExpression::source_field_type::spacedimension();

      using SourceField = typename ObservableType::source_field_type;

      using TargetField = typename TargetExpression::source_field_type;

      template <typename VectorType, typename MatrixType>
      using AffineData = WeakAffineAssembly<VectorType, MatrixType>;

      struct ScratchData : dealii::MeshWorker::ScratchData<dim, spacedim>
      {
        using dealii::MeshWorker::ScratchData<dim, spacedim>::ScratchData;

        template <typename TrialField>
        void
        prepare(
          const TrialField                                   &source,
          const std::vector<dealii::types::global_dof_index> &source_indices,
          const TargetField                                  &target,
          const std::vector<dealii::types::global_dof_index> &target_indices)
        {
          source_positions.clear();
          target_positions.clear();
          source_execution_indices.clear();
          target_execution_indices.clear();
          source_positions.reserve(source_indices.size());
          target_positions.reserve(target_indices.size());
          source_execution_indices.reserve(source_indices.size());
          target_execution_indices.reserve(target_indices.size());
          for (unsigned int j = 0; j < source_indices.size(); ++j)
            {
              const auto index = source.execution_index(source_indices[j]);
              if (index != dealii::numbers::invalid_dof_index)
                {
                  source_positions.push_back(j);
                  source_execution_indices.push_back(index);
                }
            }
          for (unsigned int i = 0; i < target_indices.size(); ++i)
            {
              const auto index = target.execution_index(target_indices[i]);
              if (index != dealii::numbers::invalid_dof_index)
                {
                  target_positions.push_back(i);
                  target_execution_indices.push_back(index);
                }
            }
          local_matrix.reinit(target_positions.size(), source_positions.size());
          trial_values.resize(source_positions.size());
          test_values.resize(target_positions.size());
        }

        std::vector<unsigned int>                    source_positions;
        std::vector<unsigned int>                    target_positions;
        std::vector<dealii::types::global_dof_index> source_execution_indices;
        std::vector<dealii::types::global_dof_index> target_execution_indices;
        dealii::FullMatrix<double>                   local_matrix;
        std::vector<typename ObservableType::value_type>   trial_values;
        std::vector<typename TargetExpression::value_type> test_values;
      };

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

      /** Reintegrate only cells whose trial coefficients have affine offsets.
       * The matrix and the homogeneous constraint graph are not rebuilt. */
      template <typename VectorType, typename MatrixType, typename Support>
      static std::shared_ptr<VectorType>
      affine_rhs(const ObservableType                &observable,
                 const TargetExpression              &target,
                 const Support                       &support,
                 bool                                 boundary,
                 std::set<dealii::types::boundary_id> boundary_ids)
      {
        const auto                        &source = observable.source();
        const auto                         row    = target.source();
        AffineData<VectorType, MatrixType> data(source.constraints(),
                                                row.constraints(),
                                                row.locally_owned_dofs(),
                                                row.space().mpi_communicator());
        if constexpr (std::is_same_v<Support,
                                     BoundaryIntegrationSupport<dim, spacedim>>)
          {
            boundary     = true;
            boundary_ids = support.boundary_ids();
          }
        const auto degree = std::max(source.space().finite_element().degree,
                                     row.space().finite_element().degree);
        const auto quadrature =
          support_quadrature(support,
                             compatibility::gauss_quadrature<dim>(
                               row.space().finite_element().reference_cell(),
                               degree + 1));
        const auto face_quadrature = compatibility::gauss_quadrature<dim - 1>(
          row.space().finite_element().reference_cell().face_reference_cell(0),
          degree + 1);
        const auto flags = dealii::update_JxW_values |
                           observable.update_flags() | target.update_flags();
        const auto volume_flags = static_cast<dealii::UpdateFlags>(
          flags & ~dealii::update_normal_vectors);
        ScratchData source_scratch(source.mapping(),
                                   source.space().finite_element(),
                                   quadrature,
                                   volume_flags,
                                   face_quadrature,
                                   flags | dealii::update_normal_vectors);
        ScratchData target_scratch(row.mapping(),
                                   row.space().finite_element(),
                                   quadrature,
                                   volume_flags,
                                   face_quadrature,
                                   flags | dealii::update_normal_vectors);
        const bool  shared_values =
          &source.dof_handler() == &row.dof_handler() &&
          &source.mapping() == &row.mapping();
        std::vector<dealii::types::global_dof_index> indices(
          source.space().finite_element().n_dofs_per_cell());
        for (const auto &cell : source.dof_handler().active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;
            if constexpr (!std::is_same_v<
                            Support,
                            BoundaryIntegrationSupport<dim, spacedim>>)
              if (!support_includes(support, cell))
                continue;
            cell->get_dof_indices(indices);
            bool has_offset = false;
            for (const auto native : indices)
              if (const auto i = source.execution_index(native);
                  i != dealii::numbers::invalid_dof_index &&
                  source.constraints().get_inhomogeneity(i) != 0.)
                {
                  has_offset = true;
                  break;
                }
            if (!has_offset)
              continue;
#ifdef IMMERSX_WEAK_TERM_TESTING
            ++weak_term_affine_cells;
#endif
            const auto target_cell =
              cell->as_dof_handler_iterator(row.dof_handler());
            if (boundary)
              {
                for (const auto f : cell->face_indices())
                  if (cell->face(f)->at_boundary() &&
                      (boundary_ids.empty() ||
                       boundary_ids.count(cell->face(f)->boundary_id())))
                    {
                      const auto &source_face = source_scratch.reinit(cell, f);
                      const auto &target_face =
                        shared_values ? source_face :
                                        target_scratch.reinit(target_cell, f);
                      assemble_face_cell<VectorType>(
                        observable,
                        source,
                        source_face,
                        target,
                        target_face,
                        source_scratch.get_local_dof_indices(),
                        shared_values ? source_scratch.get_local_dof_indices() :
                                        target_scratch.get_local_dof_indices(),
                        source_scratch,
                        static_cast<MatrixType *>(nullptr),
                        &data);
                    }
              }
            else
              {
                const auto &source_values = source_scratch.reinit(cell);
                const auto &target_values =
                  shared_values ? source_values :
                                  target_scratch.reinit(target_cell);
                assemble_cell<VectorType>(
                  observable,
                  source,
                  source_values,
                  target,
                  target_values,
                  source_scratch.get_local_dof_indices(),
                  shared_values ? source_scratch.get_local_dof_indices() :
                                  target_scratch.get_local_dof_indices(),
                  target_cell,
                  source_scratch,
                  static_cast<MatrixType *>(nullptr),
                  &data,
                  nullptr,
                  support);
              }
          }
        auto rhs = std::make_shared<VectorType>();
        data.finish(*rhs);
        return rhs;
      }

      template <typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble(const ObservableType        &observable,
               const TargetExpression      &target,
               std::shared_ptr<VectorType> *affine_rhs = nullptr,
               std::shared_ptr<AffineData<VectorType, MatrixType>>
                 *affine_data = nullptr)
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
                      (SourceField::dimension() >= dim ||
                       (SourceField::dimension() < dim && dim == spacedim)),
                    dealii::ExcMessage(
                      "Weak-term source and target must share an embedding "
                      "dimension and use a supported direct pairing."));

        return assemble_from_source<SourceField, VectorType, MatrixType>(
          observable, observable.source(), target, affine_rhs, affine_data);
      }

      template <typename VectorType, typename MatrixType, int support_dim>
      static MatrixStorage<MatrixType>
      assemble_on_support(
        const ObservableType                            &observable,
        const TargetExpression                          &target,
        const IntegrationSupport<support_dim, spacedim> &support,
        std::shared_ptr<VectorType>                     *affine_rhs = nullptr,
        std::shared_ptr<AffineData<VectorType, MatrixType>> *affine_data =
          nullptr)
      {
        const auto &source       = observable.source();
        const auto &target_field = target.source();
        const auto &support_tria = support.triangulation();
        const bool  matches_source =
          static_cast<const void *>(&support_tria) ==
          static_cast<const void *>(&source.dof_handler().get_triangulation());
        const bool matches_target =
          static_cast<const void *>(&support_tria) ==
          static_cast<const void *>(
            &target_field.dof_handler().get_triangulation());
        if constexpr (support_dim <= dim && SourceField::dimension() == dim &&
                      dim == spacedim)
          if (!matches_source && !matches_target)
            return assemble_background_trace<VectorType, MatrixType>(
              observable, target, support, affine_rhs, affine_data);
        AssertThrow(matches_source || matches_target,
                    dealii::ExcMessage(
                      "An explicit support must use the source or test "
                      "field triangulation."));
        if constexpr (SourceField::dimension() < dim)
          AssertThrow(matches_source,
                      dealii::ExcMessage(
                        "A lower-dimensional trial field defines the "
                        "integration support."));
        else if constexpr (SourceField::dimension() > dim)
          AssertThrow(matches_target,
                      dealii::ExcMessage(
                        "A lower-dimensional test field defines the "
                        "integration support."));
        return assemble_from_source<SourceField, VectorType, MatrixType>(
          observable, source, target, affine_rhs, affine_data, support);
      }

      template <typename VectorType,
                typename MatrixType,
                typename Support,
                std::enable_if_t<is_physical_quadrature_support<Support>::value,
                                 int> = 0>
      static MatrixStorage<MatrixType>
      assemble_on_support(const ObservableType        &observable,
                          const TargetExpression      &target,
                          const Support               &support,
                          std::shared_ptr<VectorType> *affine_rhs = nullptr,
                          std::shared_ptr<AffineData<VectorType, MatrixType>>
                            *affine_data = nullptr)
      {
        static_assert(Support::intrinsic_dimension > 0 &&
                        Support::intrinsic_dimension <= dim,
                      "Physical integration support must have dimension "
                      "between one and the background dimension.");
        static_assert(Support::ambient_dimension == spacedim,
                      "Integration support and weak-term fields must share "
                      "an embedding dimension.");
        return assemble_background_trace<VectorType, MatrixType>(
          observable, target, support, affine_rhs, affine_data);
      }

      /** Assemble a background-to-background pairing at physical points
       * generated by an independent integration mesh.
       *
       * The support owns the integration geometry. Its quadrature points are
       * located in the background mesh through deal.II's GridTools cache, and
       * FEValues evaluates both background bases at the resulting reference
       * points. This initial path requires the trial and test spaces to share
       * one background triangulation.
       */
      template <typename VectorType, typename MatrixType, typename Support>
      static MatrixStorage<MatrixType>
      assemble_background_trace(
        const ObservableType        &observable,
        const TargetExpression      &target,
        const Support               &support,
        std::shared_ptr<VectorType> *affine_rhs = nullptr,
        std::shared_ptr<AffineData<VectorType, MatrixType>> *affine_data =
          nullptr)
      {
        constexpr int support_dim = [] {
          if constexpr (is_physical_quadrature_support<Support>::value)
            return Support::intrinsic_dimension;
          else
            return Support::dimension();
        }();
        static_assert(
          support_dim > 0 && support_dim <= dim,
          "Independent integration support dimension must not exceed the "
          "background dimension.");
        const auto &source       = observable.source();
        const auto  target_field = target.source();
        AssertThrow(
          static_cast<const void *>(
            &source.dof_handler().get_triangulation()) ==
            static_cast<const void *>(
              &target_field.dof_handler().get_triangulation()),
          dealii::ExcMessage(
            "Independent trace integration currently requires trial and test "
            "fields on the same background triangulation."));
        AssertThrow(&source.mapping() == &target_field.mapping(),
                    dealii::ExcMessage(
                      "Independent trace integration requires trial and test "
                      "fields to use the same background mapping."));

        struct CellQuadrature
        {
          typename SourceField::space_type::DoFHandlerType::active_cell_iterator
                                          source_cell;
          std::vector<dealii::Point<dim>> reference_points;
          std::vector<double>             physical_weights;
        };

        using CellIterator = typename SourceField::space_type::DoFHandlerType::
          active_cell_iterator;
        std::map<dealii::CellId, CellQuadrature> cell_quadratures;
        std::uint64_t local_quadrature_point_count = 0;
        const auto    append_point = [&](const CellIterator       &source_cell,
                                      const dealii::Point<dim> &point,
                                      const double              weight) {
          auto &cell_quadrature       = cell_quadratures[source_cell->id()];
          cell_quadrature.source_cell = source_cell;
          cell_quadrature.reference_points.push_back(point);
          cell_quadrature.physical_weights.push_back(weight);
          ++local_quadrature_point_count;
        };
        std::vector<dealii::Point<spacedim>> positions;
        std::vector<std::vector<double>>     properties;
        const auto                          *background_tria = dynamic_cast<
          const dealii::parallel::TriangulationBase<dim, spacedim> *>(
          &source.dof_handler().get_triangulation());
        if constexpr (is_physical_quadrature_support<Support>::value)
          {
            AssertThrow(Support::ambient_dimension == spacedim,
                        dealii::ExcMessage(
                          "Physical support and background must share an "
                          "embedding dimension."));
            int communicator_relation = MPI_UNEQUAL;
            MPI_Comm_compare(support.mpi_communicator(),
                             target_field.space().mpi_communicator(),
                             &communicator_relation);
            int support_processes    = 0;
            int background_processes = 0;
            MPI_Comm_size(support.mpi_communicator(), &support_processes);
            MPI_Comm_size(target_field.space().mpi_communicator(),
                          &background_processes);
            AssertThrow(communicator_relation == MPI_IDENT ||
                          communicator_relation == MPI_CONGRUENT ||
                          (support_processes == 1 && background_processes == 1),
                        dealii::ExcMessage(
                          "Physical support and background must use "
                          "congruent MPI communicators or be serial."));
            const auto support_rank = dealii::Utilities::MPI::this_mpi_process(
              support.mpi_communicator());
            bool replicated_support     = false;
            bool support_is_distributed = false;
            if constexpr (has_support_triangulation<Support>::value)
              {
                const auto *distributed_support =
                  dynamic_cast<const dealii::parallel::TriangulationBase<
                    support_triangulation_dimension<Support>::value,
                    Support::ambient_dimension> *>(&support.triangulation());
                support_is_distributed = distributed_support != nullptr;
                replicated_support =
                  background_tria != nullptr && !support_is_distributed &&
                  dealii::Utilities::MPI::this_mpi_process(
                    target_field.space().mpi_communicator()) != 0;
              }
            const auto &physical_points =
              support.locally_owned_quadrature_points();
            positions.reserve(physical_points.size());
            properties.reserve(physical_points.size());
            for (const auto &point : physical_points)
              {
                AssertThrow(point.owner_rank == support_rank,
                            dealii::ExcMessage(
                              "Physical support points must be locally "
                              "owned."));
                if (replicated_support)
                  continue;
                positions.push_back(point.point);
                properties.push_back({point.weight});
              }
          }
        else
          {
            const unsigned int degree =
              std::max(source.space().finite_element().degree,
                       target_field.space().finite_element().degree);
            const auto physical_reference =
              detail::uniform_reference_cell(support.triangulation());
            AssertThrow(physical_reference.has_value(),
                        dealii::ExcMessage(
                          "The integration support mesh has no cells."));
            const auto reference_cell     = *physical_reference;
            const auto support_quadrature = support.quadrature(
              compatibility::gauss_quadrature<support_dim>(reference_cell,
                                                           degree + 2));
            const dealii::FE_Nothing<support_dim, spacedim> geometry_element(
              reference_cell);
            dealii::FEValues<support_dim, spacedim> support_values(
              compatibility::linear_mapping<support_dim, spacedim>(
                reference_cell),
              geometry_element,
              support_quadrature,
              dealii::update_quadrature_points | dealii::update_JxW_values);
            const auto *distributed_support = dynamic_cast<
              const dealii::parallel::TriangulationBase<support_dim, spacedim>
                *>(&support.triangulation());
            const bool replicated_support =
              background_tria != nullptr && distributed_support == nullptr &&
              dealii::Utilities::MPI::this_mpi_process(
                target_field.space().mpi_communicator()) != 0;
            positions.reserve(support.triangulation().n_active_cells() *
                              support_quadrature.size());
            properties.reserve(positions.capacity());
            for (const auto &support_cell :
                 support.triangulation().active_cell_iterators())
              if (!replicated_support && support_cell->is_locally_owned() &&
                  support.includes_cell(support_cell))
                {
                  support_values.reinit(support_cell);
                  for (const auto q : support_values.quadrature_point_indices())
                    {
                      const double selection =
                        support.point_weight(support_cell, q);
                      if (selection == 0.)
                        continue;
                      positions.push_back(support_values.quadrature_point(q));
                      properties.push_back({support_values.JxW(q) * selection});
                    }
                }
          }

        if constexpr (dim == spacedim)
          {
            if (background_tria != nullptr)
              {
                ParticleCouplingParameters<spacedim> parameters(
                  "/ImmersX/weak term/trace/" +
                  std::to_string(reinterpret_cast<std::uintptr_t>(&support)));
                ParticleCoupling<spacedim> distribution(parameters);
                distribution.initialize_particle_handler(*background_tria,
                                                         source.mapping(),
                                                         1);
                // ParticleHandler's optional ID exchange is collective but
                // branches on local ID-vector emptiness. Generated IDs avoid
                // divergent collectives when a rank has no support points.
                const auto receiving_ranks =
                  distribution.insert_points(positions, properties);
                std::size_t routed_points = 0;
                for (const auto &[rank, indices] : receiving_ranks)
                  {
                    (void)rank;
                    routed_points += indices.n_elements();
                  }
                const auto global_points = dealii::Utilities::MPI::sum(
                  static_cast<std::uint64_t>(positions.size()),
                  target_field.space().mpi_communicator());
                const auto global_routed_points = dealii::Utilities::MPI::sum(
                  static_cast<std::uint64_t>(routed_points),
                  target_field.space().mpi_communicator());
                // Missing points are intentionally omitted from assembly.
                // Keep a sanity check against unexpected duplicate routing.
                AssertThrow(
                  global_routed_points <= global_points,
                  dealii::ExcMessage(
                    "More support quadrature points were routed than supplied."));
                ImmersX::detail::report_retained_quadrature_points_once(
                  global_routed_points,
                  global_points,
                  target_field.space().mpi_communicator());

                for (const auto &particle : distribution.get_particles())
                  if (particle.get_surrounding_cell()->is_locally_owned())
                    {
                      const CellIterator source_cell(
                        *particle.get_surrounding_cell(),
                        &source.dof_handler());
                      append_point(source_cell,
                                   particle.get_reference_location(),
                                   particle.get_properties()[0]);
                    }
                const auto global_located_points = dealii::Utilities::MPI::sum(
                  local_quadrature_point_count,
                  target_field.space().mpi_communicator());

                AssertThrow(
                  global_located_points == global_routed_points,
                  dealii::ExcMessage(
                    "Some successfully routed support quadrature points were not "
                    "assigned to locally owned background cells."));
              }
            else
              {
                dealii::GridTools::Cache<dim, spacedim> background_cache(
                  source.dof_handler().get_triangulation(), source.mapping());
                typename dealii::Triangulation<dim,
                                               spacedim>::active_cell_iterator
                  cell_hint;
                for (unsigned int q = 0; q < positions.size(); ++q)
                  {
                    const auto [background_cell, reference_point] =
                      dealii::GridTools::find_active_cell_around_point(
                        background_cache, positions[q], cell_hint);
                    AssertThrow(
                      background_cell !=
                        source.dof_handler().get_triangulation().end(),
                      dealii::ExcMessage(
                        "An integration-support quadrature point lies outside "
                        "the background mesh."));
                    cell_hint = background_cell;
                    const CellIterator source_cell(*background_cell,
                                                   &source.dof_handler());
                    append_point(source_cell,
                                 reference_point,
                                 properties[q][0]);
                  }
              }
          }

        dealii::DynamicSparsityPattern sparsity(
          target_field.locally_owned_dofs().size(),
          source.locally_owned_dofs().size(),
          target_field.locally_relevant_dofs());
        // Physical point sets vary per cell; their FEValues are rebuilt below,
        // while the local integration buffers remain shared across cells.
        ScratchData                                  scratch(source.mapping(),
                            source.space().finite_element(),
                            dealii::Quadrature<dim>(),
                            dealii::update_default);
        std::vector<dealii::types::global_dof_index> source_indices(
          source.space().finite_element().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        for (const auto &[cell_id, cell_quadrature] : cell_quadratures)
          {
            (void)cell_id;
            const auto target_cell =
              cell_quadrature.source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
            cell_quadrature.source_cell->get_dof_indices(source_indices);
            target_cell->get_dof_indices(target_indices);
            const auto source_execution_indices =
              execution_indices(source, source_indices);
            const auto target_execution_indices =
              execution_indices(target_field, target_indices);
            target_field.constraints().add_entries_local_to_global(
              target_execution_indices,
              source.constraints(),
              source_execution_indices,
              sparsity,
              true);
          }

        if (dynamic_cast<
              const dealii::parallel::TriangulationBase<dim, spacedim> *>(
              &source.dof_handler().get_triangulation()) != nullptr)
          dealii::SparsityTools::distribute_sparsity_pattern(
            sparsity,
            target_field.locally_owned_dofs(),
            target_field.space().mpi_communicator(),
            target_field.locally_relevant_dofs());

        auto matrix = std::make_shared<MatrixType>();
        auto matrix_sparsity =
          initialize_weak_matrix(*matrix,
                                 target_field.locally_owned_dofs(),
                                 source.locally_owned_dofs(),
                                 sparsity,
                                 target_field.space().mpi_communicator());

        if (affine_data != nullptr)
          {
            *affine_data = std::make_shared<AffineData<VectorType, MatrixType>>(
              source.constraints(),
              target_field.constraints(),
              target_field.locally_owned_dofs(),
              target_field.space().mpi_communicator());
          }

        for (const auto &[cell_id, cell_quadrature] : cell_quadratures)
          {
            (void)cell_id;
            const auto target_cell =
              cell_quadrature.source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
            const dealii::Quadrature<dim> point_quadrature(
              cell_quadrature.reference_points,
              std::vector<double>(cell_quadrature.reference_points.size(), 1.));
            const auto flags = observable.update_flags() |
                               target.update_flags() | dealii::update_values;
            dealii::FEValues<dim, spacedim> source_values(
              source.mapping(),
              source.space().finite_element(),
              point_quadrature,
              flags);
            dealii::FEValues<dim, spacedim> target_values(
              target_field.mapping(),
              target_field.space().finite_element(),
              point_quadrature,
              flags);
            source_values.reinit(cell_quadrature.source_cell);
            target_values.reinit(target_cell);
            cell_quadrature.source_cell->get_dof_indices(source_indices);
            target_cell->get_dof_indices(target_indices);
            assemble_cell<VectorType>(observable,
                                      source,
                                      source_values,
                                      target,
                                      target_values,
                                      source_indices,
                                      target_indices,
                                      target_cell,
                                      scratch,
                                      matrix.get(),
                                      affine_data != nullptr ?
                                        affine_data->get() :
                                        nullptr,
                                      &cell_quadrature.physical_weights);
          }
        if (affine_rhs != nullptr && affine_data != nullptr)
          {
            *affine_rhs = std::make_shared<VectorType>();
            (*affine_data)->finish(**affine_rhs);
          }
        compress_weak_matrix(*matrix);
        return {std::move(matrix), std::move(matrix_sparsity)};
      }

      template <typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_on_boundary_support(
        const ObservableType                            &observable,
        const TargetExpression                          &target,
        const BoundaryIntegrationSupport<dim, spacedim> &support,
        std::shared_ptr<VectorType>                     *affine_rhs = nullptr,
        std::shared_ptr<AffineData<VectorType, MatrixType>> *affine_data =
          nullptr)
      {
        const auto &source       = observable.source();
        const auto &target_field = target.source();
        support.assert_matches(source.dof_handler().get_triangulation());
        support.assert_matches(target_field.dof_handler().get_triangulation());
        return assemble_boundary<VectorType, MatrixType>(
          observable, target, support.boundary_ids(), affine_rhs, affine_data);
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

        auto result =
          std::make_shared<PreparedMatrix<VectorType, MatrixType>>();
        result->rebuild = [observable, source, target, degree] {
          return assemble_nonmatching<SourceField, VectorType, MatrixType>(
            observable, source, target, degree);
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

      template <typename Support, typename CellIterator>
      static bool
      support_includes(const Support &support, const CellIterator &cell)
      {
        if constexpr (std::is_same_v<Support, InferredIntegrationSupport>)
          return true;
        else
          return support.includes_cell(cell);
      }

      template <typename SourceField,
                typename VectorType,
                typename MatrixType,
                typename Support = InferredIntegrationSupport>
      static MatrixStorage<MatrixType>
      assemble_from_source(const ObservableType        &observable,
                           const SourceField           &source,
                           const TargetExpression      &target,
                           std::shared_ptr<VectorType> *affine_rhs = nullptr,
                           std::shared_ptr<AffineData<VectorType, MatrixType>>
                                         *affine_data = nullptr,
                           const Support &support     = {})
      {
        const auto target_field = target.source();
        if constexpr (SourceField::dimension() != dim)
          return assemble_nonmatching<SourceField, VectorType, MatrixType>(
            observable,
            source,
            target,
            std::max(source.space().finite_element().degree,
                     target_field.space().finite_element().degree),
            support);
        else
          {
            const auto degree =
              std::max(source.space().finite_element().degree,
                       target_field.space().finite_element().degree);
            const auto quadrature = support_quadrature(
              support,
              compatibility::gauss_quadrature<dim>(
                target_field.space().finite_element().reference_cell(),
                degree + 1));
            dealii::UpdateFlags update_flags = dealii::update_JxW_values |
                                               observable.update_flags() |
                                               target.update_flags();

            if (&source.space().dof_handler().get_triangulation() !=
                &target_field.space().dof_handler().get_triangulation())
              return assemble_nonmatching<SourceField, VectorType, MatrixType>(
                observable, source, target, degree, support);

            AssertThrow(&source.mapping() == &target_field.mapping(),
                        dealii::ExcMessage(
                          "A weak term requires compatible source and target "
                          "mappings."));

            dealii::DynamicSparsityPattern sparsity(
              target_field.locally_owned_dofs().size(),
              source.locally_owned_dofs().size(),
              target_field.locally_relevant_dofs());

            if (&source.dof_handler() == &target_field.dof_handler())
              {
                for (const auto &cell :
                     target_field.dof_handler().active_cell_iterators())
                  if (cell->is_locally_owned())
                    {
                      if (!support_includes(support, cell))
                        continue;
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
                        true);
                    }
              }
            else
              for (const auto &source_cell :
                   source.dof_handler().active_cell_iterators())
                {
                  const auto target_cell = source_cell->as_dof_handler_iterator(
                    target_field.dof_handler());
                  if (!target_cell->is_locally_owned() ||
                      !support_includes(support, source_cell))
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
                    true);
                }

            if (dynamic_cast<
                  const dealii::parallel::TriangulationBase<dim, spacedim> *>(
                  &target_field.dof_handler().get_triangulation()) != nullptr)
              dealii::SparsityTools::distribute_sparsity_pattern(
                sparsity,
                target_field.locally_owned_dofs(),
                target_field.space().mpi_communicator(),
                target_field.locally_relevant_dofs());

            auto matrix = std::make_shared<MatrixType>();
            auto matrix_sparsity =
              initialize_weak_matrix(*matrix,
                                     target_field.locally_owned_dofs(),
                                     source.locally_owned_dofs(),
                                     sparsity,
                                     target_field.space().mpi_communicator());

            if (affine_data != nullptr)
              {
                *affine_data =
                  std::make_shared<AffineData<VectorType, MatrixType>>(
                    source.constraints(),
                    target_field.constraints(),
                    target_field.locally_owned_dofs(),
                    target_field.space().mpi_communicator());
              }

            if (&source.dof_handler() == &target_field.dof_handler())
              {
                ScratchData scratch(target_field.mapping(),
                                    target_field.space().finite_element(),
                                    quadrature,
                                    update_flags);
                for (const auto &cell :
                     target_field.dof_handler().active_cell_iterators())
                  {
                    if (!cell->is_locally_owned())
                      continue;
                    if (!support_includes(support, cell))
                      continue;
                    const auto &values = scratch.reinit(cell);
                    assemble_cell<VectorType>(observable,
                                              source,
                                              values,
                                              target,
                                              values,
                                              scratch.get_local_dof_indices(),
                                              scratch.get_local_dof_indices(),
                                              cell,
                                              scratch,
                                              matrix.get(),
                                              affine_data != nullptr &&
                                                  *affine_data != nullptr ?
                                                affine_data->get() :
                                                nullptr,
                                              nullptr,
                                              support);
                  }
              }
            else
              {
                ScratchData source_scratch(source.mapping(),
                                           source.space().finite_element(),
                                           quadrature,
                                           update_flags);
                ScratchData target_scratch(
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
                    if (!target_cell->is_locally_owned() ||
                        !support_includes(support, source_cell))
                      continue;
                    const auto &source_values =
                      source_scratch.reinit(source_cell);
                    const auto &target_values =
                      target_scratch.reinit(target_cell);
                    assemble_cell<VectorType>(
                      observable,
                      source,
                      source_values,
                      target,
                      target_values,
                      source_scratch.get_local_dof_indices(),
                      target_scratch.get_local_dof_indices(),
                      target_cell,
                      source_scratch,
                      matrix.get(),
                      affine_data != nullptr && *affine_data != nullptr ?
                        affine_data->get() :
                        nullptr,
                      nullptr,
                      support);
                  }
              }
            if (affine_rhs != nullptr && affine_data != nullptr &&
                *affine_data != nullptr)
              {
                *affine_rhs = std::make_shared<VectorType>();
                (*affine_data)->finish(**affine_rhs);
              }
            compress_weak_matrix(*matrix);
            return {std::move(matrix), std::move(matrix_sparsity)};
          }
      }

    public:
      /** Assemble a linear pairing on selected boundary faces. */
      template <typename VectorType, typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_boundary(
        const ObservableType                       &observable,
        const TargetExpression                     &target,
        const std::set<dealii::types::boundary_id> &boundary_ids,
        std::shared_ptr<VectorType>                *affine_rhs = nullptr,
        std::shared_ptr<AffineData<VectorType, MatrixType>> *affine_data =
          nullptr)
      {
        if constexpr (SourceField::dimension() != dim)
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "Boundary weak terms require equal-dimensional "
                          "source and target fields."));
            return {};
          }
        else
          {
            const auto &source       = observable.source();
            const auto &target_field = target.source();
            AssertThrow(
              !boundary_ids.empty(),
              dealii::ExcMessage(
                "A boundary weak term needs at least one boundary id."));
            AssertThrow(&source.dof_handler() == &target_field.dof_handler(),
                        dealii::ExcMessage(
                          "Boundary weak terms currently require a shared "
                          "source and target DoFHandler."));
            AssertThrow(
              &source.mapping() == &target_field.mapping(),
              dealii::ExcMessage(
                "A boundary weak term requires compatible mappings."));

            const unsigned int degree =
              std::max(source.space().finite_element().degree,
                       target_field.space().finite_element().degree);
            const auto quadrature =
              compatibility::gauss_quadrature<dim - 1>(target_field.space()
                                                         .finite_element()
                                                         .reference_cell()
                                                         .face_reference_cell(
                                                           0),
                                                       degree + 1);
            const auto flags =
              dealii::update_JxW_values | dealii::update_quadrature_points |
              dealii::update_normal_vectors | observable.update_flags() |
              target.update_flags();

            dealii::DynamicSparsityPattern sparsity(
              target_field.locally_owned_dofs().size(),
              source.locally_owned_dofs().size(),
              target_field.locally_relevant_dofs());
            for (const auto &cell :
                 target_field.dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  std::vector<dealii::types::global_dof_index> indices(
                    cell->get_fe().n_dofs_per_cell());
                  cell->get_dof_indices(indices);
                  target_field.constraints().add_entries_local_to_global(
                    execution_indices(target_field, indices),
                    source.constraints(),
                    execution_indices(source, indices),
                    sparsity,
                    true);
                }

            if (dynamic_cast<
                  const dealii::parallel::TriangulationBase<dim, spacedim> *>(
                  &target_field.dof_handler().get_triangulation()) != nullptr)
              dealii::SparsityTools::distribute_sparsity_pattern(
                sparsity,
                target_field.locally_owned_dofs(),
                target_field.space().mpi_communicator(),
                target_field.locally_relevant_dofs());

            auto matrix = std::make_shared<MatrixType>();
            auto matrix_sparsity =
              initialize_weak_matrix(*matrix,
                                     target_field.locally_owned_dofs(),
                                     source.locally_owned_dofs(),
                                     sparsity,
                                     target_field.space().mpi_communicator());
            if (affine_data != nullptr)
              {
                *affine_data =
                  std::make_shared<AffineData<VectorType, MatrixType>>(
                    source.constraints(),
                    target_field.constraints(),
                    target_field.locally_owned_dofs(),
                    target_field.space().mpi_communicator());
              }
            ScratchData scratch(target_field.mapping(),
                                target_field.space().finite_element(),
                                dealii::Quadrature<dim>(),
                                dealii::update_default,
                                quadrature,
                                flags);
            for (const auto &cell :
                 target_field.dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                for (const auto face : cell->face_indices())
                  if (cell->face(face)->at_boundary() &&
                      boundary_ids.count(cell->face(face)->boundary_id()))
                    {
                      const auto &values = scratch.reinit(cell, face);
                      assemble_face_cell<VectorType>(
                        observable,
                        source,
                        values,
                        target,
                        values,
                        scratch.get_local_dof_indices(),
                        scratch.get_local_dof_indices(),
                        scratch,
                        matrix.get(),
                        affine_data != nullptr && *affine_data != nullptr ?
                          affine_data->get() :
                          nullptr);
                    }
            if (affine_rhs != nullptr && affine_data != nullptr &&
                *affine_data != nullptr)
              {
                *affine_rhs = std::make_shared<VectorType>();
                (*affine_data)->finish(**affine_rhs);
              }
            compress_weak_matrix(*matrix);
            return {std::move(matrix), std::move(matrix_sparsity)};
          }
      }

    private:
      template <typename BackgroundField,
                typename EmbeddedField,
                typename VectorType,
                typename MatrixType>
      static MatrixStorage<MatrixType>
      assemble_native_nonmatching_value(const BackgroundField &background,
                                        const EmbeddedField   &embedded,
                                        const double           scale,
                                        const bool             transpose)
      {
        constexpr int background_dim = BackgroundField::dimension();
        constexpr int embedded_dim   = EmbeddedField::dimension();
        static_assert(background_dim == BackgroundField::spacedimension(),
                      "The NonMatching mass matrix needs a full-dimensional "
                      "background field.");
        static_assert(embedded_dim <= background_dim,
                      "The embedded field dimension cannot exceed the "
                      "background dimension.");

        const auto background_mask = background.component_mask();
        const auto embedded_mask   = embedded.component_mask();
        const auto quadrature = compatibility::gauss_quadrature<embedded_dim>(
          embedded.space().finite_element().reference_cell(),
          std::max(background.space().finite_element().degree,
                   embedded.space().finite_element().degree) +
            1);
        dealii::DynamicSparsityPattern sparsity(
          background.dof_handler().n_dofs(),
          embedded.dof_handler().n_dofs(),
          background.locally_relevant_dofs());
        dealii::GridTools::Cache<background_dim, background_dim> cache(
          background.dof_handler().get_triangulation(), background.mapping());
        dealii::NonMatching::create_coupling_sparsity_pattern(
          cache,
          background.dof_handler(),
          embedded.dof_handler(),
          quadrature,
          sparsity,
          background.constraints(),
          background_mask,
          embedded_mask,
          embedded.mapping(),
          embedded.constraints());
        if (dynamic_cast<const dealii::parallel::
                           TriangulationBase<background_dim, background_dim> *>(
              &background.dof_handler().get_triangulation()) != nullptr)
          dealii::SparsityTools::distribute_sparsity_pattern(
            sparsity,
            background.locally_owned_dofs(),
            background.space().mpi_communicator(),
            background.locally_relevant_dofs());

        auto matrix = std::make_shared<MatrixType>();
        auto matrix_sparsity =
          initialize_weak_matrix(*matrix,
                                 background.locally_owned_dofs(),
                                 embedded.locally_owned_dofs(),
                                 sparsity,
                                 background.space().mpi_communicator());
        dealii::NonMatching::create_coupling_mass_matrix(
          cache,
          background.dof_handler(),
          embedded.dof_handler(),
          quadrature,
          *matrix,
          background.constraints(),
          background_mask,
          embedded_mask,
          embedded.mapping(),
          embedded.constraints());
        compress_weak_matrix(*matrix);
        *matrix *= scale;
        if (transpose)
          return {ImmersX::detail::transpose_matrix(matrix), nullptr};
        return {std::move(matrix), std::move(matrix_sparsity)};
      }

      template <typename SourceField,
                typename VectorType,
                typename MatrixType,
                typename Support = InferredIntegrationSupport>
      static MatrixStorage<MatrixType>
      assemble_nonmatching(const ObservableType   &observable,
                           const SourceField      &source,
                           const TargetExpression &target,
                           const unsigned int      degree,
                           const Support          &support = {})
      {
#ifdef IMMERSX_WEAK_TERM_TESTING
        ++weak_term_nonmatching_preparations;
#endif
        if constexpr (std::is_same_v<typename ObservableType::operation_type,
                                     ValueOperation> &&
                      std::is_same_v<typename TargetExpression::operation_type,
                                     ValueOperation>)
          {
            bool use_native_nonmatching = true;
            if constexpr (!std::is_same_v<Support, InferredIntegrationSupport>)
              use_native_nonmatching = !support.has_material_filter() &&
                                       !support.has_quadrature_partition();

            if (use_native_nonmatching)
              {
                if constexpr (SourceField::dimension() == spacedim)
                  {
                    using EmbeddedField =
                      typename TargetExpression::source_field_type;
                    const auto &embedded = target.source();
                    const bool  embedded_is_distributed =
                      dynamic_cast<const dealii::parallel::distributed::
                                     Triangulation<EmbeddedField::dimension(),
                                                   spacedim> *>(
                        &embedded.dof_handler().get_triangulation()) != nullptr;
                    const bool background_needs_particle_assembly =
                      dynamic_cast<
                        const dealii::parallel::TriangulationBase<spacedim> *>(
                        &source.dof_handler().get_triangulation()) != nullptr &&
                      dealii::Utilities::MPI::n_mpi_processes(
                        source.space().mpi_communicator()) > 1;
                    if (!embedded_is_distributed &&
                        !background_needs_particle_assembly)
                      return assemble_native_nonmatching_value<SourceField,
                                                               EmbeddedField,
                                                               VectorType,
                                                               MatrixType>(
                        source,
                        embedded,
                        observable.scale() * target.scale(),
                        true);
                  }
                else if constexpr (SourceField::dimension() < dim &&
                                   dim == spacedim)
                  {
                    const auto &background = target.source();
                    const bool  immersed_is_distributed =
                      dynamic_cast<
                        const dealii::parallel::distributed::
                          Triangulation<SourceField::dimension(), spacedim> *>(
                        &source.dof_handler().get_triangulation()) != nullptr;
                    const bool background_needs_particle_assembly =
                      dynamic_cast<
                        const dealii::parallel::TriangulationBase<spacedim> *>(
                        &background.dof_handler().get_triangulation()) !=
                        nullptr &&
                      dealii::Utilities::MPI::n_mpi_processes(
                        background.space().mpi_communicator()) > 1;
                    if (!immersed_is_distributed &&
                        !background_needs_particle_assembly)
                      return assemble_native_nonmatching_value<
                        typename TargetExpression::source_field_type,
                        SourceField,
                        VectorType,
                        MatrixType>(target.source(),
                                    source,
                                    observable.scale() * target.scale(),
                                    false);
                  }
              }
          }
        if constexpr (SourceField::dimension() == spacedim)
          return assemble_nonmatching_reverse<SourceField,
                                              VectorType,
                                              MatrixType>(
            observable, source, target, degree, support);
        else if constexpr (SourceField::dimension() < dim && dim == spacedim)
          return assemble_nonmatching_forward<SourceField,
                                              VectorType,
                                              MatrixType>(
            observable, source, target, degree, support);
        else
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "This weak-term geometry combination is not "
                          "implemented."));
            return {};
          }
      }

      /** Assemble a lower-dimensional source against a full-dimensional test.
       * Source quadrature points are inserted into the distributed target
       * background.  The particle properties retain the source-cell stencil,
       * so the resulting matrix has target rows and source columns even when
       * the two meshes have different dimensions and partitions.
       */
      template <typename SourceField,
                typename VectorType,
                typename MatrixType,
                typename Support>
      static MatrixStorage<MatrixType>
      assemble_nonmatching_forward(const ObservableType   &observable,
                                   const SourceField      &source,
                                   const TargetExpression &target,
                                   const unsigned int      degree,
                                   const Support          &support)
      {
        static_assert(SourceField::dimension() < dim && dim == spacedim,
                      "Forward nonmatching assembly needs a lower-dimensional "
                      "source and a full-dimensional target.");

        const auto  target_field = target.source();
        const auto *target_tria =
          &target_field.space().dof_handler().get_triangulation();

        const auto quadrature = support_quadrature(
          support,
          compatibility::gauss_quadrature<SourceField::dimension()>(
            source.space().finite_element().reference_cell(), degree + 1));
        const unsigned int n_source_dofs =
          source.space().finite_element().n_dofs_per_cell();
        const unsigned int source_components =
          n_stored_components<typename ObservableType::value_type>();
        const unsigned int n_properties =
          1 + n_source_dofs + n_source_dofs * source_components;

        ParticleCouplingParameters<spacedim> particle_parameters(
          "/ImmersX/weak term/forward/" +
          std::to_string(reinterpret_cast<std::uintptr_t>(&source)));
        ParticleCoupling<spacedim> distribution(particle_parameters);
        distribution.initialize_particle_handler(*target_tria,
                                                 target_field.mapping(),
                                                 n_properties);

        dealii::FEValues<SourceField::dimension(), spacedim> source_values(
          source.mapping(),
          source.space().finite_element(),
          quadrature,
          dealii::update_values | dealii::update_JxW_values |
            dealii::update_quadrature_points | observable.update_flags());

        std::vector<dealii::Point<spacedim>>       positions;
        std::vector<std::vector<double>>           properties;
        std::vector<dealii::types::particle_index> ids;
        positions.reserve(source.dof_handler().n_dofs() * quadrature.size());
        properties.reserve(positions.capacity());
        ids.reserve(positions.capacity());

        std::vector<dealii::types::global_dof_index> source_indices(
          n_source_dofs);
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned() &&
              support_includes(support, source_cell))
            {
              source_values.reinit(source_cell);
              source_cell->get_dof_indices(source_indices);
              for (const auto q : source_values.quadrature_point_indices())
                {
                  std::vector<double> point_properties;
                  point_properties.reserve(n_properties);
                  const double selection =
                    support_point_weight(support, source_cell, q);
                  if (selection == 0.)
                    continue;
                  point_properties.push_back(source_values.JxW(q) * selection);
                  for (const auto index : source_indices)
                    {
                      const auto encoded = static_cast<double>(index);
                      AssertThrow(
                        static_cast<dealii::types::global_dof_index>(encoded) ==
                          index,
                        dealii::ExcMessage(
                          "A source DoF index cannot be transported exactly "
                          "through ParticleHandler properties."));
                      point_properties.push_back(encoded);
                    }
                  for (unsigned int i = 0; i < n_source_dofs; ++i)
                    flatten_value(observable.operation()(
                                    source_values[source.extractor()], i, q),
                                  point_properties);

                  positions.push_back(source_values.quadrature_point(q));
                  properties.emplace_back(std::move(point_properties));
                  const auto id = static_cast<std::uint64_t>(
                                    source_cell->global_active_cell_index()) *
                                    quadrature.size() +
                                  q;
                  ids.push_back(static_cast<dealii::types::particle_index>(id));
                }
            }
        distribution.insert_points(positions, properties, ids);

        dealii::DynamicSparsityPattern sparsity(
          target_field.dof_handler().n_dofs(),
          source.dof_handler().n_dofs(),
          target_field.locally_relevant_dofs());
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        for (const auto &particle : distribution.get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename TargetField::space_type::DoFHandlerType::
                cell_iterator target_cell(*particle.get_surrounding_cell(),
                                          &target_field.dof_handler());
              target_cell->get_dof_indices(target_indices);
              const auto &data = particle.get_properties();
              for (const auto target_index : target_indices)
                if (target_field.has_execution_index(target_index))
                  for (unsigned int j = 0; j < n_source_dofs; ++j)
                    sparsity.add(target_field.execution_index(target_index),
                                 static_cast<dealii::types::global_dof_index>(
                                   data[1 + j]));
            }
        dealii::SparsityTools::distribute_sparsity_pattern(
          sparsity,
          target_field.locally_owned_dofs(),
          target_field.space().mpi_communicator(),
          target_field.locally_relevant_dofs());

        auto matrix = std::make_shared<MatrixType>();
        auto matrix_sparsity =
          initialize_weak_matrix(*matrix,
                                 target_field.locally_owned_dofs(),
                                 source.locally_owned_dofs(),
                                 sparsity,
                                 target_field.space().mpi_communicator());

        const unsigned int n_target_dofs =
          target_field.space().finite_element().n_dofs_per_cell();
        const auto target_flags = target.update_flags();
        const auto scale        = observable.scale() * target.scale();
        for (const auto &particle : distribution.get_particles())
          if (particle.get_surrounding_cell()->is_locally_owned())
            {
              const typename TargetField::space_type::DoFHandlerType::
                cell_iterator target_cell(*particle.get_surrounding_cell(),
                                          &target_field.dof_handler());
              target_cell->get_dof_indices(target_indices);
              std::vector<unsigned int> target_positions;
              std::vector<dealii::types::global_dof_index>
                target_execution_indices;
              for (unsigned int i = 0; i < n_target_dofs; ++i)
                if (target_field.has_execution_index(target_indices[i]))
                  {
                    target_positions.push_back(i);
                    target_execution_indices.push_back(
                      target_field.execution_index(target_indices[i]));
                  }

              const dealii::Quadrature<dim> point_quadrature(
                std::vector<dealii::Point<dim>>{
                  particle.get_reference_location()});
              dealii::FEValues<dim, spacedim> target_values(
                target_field.mapping(),
                target_field.space().finite_element(),
                point_quadrature,
                target_flags);
              target_values.reinit(target_cell);
              const auto &target_view = target_values[target_field.extractor()];
              const auto &data        = particle.get_properties();
              dealii::FullMatrix<double> local(target_positions.size(),
                                               n_source_dofs);
              for (unsigned int i = 0; i < target_positions.size(); ++i)
                {
                  const auto target_value =
                    target.operation()(target_view, target_positions[i], 0);
                  for (unsigned int j = 0; j < n_source_dofs; ++j)
                    {
                      unsigned int offset =
                        1 + n_source_dofs + j * source_components;
                      const auto source_value =
                        unflatten_value<typename ObservableType::value_type>(
                          data, offset);
                      local(i, j) =
                        scale * natural_pairing(source_value, target_value) *
                        data[0];
                    }
                }

              std::vector<dealii::types::global_dof_index> source_native(
                n_source_dofs);
              for (unsigned int j = 0; j < n_source_dofs; ++j)
                source_native[j] =
                  static_cast<dealii::types::global_dof_index>(data[1 + j]);
              target_field.constraints().distribute_local_to_global(
                local,
                target_execution_indices,
                source.constraints(),
                source_native,
                *matrix);
            }
        compress_weak_matrix(*matrix);
        return {std::move(matrix), std::move(matrix_sparsity)};
      }

      /** Assemble a nonmatching pairing by transposing the natural source-row
       * matrix. Target quadrature points are searched in the full-dimensional
       * source mesh, so rows remain owned by the source background ranks and no
       * distributed matrix contribution exchange is needed before the explicit
       * transpose. */
      template <typename SourceField,
                typename VectorType,
                typename MatrixType,
                typename Support>
      static MatrixStorage<MatrixType>
      assemble_nonmatching_reverse(const ObservableType   &observable,
                                   const SourceField      &source,
                                   const TargetExpression &target,
                                   const unsigned int      degree,
                                   const Support          &support)
      {
        static_assert(SourceField::dimension() == spacedim,
                      "Nonmatching weak terms require a full-dimensional "
                      "source background.");

        const auto target_field = target.source();
        const auto quadrature   = support_quadrature(
          support,
          compatibility::gauss_quadrature<dim>(
            target_field.space().finite_element().reference_cell(),
            degree + 1));
        const auto *source_tria =
          &source.space().dof_handler().get_triangulation();

        using Point = detail::CouplingPoint<spacedim, double>;
        const unsigned int local_target_dofs = [&target_field] {
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
        // A distributed target triangulation may have no locally owned cells
        // on some ranks (for example, a coarse one-cell mesh on two ranks).
        // The particle property layout must nevertheless be identical on all
        // ranks, so recover the per-cell execution DoF count collectively.
        const unsigned int n_target_dofs =
          dealii::Utilities::MPI::max(local_target_dofs,
                                      source.space().mpi_communicator());
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
          if (cell->is_locally_owned() && support_includes(support, cell))
            {
              target_values.reinit(cell);
              cell->get_dof_indices(target_indices);
              for (const auto q : target_values.quadrature_point_indices())
                {
                  const double selection =
                    support_point_weight(support, cell, q);
                  if (selection == 0.)
                    continue;
                  Point point;
                  point.point = target_values.quadrature_point(q);
                  point.representative_point = point.point;
                  point.weight               = target_values.JxW(q) * selection;
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

      template <typename VectorType,
                typename SourceField,
                typename SourceValues,
                typename TargetValues,
                typename TargetCell,
                typename MatrixType,
                typename Support = InferredIntegrationSupport>
      static void
      assemble_cell(
        const ObservableType                               &observable,
        const SourceField                                  &source,
        const SourceValues                                 &source_values,
        const TargetExpression                             &target,
        const TargetValues                                 &target_values,
        const std::vector<dealii::types::global_dof_index> &source_indices,
        const std::vector<dealii::types::global_dof_index> &target_indices,
        const TargetCell                                   &target_cell,
        ScratchData                                        &scratch,
        MatrixType                                         *matrix,
        AffineData<VectorType, MatrixType>                 *affine_data,
        const std::vector<double> *physical_weights = nullptr,
        const Support             &support          = {})
      {
        const auto &target_field = target.source();
        scratch.prepare(source, source_indices, target_field, target_indices);
        const auto &source_positions         = scratch.source_positions;
        const auto &target_positions         = scratch.target_positions;
        const auto &source_execution_indices = scratch.source_execution_indices;
        const auto &target_execution_indices = scratch.target_execution_indices;
        auto       &local                    = scratch.local_matrix;
        auto       &trial_values             = scratch.trial_values;
        auto       &test_values              = scratch.test_values;
        const auto &trial_view = source_values[source.extractor()];
        const auto &test_view  = target_values[target_field.extractor()];
        AssertDimension(source_values.n_quadrature_points,
                        target_values.n_quadrature_points);
        if (physical_weights != nullptr)
          AssertDimension(physical_weights->size(),
                          target_values.n_quadrature_points);
        const double scale = observable.scale() * target.scale();
        for (unsigned int q = 0; q < target_values.n_quadrature_points; ++q)
          {
            const double weight =
              scale *
              (physical_weights == nullptr ? target_values.JxW(q) :
                                             (*physical_weights)[q]) *
              support_point_weight(support, target_cell, q);
            for (unsigned int j = 0; j < source_positions.size(); ++j)
              trial_values[j] =
                observable.operation()(trial_view, source_positions[j], q);
            for (unsigned int i = 0; i < target_positions.size(); ++i)
              test_values[i] =
                target.operation()(test_view, target_positions[i], q);
            for (unsigned int i = 0; i < target_positions.size(); ++i)
              for (unsigned int j = 0; j < source_positions.size(); ++j)
                local(i, j) += weight * detail::natural_pairing(trial_values[j],
                                                                test_values[i]);
          }

        const bool use_square_constraints =
          !source.is_reindexed() && !target_field.is_reindexed() &&
          &source.constraints() == &target_field.constraints() &&
          source_execution_indices == target_execution_indices;
        if (affine_data != nullptr)
          affine_data->add(local,
                           source_execution_indices,
                           target_execution_indices,
                           use_square_constraints);

        if (matrix == nullptr)
          return;

        // Preserve native constrained diagonals only for unreindexed fields.
        // Mixed block fields use condensed residual rows; adding artificial
        // diagonals there changes the semidiscrete residual and Jacobian.
        if (use_square_constraints)
          target_field.constraints().distribute_local_to_global(
            local, target_execution_indices, *matrix);
        else
          target_field.constraints().distribute_local_to_global(
            local,
            target_execution_indices,
            source.constraints(),
            source_execution_indices,
            *matrix);
      }

      template <typename VectorType,
                typename SourceField,
                typename SourceValues,
                typename TargetValues,
                typename MatrixType>
      static void
      assemble_face_cell(
        const ObservableType                               &observable,
        const SourceField                                  &source,
        const SourceValues                                 &source_values,
        const TargetExpression                             &target,
        const TargetValues                                 &target_values,
        const std::vector<dealii::types::global_dof_index> &source_indices,
        const std::vector<dealii::types::global_dof_index> &target_indices,
        ScratchData                                        &scratch,
        MatrixType                                         *matrix,
        AffineData<VectorType, MatrixType>                 *affine_data)
      {
        const auto &target_field = target.source();
        scratch.prepare(source, source_indices, target_field, target_indices);
        const auto &source_positions         = scratch.source_positions;
        const auto &target_positions         = scratch.target_positions;
        const auto &source_execution_indices = scratch.source_execution_indices;
        const auto &target_execution_indices = scratch.target_execution_indices;
        auto       &local                    = scratch.local_matrix;
        auto       &trial_values             = scratch.trial_values;
        auto       &test_values              = scratch.test_values;
        const auto &trial_view = source_values[source.extractor()];
        const auto &test_view  = target_values[target_field.extractor()];
        AssertDimension(source_values.n_quadrature_points,
                        target_values.n_quadrature_points);
        const double scale = observable.scale() * target.scale();
        for (unsigned int q = 0; q < target_values.n_quadrature_points; ++q)
          {
            const double weight = scale * target_values.JxW(q);
            for (unsigned int j = 0; j < source_positions.size(); ++j)
              {
                if constexpr (detail::is_boundary_normal_observable<
                                ObservableType>::value)
                  trial_values[j] = observable.evaluate_face(
                    trial_view, source_positions[j], q, target_values);
                else
                  trial_values[j] =
                    observable.operation()(trial_view, source_positions[j], q);
              }
            for (unsigned int i = 0; i < target_positions.size(); ++i)
              test_values[i] =
                target.operation()(test_view, target_positions[i], q);
            for (unsigned int i = 0; i < target_positions.size(); ++i)
              for (unsigned int j = 0; j < source_positions.size(); ++j)
                local(i, j) += weight * detail::natural_pairing(trial_values[j],
                                                                test_values[i]);
          }

        if (affine_data != nullptr)
          affine_data->add(local,
                           source_execution_indices,
                           target_execution_indices);

        if (matrix == nullptr)
          return;
        target_field.constraints().distribute_local_to_global(
          local,
          target_execution_indices,
          source.constraints(),
          source_execution_indices,
          *matrix);
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

      template <typename Support>
      static constexpr bool
      uses_support_quadratures()
      {
        if constexpr (is_physical_quadrature_support<Support>::value)
          return true;
        else if constexpr (is_integration_support<Support>::value)
          return Support::dimension() + 1 == dim;
        else
          return false;
      }

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
          // Assembled outputs accept off-rank writes and compress them;
          // ghost entries belong to input states used for reading.
          vector.reinit(field.locally_owned_dofs(),
                        field.space().mpi_communicator());
        else
          vector.reinit(field.dof_handler().n_dofs());
      }

      template <typename Support>
      struct PhysicalSupportQuadrature
      {
        using CellIterator = typename SourceField::space_type::DoFHandlerType::
          active_cell_iterator;

        CellIterator                    cell;
        std::vector<dealii::Point<dim>> reference_points;
        std::vector<double>             physical_weights;
      };

      template <typename Support, typename TargetField>
      static auto
      make_support_quadratures(const Support     &support,
                               const SourceField &source,
                               const TargetField &target_field)
      {
        using CellIterator = typename SourceField::space_type::DoFHandlerType::
          active_cell_iterator;
        using QuadratureData = PhysicalSupportQuadrature<Support>;

        AssertThrow(&source.dof_handler().get_triangulation() ==
                        &target_field.dof_handler().get_triangulation() &&
                      &source.mapping() == &target_field.mapping(),
                    dealii::ExcMessage(
                      "Nonlinear support requires matching source and test "
                      "meshes and mappings."));

        std::vector<dealii::Point<spacedim>> positions;
        std::vector<std::vector<double>>     properties;
        const auto                          *background_tria = dynamic_cast<
          const dealii::parallel::TriangulationBase<dim, spacedim> *>(
          &source.dof_handler().get_triangulation());
        bool replicated_support = false;
        if constexpr (is_physical_quadrature_support<Support>::value)
          {
            static_assert(Support::ambient_dimension == spacedim,
                          "Physical support and fields must share an "
                          "embedding dimension.");
            int communicator_relation = MPI_UNEQUAL;
            MPI_Comm_compare(support.mpi_communicator(),
                             target_field.space().mpi_communicator(),
                             &communicator_relation);
            int support_processes    = 0;
            int background_processes = 0;
            MPI_Comm_size(support.mpi_communicator(), &support_processes);
            MPI_Comm_size(target_field.space().mpi_communicator(),
                          &background_processes);
            AssertThrow(communicator_relation == MPI_IDENT ||
                          communicator_relation == MPI_CONGRUENT ||
                          (support_processes == 1 && background_processes == 1),
                        dealii::ExcMessage(
                          "Physical support and background must use "
                          "congruent MPI communicators or be serial."));
            const auto support_rank = dealii::Utilities::MPI::this_mpi_process(
              support.mpi_communicator());
            if constexpr (has_support_triangulation<Support>::value)
              {
                const auto *distributed_support =
                  dynamic_cast<const dealii::parallel::TriangulationBase<
                    support_triangulation_dimension<Support>::value,
                    Support::ambient_dimension> *>(&support.triangulation());
                replicated_support =
                  background_tria != nullptr &&
                  distributed_support == nullptr &&
                  dealii::Utilities::MPI::this_mpi_process(
                    target_field.space().mpi_communicator()) != 0;
              }
            const auto &physical_points =
              support.locally_owned_quadrature_points();
            positions.reserve(physical_points.size());
            properties.reserve(physical_points.size());
            for (const auto &point : physical_points)
              {
                AssertThrow(point.owner_rank == support_rank,
                            dealii::ExcMessage(
                              "Physical support points must be locally "
                              "owned."));
                if (replicated_support)
                  continue;
                positions.push_back(point.point);
                properties.push_back({point.weight});
              }
          }
        else if constexpr (is_integration_support<Support>::value)
          {
            static_assert(Support::spacedimension() == spacedim,
                          "Integration support and fields must share an "
                          "embedding dimension.");
            static_assert(Support::dimension() + 1 == dim,
                          "Independent nonlinear supports currently require "
                          "codimension one.");
            if constexpr (has_support_triangulation<Support>::value)
              {
                constexpr int support_dim = Support::dimension();
                const auto   *distributed_support =
                  dynamic_cast<const dealii::parallel::
                                 TriangulationBase<support_dim, spacedim> *>(
                    &support.triangulation());
                replicated_support =
                  background_tria != nullptr &&
                  distributed_support == nullptr &&
                  dealii::Utilities::MPI::this_mpi_process(
                    target_field.space().mpi_communicator()) != 0;
                const unsigned int degree =
                  std::max(source.space().finite_element().degree,
                           target_field.space().finite_element().degree);
                const auto physical_reference =
                  detail::uniform_reference_cell(support.triangulation());
                AssertThrow(physical_reference.has_value(),
                            dealii::ExcMessage(
                              "The integration support mesh has no cells."));
                const auto reference_cell = *physical_reference;
                const auto quadrature     = support_quadrature(
                  support,
                  compatibility::gauss_quadrature<support_dim>(reference_cell,
                                                               degree + 2));
                const dealii::FE_Nothing<support_dim, spacedim> geometry_fe(
                  reference_cell);
                dealii::FEValues<support_dim, spacedim> support_values(
                  compatibility::linear_mapping<support_dim, spacedim>(
                    reference_cell),
                  geometry_fe,
                  quadrature,
                  dealii::update_quadrature_points | dealii::update_JxW_values);
                for (const auto &cell :
                     support.triangulation().active_cell_iterators())
                  if (!replicated_support && cell->is_locally_owned() &&
                      support.includes_cell(cell))
                    {
                      support_values.reinit(cell);
                      for (const auto q :
                           support_values.quadrature_point_indices())
                        {
                          const double selection =
                            support.point_weight(cell, q);
                          if (selection == 0.)
                            continue;
                          positions.push_back(
                            support_values.quadrature_point(q));
                          properties.push_back(
                            {support_values.JxW(q) * selection});
                        }
                    }
              }
          }

        std::map<dealii::CellId, QuadratureData> cell_quadratures;
        const auto append_point = [&](const CellIterator       &source_cell,
                                      const dealii::Point<dim> &reference_point,
                                      const double              weight) {
          auto &data = cell_quadratures[source_cell->id()];
          data.cell  = source_cell;
          data.reference_points.push_back(reference_point);
          data.physical_weights.push_back(weight);
        };

        if (background_tria != nullptr)
          {
            ParticleCouplingParameters<spacedim> parameters(
              "/ImmersX/weak term/nonlinear support/" +
              std::to_string(reinterpret_cast<std::uintptr_t>(&support)));
            ParticleCoupling<spacedim> distribution(parameters);
            distribution.initialize_particle_handler(*background_tria,
                                                     source.mapping(),
                                                     1);
            // Use generated IDs for the collective point routing.
            const auto receiving_ranks =
              distribution.insert_points(positions, properties);
            std::size_t routed_points = 0;
            for (const auto &[rank, indices] : receiving_ranks)
              {
                (void)rank;
                routed_points += indices.n_elements();
              }
            const auto global_points = dealii::Utilities::MPI::sum(
              static_cast<std::uint64_t>(positions.size()),
              target_field.space().mpi_communicator());
            const auto global_routed_points = dealii::Utilities::MPI::sum(
              static_cast<std::uint64_t>(routed_points),
              target_field.space().mpi_communicator());
            // Missing points are intentionally omitted from assembly.
            // Keep a sanity check against unexpected duplicate routing.
            AssertThrow(
              global_routed_points <= global_points,
              dealii::ExcMessage(
                "More support quadrature points were routed than supplied."));
            ImmersX::detail::report_retained_quadrature_points_once(
              global_routed_points,
              global_points,
              target_field.space().mpi_communicator());


            for (const auto &particle : distribution.get_particles())
              if (particle.get_surrounding_cell()->is_locally_owned())
                {
                  const CellIterator source_cell(
                    *particle.get_surrounding_cell(), &source.dof_handler());
                  append_point(source_cell,
                               particle.get_reference_location(),
                               particle.get_properties()[0]);
                }
          }
        else
          {
            dealii::GridTools::Cache<dim, spacedim> cache(
              source.dof_handler().get_triangulation(), source.mapping());
            typename dealii::Triangulation<dim, spacedim>::active_cell_iterator
              cell_hint;
            for (unsigned int q = 0; q < positions.size(); ++q)
              {
                const auto [background_cell, reference_point] =
                  dealii::GridTools::find_active_cell_around_point(cache,
                                                                   positions[q],
                                                                   cell_hint);
                AssertThrow(
                  background_cell !=
                    source.dof_handler().get_triangulation().end(),
                  dealii::ExcMessage(
                    "A physical support quadrature point lies outside the "
                    "background mesh."));
                cell_hint = background_cell;
                const CellIterator source_cell(*background_cell,
                                               &source.dof_handler());
                append_point(source_cell, reference_point, properties[q][0]);
              }
          }
        return cell_quadratures;
      }

      template <typename Support, typename CellIterator>
      static bool
      support_includes(const Support &support, const CellIterator &cell)
      {
        if constexpr (is_integration_support<std::decay_t<Support>>::value)
          return support.includes_cell(cell);
        else
          return true;
      }

      template <typename Support, typename SourceField, typename TargetField>
      static void
      assert_support_matches(const Support     &support,
                             const SourceField &source,
                             const TargetField &target)
      {
        if constexpr (is_integration_support<std::decay_t<Support>>::value)
          {
            if constexpr (Support::dimension() == SourceField::dimension())
              {
                support.assert_matches(
                  source.dof_handler().get_triangulation());
                support.assert_matches(
                  target.dof_handler().get_triangulation());
              }
            else
              AssertThrow(
                &source.dof_handler().get_triangulation() ==
                  &target.dof_handler().get_triangulation(),
                dealii::ExcMessage(
                  "Independent nonlinear trace supports require matching "
                  "source and test background meshes."));
          }
      }

      template <typename ObservableType_>
      static void
      assert_dependencies_match_source_mesh(const ObservableType_ &observable)
      {
        const auto &source_triangulation =
          observable.source().dof_handler().get_triangulation();
        for (const auto field : observable.dependencies())
          AssertThrow(
            static_cast<const void *>(&observable.source_for(field)
                                         .dof_handler()
                                         .get_triangulation()) ==
              static_cast<const void *>(&source_triangulation),
            dealii::ExcMessage(
              "Nonlinear weak term inputs must use the source triangulation."));
      }

      /** Cache input FEValues and local coefficients on one cell. Reuse the
       * FEValues across cells when an input shares the integration mapping;
       * otherwise transfer quadrature points from physical space. */
      template <typename Context, typename Cell, typename SourceValues>
      class CellInputCache
      {
      public:
        using VectorType =
          std::decay_t<decltype(std::declval<const Context &>().state(
            FieldId()))>;
        using InputStates = std::map<FieldId, VectorType>;

        struct Entry
        {
          std::unique_ptr<dealii::FEValues<dim, spacedim>> values;
          std::vector<unsigned int>                        positions;
          std::vector<dealii::types::global_dof_index>     execution_indices;
          std::vector<double>                              execution_values;
          const dealii::DoFHandler<dim, spacedim> *dof_handler = nullptr;
          dealii::CellId                           cell_id;
          bool                                     cell_initialized = false;
          bool                                     reusable         = false;
        };

        CellInputCache(
          const Context                        &context,
          const InputStates                    &input_states,
          const Cell                           &source_cell,
          const SourceValues                   &source_values,
          const dealii::Mapping<dim, spacedim> &integration_mapping,
          const dealii::UpdateFlags             update_flags,
          const bool                            reuse_across_cells = false)
          : context_(context)
          , input_states_(input_states)
          , source_cell_(source_cell)
          , source_values_(&source_values)
          , integration_mapping_(&integration_mapping)
          , update_flags_(update_flags)
          , reuse_across_cells_(reuse_across_cells)
        {}

        void
        reinit(const Cell &source_cell, const SourceValues &source_values)
        {
          source_cell_   = source_cell;
          source_values_ = &source_values;
          if (!reuse_across_cells_)
            {
              entries_.clear();
              return;
            }

          for (auto it = entries_.begin(); it != entries_.end();)
            if (it->second->reusable)
              {
                const auto input_cell = source_cell_->as_dof_handler_iterator(
                  *it->second->dof_handler);
                if (&it->second->values->get_fe() == &input_cell->get_fe())
                  {
                    it->second->values->reinit(input_cell);
                    ++it;
                  }
                else
                  it = entries_.erase(it);
              }
            else
              it = entries_.erase(it);
        }

        template <typename ObservableType_>
        const Entry &
        get(const ObservableType_ &observable) const
        {
          const auto key = static_cast<const void *>(&observable);
          auto       it  = entries_.find(key);
          if (it != entries_.end())
            {
              if (it->second->reusable &&
                  (!it->second->cell_initialized ||
                   it->second->cell_id != source_cell_->id()))
                update_coefficients(observable, *it->second);
              return *it->second;
            }

          const auto &input_field = observable.source();
          // Frozen leaves are not dependencies. Check them before converting
          // the integration cell to their DoFHandler.
          AssertThrow(&input_field.dof_handler().get_triangulation() ==
                        &source_cell_->get_triangulation(),
                      dealii::ExcMessage(
                        "Nonlinear weak term inputs, including frozen fields, "
                        "must use the source triangulation."));
          AssertThrow(!observable.is_frozen() ||
                        &input_field.mapping() == integration_mapping_,
                      dealii::ExcMessage(
                        "Frozen nonlinear weak term inputs must use the "
                        "source mapping."));
          const auto input_cell =
            source_cell_->as_dof_handler_iterator(input_field.dof_handler());
          auto entry         = std::make_unique<Entry>();
          entry->dof_handler = &input_field.dof_handler();
          entry->reusable    = reuse_across_cells_ &&
                            &input_field.mapping() == integration_mapping_;
          std::unique_ptr<dealii::Quadrature<dim>> quadrature;
          if (!entry->reusable)
            {
              std::vector<dealii::Point<dim>> reference_points;
              reference_points.reserve(source_values_->n_quadrature_points);
              for (unsigned int q = 0; q < source_values_->n_quadrature_points;
                   ++q)
                reference_points.push_back(
                  input_field.mapping().transform_real_to_unit_cell(
                    input_cell, source_values_->quadrature_point(q)));
              quadrature = std::make_unique<dealii::Quadrature<dim>>(
                reference_points,
                std::vector<double>(reference_points.size(), 1.));
            }
          entry->values = std::make_unique<dealii::FEValues<dim, spacedim>>(
            input_field.mapping(),
            input_cell->get_fe(),
            entry->reusable ? source_values_->get_quadrature() : *quadrature,
            dealii::update_values | update_flags_);
          entry->values->reinit(input_cell);
          update_coefficients(observable, *entry);

          return *entries_.emplace(key, std::move(entry)).first->second;
        }

      private:
        template <typename ObservableType_>
        void
        update_coefficients(const ObservableType_ &observable,
                            Entry                 &entry) const
        {
          using VectorType =
            std::decay_t<decltype(context_.state(observable.source_field()))>;
          const auto &input_field = observable.source();
          const auto  input_cell =
            source_cell_->as_dof_handler_iterator(input_field.dof_handler());
          std::vector<dealii::types::global_dof_index> indices(
            input_cell->get_fe().n_dofs_per_cell());
          input_cell->get_dof_indices(indices);
          entry.positions.clear();
          entry.execution_indices.clear();
          entry.positions.reserve(indices.size());
          entry.execution_indices.reserve(indices.size());
          for (unsigned int i = 0; i < indices.size(); ++i)
            if (input_field.has_execution_index(indices[i]))
              {
                entry.positions.push_back(i);
                entry.execution_indices.push_back(
                  input_field.execution_index(indices[i]));
              }
          entry.execution_values.resize(entry.execution_indices.size());
          const auto &coefficients =
            observable.is_frozen() ?
              observable.template frozen_values<VectorType>() :
              input_states_.at(observable.source_field());
          input_field.constraints().get_dof_values(
            coefficients,
            entry.execution_indices.begin(),
            entry.execution_values.begin(),
            entry.execution_values.end());
          entry.cell_id          = source_cell_->id();
          entry.cell_initialized = true;
        }

        const Context                        &context_;
        const InputStates                    &input_states_;
        Cell                                  source_cell_;
        const SourceValues                   *source_values_;
        const dealii::Mapping<dim, spacedim> *integration_mapping_;
        dealii::UpdateFlags                   update_flags_;
        bool                                  reuse_across_cells_;
        mutable std::map<const void *, std::unique_ptr<Entry>> entries_;
      };

      template <typename VectorType, typename ObservableType_, typename Cache>
      static typename std::decay_t<ObservableType_>::value_type
      primitive_value(const ObservableType_ &observable,
                      const Cache           &input_cache,
                      const unsigned int     q)
      {
        using Observable = std::decay_t<decltype(observable)>;
        using Value      = typename Observable::value_type;

        const auto &input_field = observable.source();
        const auto &entry       = input_cache.get(observable);
        const auto &view        = (*entry.values)[input_field.extractor()];
        Value       result      = Value{};
        for (unsigned int i = 0; i < entry.positions.size(); ++i)
          result += entry.execution_values[i] *
                    observable.operation()(view, entry.positions[i], q);
        return observable.scale() * result;
      }

      template <typename ObservableType_, typename Cache>
      static typename std::decay_t<ObservableType_>::value_type
      primitive_derivative(const ObservableType_ &observable,
                           const FieldId          field,
                           const Cache           &input_cache,
                           const unsigned int     q,
                           const unsigned int     basis)
      {
        using Observable = std::decay_t<decltype(observable)>;
        using Value      = typename Observable::value_type;
        if (observable.is_frozen() || observable.source_field() != field)
          return Value{};
        const auto &input_field = observable.source();
        const auto &entry       = input_cache.get(observable);
        AssertIndexRange(basis, entry.values->dofs_per_cell);
        const auto &view = (*entry.values)[input_field.extractor()];
        return observable.operation()(view, basis, q);
      }

      template <typename VectorType, typename SourceValues, typename Cache>
      static typename std::decay_t<ObservableType>::value_type
      value_at(const ObservableType &observable,
               const SourceValues   &source_values,
               const Cache          &input_cache,
               const unsigned int    q,
               const double          time)
      {
        const auto evaluator =
          [&](const auto &input, const auto &, const double) {
            return primitive_value<VectorType>(input, input_cache, q);
          };
        return evaluate_observable_input(observable,
                                         source_values.quadrature_point(q),
                                         time,
                                         evaluator);
      }

      template <typename VectorType, typename SourceValues, typename Cache>
      static typename std::decay_t<ObservableType>::value_type
      derivative_at(const ObservableType &observable,
                    const FieldId         field,
                    const SourceValues   &source_values,
                    const Cache          &input_cache,
                    const unsigned int    q,
                    const unsigned int    basis,
                    const double          time)
      {
        const auto evaluator =
          [&](const auto &input, const auto &, const double) {
            return primitive_value<VectorType>(input, input_cache, q);
          };
        const auto derivative_evaluator = [&](const auto   &input,
                                              const FieldId input_field,
                                              const auto &,
                                              const double) {
          return primitive_derivative(
            input, input_field, input_cache, q, basis);
        };
        return linearize_observable_input(observable,
                                          field,
                                          source_values.quadrature_point(q),
                                          time,
                                          evaluator,
                                          derivative_evaluator);
      }

      template <typename VectorType, typename FieldType>
      static void
      distribute_cell_residual(
        const FieldType                                    &field,
        const std::vector<dealii::types::global_dof_index> &native_indices,
        const dealii::Vector<double>                       &cell_values,
        VectorType                                         &result)
      {
        std::vector<dealii::types::global_dof_index> indices;
        std::vector<unsigned int>                    positions;
        for (unsigned int i = 0; i < native_indices.size(); ++i)
          if (field.has_execution_index(native_indices[i]))
            {
              indices.push_back(field.execution_index(native_indices[i]));
              positions.push_back(i);
            }
        dealii::Vector<double> values(indices.size());
        for (unsigned int i = 0; i < indices.size(); ++i)
          values[i] = cell_values[positions[i]];
        field.constraints().distribute_local_to_global(values, indices, result);
      }

      // All ranks import state coefficients before entering local cell loops.
      // Empty ranks must participate in the distributed vector setup as well.
      template <typename VectorType, typename Context>
      static std::map<FieldId, VectorType>
      read_input_states(const ObservableType &observable,
                        const Context        &context)
      {
        std::map<FieldId, VectorType> result;
        for (const auto id : observable.dependencies())
          {
            const auto &field  = observable.source_for(id);
            auto       &values = result[id];
            if constexpr (has_distributed_reinit<VectorType,
                                                 SourceField>::value)
              {
                values.reinit(field.locally_owned_dofs(),
                              field.locally_relevant_dofs(),
                              field.space().mpi_communicator());
                values = context.state(id);
                values.update_ghost_values();
              }
            else
              values = context.state(id);
          }
        return result;
      }

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static std::shared_ptr<VectorType>
      residual(const ObservableType   &observable,
               const TargetExpression &target,
               const Context          &context,
               const Support          &support = {})
      {
        const auto &source       = observable.source();
        const auto  target_field = target.source();
        assert_dependencies_match_source_mesh(observable);
        assert_support_matches(support, source, target_field);
        AssertThrow(SourceField::dimension() == dim &&
                      &source.dof_handler().get_triangulation() ==
                        &target_field.dof_handler().get_triangulation(),
                    dealii::ExcMessage(
                      "Nonlinear weak terms currently require matching "
                      "source and target meshes."));

        const auto input_states =
          read_input_states<VectorType>(observable, context);

        if constexpr (uses_support_quadratures<Support>())
          {
            const auto cell_quadratures =
              make_support_quadratures(support, source, target_field);
            auto result = std::make_shared<VectorType>();
            reinit_result(*result, target_field);
            *result = 0.;
            for (const auto &[cell_id, cell_quadrature] : cell_quadratures)
              {
                (void)cell_id;
                const auto target_cell =
                  cell_quadrature.cell->as_dof_handler_iterator(
                    target_field.dof_handler());
                if (!target_cell->is_locally_owned())
                  continue;
                const dealii::Quadrature<dim> point_quadrature(
                  cell_quadrature.reference_points,
                  std::vector<double>(cell_quadrature.reference_points.size(),
                                      1.));
                const auto flags = dealii::update_quadrature_points |
                                   observable.update_flags() |
                                   target.update_flags();
                dealii::FEValues<dim, spacedim> source_values(
                  source.mapping(),
                  source.space().finite_element(),
                  point_quadrature,
                  flags);
                dealii::FEValues<dim, spacedim> target_values(
                  target_field.mapping(),
                  target_field.space().finite_element(),
                  point_quadrature,
                  flags);
                source_values.reinit(cell_quadrature.cell);
                CellInputCache input_cache(context,
                                           input_states,
                                           cell_quadrature.cell,
                                           source_values,
                                           source.mapping(),
                                           observable.update_flags());
                target_values.reinit(target_cell);
                std::vector<dealii::types::global_dof_index> target_indices(
                  target_cell->get_fe().n_dofs_per_cell());
                target_cell->get_dof_indices(target_indices);
                dealii::Vector<double> cell_residual(target_indices.size());
                const auto            &target_view =
                  target_values[target_field.extractor()];
                for (unsigned int q = 0;
                     q < cell_quadrature.reference_points.size();
                     ++q)
                  {
                    const auto value = value_at<VectorType>(observable,
                                                            source_values,
                                                            input_cache,
                                                            q,
                                                            context.time());
                    for (unsigned int i = 0; i < target_indices.size(); ++i)
                      if (target_field.has_execution_index(target_indices[i]))
                        cell_residual[i] +=
                          target.scale() *
                          natural_pairing(
                            value, target.operation()(target_view, i, q)) *
                          cell_quadrature.physical_weights[q];
                  }
                distribute_cell_residual(target_field,
                                         target_indices,
                                         cell_residual,
                                         *result);
              }
            result->compress(dealii::VectorOperation::add);
            return result;
          }

        const auto degree =
          std::max(observable.polynomial_degree(),
                   target_field.space().finite_element().degree);
        const auto quadrature = support_quadrature(
          support,
          compatibility::gauss_quadrature<dim>(
            target_field.space().finite_element().reference_cell(),
            degree + 1));
        const auto flags = dealii::update_JxW_values |
                           dealii::update_quadrature_points |
                           observable.update_flags() | target.update_flags();
        const dealii::hp::MappingCollection<dim, spacedim> source_mappings(
          source.mapping());
        const dealii::hp::MappingCollection<dim, spacedim> target_mappings(
          target_field.mapping());
        const dealii::hp::QCollection<dim>  quadratures(quadrature);
        dealii::hp::FEValues<dim, spacedim> source_hp_values(
          source_mappings,
          source.dof_handler().get_fe_collection(),
          quadratures,
          flags);
        dealii::hp::FEValues<dim, spacedim> target_hp_values(
          target_mappings,
          target_field.dof_handler().get_fe_collection(),
          quadratures,
          flags);

        auto result = std::make_shared<VectorType>();
        reinit_result(*result, target_field);
        *result = 0.;
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        using Cache = CellInputCache<
          Context,
          typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator,
          dealii::FEValues<dim, spacedim>>;
        std::unique_ptr<Cache> input_cache;
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned())
            {
              if (!support_includes(support, source_cell))
                continue;
              const auto target_cell = source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
              if (!target_cell->is_locally_owned())
                continue;
              source_hp_values.reinit(source_cell);
              target_hp_values.reinit(target_cell);
              const auto &source_values =
                source_hp_values.get_present_fe_values();
              const auto &target_values =
                target_hp_values.get_present_fe_values();
              if (input_cache)
                input_cache->reinit(source_cell, source_values);
              else
                input_cache = std::make_unique<Cache>(context,
                                                      input_states,
                                                      source_cell,
                                                      source_values,
                                                      source.mapping(),
                                                      observable.update_flags(),
                                                      true);
              target_indices.resize(target_cell->get_fe().n_dofs_per_cell());
              target_cell->get_dof_indices(target_indices);
              dealii::Vector<double> cell_residual(target_indices.size());
              const auto &target_view = target_values[target_field.extractor()];
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                {
                  if (support_point_weight(support, source_cell, q) == 0.)
                    continue;
                  const auto value = value_at<VectorType>(
                    observable, source_values, *input_cache, q, context.time());
                  for (unsigned int i = 0; i < target_indices.size(); ++i)
                    if (target_field.has_execution_index(target_indices[i]))
                      cell_residual[i] +=
                        target.scale() *
                        natural_pairing(value,
                                        target.operation()(target_view, i, q)) *
                        target_values.JxW(q) *
                        support_point_weight(support, source_cell, q);
                }
              distribute_cell_residual(target_field,
                                       target_indices,
                                       cell_residual,
                                       *result);
            }
        result->compress(dealii::VectorOperation::add);
        return result;
      }

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static MatrixStorage<MatrixType>
      jacobian(const ObservableType   &observable,
               const TargetExpression &target,
               const FieldId           field,
               const Context          &context,
               const Support          &support = {})
      {
        const auto &source        = observable.source();
        const auto &column_source = observable.source_for(field);
        const auto  target_field  = target.source();
        assert_dependencies_match_source_mesh(observable);
        assert_support_matches(support, source, target_field);
        AssertThrow(SourceField::dimension() == dim &&
                      &source.dof_handler().get_triangulation() ==
                        &target_field.dof_handler().get_triangulation() &&
                      &column_source.dof_handler().get_triangulation() ==
                        &source.dof_handler().get_triangulation(),
                    dealii::ExcMessage(
                      "Nonlinear weak terms currently require matching "
                      "source meshes."));

        const auto input_states =
          read_input_states<VectorType>(observable, context);

        if constexpr (uses_support_quadratures<Support>())
          {
            const auto cell_quadratures =
              make_support_quadratures(support, source, target_field);
            dealii::DynamicSparsityPattern sparsity(
              target_field.locally_owned_dofs().size(),
              column_source.locally_owned_dofs().size(),
              target_field.locally_relevant_dofs());
            for (const auto &[cell_id, cell_quadrature] : cell_quadratures)
              {
                (void)cell_id;
                const auto target_cell =
                  cell_quadrature.cell->as_dof_handler_iterator(
                    target_field.dof_handler());
                const auto column_cell =
                  cell_quadrature.cell->as_dof_handler_iterator(
                    column_source.dof_handler());
                std::vector<dealii::types::global_dof_index> target_indices(
                  target_cell->get_fe().n_dofs_per_cell());
                std::vector<dealii::types::global_dof_index> column_indices(
                  column_cell->get_fe().n_dofs_per_cell());
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
                  true);
              }

            if (dynamic_cast<
                  const dealii::parallel::TriangulationBase<dim, spacedim> *>(
                  &target_field.dof_handler().get_triangulation()) != nullptr)
              dealii::SparsityTools::distribute_sparsity_pattern(
                sparsity,
                target_field.locally_owned_dofs(),
                target_field.space().mpi_communicator(),
                target_field.locally_relevant_dofs());

            MatrixStorage<MatrixType> result;
            result.matrix = std::make_shared<MatrixType>();
            result.sparsity =
              initialize_weak_matrix(*result.matrix,
                                     target_field.locally_owned_dofs(),
                                     column_source.locally_owned_dofs(),
                                     sparsity,
                                     target_field.space().mpi_communicator());

            for (const auto &[cell_id, cell_quadrature] : cell_quadratures)
              {
                (void)cell_id;
                const auto target_cell =
                  cell_quadrature.cell->as_dof_handler_iterator(
                    target_field.dof_handler());
                if (!target_cell->is_locally_owned())
                  continue;
                const auto column_cell =
                  cell_quadrature.cell->as_dof_handler_iterator(
                    column_source.dof_handler());
                const dealii::Quadrature<dim> point_quadrature(
                  cell_quadrature.reference_points,
                  std::vector<double>(cell_quadrature.reference_points.size(),
                                      1.));
                const auto flags = dealii::update_quadrature_points |
                                   observable.update_flags() |
                                   target.update_flags();
                dealii::FEValues<dim, spacedim> source_values(
                  source.mapping(),
                  source.space().finite_element(),
                  point_quadrature,
                  flags);
                dealii::FEValues<dim, spacedim> target_values(
                  target_field.mapping(),
                  target_field.space().finite_element(),
                  point_quadrature,
                  flags);
                source_values.reinit(cell_quadrature.cell);
                CellInputCache input_cache(context,
                                           input_states,
                                           cell_quadrature.cell,
                                           source_values,
                                           source.mapping(),
                                           observable.update_flags());
                target_values.reinit(target_cell);
                std::vector<dealii::types::global_dof_index> target_indices(
                  target_cell->get_fe().n_dofs_per_cell());
                std::vector<dealii::types::global_dof_index> column_indices(
                  column_cell->get_fe().n_dofs_per_cell());
                target_cell->get_dof_indices(target_indices);
                column_cell->get_dof_indices(column_indices);
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
                const auto                &target_view =
                  target_values[target_field.extractor()];
                for (unsigned int q = 0;
                     q < cell_quadrature.reference_points.size();
                     ++q)
                  for (unsigned int j = 0; j < column_positions.size(); ++j)
                    {
                      const auto derivative =
                        derivative_at<VectorType>(observable,
                                                  field,
                                                  source_values,
                                                  input_cache,
                                                  q,
                                                  column_positions[j],
                                                  context.time());
                      for (unsigned int i = 0; i < target_positions.size(); ++i)
                        local(i, j) += target.scale() *
                                       natural_pairing(
                                         derivative,
                                         target.operation()(target_view,
                                                            target_positions[i],
                                                            q)) *
                                       cell_quadrature.physical_weights[q];
                    }
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

        const auto degree =
          std::max(observable.polynomial_degree(),
                   target_field.space().finite_element().degree);
        const auto quadrature = support_quadrature(
          support,
          compatibility::gauss_quadrature<dim>(
            target_field.space().finite_element().reference_cell(),
            degree + 1));
        const auto flags = dealii::update_JxW_values |
                           dealii::update_quadrature_points |
                           observable.update_flags() | target.update_flags();

        dealii::DynamicSparsityPattern sparsity(
          target_field.locally_owned_dofs().size(),
          column_source.locally_owned_dofs().size(),
          target_field.locally_relevant_dofs());
        std::vector<dealii::types::global_dof_index> target_indices(
          target_field.space().finite_element().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> column_indices(
          column_source.space().finite_element().n_dofs_per_cell());
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned())
            {
              if (!support_includes(support, source_cell))
                continue;
              const auto target_cell = source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
              if (!target_cell->is_locally_owned())
                continue;
              const auto column_cell = source_cell->as_dof_handler_iterator(
                column_source.dof_handler());
              target_indices.resize(target_cell->get_fe().n_dofs_per_cell());
              target_cell->get_dof_indices(target_indices);
              column_indices.resize(column_cell->get_fe().n_dofs_per_cell());
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
                true);
            }

        if (dynamic_cast<
              const dealii::parallel::TriangulationBase<dim, spacedim> *>(
              &target_field.dof_handler().get_triangulation()) != nullptr)
          dealii::SparsityTools::distribute_sparsity_pattern(
            sparsity,
            target_field.locally_owned_dofs(),
            target_field.space().mpi_communicator(),
            target_field.locally_relevant_dofs());

        MatrixStorage<MatrixType> result;
        result.matrix = std::make_shared<MatrixType>();
        result.sparsity =
          initialize_weak_matrix(*result.matrix,
                                 target_field.locally_owned_dofs(),
                                 column_source.locally_owned_dofs(),
                                 sparsity,
                                 target_field.space().mpi_communicator());

        const dealii::hp::MappingCollection<dim, spacedim> source_mappings(
          source.mapping());
        const dealii::hp::MappingCollection<dim, spacedim> target_mappings(
          target_field.mapping());
        const dealii::hp::QCollection<dim>  quadratures(quadrature);
        dealii::hp::FEValues<dim, spacedim> source_hp_values(
          source_mappings,
          source.dof_handler().get_fe_collection(),
          quadratures,
          flags);
        dealii::hp::FEValues<dim, spacedim> target_hp_values(
          target_mappings,
          target_field.dof_handler().get_fe_collection(),
          quadratures,
          flags);
        using Cache = CellInputCache<
          Context,
          typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator,
          dealii::FEValues<dim, spacedim>>;
        std::unique_ptr<Cache> input_cache;
        for (const auto &source_cell :
             source.dof_handler().active_cell_iterators())
          if (source_cell->is_locally_owned())
            {
              if (!support_includes(support, source_cell))
                continue;
              const auto target_cell = source_cell->as_dof_handler_iterator(
                target_field.dof_handler());
              if (!target_cell->is_locally_owned())
                continue;
              const auto column_cell = source_cell->as_dof_handler_iterator(
                column_source.dof_handler());
              source_hp_values.reinit(source_cell);
              target_hp_values.reinit(target_cell);
              const auto &source_values =
                source_hp_values.get_present_fe_values();
              const auto &target_values =
                target_hp_values.get_present_fe_values();
              if (input_cache)
                input_cache->reinit(source_cell, source_values);
              else
                input_cache = std::make_unique<Cache>(context,
                                                      input_states,
                                                      source_cell,
                                                      source_values,
                                                      source.mapping(),
                                                      observable.update_flags(),
                                                      true);
              target_indices.resize(target_cell->get_fe().n_dofs_per_cell());
              target_cell->get_dof_indices(target_indices);
              column_indices.resize(column_cell->get_fe().n_dofs_per_cell());
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
                {
                  if (support_point_weight(support, source_cell, q) == 0.)
                    continue;
                  for (unsigned int j = 0; j < column_positions.size(); ++j)
                    {
                      const auto derivative =
                        derivative_at<VectorType>(observable,
                                                  field,
                                                  source_values,
                                                  *input_cache,
                                                  q,
                                                  column_positions[j],
                                                  context.time());
                      for (unsigned int i = 0; i < target_positions.size(); ++i)
                        local(i, j) +=
                          target.scale() *
                          natural_pairing(
                            derivative,
                            target.operation()(target_view,
                                               target_positions[i],
                                               q)) *
                          target_values.JxW(q) *
                          support_point_weight(support, source_cell, q);
                    }
                }
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

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static MaterializedOperator<VectorType, MatrixType>
      linearize_matrix(const ObservableType   &observable,
                       const TargetExpression &target,
                       const FieldId           field,
                       const Context          &context,
                       const Support          &support = {})
      {
        const auto storage = jacobian<VectorType, MatrixType>(
          observable, target, field, context, support);
        const auto matrix   = storage.matrix;
        const auto sparsity = storage.sparsity;
        const auto matrix_view =
          ImmersX::matrix_operator<VectorType, MatrixType>(*matrix).view;
        MaterializedOperator<VectorType, MatrixType> result;
        result.view.reinit_range_vector =
          [matrix, sparsity, matrix_view](VectorType &vector, const bool omit) {
            matrix_view.reinit_range_vector(vector, omit);
          };
        result.view.reinit_domain_vector =
          [matrix, sparsity, matrix_view](VectorType &vector, const bool omit) {
            matrix_view.reinit_domain_vector(vector, omit);
          };
        result.view.vmult =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.vmult(destination, source);
          };
        result.view.vmult_add =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.vmult_add(destination, source);
          };
        result.view.Tvmult =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.Tvmult(destination, source);
          };
        result.view.Tvmult_add =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.Tvmult_add(destination, source);
          };
        result.materialize      = [matrix]() { return matrix; };
        result.materialize_into = [matrix](MatrixType &destination) {
          copy_matrix(destination, *matrix);
        };
        return result;
      }

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static dealii::LinearOperator<VectorType, VectorType>
      linearize(const ObservableType   &observable,
                const TargetExpression &target,
                const FieldId           field,
                const Context          &context,
                const Support          &support = {})
      {
        return linearize_matrix<VectorType, MatrixType>(
                 observable, target, field, context, support)
          .view;
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
                 cell->get_fe(),
                 quadrature,
                 target.update_flags());
        values.reinit(cell);
        const auto         &view = values[target.source().extractor()];
        std::vector<double> result(cell->get_fe().n_dofs_per_cell());
        for (unsigned int i = 0; i < result.size(); ++i)
          result[i] = target.operation()(view, i, 0);
        return result;
      }

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static std::shared_ptr<VectorType>
      residual(const ObservableType   &observable,
               const TargetExpression &target,
               const Context          &context,
               const Support          &support = {})
      {
        const auto &source       = observable.source();
        const auto  target_field = target.source();
        assert_lifted_support_matches(support, target_field);
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
                      cell->global_active_cell_index() &&
                    support_lifted_point(support, points[q]))
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

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static MatrixStorage<MatrixType>
      jacobian(const ObservableType   &observable,
               const TargetExpression &target,
               const FieldId           field,
               const Context          &context,
               const Support          &support = {})
      {
        const auto &source        = observable.source();
        const auto &column_source = observable.source_for(field);
        const auto  target_field  = target.source();
        assert_lifted_support_matches(support, target_field);
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
          target_field.locally_relevant_dofs());
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
              target_indices.resize(cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(target_indices);
              const auto column_cell =
                cell->as_dof_handler_iterator(column_source.dof_handler());
              column_indices.resize(column_cell->get_fe().n_dofs_per_cell());
              column_cell->get_dof_indices(column_indices);
              std::vector<dealii::types::global_dof_index> rows;
              rows.reserve(target_indices.size());
              for (const auto index : target_indices)
                {
                  const auto row = target_field.execution_index(index);
                  if (row != std::numeric_limits<
                               dealii::types::global_dof_index>::max())
                    rows.push_back(row);
                }
              std::vector<dealii::types::global_dof_index> columns;
              columns.reserve(column_indices.size());
              for (const auto index : column_indices)
                if (column_source.has_execution_index(index))
                  columns.push_back(column_source.execution_index(index));
              target_field.constraints().add_entries_local_to_global(
                rows, column_source.constraints(), columns, sparsity, true);
            }

        if (dynamic_cast<const dealii::parallel::TriangulationBase<
              TargetField::dimension(),
              TargetField::spacedimension()> *>(
              &target_field.dof_handler().get_triangulation()) != nullptr)
          dealii::SparsityTools::distribute_sparsity_pattern(
            sparsity,
            target_field.locally_owned_dofs(),
            target_field.space().mpi_communicator(),
            target_field.locally_relevant_dofs());

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
              target_indices.resize(cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(target_indices);
              const auto column_cell =
                cell->as_dof_handler_iterator(column_source.dof_handler());
              column_indices.resize(column_cell->get_fe().n_dofs_per_cell());
              column_cell->get_dof_indices(column_indices);
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
              std::vector<unsigned int>                    active_columns;
              std::vector<dealii::types::global_dof_index> columns;
              for (unsigned int j = 0; j < column_indices.size(); ++j)
                if (column_source.has_execution_index(column_indices[j]))
                  {
                    active_columns.push_back(j);
                    columns.push_back(
                      column_source.execution_index(column_indices[j]));
                  }
              dealii::FullMatrix<double> local(active_rows.size(),
                                               active_columns.size());
              for (std::size_t q = 0; q < points.size(); ++q)
                if (points[q].source_entity_id ==
                      cell->global_active_cell_index() &&
                    support_lifted_point(support, points[q]))
                  {
                    const auto basis = target_basis(target, cell, points[q]);
                    if (active_rows.empty())
                      continue;
                    for (unsigned int j = 0; j < active_columns.size(); ++j)
                      {
                        // The trial derivative is shared by all test rows.
                        const auto derivative =
                          radial_value(observable.linearize_point(
                                         context, field, q, active_columns[j]),
                                       points[q]);
                        for (unsigned int row = 0; row < active_rows.size();
                             ++row)
                          local(row, j) += target.scale() * derivative *
                                           basis[active_rows[row]] *
                                           points[q].weight;
                      }
                  }
              target_field.constraints().distribute_local_to_global(
                local,
                rows,
                column_source.constraints(),
                columns,
                *result.matrix);
            }
        compress_weak_matrix(*result.matrix);
        return result;
      }

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static MaterializedOperator<VectorType, MatrixType>
      linearize_matrix(const ObservableType   &observable,
                       const TargetExpression &target,
                       const FieldId           field,
                       const Context          &context,
                       const Support          &support = {})
      {
        const auto storage = jacobian<VectorType, MatrixType>(
          observable, target, field, context, support);
        const auto matrix   = storage.matrix;
        const auto sparsity = storage.sparsity;
        const auto matrix_view =
          ImmersX::matrix_operator<VectorType, MatrixType>(*matrix).view;
        MaterializedOperator<VectorType, MatrixType> result;
        result.view.reinit_range_vector =
          [matrix, sparsity, matrix_view](VectorType &vector, const bool omit) {
            matrix_view.reinit_range_vector(vector, omit);
          };
        result.view.reinit_domain_vector =
          [matrix, sparsity, matrix_view](VectorType &vector, const bool omit) {
            matrix_view.reinit_domain_vector(vector, omit);
          };
        result.view.vmult =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.vmult(destination, source);
          };
        result.view.vmult_add =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.vmult_add(destination, source);
          };
        result.view.Tvmult =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.Tvmult(destination, source);
          };
        result.view.Tvmult_add =
          [matrix, sparsity, matrix_view](VectorType       &destination,
                                          const VectorType &source) {
            matrix_view.Tvmult_add(destination, source);
          };
        result.materialize      = [matrix]() { return matrix; };
        result.materialize_into = [matrix](MatrixType &destination) {
          copy_matrix(destination, *matrix);
        };
        return result;
      }

      template <typename VectorType,
                typename MatrixType,
                typename Context,
                typename Support = InferredIntegrationSupport>
      static dealii::LinearOperator<VectorType, VectorType>
      linearize(const ObservableType   &observable,
                const TargetExpression &target,
                const FieldId           field,
                const Context          &context,
                const Support          &support = {})
      {
        return linearize_matrix<VectorType, MatrixType>(
                 observable, target, field, context, support)
          .view;
      }
    };

    /** Assemble a full-dimensional FE quantity against a lifted test. */
    template <typename ObservableType, typename TargetLift>
    struct LiftedTargetWeakAssembly
    {
      using TargetField = typename TargetLift::source_field_type;
      using Point       = typename TargetLift::Point;

      template <typename VectorType, typename MatrixType>
      using AffineData = WeakAffineAssembly<VectorType, MatrixType>;

      template <typename MatrixType>
      struct MatrixStorage
      {
        std::shared_ptr<MatrixType>              matrix;
        std::shared_ptr<dealii::SparsityPattern> sparsity;
      };

      template <typename VectorType,
                typename MatrixType,
                typename Support = InferredIntegrationSupport>
      static MatrixStorage<MatrixType>
      assemble(const ObservableType        &observable,
               const TargetLift            &target,
               std::shared_ptr<VectorType> *affine_rhs = nullptr,
               std::shared_ptr<AffineData<VectorType, MatrixType>>
                             *affine_data = nullptr,
               const Support &support     = {})
      {
        using SourceField = typename ObservableType::source_field_type;
        static_assert(SourceField::dimension() == SourceField::spacedimension(),
                      "A lifted test needs a full-dimensional trial field.");
        const auto &source       = observable.source();
        const auto &target_field = target.source();
        assert_lifted_support_matches(support, target_field);
        const auto *source_tria = &source.dof_handler().get_triangulation();

        const auto         points = target.lifted_points();
        const unsigned int local_target_dofs =
          points.empty() ? 0u : points.front().dof_indices.size();
        const unsigned int n_target_dofs =
          dealii::Utilities::MPI::max(local_target_dofs,
                                      target_field.space().mpi_communicator());
        AssertThrow(n_target_dofs > 0,
                    dealii::ExcMessage(
                      "A lifted target has no active target DoFs."));
        for (const auto &point : points)
          AssertDimension(point.dof_indices.size(), n_target_dofs);
        const unsigned int n_properties =
          1 + 2 * n_target_dofs + SourceField::spacedimension();
        std::vector<dealii::types::global_dof_index> target_indices(
          n_target_dofs);

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
            if (!support_lifted_point(support, point))
              continue;
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
        const auto expected_points =
          dealii::Utilities::MPI::sum(physical_points.size(),
                                      source.space().mpi_communicator());
        AssertThrow(
          distribution.get_particles().n_global_particles() == expected_points,
          dealii::ExcMessage(
            "Some lifted integration points lie outside the background mesh."));

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
              std::vector<dealii::types::global_dof_index> target_indices(
                n_target_dofs);
              for (unsigned int i = 0; i < n_target_dofs; ++i)
                target_indices[i] =
                  static_cast<dealii::types::global_dof_index>(data[1 + i]);
              source.constraints().add_entries_local_to_global(
                source_indices,
                target_field.constraints(),
                target_indices,
                sparsity,
                true);
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

        if (affine_data != nullptr)
          {
            *affine_data = std::make_shared<AffineData<VectorType, MatrixType>>(
              source.constraints(),
              target_field.constraints(),
              target_field.locally_owned_dofs(),
              target_field.space().mpi_communicator());
          }
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
                  {
                    const auto trial = observable.operation()(view, i, 0);
                    if constexpr (std::is_arithmetic_v<decltype(trial)>)
                      local(i, j) =
                        data[0] * data[1 + n_target_dofs + j] * trial;
                    else
                      local(i, j) = data[0] * data[1 + n_target_dofs + j] *
                                    (trial * normal);
                  }
              for (unsigned int j = 0; j < n_target_dofs; ++j)
                target_indices[j] =
                  static_cast<dealii::types::global_dof_index>(data[1 + j]);
              if (affine_data != nullptr)
                {
                  dealii::FullMatrix<double> transposed;
                  transposed.copy_transposed(local);
                  (*affine_data)
                    ->add(transposed, source_indices, target_indices);
                }
              source.constraints().distribute_local_to_global(
                local,
                source_indices,
                target_field.constraints(),
                target_indices,
                *result.matrix);
            }
        compress_weak_matrix(*result.matrix);
        result.matrix = ImmersX::detail::transpose_matrix(result.matrix);
        if (affine_rhs != nullptr && affine_data != nullptr)
          {
            *affine_rhs = std::make_shared<VectorType>();
            (*affine_data)->finish(**affine_rhs);
          }
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
      auto point_indices = [&load, n = source_points.size()] {
        std::vector<dealii::types::global_dof_index> result;
        result.reserve(n);
        for (std::size_t q = 0; q < n; ++q)
          result.push_back(load.point_index(q));
        return result;
      }();

      const auto point_indices_storage =
        std::make_shared<const decltype(point_indices)>(
          std::move(point_indices));

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
        [distribution, point_indices_storage](const QuantityVector &source) {
          const auto            &point_indices = *point_indices_storage;
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
      result.Tvmult = [entries, distribution, point_indices_storage](
                        QuantityVector &destination, const VectorType &source) {
        const auto &point_indices = *point_indices_storage;
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
      result.Tvmult_add = [entries,
                           distribution,
                           point_indices_storage](QuantityVector   &destination,
                                                  const VectorType &source) {
        const auto &point_indices = *point_indices_storage;
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

  /** Region used by a solver-neutral FE weak term. */
  struct WeakTermRegion
  {
    enum class Kind
    {
      volume,
      boundary
    };

    Kind                                 kind = Kind::volume;
    std::set<dealii::types::boundary_id> boundary_ids;
  };

  /** A solver-neutral FE weak term contributed to a residual row. */
  template <typename TrialExpression,
            typename TestExpression,
            typename Support = InferredIntegrationSupport>
  class WeakTerm
  {
  public:
    using target_type      = TestExpression;
    using trial_field_type = typename TrialExpression::source_field_type;

    WeakTerm(TrialExpression observable, TestExpression target)
      : observable_(std::move(observable))
      , target_(std::move(target))
    {}

    WeakTerm(TrialExpression observable, TestExpression target, Support support)
      : observable_(std::move(observable))
      , target_(std::move(target))
      , support_(std::move(support))
    {}

    template <int support_dim, int support_spacedim>
    auto
    on(IntegrationSupport<support_dim, support_spacedim> support) const
    {
      static_assert(
        support_dim == TestExpression::dimension() ||
          support_dim == trial_field_type::dimension() ||
          (support_dim > 0 && support_dim <= TestExpression::dimension() &&
           TestExpression::dimension() == trial_field_type::dimension()),
        "Integration support must match a trial or test dimension, or "
        "integrate a same-dimensional trial/test pairing.");
      static_assert(support_spacedim == TestExpression::spacedimension(),
                    "Integration support and test field must share an "
                    "embedding dimension.");
      auto result = WeakTerm<TrialExpression,
                             TestExpression,
                             IntegrationSupport<support_dim, support_spacedim>>(
        observable_, target_, std::move(support));
      result.cache_assembly_ = cache_assembly_;
      return result;
    }

    template <int support_dim, int support_spacedim>
    auto
    on(BoundaryIntegrationSupport<support_dim, support_spacedim> support) const
    {
      static_assert(support_dim == TestExpression::dimension(),
                    "Boundary support dimension must match the test field "
                    "dimension.");
      static_assert(support_spacedim == TestExpression::spacedimension(),
                    "Boundary support and test field must share an "
                    "embedding dimension.");
      auto result =
        WeakTerm<TrialExpression,
                 TestExpression,
                 BoundaryIntegrationSupport<support_dim, support_spacedim>>(
          observable_, target_, std::move(support));
      result.cache_assembly_ = cache_assembly_;
      return result;
    }

    template <typename PhysicalSupport,
              std::enable_if_t<is_physical_quadrature_support<
                                 std::decay_t<PhysicalSupport>>::value,
                               int> = 0>
    auto
    on(PhysicalSupport support) const
    {
      static_assert(PhysicalSupport::ambient_dimension ==
                      TestExpression::spacedimension(),
                    "Physical support and test field must share an "
                    "embedding dimension.");
      static_assert(PhysicalSupport::intrinsic_dimension <=
                      TestExpression::dimension(),
                    "Physical support dimension cannot exceed the test "
                    "field dimension.");
      auto result = WeakTerm<TrialExpression, TestExpression, PhysicalSupport>(
        observable_, target_, std::move(support));
      result.cache_assembly_ = cache_assembly_;
      return result;
    }

    WeakTerm
    on_boundary(const dealii::types::boundary_id id) const
    {
      return on_boundary(std::set<dealii::types::boundary_id>{id});
    }

    WeakTerm
    on_boundary(const std::set<dealii::types::boundary_id> &boundary_ids) const
    {
      auto result                 = *this;
      result.region_.kind         = WeakTermRegion::Kind::boundary;
      result.region_.boundary_ids = boundary_ids;
      return result;
    }

    WeakTerm
    in_volume() const
    {
      auto result         = *this;
      result.region_.kind = WeakTermRegion::Kind::volume;
      result.region_.boundary_ids.clear();
      return result;
    }

    const Support &
    integration_support() const
    {
      return support_;
    }

    /** Retain a linear pairing after its first assembly.
     * Meshes, mappings, lifted reference geometry, integration support, FE
     * spaces and homogeneous constraint coefficients must stay fixed while
     * this term is used. State values and affine constraint offsets may change.
     * Rebuild the term after changing any fixed data. Nonlinear trials are
     * rejected; the default keeps assembling the current pairing on demand.
     */
    WeakTerm
    with_cached_assembly() const
    {
      AssertThrow(detail::has_linear_cached_trial<TrialExpression>::value,
                  dealii::ExcMessage(
                    "Cached weak-term assembly requires a linear trial."));
      auto result            = *this;
      result.cache_assembly_ = true;
      return result;
    }

    template <typename VectorType, typename MatrixType>
    FieldId
    add(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      if (context_update_)
        builder.context_update([update = context_update_](const auto &context) {
          update(context.time());
        });
      using Model          = SemiDiscreteModel<VectorType, MatrixType>;
      const auto target_id = target_.source().field_id();
      auto       term      = builder.term(target_id, "weak_term");
      if constexpr (detail::is_lifted_observable<TrialExpression>::value &&
                    trial_field_type::dimension() <
                      TestExpression::source_field_type::dimension())
        {
          static_assert(
            std::is_same_v<TestExpression, decltype(test(target_.source()))>,
            "A lifted reaction currently pairs with a value test.");
          const auto bulk = value(target_.source());
          auto       reverse =
            WeakTerm<decltype(bulk), TrialExpression, Support>(bulk,
                                                               observable_,
                                                               support_);
          reverse.cache_assembly_ = cache_assembly_;
          const auto pairing =
            reverse.make_pairing(builder, observable_, region_);
          const auto reaction =
            observable_.scale() * target_.scale() *
            ImmersX::transpose_operator(pairing.operator_with_matrix);
          term
            .residual(
              [reaction, observable = observable_](const auto &context) {
                return reaction.view * context.state(observable.source_field());
              })
            .state(observable_.source_field(), reaction);
        }
      else if constexpr (!detail::is_linear_observable<TrialExpression>::value)
        {
          {
            AssertThrow(
              (std::is_same_v<Support, InferredIntegrationSupport> ||
               is_integration_support<Support>::value ||
               is_physical_quadrature_support<Support>::value),
              dealii::ExcMessage(
                "Nonlinear weak terms require cell or physical quadrature "
                "support."));
            if constexpr (is_integration_support<Support>::value)
              {
                if constexpr (Support::dimension() ==
                              trial_field_type::dimension())
                  {
                    support_.assert_matches(
                      observable_.source().dof_handler().get_triangulation());
                    support_.assert_matches(
                      target_.source().dof_handler().get_triangulation());
                  }
              }
          }
          AssertThrow(region_.kind == WeakTermRegion::Kind::volume,
                      dealii::ExcMessage(
                        "Boundary regions currently require linear weak "
                        "terms."));
          using Assembly = std::conditional_t<
            detail::is_lifted_observable<TrialExpression>::value,
            detail::LiftedWeakAssembly<TrialExpression, TestExpression>,
            detail::NonlinearWeakAssembly<TrialExpression, TestExpression>>;
          term.residual([observable = observable_,
                         target     = target_,
                         support    = support_](const auto &context) {
            const auto values = [&] {
              if constexpr (detail::is_lifted_observable<
                              TrialExpression>::value)
                {
                  return Assembly::template residual<VectorType, MatrixType>(
                    observable, target, context, support);
                }
              else
                return Assembly::template residual<VectorType, MatrixType>(
                  observable, target, context, support);
            }();
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
          using MatrixOperatorFactory = typename Model::MatrixOperatorFactory;
          for (const auto field : observable_.dependencies())
            term.state(
              field,
              MatrixOperatorFactory([observable = observable_,
                                     target     = target_,
                                     field,
                                     support = support_](const auto &context) {
                if constexpr (detail::is_lifted_observable<
                                TrialExpression>::value)
                  {
                    return Assembly::template linearize_matrix<VectorType,
                                                               MatrixType>(
                      observable, target, field, context, support);
                  }
                else
                  return Assembly::template linearize_matrix<VectorType,
                                                             MatrixType>(
                    observable, target, field, context, support);
              }));
        }
      else if (observable_.is_frozen())
        {
          const auto pairing = make_pairing(builder, target_, region_);
          const auto frozen  = observable_.template frozen_values<VectorType>();
          term.residual([pairing, frozen](const auto &) {
            const auto                affine_rhs = pairing.affine_rhs_factory ?
                                                     pairing.affine_rhs_factory() :
                                                     pairing.affine_rhs;
            typename Model::Operation result;
            result.reinit_vector =
              pairing.operator_with_matrix.view.reinit_range_vector;
            result.apply =
              [pairing, frozen, affine_rhs](VectorType &destination) {
                pairing.operator_with_matrix.view.vmult(destination, frozen);
                if (affine_rhs != nullptr)
                  destination -= *affine_rhs;
              };
            result.apply_add = [pairing, frozen, affine_rhs](
                                 VectorType &destination) {
              pairing.operator_with_matrix.view.vmult_add(destination, frozen);
              if (affine_rhs != nullptr)
                destination -= *affine_rhs;
            };
            return result;
          });
        }
      else
        {
          const auto pairing   = make_pairing(builder, target_, region_);
          const auto source_id = pairing.source_id;
          typename Model::MatrixOperatorFactory state_factory =
            [pairing](const typename Model::Context &) {
              (void)pairing.matrix;
              (void)pairing.sparsity;
              return pairing.operator_with_matrix;
            };
          term
            .residual([pairing, source_id](const auto &ctx) {
              const auto affine_rhs = pairing.affine_rhs_factory ?
                                        pairing.affine_rhs_factory() :
                                        pairing.affine_rhs;
              if (affine_rhs == nullptr)
                return pairing.operator_with_matrix.view * ctx.state(source_id);

              typename Model::Operation result;
              const auto               *state = &ctx.state(source_id);
              result.reinit_vector =
                pairing.operator_with_matrix.view.reinit_range_vector;
              result.apply =
                [pairing, state, affine_rhs](VectorType &destination) {
                  pairing.operator_with_matrix.view.vmult(destination, *state);
                  destination -= *affine_rhs;
                };
              result.apply_add =
                [pairing, state, affine_rhs](VectorType &destination) {
                  pairing.operator_with_matrix.view.vmult_add(destination,
                                                              *state);
                  destination -= *affine_rhs;
                };
              return result;
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
                                    sign,
                                    support = support_](const auto &context) {
            const auto values =
              Assembly::template residual<VectorType, MatrixType>(
                observable, multiplier_expression, context, support);
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
              OperatorFactory([observable = observable_,
                               multiplier_expression,
                               field,
                               sign,
                               support = support_](const auto &context) {
                return sign *
                       Assembly::template linearize<VectorType, MatrixType>(
                         observable,
                         multiplier_expression,
                         field,
                         context,
                         support);
              }));

          for (const auto field : observable_.dependencies())
            {
              auto participant_term =
                builder.term(field, suffix + ".participant");
              const auto participant_field = observable_.source_for(field);
              participant_term.residual([observable = observable_,
                                         multiplier_expression,
                                         field,
                                         multiplier,
                                         sign,
                                         participant_field,
                                         support =
                                           support_](const auto &context) {
                const auto jacobian =
                  Assembly::template linearize<VectorType, MatrixType>(
                    observable, multiplier_expression, field, context, support);
                const auto  reaction = dealii::transpose_operator(jacobian);
                const auto *multiplier_state = &context.state(multiplier);
                typename Model::Operation result;
                result.reinit_vector =
                  [owned        = participant_field.locally_owned_dofs(),
                   communicator = participant_field.space().mpi_communicator()](
                    VectorType &vector, const bool omit) {
                    if constexpr (detail::NonlinearWeakAssembly<
                                    TrialExpression,
                                    decltype(multiplier_expression)>::
                                    template has_distributed_reinit<
                                      VectorType,
                                      decltype(participant_field)>::value)
                      vector.reinit(owned, communicator, omit);
                    else
                      vector.reinit(owned.size(), omit);
                  };
                result.apply =
                  [reaction, multiplier_state, sign](VectorType &destination) {
                    reaction.vmult(destination, *multiplier_state);
                    destination *= sign;
                  };
                result.apply_add =
                  [reaction,
                   multiplier_state,
                   sign,
                   owned        = participant_field.locally_owned_dofs(),
                   communicator = participant_field.space().mpi_communicator()](
                    VectorType &destination) {
                    VectorType contribution;
                    if constexpr (detail::NonlinearWeakAssembly<
                                    TrialExpression,
                                    decltype(multiplier_expression)>::
                                    template has_distributed_reinit<
                                      VectorType,
                                      decltype(participant_field)>::value)
                      contribution.reinit(owned, communicator, false);
                    else
                      contribution.reinit(owned.size(), false);
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
                                 sign,
                                 support = support_](const auto &context) {
                  const auto jacobian =
                    Assembly::template linearize<VectorType, MatrixType>(
                      observable,
                      multiplier_expression,
                      field,
                      context,
                      support);
                  return sign * dealii::transpose_operator(jacobian);
                }));
            }
        }
      else
        {
          const auto pairing =
            make_pairing(builder, multiplier_expression, region_);
          const auto reaction =
            ImmersX::transpose_operator(pairing.operator_with_matrix);
          const auto source_id = pairing.source_id;
          builder.term(multiplier, suffix + ".multiplier")
            .residual([pairing, source_id, sign](const auto &context) {
              auto       result = sign * (pairing.operator_with_matrix.view *
                                    context.state(source_id));
              const auto affine_rhs = pairing.affine_rhs_factory ?
                                        pairing.affine_rhs_factory() :
                                        pairing.affine_rhs;
              if (affine_rhs != nullptr)
                {
                  const auto apply     = result.apply;
                  const auto apply_add = result.apply_add;
                  result.apply =
                    [apply, affine_rhs, sign](VectorType &destination) {
                      apply(destination);
                      destination.add(-sign, *affine_rhs);
                    };
                  result.apply_add =
                    [apply_add, affine_rhs, sign](VectorType &destination) {
                      apply_add(destination);
                      destination.add(-sign, *affine_rhs);
                    };
                }
              return result;
            })
            .state(source_id, sign * pairing.operator_with_matrix);

          const auto multiplier_field = multiplier_expression.source();
          detail::assert_homogeneous_weak_constraints(
            multiplier_field,
            "Linear constraint reactions require homogeneous multiplier "
            "constraints.");
          builder.term(source_id, suffix + ".participant")
            .residual([reaction, multiplier, sign, multiplier_field](
                        const auto &context) {
              detail::assert_homogeneous_weak_constraints(
                multiplier_field,
                "Linear constraint reactions require homogeneous multiplier "
                "constraints.");
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

    /** Return this term with a callback run before each evaluation context. */
    template <typename Callback>
    WeakTerm
    with_context_update(Callback callback) const
    {
      WeakTerm result        = *this;
      result.context_update_ = std::function<void(double)>(std::move(callback));
      return result;
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
    template <typename, typename, typename>
    friend class WeakTerm;

    template <typename VectorType, typename MatrixType>
    struct Pairing
    {
      using Model = SemiDiscreteModel<VectorType, MatrixType>;

      std::shared_ptr<MatrixType>                  matrix;
      std::shared_ptr<dealii::SparsityPattern>     sparsity;
      std::shared_ptr<VectorType>                  affine_rhs;
      std::function<std::shared_ptr<VectorType>()> affine_rhs_factory;
      typename Model::MatrixOperator               operator_with_matrix;
      FieldId                                      source_id;
      FieldId                                      target_id;
    };

    template <typename VectorType, typename MatrixType, typename Target>
    Pairing<VectorType, MatrixType>
    make_pairing(SemidiscreteBuilder<VectorType, MatrixType> &,
                 const Target         &target,
                 const WeakTermRegion &region) const
    {
      using Pair = Pairing<VectorType, MatrixType>;
      Pair result;
      result.source_id = observable_.source_field();
      result.target_id = target.source().field_id();
      const auto cached =
        cache_assembly_ ? std::make_shared<std::optional<Pair>>() : nullptr;
      const auto owned_recipe = std::make_shared<const WeakTerm>(*this);
      const auto owned_target = std::make_shared<const Target>(target);
      auto       build_action = [owned_recipe, owned_target, region, cached] {
        const auto &recipe = *owned_recipe;
        const auto &target = *owned_target;
        if (cached == nullptr)
          return recipe.template assemble_pairing<VectorType, MatrixType>(
            target, region, false);
        if (!cached->has_value())
          cached->emplace(
            recipe.template assemble_pairing<VectorType, MatrixType>(target,
                                                                     region,
                                                                     false));
        return cached->value();
      };
      const auto build =
        std::make_shared<const decltype(build_action)>(std::move(build_action));
      const auto source            = observable_.source();
      const auto row               = target.source();
      auto      &op                = result.operator_with_matrix;
      op.view.reinit_domain_vector = [source](VectorType &v, const bool) {
        detail::initialize_weak_vector(v,
                                       source.locally_owned_dofs(),
                                       source.space().mpi_communicator());
      };
      op.view.reinit_range_vector = [row](VectorType &v, const bool) {
        detail::initialize_weak_vector(v,
                                       row.locally_owned_dofs(),
                                       row.space().mpi_communicator());
      };
      op.view.vmult = [build, source, row](VectorType       &dst,
                                           const VectorType &src) {
        AssertDimension(src.size(), source.locally_owned_dofs().size());
        AssertDimension(dst.size(), row.locally_owned_dofs().size());
        if (detail::weak_vector_is_zero(src, source.space().mpi_communicator()))
          {
            dst = 0.;
            return;
          }
        const auto pairing = (*build)();
        pairing.operator_with_matrix.view.vmult(dst, src);
      };
      op.view.vmult_add = [build, source, row](VectorType       &dst,
                                               const VectorType &src) {
        AssertDimension(src.size(), source.locally_owned_dofs().size());
        AssertDimension(dst.size(), row.locally_owned_dofs().size());
        if (detail::weak_vector_is_zero(src, source.space().mpi_communicator()))
          {
            return;
          }
        const auto pairing = (*build)();
        pairing.operator_with_matrix.view.vmult_add(dst, src);
      };
      op.view.Tvmult = [build, source, row](VectorType       &dst,
                                            const VectorType &src) {
        AssertDimension(src.size(), row.locally_owned_dofs().size());
        AssertDimension(dst.size(), source.locally_owned_dofs().size());
        if (detail::weak_vector_is_zero(src, source.space().mpi_communicator()))
          {
            dst = 0.;
            return;
          }
        const auto pairing = (*build)();
        pairing.operator_with_matrix.view.Tvmult(dst, src);
      };
      op.view.Tvmult_add = [build, source, row](VectorType       &dst,
                                                const VectorType &src) {
        AssertDimension(src.size(), row.locally_owned_dofs().size());
        AssertDimension(dst.size(), source.locally_owned_dofs().size());
        if (detail::weak_vector_is_zero(src, source.space().mpi_communicator()))
          {
            return;
          }
        const auto pairing = (*build)();
        pairing.operator_with_matrix.view.Tvmult_add(dst, src);
      };
      op.materialize = [build] {
        auto pairing = std::make_shared<Pair>((*build)());
        return std::shared_ptr<MatrixType>(pairing, pairing->matrix.get());
      };
      op.materialize_into = [build](MatrixType &dst) {
        const auto pairing = (*build)();
        if constexpr (detail::has_sparse_reinit<MatrixType>::value &&
                      !detail::has_distributed_partitions<MatrixType>::value)
          {
            AssertThrow(
              dst.m() == pairing.matrix->m() && dst.n() == pairing.matrix->n(),
              dealii::ExcMessage(
                "Initialize the destination sparsity before assembling a serial sparse matrix."));
            dst = 0.;
            detail::set_matrix_values(dst, *pairing.matrix);
          }
        else
          pairing.operator_with_matrix.materialize_into_matrix(dst);
      };
      result.affine_rhs_factory = [owned_recipe, owned_target, region] {
        const auto &recipe = *owned_recipe;
        const auto &target = *owned_target;
        const auto  source = recipe.observable_.source();
        if (dealii::Utilities::MPI::max(
              source.constraints().has_inhomogeneities() ? 1u : 0u,
              source.space().mpi_communicator()) == 0)
          return std::shared_ptr<VectorType>();
        if constexpr (!detail::is_lifted_observable<
                        std::decay_t<Target>>::value &&
                      trial_field_type::dimension() == Target::dimension() &&
                      (std::is_same_v<Support, InferredIntegrationSupport> ||
                       is_integration_support<Support>::value ||
                       std::is_same_v<
                         Support,
                         BoundaryIntegrationSupport<Target::dimension(),
                                                    Target::spacedimension()>>))
          {
            using Assembly = detail::WeakAssembly<TrialExpression, Target>;
            if (!Assembly::geometry_is_nonmatching(recipe.observable_, target))
              {
                constexpr bool matching_support = [] {
                  if constexpr (is_integration_support<Support>::value)
                    return Support::dimension() == Target::dimension();
                  else
                    return true;
                }();
                if constexpr (matching_support)
                  {
                    bool native_support = true;
                    if constexpr (is_integration_support<Support>::value)
                      native_support =
                        static_cast<const void *>(
                          &recipe.support_.triangulation()) ==
                        static_cast<const void *>(
                          &source.dof_handler().get_triangulation());
                    if (native_support)
                      return Assembly::template affine_rhs<VectorType,
                                                           MatrixType>(
                        recipe.observable_,
                        target,
                        recipe.support_,
                        region.kind == WeakTermRegion::Kind::boundary,
                        region.boundary_ids);
                  }
              }
          }
        const auto pairing =
          recipe.template assemble_pairing<VectorType, MatrixType>(target,
                                                                   region);
        return pairing.affine_rhs_factory ? pairing.affine_rhs_factory() :
                                            pairing.affine_rhs;
      };
      return result;
    }

    template <typename VectorType, typename MatrixType, typename Target>
    Pairing<VectorType, MatrixType>
    assemble_pairing(const Target         &target,
                     const WeakTermRegion &region,
                     const bool            include_affine = true) const
    {
      Pairing<VectorType, MatrixType> result;
      std::shared_ptr<detail::WeakAffineAssembly<VectorType, MatrixType>>
            affine_data;
      auto *affine_data_output = include_affine ? &affine_data : nullptr;
      auto *affine_rhs_output  = include_affine ? &result.affine_rhs : nullptr;
      if constexpr (detail::is_lifted_observable<std::decay_t<Target>>::value)
        {
          AssertThrow(region.kind == WeakTermRegion::Kind::volume,
                      dealii::ExcMessage(
                        "Lifted weak terms use physical sample integrals."));
          using Assembly =
            detail::LiftedTargetWeakAssembly<TrialExpression, Target>;
          const auto storage =
            Assembly::template assemble<VectorType, MatrixType>(
              observable_,
              target,
              affine_rhs_output,
              affine_data_output,
              support_);
          result.matrix   = storage.matrix;
          result.sparsity = storage.sparsity;
          result.operator_with_matrix =
            ImmersX::matrix_operator<VectorType>(*result.matrix);
        }
      else
        {
          using Assembly = detail::WeakAssembly<TrialExpression, Target>;
          const bool nonmatching_geometry =
            Assembly::geometry_is_nonmatching(observable_, target);
          if constexpr (detail::is_boundary_normal_observable<
                          TrialExpression>::value)
            {
              constexpr bool has_explicit_boundary_support = std::is_same_v<
                Support,
                BoundaryIntegrationSupport<TestExpression::dimension(),
                                           TestExpression::spacedimension()>>;
              AssertThrow(has_explicit_boundary_support ||
                            region.kind == WeakTermRegion::Kind::boundary,
                          dealii::ExcMessage(
                            "A normal-contracted weak term requires a "
                            "boundary support."));
            }
          if constexpr (!std::is_same_v<Support, InferredIntegrationSupport>)
            {
              AssertThrow(region.kind == WeakTermRegion::Kind::volume,
                          dealii::ExcMessage(
                            "An explicit integration support cannot be "
                            "combined with an on_boundary region."));
              if constexpr (std::is_same_v<Support,
                                           BoundaryIntegrationSupport<
                                             TestExpression::dimension(),
                                             TestExpression::spacedimension()>>)
                AssertThrow(!nonmatching_geometry,
                            dealii::ExcMessage(
                              "Boundary weak terms require matching source "
                              "and test meshes."));
              const auto storage = [&] {
                if constexpr (is_physical_quadrature_support<Support>::value)
                  return Assembly::template assemble_on_support<VectorType,
                                                                MatrixType>(
                    observable_,
                    target,
                    support_,
                    affine_rhs_output,
                    affine_data_output);
                else if constexpr (std::is_same_v<
                                     Support,
                                     BoundaryIntegrationSupport<
                                       TestExpression::dimension(),
                                       TestExpression::spacedimension()>>)
                  return Assembly::template assemble_on_boundary_support<
                    VectorType,
                    MatrixType>(observable_,
                                target,
                                support_,
                                affine_rhs_output,
                                affine_data_output);
                else
                  return Assembly::template assemble_on_support<VectorType,
                                                                MatrixType>(
                    observable_,
                    target,
                    support_,
                    affine_rhs_output,
                    affine_data_output);
              }();
              result.matrix   = storage.matrix;
              result.sparsity = storage.sparsity;
              result.operator_with_matrix =
                ImmersX::matrix_operator<VectorType>(*result.matrix);
              if constexpr (is_integration_support<Support>::value &&
                            std::is_constructible_v<TrialExpression,
                                                    trial_field_type>)
                {
                  const auto source      = observable_.source();
                  const auto constrained = dealii::Utilities::MPI::max(
                    source.constraints().n_constraints() != 0 ? 1u : 0u,
                    source.space().mpi_communicator());
                  if (include_affine && nonmatching_geometry &&
                      !source.is_reindexed() && constrained != 0)
                    {
                      auto empty =
                        std::make_shared<dealii::AffineConstraints<double>>();
                      empty->reinit(source.locally_owned_dofs(),
                                    source.locally_relevant_dofs());
                      empty->close();
                      const auto unconstrained =
                        TrialExpression(source.with_constraints(empty))
                          .scaled(observable_.scale());
                      const auto storage =
                        Assembly::template assemble_on_support<VectorType,
                                                               MatrixType>(
                          unconstrained, target, support_);
                      const auto affine_operator =
                        ImmersX::matrix_operator<VectorType>(*storage.matrix);
                      // Source offsets change in time; geometry and the
                      // selected samples stay fixed.
                      result.affine_rhs_factory =
                        [source, storage, affine_operator] {
                          VectorType offset;
                          affine_operator.view.reinit_domain_vector(offset,
                                                                    false);
                          offset = 0.;
                          for (const auto i : source.locally_owned_dofs())
                            offset[i] +=
                              source.constraints().get_inhomogeneity(i);
                          detail::compress_weak_vector(offset);
                          auto rhs = std::make_shared<VectorType>();
                          affine_operator.view.reinit_range_vector(*rhs, false);
                          affine_operator.view.vmult(*rhs, offset);
                          *rhs *= -1.;
                          return rhs;
                        };
                    }
                }
            }
          else
            {
              AssertThrow(region.kind != WeakTermRegion::Kind::boundary ||
                            !nonmatching_geometry,
                          dealii::ExcMessage(
                            "Boundary weak terms require matching source and "
                            "target geometry."));
              if (nonmatching_geometry)
                {
                  const auto prepared =
                    Assembly::template prepare<VectorType, MatrixType>(
                      observable_, target);
                  result.matrix   = prepared->storage.matrix;
                  result.sparsity = prepared->storage.sparsity;
                  result.operator_with_matrix =
                    Assembly::template dynamic_matrix_operator<VectorType,
                                                               MatrixType>(
                      prepared);
                  // Condensed nonmatching matrices omit essential columns.
                  // Keep their action on the affine offset using the same
                  // backend with unconstrained source columns.
                  if constexpr (std::is_constructible_v<TrialExpression,
                                                        trial_field_type>)
                    {
                      const auto source      = observable_.source();
                      const auto constrained = dealii::Utilities::MPI::max(
                        source.constraints().n_constraints() != 0 ? 1u : 0u,
                        source.space().mpi_communicator());
                      if (include_affine && !source.is_reindexed() &&
                          constrained != 0)
                        {
                          auto empty = std::make_shared<
                            dealii::AffineConstraints<double>>();
                          empty->reinit(source.locally_owned_dofs(),
                                        source.locally_relevant_dofs());
                          empty->close();
                          const auto unconstrained =
                            TrialExpression(source.with_constraints(empty))
                              .scaled(observable_.scale());
                          const auto affine_prepared =
                            Assembly::template prepare<VectorType, MatrixType>(
                              unconstrained, target);
                          const auto affine_operator =
                            Assembly::template dynamic_matrix_operator<
                              VectorType,
                              MatrixType>(affine_prepared);
                          result.affine_rhs_factory = [source,
                                                       affine_operator] {
                            VectorType offset;
                            affine_operator.view.reinit_domain_vector(offset,
                                                                      false);
                            offset = 0.;
                            for (const auto i : source.locally_owned_dofs())
                              offset[i] +=
                                source.constraints().get_inhomogeneity(i);
                            detail::compress_weak_vector(offset);
                            auto rhs = std::make_shared<VectorType>();
                            affine_operator.view.reinit_range_vector(*rhs,
                                                                     false);
                            affine_operator.view.vmult(*rhs, offset);
                            *rhs *= -1.;
                            return rhs;
                          };
                        }
                    }
                }
              else
                {
                  const auto storage = [&] {
                    if (region.kind == WeakTermRegion::Kind::boundary)
                      {
                        AssertThrow(!detail::is_lifted_observable<
                                      std::decay_t<Target>>::value,
                                    dealii::ExcMessage(
                                      "Boundary weak terms do not support "
                                      "lifted targets."));
                        return Assembly::template assemble_boundary<VectorType,
                                                                    MatrixType>(
                          observable_,
                          target,
                          region.boundary_ids,
                          affine_rhs_output,
                          affine_data_output);
                      }
                    return Assembly::template assemble<VectorType, MatrixType>(
                      observable_,
                      target,
                      affine_rhs_output,
                      affine_data_output);
                  }();
                  result.matrix   = storage.matrix;
                  result.sparsity = storage.sparsity;
                  result.operator_with_matrix =
                    ImmersX::matrix_operator<VectorType>(*result.matrix);
                }
            }
        }
      if (!include_affine)
        return result;
      if (affine_data != nullptr)
        result.affine_rhs_factory = [rhs = result.affine_rhs] { return rhs; };
      if (!result.affine_rhs_factory)
        {
          const auto source         = observable_.source();
          result.affine_rhs_factory = [source] {
            detail::assert_homogeneous_weak_constraints(
              source,
              "This weak-term backend does not support inhomogeneous source "
              "constraints. Use matching cell or boundary supports.");
            return std::shared_ptr<VectorType>();
          };
          result.affine_rhs_factory();
        }
      result.source_id = observable_.source_field();
      result.target_id = target.source().field_id();
      return result;
    }

    TrialExpression             observable_;
    TestExpression              target_;
    Support                     support_;
    WeakTermRegion              region_;
    bool                        cache_assembly_ = false;
    std::function<void(double)> context_update_;
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

  template <typename Trial, typename Test, int dim, int spacedim>
  auto
  weak_term(Trial                             trial,
            Test                              test_expression,
            IntegrationSupport<dim, spacedim> support)
  {
    return weak_term(std::move(trial), std::move(test_expression))
      .on(std::move(support));
  }

  template <typename Trial, typename Test, int dim, int spacedim>
  auto
  weak_term(Trial                                     trial,
            Test                                      test_expression,
            BoundaryIntegrationSupport<dim, spacedim> support)
  {
    return weak_term(std::move(trial), std::move(test_expression))
      .on(std::move(support));
  }

  template <typename Trial,
            typename Test,
            typename Support,
            std::enable_if_t<
              is_physical_quadrature_support<std::decay_t<Support>>::value,
              int> = 0>
  auto
  weak_term(Trial trial, Test test_expression, Support support)
  {
    return weak_term(std::move(trial), std::move(test_expression))
      .on(std::move(support));
  }

  template <typename Trial, typename FieldType, typename Operation>
  auto
  operator*(Trial trial, TestExpression<FieldType, Operation> test_expression)
    -> std::enable_if_t<detail::is_linear_observable<Trial>::value,
                        WeakTerm<Trial, TestExpression<FieldType, Operation>>>
  {
    return weak_term(std::move(trial), std::move(test_expression));
  }

  template <typename Trial,
            typename Test,
            typename Support,
            int dim,
            int spacedim>
  auto
  operator*(WeakTerm<Trial, Test, Support>    term,
            IntegrationSupport<dim, spacedim> support)
  {
    return term.on(std::move(support));
  }

  template <typename Trial,
            typename Test,
            typename CurrentSupport,
            typename Support,
            std::enable_if_t<
              is_physical_quadrature_support<std::decay_t<Support>>::value,
              int> = 0>
  auto
  operator*(WeakTerm<Trial, Test, CurrentSupport> term, Support support)
  {
    return term.on(std::move(support));
  }

  template <typename Trial,
            typename Test,
            typename Support,
            int dim,
            int spacedim>
  auto
  operator*(WeakTerm<Trial, Test, Support>            term,
            BoundaryIntegrationSupport<dim, spacedim> support)
  {
    return term.on(std::move(support));
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
