// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_known_term_h
#define immersx_known_term_h

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <immersx/core/weak_term.h>

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>

namespace ImmersX
{
  /** Data supplied to a known volume or boundary source. */
  template <int spacedim>
  struct KnownTermContext
  {
    dealii::Point<spacedim>                   point;
    double                                    time        = 0.;
    dealii::types::material_id                material_id = 0;
    std::optional<dealii::types::boundary_id> boundary_id;
    dealii::Tensor<1, spacedim>               normal;
    double                                    h = 0.;

    bool
    on_boundary() const
    {
      return boundary_id.has_value();
    }
  };

  /**
   * A solver-neutral, context-aware known source.
   *
   * The material and boundary maps are overrides.  When no override is
   * registered, the default callback is used.  This gives volume and face
   * terms the same fallback semantics as the elasticity problem parameters.
   */
  template <typename Value, int spacedim>
  class KnownSource
  {
  public:
    using value_type = Value;
    using Context    = KnownTermContext<spacedim>;
    using Callback   = std::function<Value(const Context &)>;

    KnownSource() = default;

    explicit KnownSource(Callback callback)
      : default_(std::move(callback))
    {}

    KnownSource &
    on_material(const dealii::types::material_id id, Callback callback)
    {
      material_overrides_[id] = std::move(callback);
      return *this;
    }

    KnownSource &
    on_boundary(const dealii::types::boundary_id id, Callback callback)
    {
      boundary_overrides_[id] = std::move(callback);
      return *this;
    }

    Value
    evaluate(const Context &context) const
    {
      if (context.boundary_id.has_value())
        if (const auto it = boundary_overrides_.find(*context.boundary_id);
            it != boundary_overrides_.end())
          return it->second(context);

      if (const auto it = material_overrides_.find(context.material_id);
          it != material_overrides_.end())
        return it->second(context);

      AssertThrow(static_cast<bool>(default_),
                  dealii::ExcMessage("A KnownSource has no default callback."));
      return default_(context);
    }

  private:
    Callback                                       default_;
    std::map<dealii::types::material_id, Callback> material_overrides_;
    std::map<dealii::types::boundary_id, Callback> boundary_overrides_;
  };

  namespace detail
  {
    template <typename FunctionType, typename = void>
    struct has_set_time : std::false_type
    {};

    template <typename FunctionType>
    struct has_set_time<
      FunctionType,
      std::void_t<decltype(std::declval<FunctionType &>().set_time(0.))>>
      : std::true_type
    {};

    template <typename FunctionType>
    void
    set_function_time(const FunctionType &function, const double time)
    {
      if constexpr (has_set_time<FunctionType>::value)
        const_cast<FunctionType &>(function).set_time(time);
    }

    template <typename Type>
    struct is_known_source : std::false_type
    {};

    template <typename Value, int spacedim>
    struct is_known_source<KnownSource<Value, spacedim>> : std::true_type
    {};

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
    void
    reinit_known_vector(VectorType &vector, const FieldType &field)
    {
      if constexpr (has_distributed_reinit<VectorType, FieldType>::value)
        vector.reinit(field.locally_owned_dofs(),
                      field.locally_relevant_dofs(),
                      field.space().mpi_communicator());
      else
        vector.reinit(field.dof_handler().n_dofs());
    }
  } // namespace detail

  /** Wrap a deal.II scalar Function with time-aware known-source semantics. */
  template <int spacedim,
            typename FunctionType,
            std::enable_if_t<
              std::is_base_of_v<dealii::Function<spacedim>, FunctionType>,
              int> = 0>
  KnownSource<double, spacedim>
  known_function(const FunctionType &function)
  {
    return KnownSource<double, spacedim>(
      [&function](const KnownTermContext<spacedim> &context) {
        detail::set_function_time(function, context.time);
        return function.value(context.point);
      });
  }

  /** Wrap a deal.II vector Function with time-aware known-source semantics. */
  template <int spacedim,
            typename FunctionType,
            std::enable_if_t<
              std::is_base_of_v<dealii::Function<spacedim>, FunctionType>,
              int> = 0>
  KnownSource<dealii::Tensor<1, spacedim>, spacedim>
  known_vector_function(const FunctionType &function)
  {
    return KnownSource<dealii::Tensor<1, spacedim>, spacedim>(
      [&function](const KnownTermContext<spacedim> &context) {
        detail::set_function_time(function, context.time);
        dealii::Tensor<1, spacedim> result;
        for (unsigned int component = 0; component < spacedim; ++component)
          result[component] = function.value(context.point, component);
        return result;
      });
  }

  /** Integration region selected by a KnownTerm. */
  struct KnownTermRegion
  {
    enum class Kind
    {
      volume,
      boundary
    };

    Kind                                 kind = Kind::volume;
    std::set<dealii::types::boundary_id> boundary_ids;
  };

  /**
   * A dependency-free FE term generated by a known source.
   *
   * A frozen Observable is accepted as a source as well.  In that case its
   * existing FE operation and frozen vector are evaluated directly, so the
   * term reuses the ordinary FE expression machinery without registering a
   * state dependency or duplicating a field representation.
   */
  template <typename Source, typename TestExpression>
  class KnownTerm
  {
  public:
    using target_type = TestExpression;

    KnownTerm(Source source, TestExpression target)
      : source_(std::move(source))
      , target_(std::move(target))
    {}

    const std::vector<FieldId> &
    dependencies() const
    {
      static const std::vector<FieldId> none;
      return none;
    }

    KnownTerm
    on_boundary(const dealii::types::boundary_id id) const
    {
      return on_boundary(std::set<dealii::types::boundary_id>{id});
    }

    KnownTerm
    on_boundary(const std::set<dealii::types::boundary_id> &boundary_ids) const
    {
      auto result                 = *this;
      result.region_.kind         = KnownTermRegion::Kind::boundary;
      result.region_.boundary_ids = boundary_ids;
      return result;
    }

    KnownTerm
    in_volume() const
    {
      auto result         = *this;
      result.region_.kind = KnownTermRegion::Kind::volume;
      result.region_.boundary_ids.clear();
      return result;
    }

    template <typename VectorType, typename MatrixType>
    FieldId
    add(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      using Model          = SemiDiscreteModel<VectorType, MatrixType>;
      const auto target_id = target_.source().field_id();
      AssertThrow(target_id.is_valid(),
                  dealii::ExcMessage("A KnownTerm target must be registered."));

      auto term = builder.term(target_id, "known_term");
      term.residual([source = source_, target = target_, region = region_](
                      const auto &context) {
        const auto values =
          assemble<VectorType>(source, target, region, context);
        typename Model::Operation result;
        result.reinit_vector = [values](VectorType &vector, const bool omit) {
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
      return target_id;
    }

    template <typename VectorType, typename MatrixType>
    FieldId
    operator()(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      return add(builder);
    }

    const Source &
    source() const
    {
      return source_;
    }

    const TestExpression &
    target() const
    {
      return target_;
    }

  private:
    template <typename VectorType, typename Context>
    static std::shared_ptr<VectorType>
    assemble(const Source          &source,
             const TestExpression  &target,
             const KnownTermRegion &region,
             const Context         &context)
    {
      using TargetField = typename TestExpression::source_field_type;
      using Value       = typename std::decay_t<Source>::value_type;

      const auto &target_field = target.source();
      auto        result       = std::make_shared<VectorType>();
      detail::reinit_known_vector(*result, target_field);
      *result = 0.;

      const auto degree = target_field.space().finite_element().degree + 1;
      AssertThrow(region.kind == KnownTermRegion::Kind::volume ||
                    !region.boundary_ids.empty(),
                  dealii::ExcMessage(
                    "A boundary KnownTerm needs at least one boundary id."));

      if constexpr (detail::is_known_source<std::decay_t<Source>>::value)
        {
          assemble_callback<VectorType, Value>(
            source, target, region, context, degree, *result);
        }
      else if constexpr (detail::is_observable<std::decay_t<Source>>::value)
        {
          AssertThrow(source.is_frozen(),
                      dealii::ExcMessage(
                        "KnownTerm FE sources must be frozen Observables."));
          assemble_frozen<VectorType, std::decay_t<Source>, Value>(
            source, target, region, context, degree, *result);
        }
      else
        static_assert(detail::is_known_source<std::decay_t<Source>>::value,
                      "KnownTerm requires KnownSource or a frozen Observable.");

      detail::compress_weak_vector(*result);
      return result;
    }

    template <typename VectorType, typename Value, typename Context>
    static void
    assemble_callback(
      const KnownSource<Value, TestExpression::spacedimension()> &source,
      const TestExpression                                       &target,
      const KnownTermRegion                                      &region,
      const Context                                              &context,
      const unsigned int                                          degree,
      VectorType                                                 &result)
    {
      using TargetField          = typename TestExpression::source_field_type;
      constexpr int dim          = TargetField::dimension();
      constexpr int spacedim     = TargetField::spacedimension();
      const auto   &target_field = target.source();
      const auto    quadrature   = dealii::QGauss<dim>(degree);
      const auto    flags = target.update_flags() | dealii::update_JxW_values |
                         dealii::update_quadrature_points;

      if (region.kind == KnownTermRegion::Kind::volume)
        {
          dealii::FEValues<dim, spacedim> values(
            target_field.mapping(),
            target_field.space().finite_element(),
            quadrature,
            flags);
          for (const auto &cell :
               target_field.dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                values.reinit(cell);
                add_callback_cell(source,
                                  target,
                                  context,
                                  values,
                                  cell,
                                  cell,
                                  std::nullopt,
                                  result);
              }
        }
      else
        {
          const dealii::QGauss<dim - 1> face_quadrature(degree);
          const auto                    face_flags =
            target.update_flags() | dealii::update_JxW_values |
            dealii::update_quadrature_points | dealii::update_normal_vectors;
          dealii::FEFaceValues<dim, spacedim> face_values(
            target_field.mapping(),
            target_field.space().finite_element(),
            face_quadrature,
            face_flags);
          for (const auto &cell :
               target_field.dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary() &&
                    region.boundary_ids.count(cell->face(face)->boundary_id()))
                  {
                    face_values.reinit(cell, face);
                    add_callback_cell(source,
                                      target,
                                      context,
                                      face_values,
                                      cell,
                                      cell,
                                      cell->face(face)->boundary_id(),
                                      result);
                  }
        }
    }

    template <typename VectorType,
              typename SourceType,
              typename Value,
              typename Context>
    static void
    assemble_frozen(const SourceType      &source,
                    const TestExpression  &target,
                    const KnownTermRegion &region,
                    const Context         &context,
                    const unsigned int     degree,
                    VectorType            &result)
    {
      using SourceField          = typename SourceType::source_field_type;
      using TargetField          = typename TestExpression::source_field_type;
      constexpr int dim          = TargetField::dimension();
      constexpr int spacedim     = TargetField::spacedimension();
      const auto   &source_field = source.source();
      AssertThrow(&source_field.dof_handler().get_triangulation() ==
                    &target.source().dof_handler().get_triangulation(),
                  dealii::ExcMessage(
                    "A frozen KnownTerm source and target must share a mesh."));
      AssertThrow(
        &source_field.mapping() == &target.source().mapping(),
        dealii::ExcMessage(
          "A frozen KnownTerm source and target must share a mapping."));
      const auto quadrature = dealii::QGauss<dim>(degree);
      const auto flags      = source.update_flags() | target.update_flags() |
                         dealii::update_JxW_values |
                         dealii::update_quadrature_points;
      const auto &frozen = source.template frozen_values<VectorType>();

      if (region.kind == KnownTermRegion::Kind::volume)
        {
          dealii::FEValues<dim, spacedim> values(
            target.source().mapping(),
            target.source().space().finite_element(),
            quadrature,
            flags);
          for (const auto &cell :
               target.source().dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                values.reinit(cell);
                add_frozen_cell(source,
                                target,
                                context,
                                values,
                                values,
                                cell,
                                cell,
                                std::nullopt,
                                frozen,
                                result);
              }
        }
      else
        {
          const dealii::QGauss<dim - 1> face_quadrature(degree);
          const auto                    face_flags =
            source.update_flags() | target.update_flags() |
            dealii::update_JxW_values | dealii::update_quadrature_points |
            dealii::update_normal_vectors;
          dealii::FEFaceValues<dim, spacedim> face_values(
            target.source().mapping(),
            target.source().space().finite_element(),
            face_quadrature,
            face_flags);
          for (const auto &cell :
               target.source().dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              for (const auto face : cell->face_indices())
                if (cell->face(face)->at_boundary() &&
                    region.boundary_ids.count(cell->face(face)->boundary_id()))
                  {
                    face_values.reinit(cell, face);
                    add_frozen_cell(source,
                                    target,
                                    context,
                                    face_values,
                                    face_values,
                                    cell,
                                    cell,
                                    cell->face(face)->boundary_id(),
                                    frozen,
                                    result);
                  }
        }
    }

    template <typename VectorType,
              typename Value,
              typename Values,
              typename Cell>
    static void
    add_callback_cell(
      const KnownSource<Value, TestExpression::spacedimension()> &source,
      const TestExpression                                       &target,
      const EvaluationContext<VectorType>                        &context,
      const Values                                               &values,
      const Cell                                                 &cell,
      const Cell &,
      const std::optional<dealii::types::boundary_id> boundary_id,
      VectorType                                     &result)
    {
      using TargetField = typename TestExpression::source_field_type;
      std::vector<dealii::types::global_dof_index> indices(
        cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(indices);
      std::vector<unsigned int>                    positions;
      std::vector<dealii::types::global_dof_index> execution_indices;
      for (unsigned int i = 0; i < indices.size(); ++i)
        if (target.source().has_execution_index(indices[i]))
          {
            positions.push_back(i);
            execution_indices.push_back(
              target.source().execution_index(indices[i]));
          }
      dealii::Vector<double> local(positions.size());
      local            = 0.;
      const auto &view = values[target.source().extractor()];
      for (const auto q : values.quadrature_point_indices())
        {
          KnownTermContext<TestExpression::spacedimension()> data;
          data.point       = values.quadrature_point(q);
          data.time        = context.time();
          data.material_id = cell->material_id();
          data.boundary_id = boundary_id;
          data.h           = cell->diameter();
          if (boundary_id.has_value())
            data.normal = values.normal_vector(q);
          const auto known = source.evaluate(data);
          for (unsigned int i = 0; i < positions.size(); ++i)
            local[i] += detail::natural_pairing(
                          known, target.operation()(view, positions[i], q)) *
                        values.JxW(q);
        }
      target.source().constraints().distribute_local_to_global(
        local, execution_indices, result);
    }

    template <typename VectorType,
              typename SourceType,
              typename Values,
              typename Cell>
    static void
    add_frozen_cell(const SourceType                    &source,
                    const TestExpression                &target,
                    const EvaluationContext<VectorType> &context,
                    const Values                        &source_values,
                    const Values                        &target_values,
                    const Cell                          &source_cell,
                    const Cell                          &target_cell,
                    const std::optional<dealii::types::boundary_id> boundary_id,
                    const VectorType                               &frozen,
                    VectorType                                     &result)
    {
      std::vector<dealii::types::global_dof_index> source_indices(
        source_cell->get_fe().n_dofs_per_cell());
      std::vector<dealii::types::global_dof_index> target_indices(
        target_cell->get_fe().n_dofs_per_cell());
      source_cell->get_dof_indices(source_indices);
      target_cell->get_dof_indices(target_indices);
      std::vector<unsigned int>                    source_positions;
      std::vector<unsigned int>                    target_positions;
      std::vector<dealii::types::global_dof_index> target_execution_indices;
      for (unsigned int j = 0; j < source_indices.size(); ++j)
        if (source.source().has_execution_index(source_indices[j]))
          source_positions.push_back(j);
      for (unsigned int i = 0; i < target_indices.size(); ++i)
        if (target.source().has_execution_index(target_indices[i]))
          {
            target_positions.push_back(i);
            target_execution_indices.push_back(
              target.source().execution_index(target_indices[i]));
          }
      dealii::Vector<double> local(target_positions.size());
      local                   = 0.;
      const auto &source_view = source_values[source.source().extractor()];
      const auto &target_view = target_values[target.source().extractor()];
      for (const auto q : target_values.quadrature_point_indices())
        {
          typename SourceType::value_type known{};
          for (const auto j : source_positions)
            known +=
              frozen[source.source().execution_index(source_indices[j])] *
              source.operation()(source_view, j, q);
          for (unsigned int i = 0; i < target_positions.size(); ++i)
            local[i] +=
              detail::natural_pairing(known,
                                      target.operation()(target_view,
                                                         target_positions[i],
                                                         q)) *
              target_values.JxW(q);
        }
      target.source().constraints().distribute_local_to_global(
        local, target_execution_indices, result);
      (void)context;
      (void)boundary_id;
    }

    Source          source_;
    TestExpression  target_;
    KnownTermRegion region_;
  };

  template <typename Value, int spacedim, typename TestExpression>
  auto
  known_term(const KnownSource<Value, spacedim> &source,
             const TestExpression               &target)
  {
    return KnownTerm<KnownSource<Value, spacedim>, TestExpression>(source,
                                                                   target);
  }

  template <typename SourceFieldType,
            typename Operation,
            typename TestExpression>
  auto
  known_term(const Observable<SourceFieldType, Operation> &source,
             const TestExpression                         &target)
  {
    return KnownTerm<Observable<SourceFieldType, Operation>, TestExpression>(
      source, target);
  }
} // namespace ImmersX

#endif
