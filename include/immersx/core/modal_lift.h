// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_modal_lift_h
#define immersx_modal_lift_h

#include <deal.II/base/function_parser.h>

#include <deal.II/particles/data_out.h>

#include <immersx/core/constraint.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/overlap_integration_support.h>
#include <immersx/coupling/lifted_modal_load.h>
#include <immersx/io/lifted_particle_output.h>

#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** Modal coefficient values at one representative-cell quadrature point. */
  using ModalDataEvaluator =
    std::function<std::vector<double>(const dealii::CellId &,
                                      const dealii::Point<3> &)>;

  /** Scalar surface quantity whose representative FE components are modal
   * coefficients, not ambient vector components. Geometry is fixed during a
   * solve. Rebuild this value after changing either mesh or the radius field.
   */
  template <typename FieldType>
  class ModalLiftedObservable
  {
  public:
    static_assert(FieldType::dimension() == 1 &&
                    FieldType::spacedimension() == 3,
                  "Modal lifting currently requires a curve in 3D.");
    using source_field_type = FieldType;
    using Point             = detail::CouplingPoint<3, double>;
    using value_type        = double;
    using Lift              = TensorProductLift<1, 2, 3>;
    using Radius =
      std::function<double(const dealii::CellId &, const dealii::Point<3> &)>;
    static constexpr unsigned int support_dimension        = 2;
    static constexpr unsigned int ambient_dimension        = 3;
    static constexpr unsigned int representative_dimension = 1;

    ModalLiftedObservable(FieldType field, const Lift &lift, Radius radius = {})
      : field_(std::move(field))
      , support_(lift.parameters(), field_.finite_element().degree)
      , radius_(std::move(radius))
    {
      if (!radius_)
        {
          auto radius_function = std::make_shared<dealii::FunctionParser<3>>();
          radius_function->initialize("x,y,z,t",
                                      lift.thickness,
                                      {{"pi", dealii::numbers::PI},
                                       {"E", dealii::numbers::E}},
                                      true);
          radius_ = [radius_function](const auto &, const auto &point) {
            return radius_function->value(point);
          };
        }
      AssertThrow(
        field_.finite_element().n_components() ==
          support_.selected_modes().size(),
        dealii::ExcMessage(
          "The modal FE component count must equal the selected-mode count."));
      AssertThrow(field_.finite_element().is_primitive(),
                  dealii::ExcMessage(
                    "Modal coefficients require a primitive finite element."));
      const auto            &quadrature = support_.representative_quadrature();
      dealii::FEValues<1, 3> values(field_.mapping(),
                                    field_.finite_element(),
                                    quadrature,
                                    dealii::update_values |
                                      dealii::update_JxW_values |
                                      dealii::update_quadrature_points |
                                      dealii::update_jacobians);
      for (const auto &cell : field_.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            std::vector<dealii::types::global_dof_index> indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices);
            for (const auto q : values.quadrature_point_indices())
              {
                dealii::Tensor<1, 3> tangent;
                for (unsigned int d = 0; d < 3; ++d)
                  tangent[d] = values.jacobian(q)[d][0];
                tangent /= tangent.norm();
                const auto   p = values.quadrature_point(q);
                const double r =
                  radius_ ? radius_(cell->id(), p) : support_.thickness(p, 0.);
                AssertThrow(std::isfinite(r) && r > 0.,
                            dealii::ExcMessage(
                              "Radius must be finite and positive."));
                const auto transformed =
                  support_.transform(p, tangent, values.JxW(q), r, q);
                for (unsigned int section_q = 0; section_q < transformed.size();
                     ++section_q)
                  {
                    const auto &lifted = transformed[section_q];
                    Point       point;
                    point.point                = lifted.point;
                    point.representative_point = p;
                    point.source_cell_id       = cell->id();
                    point.source_entity_id = cell->global_active_cell_index();
                    point.weight           = lifted.weight;
                    point.representative_qpoint = q;
                    point.section_qpoint        = section_q;
                    point.stable_id =
                      static_cast<std::uint64_t>(point.source_entity_id) *
                        quadrature.size() * transformed.size() +
                      q * transformed.size() + section_q;
                    point.dof_indices = indices;
                    point.mode_values = lifted.mode_values;
                    point.source_basis_values.resize(indices.size());
                    for (unsigned int i = 0; i < indices.size(); ++i)
                      {
                        const auto component =
                          cell->get_fe().system_to_component_index(i).first;
                        point.source_basis_values[i] =
                          values[dealii::FEValuesExtractors::Scalar(component)]
                            .value(i, q) *
                          lifted.mode_values[component];
                      }
                    point.basis_values = point.source_basis_values;
                    points_.push_back(std::move(point));
                  }
              }
          }
    }

    const FieldType &
    source() const
    {
      return field_;
    }
    FieldId
    source_field() const
    {
      return field_.id();
    }
    const std::vector<FieldId> &
    dependencies() const
    {
      return dependencies_;
    }
    bool
    is_frozen() const
    {
      return false;
    }
    const std::vector<Point> &
    lifted_points() const
    {
      return points_;
    }
    ModalLiftedObservable
    with_id(FieldId id) const
    {
      auto result          = *this;
      result.field_        = field_.with_id(id);
      result.dependencies_ = {id};
      return result;
    }
    static constexpr unsigned int
    dimension()
    {
      return 1;
    }
    static constexpr unsigned int
    spacedimension()
    {
      return 3;
    }
    const auto &
    support() const
    {
      return support_;
    }

    template <typename Context>
    double
    evaluate_point(const Context &context, std::size_t q) const
    {
      const auto         &point = points_.at(q);
      std::vector<double> local(point.dof_indices.size());
      field_.constraints().get_dof_values(context.state(field_.id()),
                                          point.dof_indices.begin(),
                                          local.begin(),
                                          local.end());
      double result = 0.;
      for (unsigned int i = 0; i < local.size(); ++i)
        result += local[i] * point.basis_values[i];
      return result;
    }

    /** Write the lifted wall with the lifted value and quadrature weight. */
    template <typename BulkField, typename VectorType>
    void
    write_lifted_output(const BulkField   &bulk,
                        const VectorType  &coefficients,
                        const std::string &filename) const
    {
      const LiftedParticleOutputParameters defaults(
        "/Lifted output particles/");
      write_lifted_output(bulk, coefficients, filename, defaults);
    }

    /** Write the lifted wall as particles carrying the properties selected in
     * options. The lifted value is named lambda and the modal coefficients
     * lambda_k, with k the selected cross-section coefficient. data is only
     * needed for the modal_data property, bulk_values only for bulk_field;
     * bulk_values must be ghosted with constraints distributed.
     */
    template <typename BulkField, typename VectorType>
    void
    write_lifted_output(const BulkField                      &bulk,
                        const VectorType                     &coefficients,
                        const std::string                    &filename,
                        const LiftedParticleOutputParameters &options,
                        const ModalDataEvaluator             &data = {},
                        const VectorType *bulk_values = nullptr) const
    {
      const auto           communicator = field_.space().mpi_communicator();
      const auto          &modes        = support_.selected_modes();
      const auto           n_modes      = modes.size();
      LiftedParticleLabels labels;
      labels.value_name = "lambda";
      for (const auto mode : modes)
        labels.mode_labels.push_back(std::to_string(mode));

      std::vector<double> modal_values;
      if (options.selected("modal_coefficients") ||
          options.selected("lifted_value"))
        {
          // The solver state need not satisfy the hanging-node constraints;
          // the sampled FE function must.
          VectorType owned;
          owned.reinit(field_.locally_owned_dofs(), communicator);
          owned = coefficients;
          field_.constraints().distribute(owned);
          VectorType relevant;
          relevant.reinit(field_.locally_owned_dofs(),
                          field_.locally_relevant_dofs(),
                          communicator);
          relevant = owned;
          relevant.update_ghost_values();
          modal_values = sample_at_representative_points(field_.dof_handler(),
                                                         field_.mapping(),
                                                         relevant,
                                                         points_);
        }

      std::vector<double> modal_data;
      if (options.selected("modal_data"))
        {
          AssertThrow(data,
                      dealii::ExcMessage(
                        "The modal_data particle property needs the modal "
                        "data evaluator."));
          modal_data.reserve(points_.size() * n_modes);
          std::map<std::pair<dealii::CellId, unsigned int>, std::vector<double>>
            cache;
          for (const auto &point : points_)
            {
              const auto key   = std::make_pair(point.source_cell_id,
                                              point.representative_qpoint);
              auto       found = cache.find(key);
              if (found == cache.end())
                found = cache
                          .emplace(key,
                                   data(point.source_cell_id,
                                        point.representative_point))
                          .first;
              AssertThrow(found->second.size() == n_modes,
                          dealii::ExcMessage(
                            "Provide one modal data value per selected mode."));
              modal_data.insert(modal_data.end(),
                                found->second.begin(),
                                found->second.end());
            }
        }

      const auto columns = make_lifted_particle_columns(
        options, labels, points_, modal_values, modal_data);
      const LiftedBulkField<3, VectorType> bulk_field{
        "u", bulk.dof_handler(), bulk.mapping(), bulk_values, 0, 1};
      write_lifted_particles(options, columns, points_, bulk_field, filename);
    }

    template <typename Support>
    auto
    assemble_overlap(const ModalDataEvaluator &data,
                     const Support            &overlap) const
    {
      overlap.assert_matches(field_.dof_handler().get_triangulation());
      AssertThrow(!overlap.has_material_filter(),
                  dealii::ExcMessage(
                    "A modal Gram matrix requires the full source mesh."));
      const auto basis = [&](const std::size_t q, const unsigned int i) {
        return points_[q].basis_values[i];
      };
      const auto datum = [&](const std::size_t q) {
        const auto &point = points_[q];
        const auto  amplitudes =
          data(point.source_cell_id, point.representative_point);
        AssertDimension(amplitudes.size(), point.mode_values.size());
        double value = 0.;
        for (unsigned int c = 0; c < amplitudes.size(); ++c)
          value += amplitudes[c] * point.mode_values[c];
        return value;
      };
      return detail::assemble_modal_gram(
        field_, points_, basis, datum, [&](const auto &point) {
          return overlap.includes_lifted_point(point);
        });
    }

    /** Production modal data are coefficients of the projected trace. Their
     * load uses the section measure, not the integral of each transverse mode.
     */
    template <typename VectorType, typename MatrixType>
    std::shared_ptr<dealii::SparsityPattern>
    assemble_representative(MatrixType               &metric,
                            VectorType               &rhs,
                            const ModalDataEvaluator &data) const
    {
      AssertThrow(data,
                  dealii::ExcMessage("Modal data evaluator cannot be empty."));
      dealii::DynamicSparsityPattern sparsity(
        field_.locally_owned_dofs().size(),
        field_.locally_owned_dofs().size(),
        field_.locally_relevant_dofs());
      dealii::DoFTools::make_sparsity_pattern(field_.dof_handler(),
                                              sparsity,
                                              field_.constraints(),
                                              false);
      auto retained_sparsity =
        detail::initialize_weak_matrix(metric,
                                       field_.locally_owned_dofs(),
                                       field_.locally_owned_dofs(),
                                       sparsity,
                                       field_.space().mpi_communicator());
      rhs.reinit(field_.locally_owned_dofs(),
                 field_.space().mpi_communicator());
      const auto            &quadrature = support_.representative_quadrature();
      dealii::FEValues<1, 3> values(field_.mapping(),
                                    field_.finite_element(),
                                    quadrature,
                                    dealii::update_values |
                                      dealii::update_JxW_values |
                                      dealii::update_quadrature_points);
      const unsigned int     n = field_.finite_element().n_dofs_per_cell();
      dealii::FullMatrix<double>                   local(n, n);
      dealii::Vector<double>                       load(n);
      std::vector<dealii::types::global_dof_index> indices(n);
      for (const auto &cell : field_.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            local = 0.;
            load  = 0.;
            cell->get_dof_indices(indices);
            for (const auto q : values.quadrature_point_indices())
              {
                const auto p           = values.quadrature_point(q);
                const auto data_values = data(cell->id(), p);
                AssertThrow(
                  data_values.size() == field_.finite_element().n_components(),
                  dealii::ExcMessage(
                    "Provide one modal data value per selected mode."));
                const auto r =
                  radius_ ? radius_(cell->id(), p) : support_.thickness(p, 0.);
                const double weight =
                  values.JxW(q) * support_.section_measure(r);
                // The modes have maximum one, so the Gram matrix of mode ci
                // on the cross section is |D| times its mean square, and the
                // datum is the amplitude of the mode (see ReducedCoupling).
                const auto &mean_squares =
                  support_.reference_cross_section().get_basis_mean_squares();
                for (unsigned int i = 0; i < n; ++i)
                  {
                    const auto ci =
                      cell->get_fe().system_to_component_index(i).first;
                    const auto vi =
                      values[dealii::FEValuesExtractors::Scalar(ci)].value(i,
                                                                           q);
                    load[i] += vi * data_values[ci] * weight * mean_squares[ci];
                    for (unsigned int j = 0; j < n; ++j)
                      if (ci ==
                          cell->get_fe().system_to_component_index(j).first)
                        local(i, j) +=
                          vi *
                          values[dealii::FEValuesExtractors::Scalar(ci)].value(
                            j, q) *
                          weight * mean_squares[ci];
                  }
              }
            field_.constraints().distribute_local_to_global(local,
                                                            indices,
                                                            metric);
            field_.constraints().distribute_local_to_global(load, indices, rhs);
          }
      detail::compress_weak_matrix(metric);
      rhs.compress(dealii::VectorOperation::add);
      return retained_sparsity;
    }

  private:
    FieldType                         field_;
    TensorProductLiftSupport<1, 2, 3> support_;
    Radius                            radius_;
    std::vector<Point>                points_;
    std::vector<FieldId>              dependencies_;
  };

  namespace detail
  {
    template <typename FieldType>
    struct is_lifted_observable<ModalLiftedObservable<FieldType>>
      : std::true_type
    {};
    template <typename FieldType>
    struct is_observable<ModalLiftedObservable<FieldType>> : std::true_type
    {};
  } // namespace detail

  template <typename FieldType>
  auto
  modal_lift(FieldType                                         field,
             const TensorProductLift<1, 2, 3>                 &lift,
             typename ModalLiftedObservable<FieldType>::Radius radius = {})
  {
    return ModalLiftedObservable<FieldType>(std::move(field),
                                            lift,
                                            std::move(radius));
  }

  /** Modal multiplier interaction. It owns no global block numbers. */
  template <typename BulkField, typename ModalLift>
  class LiftedMultiplierInteraction
  {
  public:
    LiftedMultiplierInteraction(BulkField                                  bulk,
                                ModalLift                                  lift,
                                std::shared_ptr<const dealii::Function<3>> data)
      : bulk_(std::move(bulk))
      , lift_(std::move(lift))
    {
      AssertThrow(data != nullptr,
                  dealii::ExcMessage("Modal data cannot be null."));
      AssertThrow(data->n_components ==
                    lift_.source().finite_element().n_components(),
                  dealii::ExcMessage(
                    "Provide one modal data expression per selected mode."));
      data_ = [data](const auto &, const auto &point) {
        std::vector<double> values(data->n_components);
        for (unsigned int c = 0; c < values.size(); ++c)
          values[c] = data->value(point, c);
        return values;
      };
    }

    LiftedMultiplierInteraction(BulkField          bulk,
                                ModalLift          lift,
                                ModalDataEvaluator data)
      : bulk_(std::move(bulk))
      , lift_(std::move(lift))
      , data_(std::move(data))
    {
      AssertThrow(data_,
                  dealii::ExcMessage("Modal data evaluator cannot be empty."));
    }

    template <typename VectorType, typename MatrixType>
    auto
    operator()(SemidiscreteBuilder<VectorType, MatrixType> &builder) const
    {
      const auto                 &field = lift_.source();
      const auto                  id    = builder.algebraic_field(field.name(),
                                              field.locally_owned_dofs(),
                                              field.locally_relevant_dofs());
      dealii::GridTools::Cache<3> background(
        bulk_.dof_handler().get_triangulation(), bulk_.mapping());
      const auto supports =
        make_overlap_supports(IntegrationSupport<1, 3>(
                                field.dof_handler().get_triangulation()),
                              field.mapping(),
                              lift_.support().representative_quadrature(),
                              lift_.lifted_points(),
                              background);
      // Named copies instead of structured bindings: the lambdas below capture
      // them, and capturing a structured binding requires C++20.
      const auto assembled = lift_.assemble_overlap(data_, supports.overlap);
      const auto metric    = std::get<0>(assembled);
      const auto outside   = std::get<1>(assembled);
      const auto rhs       = std::get<2>(assembled);
      auto       metric_operator       = builder.matrix_operator(*metric);
      metric_operator.materialize      = [metric] { return metric; };
      metric_operator.materialize_into = [metric](MatrixType &out) {
        out.copy_from(*metric);
      };
      builder.saddle_point(id, {bulk_.id()}, metric_operator);
      weak_term(value(bulk_), test(lift_))
        .on(supports.overlap)
        .add_constraint_terms(builder, id, 1., 0);
      detail::add_modal_outside_mass(builder, field.with_id(id), outside);
      builder.term(id, "modal data").residual([rhs](const auto &) {
        typename SemiDiscreteModel<VectorType, MatrixType>::Operation result;
        result.reinit_vector = [rhs](VectorType &out, bool omit) {
          out.reinit(*rhs, omit);
        };
        result.apply = [rhs](VectorType &out) {
          out = *rhs;
          out *= -1.;
        };
        result.apply_add = [rhs](VectorType &out) { out -= *rhs; };
        return result;
      });
      return ConstraintFields<FieldTypeAlias>{field.with_id(id)};
    }

  private:
    using FieldTypeAlias = typename ModalLift::source_field_type;
    BulkField          bulk_;
    ModalLift          lift_;
    ModalDataEvaluator data_;
  };
} // namespace ImmersX
#endif
