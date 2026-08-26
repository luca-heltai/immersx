// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License or (at your option) any later version. The
// full text of the license can be found in the LICENSE.md file at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_representation_h
#define immersx_representation_h

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/linear_operator.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/field.h>
#include <immersx/core/problem_handle.h>
#include <immersx/core/state.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/coupling/tensor_product_space.h>

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
  /** A minimal description of a Representation's physical support. */
  struct RepresentationDomain
  {
    RepresentationDomain(const unsigned int dimension   = 0,
                         const unsigned int spacedim    = 0,
                         std::string        geometry_id = "algebraic")
      : dimension(dimension)
      , spacedim(spacedim)
      , geometry_id(std::move(geometry_id))
    {}

    static RepresentationDomain
    algebraic()
    {
      return RepresentationDomain();
    }

    unsigned int dimension;
    unsigned int spacedim;
    std::string  geometry_id;

    friend bool
    operator==(const RepresentationDomain &left,
               const RepresentationDomain &right)
    {
      return left.dimension == right.dimension &&
             left.spacedim == right.spacedim &&
             left.geometry_id == right.geometry_id;
    }

    friend bool
    operator!=(const RepresentationDomain &left,
               const RepresentationDomain &right)
    {
      return !(left == right);
    }
  };

  /** Points and policy used for one Representation evaluation. */
  struct EvaluationRequest
  {
    using Point = dealii::Point<3>;

    EvaluationRequest(std::vector<Point>         points        = {},
                      std::optional<std::string> evaluation_id = {})
      : points(std::move(points))
      , evaluation_id(std::move(evaluation_id))
    {}

    std::vector<Point>         points;
    std::optional<std::string> evaluation_id;

    friend bool
    operator==(const EvaluationRequest &left, const EvaluationRequest &right)
    {
      return left.points == right.points &&
             left.evaluation_id == right.evaluation_id;
    }

    friend bool
    operator!=(const EvaluationRequest &left, const EvaluationRequest &right)
    {
      return !(left == right);
    }
  };

  /**
   * A lightweight description of the space occupied by physical quantity
   * values.
   *
   * The value type is part of the C++ type; the geometric support is carried
   * by the RepresentationDomain.  This object owns no discretization or
   * execution data.
   */
  template <typename ValueType>
  class QuantitySpace
  {
  public:
    using value_type = ValueType;

    explicit QuantitySpace(RepresentationDomain domain = {})
      : domain_(std::move(domain))
    {}

    const RepresentationDomain &
    domain() const
    {
      return domain_;
    }

    unsigned int
    dimension() const
    {
      return domain_.dimension;
    }

    unsigned int
    spacedim() const
    {
      return domain_.spacedim;
    }

    friend bool
    operator==(const QuantitySpace &left, const QuantitySpace &right)
    {
      return left.domain_ == right.domain_;
    }

    friend bool
    operator!=(const QuantitySpace &left, const QuantitySpace &right)
    {
      return !(left == right);
    }

  private:
    RepresentationDomain domain_;
  };

  /**
   * A Representation derivative maps state perturbations to quantity
   * perturbations.  The two vector types are intentionally independent.
   */
  template <typename QuantityVectorType, typename StateVectorType>
  using RepresentationOperator =
    dealii::LinearOperator<QuantityVectorType, StateVectorType>;

  template <typename VectorType>
  class ScaledRepresentation;

  /** A non-owning selection of components from one semantic Field. */
  class FieldComponentView
  {
  public:
    FieldComponentView(const FieldId source, dealii::IndexSet components)
      : source_(source)
      , components_(std::move(components))
    {}

    FieldId
    source() const
    {
      return source_;
    }

    const dealii::IndexSet &
    components() const
    {
      return components_;
    }

  private:
    FieldId          source_;
    dealii::IndexSet components_;
  };

  /** A non-owning evaluated value for a FieldComponentView. */
  template <typename VectorType>
  class FieldComponentValues
  {
  public:
    FieldComponentValues(const VectorType       &values,
                         const dealii::IndexSet &components)
      : values_(&values)
      , components_(&components)
    {}

    const VectorType &
    vector() const
    {
      return *values_;
    }

    bool
    contains(const dealii::types::global_dof_index index) const
    {
      return components_->is_element(index);
    }

    std::size_t
    size() const
    {
      return components_->n_elements();
    }

    auto
    operator[](const dealii::types::global_dof_index index) const
    {
      AssertThrow(contains(index),
                  dealii::ExcMessage("Index is not in the component view."));
      return (*values_)[index];
    }

  private:
    const VectorType       *values_;
    const dealii::IndexSet *components_;
  };

  /**
   * Identity observable for a FieldComponentView.
   *
   * Values remain in the source vector's storage.  The linearization is the
   * corresponding projection in the source Field's vector space, so no state
   * block or compacted component vector is created.
   */
  template <typename VectorType>
  class ComponentRepresentation
  {
  public:
    using value_type          = FieldComponentValues<VectorType>;
    using state_type          = VectorType;
    using quantity_space_type = QuantitySpace<value_type>;
    using Operator            = dealii::LinearOperator<VectorType, VectorType>;

    explicit ComponentRepresentation(
      const FieldComponentView  &source,
      const RepresentationDomain domain = RepresentationDomain::algebraic())
      : source_(source)
      , domain_(domain)
    {}

    const FieldComponentView &
    source() const
    {
      return source_;
    }

    const RepresentationDomain &
    domain() const
    {
      return domain_;
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain_);
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    value_type
    evaluate(const EvaluationContext<VectorType> &context,
             const EvaluationRequest & /*request*/ = {}) const
    {
      return value_type(context.state(source_.source()), source_.components());
    }

    Operator
    linearize(const EvaluationContext<VectorType> &context,
              const EvaluationRequest & /*request*/ = {}) const
    {
      const auto *reference = &context.state(source_.source());
      const auto *mask      = &source_.components();

      Operator result;
      result.reinit_range_vector = [reference](VectorType &vector, bool omit) {
        vector.reinit(*reference, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [mask](VectorType &destination, const VectorType &source) {
        destination = 0.;
        for (const auto index : *mask)
          destination[index] = source[index];
      };
      result.vmult_add = [mask](VectorType       &destination,
                                const VectorType &source) {
        for (const auto index : *mask)
          destination[index] += source[index];
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
      return result;
    }

  private:
    FieldComponentView   source_;
    RepresentationDomain domain_;
  };

  /**
   * A lightweight semantic representation of one Field.
   *
   * This identity representation owns no state and creates no execution
   * storage.  It evaluates by returning the source field from an
   * EvaluationContext and linearizes to the identity operator on that field.
   * More general representations can use the same value-object style while
   * depending on one or more Fields.
   */
  template <typename VectorType>
  class Representation
  {
  public:
    using value_type          = VectorType;
    using state_type          = VectorType;
    using quantity_space_type = QuantitySpace<value_type>;
    using Operator            = RepresentationOperator<value_type, state_type>;

    explicit Representation(
      const FieldId              source,
      const RepresentationDomain domain = RepresentationDomain::algebraic())
      : source_(source)
      , domain_(domain)
    {}

    FieldId
    source() const
    {
      return source_;
    }

    /** Return the physical evaluation domain, independent of the source Field.
     */
    const RepresentationDomain &
    domain() const
    {
      return domain_;
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain_);
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    /** Return the scalar observable q = factor * u. */
    ScaledRepresentation<VectorType>
    scaled(const double factor) const;

    const VectorType &
    evaluate(const EvaluationContext<VectorType> &context,
             const EvaluationRequest & /*request*/ = {}) const
    {
      return context.state(source_);
    }

    Operator
    linearize(const EvaluationContext<VectorType> &context,
              const EvaluationRequest & /*request*/ = {}) const
    {
      const auto *reference = &context.state(source_);

      Operator result;
      result.reinit_range_vector = [reference](VectorType &vector, bool omit) {
        vector.reinit(*reference, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult = [](VectorType &destination, const VectorType &source) {
        destination = source;
      };
      result.vmult_add = [](VectorType &destination, const VectorType &source) {
        destination += source;
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
      return result;
    }

  private:
    FieldId              source_;
    RepresentationDomain domain_;
  };

  /**
   * A scalar transform of an algebraic Representation.
   *
   * This value type owns no state and does not know the target Problem.  Its
   * evaluation is an owned vector value because a transformed observable no
   * longer aliases the source Field; its linearization remains an operator on
   * the source Field's vector space.
   */
  template <typename VectorType>
  class ScaledRepresentation
  {
  public:
    using value_type          = VectorType;
    using state_type          = VectorType;
    using quantity_space_type = QuantitySpace<value_type>;
    using Operator            = RepresentationOperator<value_type, state_type>;

    ScaledRepresentation(const Representation<VectorType> &source,
                         const double                      factor)
      : source_(source)
      , factor_(factor)
    {}

    FieldId
    source() const
    {
      return source_.source();
    }

    const RepresentationDomain &
    domain() const
    {
      return source_.domain();
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain());
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    double
    factor() const
    {
      return factor_;
    }

    value_type
    evaluate(const EvaluationContext<VectorType> &context,
             const EvaluationRequest             &request = {}) const
    {
      value_type result = source_.evaluate(context, request);
      result *= factor_;
      return result;
    }

    Operator
    linearize(const EvaluationContext<VectorType> &context,
              const EvaluationRequest & /*request*/ = {}) const
    {
      const auto *reference = &context.state(source_.source());
      const auto  factor    = factor_;

      Operator result;
      result.reinit_range_vector = [reference](VectorType &vector, bool omit) {
        vector.reinit(*reference, omit);
      };
      result.reinit_domain_vector = result.reinit_range_vector;
      result.vmult                = [factor](VectorType       &destination,
                              const VectorType &source) {
        destination = source;
        destination *= factor;
      };
      result.vmult_add = [factor](VectorType       &destination,
                                  const VectorType &source) {
        VectorType contribution = source;
        contribution *= factor;
        destination += contribution;
      };
      result.Tvmult     = result.vmult;
      result.Tvmult_add = result.vmult_add;
      return result;
    }

  private:
    Representation<VectorType> source_;
    double                     factor_;
  };

  template <typename VectorType>
  ScaledRepresentation<VectorType>
  Representation<VectorType>::scaled(const double factor) const
  {
    return ScaledRepresentation<VectorType>(*this, factor);
  }

  /**
   * One physical quadrature point together with the algebraic data that
   * represents its physical field value.
   *
   * A direct finite-element representation fills this structure with the DoFs
   * on the local FE cell and their basis values. A later TensorProductSpace
   * adapter can provide the same data with representative DoFs instead, without
   * materializing an explicit R or R^T matrix.
   *
   * `ValueType` is the physical value carried by one basis function. It is
   * scalar for the existing identity and tensor-product coupling path, but can
   * be a `dealii::Tensor` for vector- or tensor-valued observables. The local
   * DoF slots are retained even when an extractor gives a zero value for a
   * different field in a mixed element; this preserves the FE indexing needed
   * by later assembly adapters while keeping the physical value typed.
   */
  template <int spacedim, typename ValueType = double>
  struct RepresentationQuadraturePoint
  {
    using value_type = ValueType;

    dealii::Point<spacedim>     point;
    dealii::Point<spacedim>     representative_point;
    dealii::Tensor<1, spacedim> tangent;
    double                      weight = 0.;
    /** Stable source entity identity, when the source has geometric cells. */
    dealii::types::global_cell_index source_entity_id =
      dealii::numbers::invalid_unsigned_int;
    /** Quadrature slot on the source representative entity. */
    unsigned int representative_qpoint = dealii::numbers::invalid_unsigned_int;
    /** Quadrature slot on a lifted cross-section. */
    unsigned int section_qpoint = dealii::numbers::invalid_unsigned_int;
    /** Stable identity used when this point is redistributed. */
    std::uint64_t stable_id = std::numeric_limits<std::uint64_t>::max();
    std::vector<dealii::types::global_dof_index> dof_indices;
    std::vector<ValueType>                       basis_values;
  };


  /** Semantic fields consumed by a physical representation. */
  struct RepresentationMetadata
  {
    std::vector<FieldId> dependencies;
  };


  /**
   * Compile-time description of the interaction-facing representation API.
   *
   * ImmersX deliberately keeps this as a lightweight, template-based contract
   * rather than introducing a virtual representation hierarchy.  A compliant
   * representation provides an algebraic DoF space and constraints, a mapping
   * used by the interaction's point search, and
   * `locally_owned_quadrature_points()`.  The latter returns physical points,
   * physical integration weights, representative algebraic DoF indices, and
   * the corresponding basis values.  In particular, the algebraic dimension
   * need not equal `support_dimension`: a TensorProductSpace representation can
   * expose reduced coefficients on a higher-dimensional physical support.
   */
  template <typename Representation, typename = void>
  struct RepresentationConcept : std::false_type
  {};


  template <typename Representation>
  struct RepresentationConcept<
    Representation,
    std::void_t<
      decltype(Representation::support_dimension),
      decltype(Representation::ambient_dimension),
      typename Representation::value_type,
      typename Representation::ExtractorType,
      typename Representation::TriangulationType,
      typename Representation::DoFHandlerType,
      typename Representation::QuadraturePoint,
      decltype(std::declval<const Representation &>().triangulation()),
      decltype(std::declval<const Representation &>().dof_handler()),
      decltype(std::declval<const Representation &>().finite_element()),
      decltype(std::declval<const Representation &>().mapping()),
      decltype(std::declval<const Representation &>().locally_owned_dofs()),
      decltype(std::declval<const Representation &>().locally_relevant_dofs()),
      decltype(std::declval<const Representation &>().constraints()),
      decltype(std::declval<const Representation &>().mpi_communicator()),
      decltype(std::declval<const Representation &>().n_dofs_per_cell()),
      decltype(std::declval<const Representation &>().metadata()),
      decltype(std::declval<const Representation &>().dependencies()),
      decltype(std::declval<const Representation &>().geometry_version()),
      decltype(std::declval<const Representation &>()
                 .locally_owned_quadrature_points(
                   std::declval<const dealii::Quadrature<
                     Representation::support_dimension> &>()))>>
    : std::true_type
  {};


  /**
   * A non-owning view of a scalar finite-element space used as a physical
   * representation.
   *
   * This is the first, identity case of the representation abstraction: the
   * algebraic DoFs and the physical FE basis are the same, so R=I. It owns no
   * PDE state and has no coupling pointer. Interactions can freely construct
   * more than one view of the same problem and reuse a view in more than one
   * interaction. The triangulation, DoFHandler, index sets, constraints, and
   * mapping supplied to the view are non-owning and must outlive it.
   *
   * The interaction-facing contract is intentionally expressed through
   * `locally_owned_quadrature_points()`: a representation supplies physical
   * points, weights, algebraic representative DoF indices, and represented
   * basis values. A TensorProductSpace adapter should implement the same
   * contract for lifted fields on a coupling support.
   */
  template <int dim,
            int spacedim       = dim,
            typename ValueType = double,
            typename Extractor = dealii::FEValuesExtractors::Scalar>
  class FiniteElementRepresentation
  {
  public:
    static constexpr unsigned int support_dimension        = dim;
    static constexpr unsigned int ambient_dimension        = spacedim;
    static constexpr unsigned int representative_dimension = dim;

    using value_type    = ValueType;
    using ExtractorType = Extractor;
    /**
     * Algebraic coefficient space of the represented field. Geometry-only
     * views do not implement evaluate/linearize; the typedef satisfies the
     * lifting contract so that views can be composed with tensor-product
     * lifts without owning field state.
     */
    using state_type = ImmersXLA::MPI::Vector;
    using TriangulationType =
      dealii::parallel::TriangulationBase<dim, spacedim>;
    using DoFHandlerType  = dealii::DoFHandler<dim, spacedim>;
    using QuadraturePoint = RepresentationQuadraturePoint<spacedim, ValueType>;

    FiniteElementRepresentation(
      const TriangulationType                 &triangulation,
      const DoFHandlerType                    &dof_handler,
      const dealii::IndexSet                  &locally_owned_dofs,
      const dealii::IndexSet                  &locally_relevant_dofs,
      const dealii::AffineConstraints<double> &constraints,
      const dealii::Mapping<dim, spacedim>    &mapping =
        dealii::StaticMappingQ1<dim, spacedim>::mapping,
      const ExtractorType                   &extractor      = ExtractorType(0),
      const ImmersX::RepresentationMetadata &metadata       = {},
      const bool                             all_components = false)
      : triangulation_(triangulation)
      , dof_handler_(dof_handler)
      , locally_owned_dofs_(locally_owned_dofs)
      , locally_relevant_dofs_(locally_relevant_dofs)
      , constraints_(constraints)
      , mapping_(mapping)
      , extractor_(extractor)
      , metadata_(metadata)
      , all_components_(all_components)
    {}

    const TriangulationType &
    triangulation() const
    {
      return triangulation_;
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return dof_handler_;
    }

    const dealii::FiniteElement<dim, spacedim> &
    finite_element() const
    {
      return dof_handler_.get_fe();
    }

    const dealii::Mapping<dim, spacedim> &
    mapping() const
    {
      return mapping_;
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return locally_owned_dofs_;
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return locally_relevant_dofs_;
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return constraints_;
    }

    const ExtractorType &
    extractor() const
    {
      return extractor_;
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    const ImmersX::RepresentationMetadata &
    metadata() const
    {
      return metadata_;
    }

    const std::vector<ImmersX::FieldId> &
    dependencies() const
    {
      return metadata_.dependencies;
    }

    /**
     * Return the geometry generation used by cached interaction data.
     *
     * The representation does not cache quadrature values, so every evaluation
     * reads the current triangulation/mapping. Callers that move or otherwise
     * update the geometry can invalidate dependent interaction data explicitly
     * and then reassemble it.
     */
    std::uint64_t
    geometry_version() const
    {
      return geometry_version_;
    }

    void
    invalidate_geometry() const
    {
      ++geometry_version_;
    }

    MPI_Comm
    mpi_communicator() const
    {
      return triangulation_.get_mpi_communicator();
    }

    unsigned int
    n_dofs_per_cell() const
    {
      return finite_element().n_dofs_per_cell();
    }

    /**
     * Extract the local physical representation at the supplied quadrature.
     *
     * The returned data is local to the calling rank. Point ownership is the
     * ownership of the FE cells in this view, which lets an Interaction layer
     * hand the points to its own distributed geometric search machinery.
     */
    std::vector<QuadraturePoint>
    locally_owned_quadrature_points(
      const dealii::Quadrature<dim> &quadrature) const
    {
      dealii::UpdateFlags update_flags = dealii::update_values |
                                         dealii::update_quadrature_points |
                                         dealii::update_JxW_values;
      if constexpr (dim > 1 && dim < spacedim)
        update_flags |= dealii::update_normal_vectors;
      dealii::FEValues<dim, spacedim> fe_values(mapping_,
                                                finite_element(),
                                                quadrature,
                                                update_flags);

      std::vector<QuadraturePoint> points;
      points.reserve(triangulation_.n_locally_owned_active_cells() *
                     quadrature.size());

      std::vector<dealii::types::global_dof_index> dof_indices(
        n_dofs_per_cell());
      for (const auto &cell : dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (const auto q : fe_values.quadrature_point_indices())
              {
                QuadraturePoint point;
                point.point                 = fe_values.quadrature_point(q);
                point.source_entity_id      = cell->global_active_cell_index();
                point.representative_qpoint = q;
                if constexpr (dim == 1)
                  point.tangent = cell->vertex(1) - cell->vertex(0);
                else if constexpr (dim > 1 && dim < spacedim)
                  point.tangent = fe_values.normal_vector(q);
                point.weight      = fe_values.JxW(q);
                point.dof_indices = dof_indices;
                point.basis_values.resize(n_dofs_per_cell());
                if (all_components_)
                  {
                    // A modal finite element (e.g. FESystem(scalar_fe,
                    // n_modes)) exposes one algebraic slot per (local DoF,
                    // mode). The plain shape value of each slot is the
                    // reduced basis of that slot's component, which is what a
                    // tensor-product lift multiplies by the section mode.
                    for (unsigned int i = 0; i < n_dofs_per_cell(); ++i)
                      point.basis_values[i] = fe_values.shape_value(i, q);
                  }
                else
                  for (unsigned int i = 0; i < n_dofs_per_cell(); ++i)
                    point.basis_values[i] = fe_values[extractor_].value(i, q);
                points.emplace_back(std::move(point));
              }
          }

      return points;
    }

  private:
    const TriangulationType                 &triangulation_;
    const DoFHandlerType                    &dof_handler_;
    const dealii::IndexSet                  &locally_owned_dofs_;
    const dealii::IndexSet                  &locally_relevant_dofs_;
    const dealii::AffineConstraints<double> &constraints_;
    const dealii::Mapping<dim, spacedim>    &mapping_;
    const ExtractorType                      extractor_;
    ImmersX::RepresentationMetadata          metadata_;
    const bool                               all_components_;
    mutable std::uint64_t                    geometry_version_ = 0;
  };


  /** Explicit name for the R=I finite-element representation. */
  template <int dim, int spacedim = dim>
  using IdentityRepresentation = FiniteElementRepresentation<dim, spacedim>;


  /** Direct FE representation of a selected vector-valued field. */
  template <int dim, int spacedim = dim>
  using VectorFiniteElementRepresentation =
    FiniteElementRepresentation<dim,
                                spacedim,
                                dealii::Tensor<1, spacedim>,
                                dealii::FEValuesExtractors::Vector>;


  /**
   * Direct FE representation of a selected general tensor-valued field.
   *
   * This alias is intentionally small: the same typed quadrature payload and
   * extractor-based evaluation path can support tensor observables without a
   * new representation hierarchy.
   */
  template <int dim, int spacedim = dim, int rank = 2>
  using TensorFiniteElementRepresentation =
    FiniteElementRepresentation<dim,
                                spacedim,
                                dealii::Tensor<rank, spacedim>,
                                dealii::FEValuesExtractors::Tensor<rank>>;


  /**
   * Tensor-product physical representation backed by reduced-dimensional DoFs.
   *
   * The TensorProductSpace supplies the lifted physical quadrature on the
   * cross-section, while the supplied DoFHandler supplies the algebraic
   * representative coefficients.  This keeps the adapter a reusable view: it
   * owns neither a PDE nor an Interaction and can be exposed to more than one
   * coupling.
   */
  template <int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components   = 1,
            typename ValueType = double,
            typename Extractor = dealii::FEValuesExtractors::Scalar>
  class TensorProductRepresentation
  {
  public:
    static constexpr unsigned int support_dimension        = surface_dim;
    static constexpr unsigned int ambient_dimension        = spacedim;
    static constexpr unsigned int representative_dimension = reduced_dim;

    using value_type    = ValueType;
    using ExtractorType = Extractor;
    using TensorProductSpaceType =
      TensorProductSpace<reduced_dim, surface_dim, spacedim, n_components>;
    using TriangulationType =
      dealii::parallel::TriangulationBase<reduced_dim, spacedim>;
    using DoFHandlerType  = dealii::DoFHandler<reduced_dim, spacedim>;
    using QuadraturePoint = RepresentationQuadraturePoint<spacedim, ValueType>;

    TensorProductRepresentation(
      const TensorProductSpaceType                 &space,
      const DoFHandlerType                         &representative_dof_handler,
      const dealii::IndexSet                       &locally_owned_dofs,
      const dealii::IndexSet                       &locally_relevant_dofs,
      const dealii::AffineConstraints<double>      &constraints,
      const dealii::Mapping<reduced_dim, spacedim> &mapping =
        dealii::StaticMappingQ1<reduced_dim, spacedim>::mapping,
      const ExtractorType                   &extractor = ExtractorType(0),
      const ImmersX::RepresentationMetadata &metadata  = {})
      : space_(space)
      , dof_handler_(representative_dof_handler)
      , locally_owned_dofs_(locally_owned_dofs)
      , locally_relevant_dofs_(locally_relevant_dofs)
      , constraints_(constraints)
      , mapping_(mapping)
      , extractor_(extractor)
      , metadata_(metadata)
    {}

    /**
     * Construct a lifting from an existing representative Representation.
     *
     * The representative view supplies the reduced Problem's DoF space and
     * geometry-dependent metadata. TensorProductSpace remains accepted as a
     * compatibility helper for the reduced mesh and lifted quadrature cache;
     * the returned object is the reusable lifting consumed by Interactions.
     */
    template <typename Representative>
    TensorProductRepresentation(const TensorProductSpaceType &space,
                                const Representative         &representative)
      : TensorProductRepresentation(space,
                                    representative.dof_handler(),
                                    representative.locally_owned_dofs(),
                                    representative.locally_relevant_dofs(),
                                    representative.constraints(),
                                    representative.mapping(),
                                    representative.extractor(),
                                    representative.metadata())
    {}

    const TriangulationType &
    triangulation() const
    {
      return space_.get_triangulation();
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return dof_handler_;
    }

    const dealii::FiniteElement<reduced_dim, spacedim> &
    finite_element() const
    {
      return dof_handler_.get_fe();
    }

    const dealii::Mapping<reduced_dim, spacedim> &
    mapping() const
    {
      return mapping_;
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return locally_owned_dofs_;
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return locally_relevant_dofs_;
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return constraints_;
    }

    const ExtractorType &
    extractor() const
    {
      return extractor_;
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    const ImmersX::RepresentationMetadata &
    metadata() const
    {
      return metadata_;
    }

    const std::vector<ImmersX::FieldId> &
    dependencies() const
    {
      return metadata_.dependencies;
    }

    std::uint64_t
    geometry_version() const
    {
      return geometry_version_;
    }

    /** Invalidate interaction quadrature after an external geometry update. */
    void
    invalidate_geometry() const
    {
      ++geometry_version_;
    }

    MPI_Comm
    mpi_communicator() const
    {
      return space_.get_triangulation().get_mpi_communicator();
    }

    unsigned int
    n_dofs_per_cell() const
    {
      return finite_element().n_dofs_per_cell();
    }

    /**
     * Return physical surface quadrature points represented by reduced DoFs.
     *
     * The interaction chooses its quadrature from `dimension == surface_dim`.
     * TensorProductSpace has already built the lifted quadrature, so the
     * surface quadrature argument is intentionally ignored here.  For every
     * lifted cross-section point we attach the FE basis values evaluated on the
     * reduced representative cell, with no explicit surface DoFHandler.
     */
    std::vector<QuadraturePoint>
    locally_owned_quadrature_points(
      const dealii::Quadrature<surface_dim> &surface_quadrature) const
    {
      (void)surface_quadrature;

      const auto &lifted_points  = space_.get_locally_owned_qpoints();
      const auto &lifted_weights = space_.get_locally_owned_weights();
      const auto &representative_quadrature = space_.get_quadrature();
      const auto &section = space_.get_reference_cross_section();

      AssertDimension(lifted_points.size(), lifted_weights.size());

      dealii::FEValues<reduced_dim, spacedim> fe_values(
        mapping_,
        finite_element(),
        representative_quadrature,
        dealii::update_values);

      std::map<dealii::types::global_cell_index,
               typename DoFHandlerType::cell_iterator>
        representative_cells;
      for (const auto &cell : dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          representative_cells.emplace(cell->global_active_cell_index(), cell);

      std::vector<QuadraturePoint> points;
      points.reserve(lifted_points.size());
      std::vector<dealii::types::global_dof_index> dof_indices(
        n_dofs_per_cell());

      std::size_t point_index = 0;
      for (const auto &cell :
           space_.get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto representative_cell =
              representative_cells.find(cell->global_active_cell_index());
            AssertThrow(representative_cell != representative_cells.end(),
                        dealii::ExcMessage(
                          "Tensor-product and representative meshes must use "
                          "the same local cell partition."));

            fe_values.reinit(representative_cell->second);
            representative_cell->second->get_dof_indices(dof_indices);
            for (const auto q : fe_values.quadrature_point_indices())
              for (unsigned int section_q = 0;
                   section_q < section.n_quadrature_points();
                   ++section_q)
                {
                  AssertIndexRange(point_index, lifted_points.size());
                  AssertThrow(!lifted_weights[point_index].empty(),
                              dealii::ExcMessage(
                                "A tensor-product quadrature point has no "
                                "physical weight."));

                  QuadraturePoint point;
                  point.point            = lifted_points[point_index];
                  point.weight           = lifted_weights[point_index][0];
                  point.source_entity_id = cell->global_active_cell_index();
                  point.representative_qpoint = q;
                  point.section_qpoint        = section_q;
                  if constexpr (reduced_dim == 1)
                    point.tangent = cell->vertex(1) - cell->vertex(0);
                  point.dof_indices = dof_indices;
                  point.basis_values.resize(n_dofs_per_cell());
                  for (unsigned int i = 0; i < n_dofs_per_cell(); ++i)
                    point.basis_values[i] = fe_values[extractor_].value(i, q);
                  points.emplace_back(std::move(point));
                  ++point_index;
                }
          }

      AssertDimension(point_index, lifted_points.size());
      return points;
    }

  private:
    const TensorProductSpaceType                 &space_;
    const DoFHandlerType                         &dof_handler_;
    const dealii::IndexSet                       &locally_owned_dofs_;
    const dealii::IndexSet                       &locally_relevant_dofs_;
    const dealii::AffineConstraints<double>      &constraints_;
    const dealii::Mapping<reduced_dim, spacedim> &mapping_;
    const ExtractorType                           extractor_;
    ImmersX::RepresentationMetadata               metadata_;
    mutable std::uint64_t                         geometry_version_ = 0;
  };

  namespace detail
  {
    /** Evaluate one retained FE stencil without locating its physical cell. */
    template <typename VectorType>
    double
    evaluate_stencil(
      const VectorType                                   &source,
      const std::vector<dealii::types::global_dof_index> &dof_indices,
      const std::vector<double>                          &basis_values)
    {
      AssertDimension(dof_indices.size(), basis_values.size());
      double result = 0.;
      for (unsigned int i = 0; i < dof_indices.size(); ++i)
        result += basis_values[i] * source[dof_indices[i]];
      return result;
    }

    /** Detect a source-provided symbolic thickness evaluation. */
    template <typename Source, int spacedim, typename = void>
    struct has_source_thickness : std::false_type
    {};

    template <typename Source, int spacedim>
    struct has_source_thickness<
      Source,
      spacedim,
      std::void_t<decltype(std::declval<const Source &>().evaluate_thickness(
        std::declval<const dealii::Point<spacedim> &>(),
        std::declval<double>(),
        std::declval<const std::vector<double> &>()))>> : std::true_type
    {};
  } // namespace detail


  /**
   * A state-independent finite-element sampling plan.
   *
   * The plan is a retained collection of FE stencils.  It owns the point
   * payload copied from a Representation, including the source DoF indices
   * and basis values, but it never stores a state vector and never searches
   * for a cell.  Point values are distributed according to the rank that
   * retained each stencil, which makes the output a regular distributed
   * vector and keeps the transpose local except for the normal DoF
   * compression.
   */
  template <int spacedim,
            typename StateVectorType    = ImmersXLA::MPI::Vector,
            typename QuantityVectorType = ImmersXLA::MPI::Vector>
  class RetainedSamplingPlan
  {
  public:
    using Point = RepresentationQuadraturePoint<spacedim, double>;
    using Operator =
      RepresentationOperator<QuantityVectorType, StateVectorType>;

    RetainedSamplingPlan(std::vector<Point>      points,
                         const dealii::IndexSet &source_owned,
                         const dealii::IndexSet &source_relevant,
                         const dealii::AffineConstraints<double> *constraints,
                         const MPI_Comm                           communicator)
      : points_(std::move(points))
      , source_owned_(source_owned)
      , source_relevant_(source_relevant)
      , constraints_(constraints)
      , communicator_(communicator)
    {
      const auto [offset, size] =
        dealii::Utilities::MPI::partial_and_total_sum(points_.size(),
                                                      communicator_);
      point_owned_.set_size(size);
      point_owned_.add_range(offset, offset + points_.size());
      point_owned_.compress();
      point_relevant_ = point_owned_;
      point_indices_.reserve(points_.size());
      for (std::size_t q = 0; q < points_.size(); ++q)
        point_indices_.push_back(offset + q);
    }

    const std::vector<Point> &
    points() const
    {
      return points_;
    }

    const dealii::IndexSet &
    locally_owned_points() const
    {
      return point_owned_;
    }

    const dealii::IndexSet &
    locally_relevant_points() const
    {
      return point_relevant_;
    }

    dealii::types::global_dof_index
    point_index(const std::size_t local_point) const
    {
      AssertIndexRange(local_point, point_indices_.size());
      return point_indices_[local_point];
    }

    MPI_Comm
    mpi_communicator() const
    {
      return communicator_;
    }

    Operator
    linearize(const StateVectorType &prototype) const
    {
      const auto points          = points_;
      const auto point_indices   = point_indices_;
      const auto point_owned     = point_owned_;
      const auto point_relevant  = point_relevant_;
      const auto source_owned    = source_owned_;
      const auto source_relevant = source_relevant_;
      const auto constraints     = constraints_;
      const auto communicator    = communicator_;

      Operator result;
      result.reinit_range_vector =
        [point_owned, point_relevant, communicator](QuantityVectorType &vector,
                                                    const bool          omit) {
          vector.reinit(point_owned, point_relevant, communicator);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector = [prototype](StateVectorType &vector,
                                                const bool       omit) {
        vector.reinit(prototype, omit);
      };
      result.vmult = [points,
                      point_indices,
                      source_owned,
                      source_relevant,
                      constraints,
                      communicator](QuantityVectorType    &destination,
                                    const StateVectorType &source) {
        const auto relevant = make_relevant_source(
          source, source_owned, source_relevant, constraints, communicator);
        for (std::size_t q = 0; q < points.size(); ++q)
          destination[point_indices[q]] =
            detail::evaluate_stencil(relevant,
                                     points[q].dof_indices,
                                     points[q].basis_values);
      };
      result.vmult_add = [points,
                          point_indices,
                          source_owned,
                          source_relevant,
                          constraints,
                          communicator](QuantityVectorType    &destination,
                                        const StateVectorType &source) {
        const auto relevant = make_relevant_source(
          source, source_owned, source_relevant, constraints, communicator);
        for (std::size_t q = 0; q < points.size(); ++q)
          destination[point_indices[q]] +=
            detail::evaluate_stencil(relevant,
                                     points[q].dof_indices,
                                     points[q].basis_values);
      };
      result.Tvmult = [points,
                       point_indices,
                       source_owned,
                       source_relevant,
                       constraints,
                       communicator](StateVectorType          &destination,
                                     const QuantityVectorType &source) {
        apply_transpose(points,
                        point_indices,
                        source_owned,
                        source_relevant,
                        constraints,
                        communicator,
                        source,
                        destination,
                        false);
      };
      result.Tvmult_add = [points,
                           point_indices,
                           source_owned,
                           source_relevant,
                           constraints,
                           communicator](StateVectorType          &destination,
                                         const QuantityVectorType &source) {
        apply_transpose(points,
                        point_indices,
                        source_owned,
                        source_relevant,
                        constraints,
                        communicator,
                        source,
                        destination,
                        true);
      };
      return result;
    }

  private:
    static StateVectorType
    make_relevant_source(const StateVectorType  &source,
                         const dealii::IndexSet &source_owned,
                         const dealii::IndexSet &source_relevant,
                         const dealii::AffineConstraints<double> *constraints,
                         const MPI_Comm                           communicator)
    {
      StateVectorType owned;
      owned.reinit(source_owned, communicator);
      owned = source;
      if (constraints != nullptr)
        constraints->distribute(owned);

      StateVectorType relevant;
      relevant.reinit(source_owned, source_relevant, communicator);
      relevant = owned;
      relevant.update_ghost_values();
      return relevant;
    }

    static void
    apply_transpose(
      const std::vector<Point>                           &points,
      const std::vector<dealii::types::global_dof_index> &point_indices,
      const dealii::IndexSet                             &source_owned,
      const dealii::IndexSet                             &source_relevant,
      const dealii::AffineConstraints<double>            *constraints,
      const MPI_Comm                                      communicator,
      const QuantityVectorType                           &source,
      StateVectorType                                    &destination,
      const bool                                          add)
    {
      dealii::LinearAlgebra::distributed::Vector<double> contribution;
      contribution.reinit(source_owned, source_relevant, communicator);
      contribution = 0.;
      for (std::size_t q = 0; q < points.size(); ++q)
        for (std::size_t i = 0; i < points[q].dof_indices.size(); ++i)
          contribution[points[q].dof_indices[i]] +=
            points[q].basis_values[i] * source[point_indices[q]];
      contribution.compress(dealii::VectorOperation::add);

      if (constraints != nullptr)
        {
          dealii::LinearAlgebra::distributed::Vector<double> correction;
          correction.reinit(source_owned, source_relevant, communicator);
          correction = 0.;
          for (const auto &line : constraints->get_lines())
            if (source_owned.is_element(line.index))
              {
                const double constrained = contribution[line.index];
                correction[line.index] -= constrained;
                for (const auto &[master, coefficient] : line.entries)
                  correction[master] += coefficient * constrained;
              }
          correction.compress(dealii::VectorOperation::add);
          for (const auto index : source_owned)
            contribution[index] += correction[index];
        }

      if (!add)
        destination = 0.;
      for (const auto index : source_owned)
        destination[index] += contribution[index];
    }

    std::vector<Point>                           points_;
    dealii::IndexSet                             source_owned_;
    dealii::IndexSet                             source_relevant_;
    dealii::IndexSet                             point_owned_;
    dealii::IndexSet                             point_relevant_;
    std::vector<dealii::types::global_dof_index> point_indices_;
    const dealii::AffineConstraints<double>     *constraints_;
    MPI_Comm                                     communicator_;
  };


  /** Build a retained value-sampling plan from a finite-element view. */
  template <int dim, int spacedim, typename ValueType, typename Extractor>
  auto
  make_retained_sampling_plan(
    const FiniteElementRepresentation<dim, spacedim, ValueType, Extractor>
                                  &representation,
    const dealii::Quadrature<dim> &quadrature) -> RetainedSamplingPlan<spacedim>
  {
    static_assert(std::is_same_v<ValueType, double>,
                  "Retained scalar sampling requires a scalar FE view.");
    return RetainedSamplingPlan<spacedim>(
      representation.locally_owned_quadrature_points(quadrature),
      representation.locally_owned_dofs(),
      representation.locally_relevant_dofs(),
      &representation.constraints(),
      representation.mpi_communicator());
  }


  /**
   * A parameterized tensor-product lift of a source Representation.
   *
   * The source view remains the sole owner of the representative mesh, FE,
   * DoF space, and dependencies. This class owns only the lifting support and
   * the local tensor-product quadrature metadata.
   *
   * Two algebraic semantics are supported and are never silently conflated:
   *
   * - Physical source (CASE A): the source is one scalar (or vector)
   *   coefficient set, e.g. a pressure field lifted as constant across the
   *   cross-section. Only mode 0 is meaningful as a new coefficient; the
   *   default (non-modal) construction enforces that.
   *
   * - Modal source (CASE B): the source field itself owns independent
   *   coefficients indexed by (representative DoF, mode), e.g. an RLM
   *   multiplier lambda_{i,m}. The source representation must expose one
   *   algebraic slot per (local DoF, mode) in deal.II `FESystem` order (mode
   *   varying fastest). The modal construction pairs slot (i, mode) with
   *   section mode `mode` and keeps the slot's algebraic index, so mode 0
   *   and mode 1 are distinct unknowns.
   */
  template <typename SourceRepresentation,
            int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components = 1>
  class TensorProductLiftRepresentation
  {
  public:
    static constexpr unsigned int support_dimension        = surface_dim;
    static constexpr unsigned int ambient_dimension        = spacedim;
    static constexpr unsigned int representative_dimension = reduced_dim;

    using value_type          = typename SourceRepresentation::value_type;
    using state_type          = typename SourceRepresentation::state_type;
    using quantity_space_type = QuantitySpace<value_type>;
    using Operator            = RepresentationOperator<value_type, state_type>;
    using Parameters          = TensorProductLiftParameters<reduced_dim,
                                                   surface_dim,
                                                   spacedim,
                                                   n_components>;
    using Lift =
      TensorProductLift<reduced_dim, surface_dim, spacedim, n_components>;
    using Support = TensorProductLiftSupport<reduced_dim,
                                             surface_dim,
                                             spacedim,
                                             n_components>;
    using SourceQuadraturePoint =
      typename SourceRepresentation::QuadraturePoint;
    using TriangulationType = typename SourceRepresentation::TriangulationType;
    using DoFHandlerType    = typename SourceRepresentation::DoFHandlerType;
    using ExtractorType     = typename SourceRepresentation::ExtractorType;
    using QuadraturePoint   = RepresentationQuadraturePoint<spacedim, double>;

    TensorProductLiftRepresentation(const SourceRepresentation &source,
                                    const Lift                 &lift,
                                    const bool                  modal = false)
      : source_(source)
      , lift_(lift)
      , support_(lift.parameters())
      , modal_(modal)
    {
      install_source_thickness_provider();
      lifted_points_ = build_lifted_points();
    }

    FieldId
    source() const
    {
      return source_.source();
    }

    const RepresentationDomain &
    domain() const
    {
      static const RepresentationDomain domain(surface_dim,
                                               spacedim,
                                               "tensor-product-lift");
      return domain;
    }

    quantity_space_type
    quantity_space() const
    {
      return quantity_space_type(domain());
    }

    template <typename Geometry>
    decltype(auto)
    lift(const Geometry &geometry) const
    {
      return detail::invoke_lift(*this, geometry, 0);
    }

    const SourceRepresentation &
    source_representation() const
    {
      return source_;
    }

    const Lift &
    lift_descriptor() const
    {
      return lift_;
    }

    /**
     * Return whether this lift treats the source slots as independent modal
     * coefficients (CASE B) or as one physical coefficient set (CASE A).
     */
    bool
    modal() const
    {
      return modal_;
    }

    const Support &
    support() const
    {
      return support_;
    }

    const std::vector<TensorProductLiftPoint<surface_dim, spacedim>> &
    lifted_points() const
    {
      return lifted_points_;
    }

    /** Evaluate all lifted points using their retained source FE stencils. */
    value_type
    evaluate_stencils(const state_type &source) const
    {
      return evaluate_stencils(source,
                               lifted_points_,
                               source_.locally_owned_dofs(),
                               source_.locally_relevant_dofs(),
                               source_.mpi_communicator());
    }

    const TriangulationType &
    triangulation() const
    {
      return source_.triangulation();
    }

    const DoFHandlerType &
    dof_handler() const
    {
      return source_.dof_handler();
    }

    const dealii::FiniteElement<reduced_dim, spacedim> &
    finite_element() const
    {
      return source_.finite_element();
    }

    const dealii::Mapping<reduced_dim, spacedim> &
    mapping() const
    {
      return source_.mapping();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return source_.locally_owned_dofs();
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return source_.locally_relevant_dofs();
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return source_.constraints();
    }

    const ExtractorType &
    extractor() const
    {
      return source_.extractor();
    }

    const ImmersX::RepresentationMetadata &
    metadata() const
    {
      return source_.metadata();
    }

    const std::vector<ImmersX::FieldId> &
    dependencies() const
    {
      return source_.dependencies();
    }

    std::uint64_t
    geometry_version() const
    {
      return source_.geometry_version();
    }

    MPI_Comm
    mpi_communicator() const
    {
      return source_.mpi_communicator();
    }

    unsigned int
    n_dofs_per_cell() const
    {
      return source_.n_dofs_per_cell();
    }

    std::vector<QuadraturePoint>
    locally_owned_quadrature_points(
      const dealii::Quadrature<surface_dim> &surface_quadrature) const
    {
      (void)surface_quadrature;
      std::vector<QuadraturePoint> result;
      result.reserve(lifted_points_.size());
      const unsigned int n_modes =
        support_.reference_cross_section().n_selected_basis() * n_components;

      for (const auto &lifted : lifted_points_)
        {
          QuadraturePoint point;
          point.point                 = lifted.point;
          point.representative_point  = lifted.representative_point;
          point.weight                = lifted.weight;
          point.source_entity_id      = lifted.source_entity_id;
          point.representative_qpoint = lifted.source_representative_qpoint;
          point.section_qpoint        = lifted.section_qpoint;
          point.stable_id             = lifted.stable_id;
          point.dof_indices.reserve(lifted.source_dof_indices.size());
          point.basis_values.reserve(lifted.source_dof_indices.size());

          if (modal_)
            {
              // The source slots follow the deal.II FESystem layout:
              // (local DoF, component) with the component varying fastest.
              const unsigned int n_slots =
                static_cast<unsigned int>(lifted.source_dof_indices.size());
              AssertThrow(n_slots % n_modes == 0,
                          dealii::ExcMessage(
                            "A modal tensor-product lift requires the source "
                            "representation to expose one algebraic slot per "
                            "(local DoF, mode) in deal.II FESystem order (mode "
                            "varying fastest)."));
              const unsigned int n_scalar_dofs = n_slots / n_modes;
              for (unsigned int i = 0; i < n_scalar_dofs; ++i)
                for (unsigned int mode = 0; mode < n_modes; ++mode)
                  {
                    const unsigned int slot = i * n_modes + mode;
                    point.dof_indices.push_back(
                      lifted.source_dof_indices[slot]);
                    point.basis_values.push_back(
                      lifted.source_basis_values[slot] *
                      lifted.mode_values[mode]);
                  }
            }
          else
            {
              // CASE A: one physical coefficient set. A scalar source field
              // cannot silently become n_modes independent unknowns.
              AssertThrow(
                n_modes == 1,
                dealii::ExcMessage(
                  "A physical (non-modal) tensor-product lift supports "
                  "only mode 0; selected modes beyond 0 require a modal "
                  "source representation that owns one coefficient per "
                  "(representative DoF, mode)."));
              for (unsigned int i = 0; i < lifted.source_dof_indices.size();
                   ++i)
                {
                  point.dof_indices.push_back(lifted.source_dof_indices[i]);
                  point.basis_values.push_back(lifted.source_basis_values[i] *
                                               lifted.mode_values[0]);
                }
            }
          result.emplace_back(std::move(point));
        }
      return result;
    }

    value_type
    evaluate(const EvaluationContext<state_type> &context,
             const EvaluationRequest             &request = {}) const
    {
      (void)request;
      return evaluate_stencils(context.state(source_.source()));
    }

    Operator
    linearize(const EvaluationContext<state_type> &context,
              const EvaluationRequest             &request = {}) const
    {
      const auto  lifted_points   = lifted_points_;
      const auto  source_owned    = source_.locally_owned_dofs();
      const auto  source_relevant = source_.locally_relevant_dofs();
      const auto  communicator    = source_.mpi_communicator();
      const auto *reference       = &context.state(source_.source());

      Operator result;
      result.reinit_range_vector =
        [n = lifted_points.size()](value_type &vector, bool omit) {
          vector.reinit(n);
          if (!omit)
            vector = 0.;
        };
      result.reinit_domain_vector = [reference](state_type &vector,
                                                const bool  omit) {
        vector.reinit(*reference, omit);
      };
      result.vmult = [lifted_points,
                      source_owned,
                      source_relevant,
                      communicator](value_type       &destination,
                                    const state_type &source) {
        destination = evaluate_stencils(
          source, lifted_points, source_owned, source_relevant, communicator);
      };
      result.vmult_add = [lifted_points,
                          source_owned,
                          source_relevant,
                          communicator](value_type       &destination,
                                        const state_type &source) {
        const auto values = evaluate_stencils(
          source, lifted_points, source_owned, source_relevant, communicator);
        for (unsigned int q = 0; q < values.size(); ++q)
          destination[q] += values[q];
      };
      result.Tvmult = [lifted_points,
                       source_owned,
                       source_relevant,
                       communicator](state_type       &destination,
                                     const value_type &source) {
        apply_stencil_transpose(lifted_points,
                                source_owned,
                                source_relevant,
                                communicator,
                                source,
                                destination,
                                false);
      };
      result.Tvmult_add = [lifted_points,
                           source_owned,
                           source_relevant,
                           communicator](state_type       &destination,
                                         const value_type &source) {
        apply_stencil_transpose(lifted_points,
                                source_owned,
                                source_relevant,
                                communicator,
                                source,
                                destination,
                                true);
      };
      (void)request;
      return result;
    }

  private:
    static value_type
    evaluate_stencils(
      const state_type                                                 &source,
      const std::vector<TensorProductLiftPoint<surface_dim, spacedim>> &points,
      const dealii::IndexSet                                           &owned,
      const dealii::IndexSet &relevant,
      const MPI_Comm          communicator)
    {
      state_type relevant_source;
      relevant_source.reinit(owned, relevant, communicator);
      relevant_source = source;
      relevant_source.update_ghost_values();

      value_type result;
      result.reinit(points.size());
      for (unsigned int q = 0; q < points.size(); ++q)
        result[q] = detail::evaluate_stencil(relevant_source,
                                             points[q].source_dof_indices,
                                             points[q].source_basis_values);
      return result;
    }

    static void
    apply_stencil_transpose(
      const std::vector<TensorProductLiftPoint<surface_dim, spacedim>> &points,
      const dealii::IndexSet                                           &owned,
      const dealii::IndexSet &relevant,
      const MPI_Comm          communicator,
      const value_type       &source,
      state_type             &destination,
      const bool              add)
    {
      dealii::LinearAlgebra::distributed::Vector<double> contribution;
      contribution.reinit(owned, relevant, communicator);
      contribution = 0.;
      for (unsigned int q = 0; q < points.size(); ++q)
        for (unsigned int i = 0; i < points[q].source_dof_indices.size(); ++i)
          contribution[points[q].source_dof_indices[i]] +=
            points[q].source_basis_values[i] * source[q];
      contribution.compress(dealii::VectorOperation::add);
      if (add)
        for (const auto index : destination.locally_owned_elements())
          destination[index] += contribution[index];
      else
        for (const auto index : destination.locally_owned_elements())
          destination[index] = contribution[index];
    }

    /**
     * Forward the source's own thickness evaluation to the lifting support.
     *
     * The modern path never reads VTK files and never stores imported fields:
     * a source Problem that naturally owns properties provides the symbolic
     * thickness through this duck-typed seam.
     */
    void
    install_source_thickness_provider()
    {
      if constexpr (detail::has_source_thickness<SourceRepresentation,
                                                 spacedim>::value)
        support_.set_thickness_evaluator(
          [this](const dealii::Point<spacedim> &point,
                 const double                   time,
                 const std::vector<double>     &properties) {
            return source_.evaluate_thickness(point, time, properties);
          });
    }

    std::vector<TensorProductLiftPoint<surface_dim, spacedim>>
    build_lifted_points() const
    {
      const auto source_points = source_.locally_owned_quadrature_points(
        support_.representative_quadrature());
      std::vector<TensorProductLiftPoint<surface_dim, spacedim>> result;
      result.reserve(source_points.size() *
                     support_.reference_cross_section().n_quadrature_points());
      for (unsigned int q = 0; q < source_points.size(); ++q)
        {
          const auto transformed =
            support_.transform(source_points[q].point,
                               source_points[q].tangent,
                               source_points[q].weight,
                               support_.thickness(source_points[q].point, 0.),
                               q,
                               {});
          for (unsigned int section_q = 0; section_q < transformed.size();
               ++section_q)
            {
              auto point             = transformed[section_q];
              point.source_entity_id = source_points[q].source_entity_id;
              point.source_representative_qpoint =
                source_points[q].representative_qpoint ==
                    dealii::numbers::invalid_unsigned_int ?
                  q :
                  source_points[q].representative_qpoint;
              point.section_qpoint = section_q;
              point.source_dof_indices.assign(
                source_points[q].dof_indices.begin(),
                source_points[q].dof_indices.end());
              point.source_basis_values.assign(
                source_points[q].basis_values.begin(),
                source_points[q].basis_values.end());

              const auto n_representative_qpoints =
                support_.representative_quadrature().size();
              const auto n_section_qpoints =
                support_.reference_cross_section().n_quadrature_points();
              if (point.source_entity_id !=
                  dealii::numbers::invalid_unsigned_int)
                point.stable_id =
                  static_cast<std::uint64_t>(point.source_entity_id) *
                    n_representative_qpoints * n_section_qpoints +
                  point.source_representative_qpoint * n_section_qpoints +
                  section_q;
              else if (source_points[q].stable_id !=
                       std::numeric_limits<std::uint64_t>::max())
                point.stable_id =
                  source_points[q].stable_id * n_section_qpoints + section_q;
              result.emplace_back(std::move(point));
            }
        }
      return result;
    }

    SourceRepresentation                                       source_;
    const Lift                                                &lift_;
    Support                                                    support_;
    const bool                                                 modal_;
    std::vector<TensorProductLiftPoint<surface_dim, spacedim>> lifted_points_;
  };

  /**
   * Build a tensor-product lifting from a representative view and reduced
   * geometry. This is the representation-first spelling; the direct
   * TensorProductSpace constructor remains available for existing applications.
   */
  template <int reduced_dim,
            int surface_dim,
            int spacedim,
            int n_components,
            typename Representative>
  auto
  make_tensor_product_representation(
    const Representative &representative,
    const TensorProductSpace<reduced_dim, surface_dim, spacedim, n_components>
      &space) -> TensorProductRepresentation<reduced_dim,
                                             surface_dim,
                                             spacedim,
                                             n_components>
  {
    return TensorProductRepresentation<reduced_dim,
                                       surface_dim,
                                       spacedim,
                                       n_components>(space, representative);
  }


  /**
   * Prescribed coefficients on a representation.
   *
   * A prescribed datum is deliberately not a Problem: it owns only the
   * representative coefficients of a physical field and can therefore be
   * attached to any number of interactions exposing the same representation.
   * The interaction converts these coefficients to its multiplier-dual right
   * hand side through its physical pairing matrix.
   */
  template <typename Representation>
  class PrescribedFieldDatum
  {
  public:
    using VectorType = ImmersXLA::MPI::Vector;

    PrescribedFieldDatum(const Representation &representation,
                         const VectorType     &coefficients)
      : representation_(representation)
      , coefficients_(coefficients)
    {}

    const Representation &
    representation() const
    {
      return representation_;
    }

    const VectorType &
    coefficients() const
    {
      return coefficients_;
    }

  private:
    const Representation &representation_;
    const VectorType     &coefficients_;
  };

} // namespace ImmersX

#endif // immersx_representation_h
