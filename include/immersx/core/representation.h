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
#include <immersx/core/state.h>
#include <immersx/coupling/tensor_product_space.h>

#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImmersX
{
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
    using value_type = VectorType;
    using Operator   = dealii::LinearOperator<VectorType, VectorType>;

    explicit Representation(const FieldId source)
      : source_(source)
    {}

    FieldId
    source() const
    {
      return source_;
    }

    const VectorType &
    evaluate(const EvaluationContext<VectorType> &context) const
    {
      return context.state(source_);
    }

    Operator
    linearize(const EvaluationContext<VectorType> &context) const
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
    FieldId source_;
  };

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

    dealii::Point<spacedim>                      point;
    double                                       weight = 0.;
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
      const ExtractorType                   &extractor = ExtractorType(0),
      const ImmersX::RepresentationMetadata &metadata  = {})
      : triangulation_(triangulation)
      , dof_handler_(dof_handler)
      , locally_owned_dofs_(locally_owned_dofs)
      , locally_relevant_dofs_(locally_relevant_dofs)
      , constraints_(constraints)
      , mapping_(mapping)
      , extractor_(extractor)
      , metadata_(metadata)
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
      const auto update_flags = dealii::update_values |
                                dealii::update_quadrature_points |
                                dealii::update_JxW_values;
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
                point.point       = fe_values.quadrature_point(q);
                point.weight      = fe_values.JxW(q);
                point.dof_indices = dof_indices;
                point.basis_values.resize(n_dofs_per_cell());
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
                  point.point       = lifted_points[point_index];
                  point.weight      = lifted_weights[point_index][0];
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
