// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_coupling_detail_fe_stencil_h
#define immersx_coupling_detail_fe_stencil_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector_operation.h>

#include <vector>

namespace ImmersX::detail
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

  /** Evaluate a retained stencil with coefficients supplied by its point. */
  template <typename VectorType,
            typename PointType,
            typename DofIndices,
            typename Coefficient>
  double
  evaluate_stencil(const VectorType &source,
                   const PointType  &point,
                   DofIndices        dof_indices,
                   Coefficient       coefficient)
  {
    const auto &indices = dof_indices(point);
    double      result  = 0.;
    for (unsigned int i = 0; i < indices.size(); ++i)
      result += coefficient(point, i) * source[indices[i]];
    return result;
  }

  /** Build the relevant source vector for a retained linear map. */
  template <typename StateVectorType>
  StateVectorType
  make_relevant_stencil_source(
    const StateVectorType                   &source,
    const dealii::IndexSet                  &source_owned,
    const dealii::IndexSet                  &source_relevant,
    const dealii::AffineConstraints<double> *constraints,
    const MPI_Comm                           communicator,
    const bool                               use_inhomogeneities)
  {
    StateVectorType owned;
    owned.reinit(source_owned, communicator);
    owned = source;
    if (constraints != nullptr)
      {
        if (use_inhomogeneities || !constraints->has_inhomogeneities())
          constraints->distribute(owned);
        else
          {
            // The derivative of an affine constraint is its homogeneous
            // linear part. Reuse deal.II's distribution logic after removing
            // only the affine offsets.
            dealii::AffineConstraints<double> homogeneous_constraints(
              *constraints);
            for (const auto &line : homogeneous_constraints.get_lines())
              homogeneous_constraints.set_inhomogeneity(line.index, 0.);
            homogeneous_constraints.distribute(owned);
          }
      }

    StateVectorType relevant;
    relevant.reinit(source_owned, source_relevant, communicator);
    relevant = owned;
    relevant.update_ghost_values();
    return relevant;
  }

  /** Apply retained FE stencils in the forward direction. */
  template <typename PointType,
            typename SourceVectorType,
            typename TargetVectorType,
            typename TargetIndexType,
            typename DofIndices,
            typename Coefficient>
  void
  apply_stencils(const std::vector<PointType>            &points,
                 const std::vector<TargetIndexType>      &target_indices,
                 const dealii::IndexSet                  &source_owned,
                 const dealii::IndexSet                  &source_relevant,
                 const dealii::AffineConstraints<double> *constraints,
                 const MPI_Comm                           communicator,
                 const SourceVectorType                  &source,
                 TargetVectorType                        &destination,
                 DofIndices                               dof_indices,
                 Coefficient                              coefficient,
                 const bool                               use_inhomogeneities,
                 const bool                               add)
  {
    AssertDimension(points.size(), target_indices.size());
    const auto relevant = make_relevant_stencil_source(source,
                                                       source_owned,
                                                       source_relevant,
                                                       constraints,
                                                       communicator,
                                                       use_inhomogeneities);

    if (!add)
      destination = 0.;
    for (std::size_t q = 0; q < points.size(); ++q)
      destination[target_indices[q]] +=
        evaluate_stencil(relevant, points[q], dof_indices, coefficient);
  }

  /** Apply stencils to an owned destination and publish its ghosted view. */
  template <typename PointType,
            typename SourceVectorType,
            typename TargetVectorType,
            typename TargetIndexType,
            typename DofIndices,
            typename Coefficient>
  void
  apply_stencils(const std::vector<PointType>            &points,
                 const std::vector<TargetIndexType>      &target_indices,
                 const dealii::IndexSet                  &source_owned,
                 const dealii::IndexSet                  &source_relevant,
                 const dealii::IndexSet                  &target_owned,
                 const dealii::IndexSet                  &target_relevant,
                 const dealii::AffineConstraints<double> *constraints,
                 const MPI_Comm                           communicator,
                 const SourceVectorType                  &source,
                 TargetVectorType                        &destination,
                 DofIndices                               dof_indices,
                 Coefficient                              coefficient,
                 const bool                               use_inhomogeneities,
                 const bool                               add)
  {
    AssertDimension(points.size(), target_indices.size());
    const auto relevant = make_relevant_stencil_source(source,
                                                       source_owned,
                                                       source_relevant,
                                                       constraints,
                                                       communicator,
                                                       use_inhomogeneities);

    TargetVectorType owned_destination;
    owned_destination.reinit(target_owned, communicator);
    if (add)
      owned_destination = destination;
    else
      owned_destination = 0.;
    for (std::size_t q = 0; q < points.size(); ++q)
      if (target_owned.is_element(target_indices[q]))
        owned_destination[target_indices[q]] +=
          evaluate_stencil(relevant, points[q], dof_indices, coefficient);

    destination.reinit(target_owned, target_relevant, communicator);
    destination = owned_destination;
    destination.update_ghost_values();
  }

  /** Apply the algebraic transpose of retained FE stencils. */
  template <typename PointType,
            typename SourceVectorType,
            typename StateVectorType,
            typename SourceIndexType,
            typename DofIndices,
            typename Coefficient>
  void
  apply_stencils_transpose(const std::vector<PointType>       &points,
                           const std::vector<SourceIndexType> &source_indices,
                           const dealii::IndexSet             &source_owned,
                           const dealii::AffineConstraints<double> *constraints,
                           const MPI_Comm          communicator,
                           const SourceVectorType &source,
                           StateVectorType        &destination,
                           DofIndices              dof_indices,
                           Coefficient             coefficient,
                           const bool              add)
  {
    AssertDimension(points.size(), source_indices.size());
    StateVectorType contribution;
    contribution.reinit(source_owned, communicator);
    contribution = 0.;
    std::vector<dealii::types::global_dof_index> contribution_indices;
    std::vector<double>                          contribution_values;
    for (std::size_t q = 0; q < points.size(); ++q)
      {
        const auto &indices = dof_indices(points[q]);
        for (unsigned int i = 0; i < indices.size(); ++i)
          {
            contribution_indices.push_back(indices[i]);
            contribution_values.push_back(coefficient(points[q], i) *
                                          source[source_indices[q]]);
          }
      }
    if (!contribution_indices.empty())
      contribution.add(contribution_indices, contribution_values);
    contribution.compress(dealii::VectorOperation::add);

    StateVectorType correction;
    if (constraints != nullptr)
      {
        correction.reinit(source_owned, communicator);
        correction = 0.;
        std::vector<dealii::types::global_dof_index> correction_indices;
        std::vector<double>                          correction_values;
        for (const auto &line : constraints->get_lines())
          if (source_owned.is_element(line.index))
            {
              const double constrained = contribution[line.index];
              correction_indices.push_back(line.index);
              correction_values.push_back(-constrained);
              for (const auto &[master, constraint_coefficient] : line.entries)
                {
                  correction_indices.push_back(master);
                  correction_values.push_back(constraint_coefficient *
                                              constrained);
                }
            }
        if (!correction_indices.empty())
          correction.add(correction_indices, correction_values);
        correction.compress(dealii::VectorOperation::add);
      }

    StateVectorType owned_destination;
    owned_destination.reinit(source_owned, communicator);
    if (add)
      owned_destination = destination;
    else
      owned_destination = 0.;
    for (const auto index : source_owned)
      owned_destination[index] +=
        contribution[index] + (constraints != nullptr ? correction[index] : 0.);

    destination = owned_destination;
    destination.update_ghost_values();
  }
} // namespace ImmersX::detail

#endif // immersx_coupling_detail_fe_stencil_h
