// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------

/**
 * @brief Header with the interface of the CouplingOperator class.
 */
#ifndef rdlm_mf_utils
#define rdlm_mf_utils


#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <inclusions.h>

#include <fstream>

using namespace dealii;

#if DEAL_II_VERSION_GTE(9, 7, 0)
#else
#  include <deal.II/base/smartpointer.h>
template <typename T, typename P = void>
using ObserverPointer = SmartPointer<T, P>;
#endif



/**
 * Class responsible to provide the action of the coupling operator in a
 * matrix-free fashion.
 */
template <int dim, typename number, int n_components = 1>
class CouplingOperator
{
public:
  /**
   * Distributed vector type used by matrix-free operator application.
   */
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  /**
   * Constructor. Takes an @p Inclusions instance, a reference to the background
   * triangulation and an optional mapping to initialize the evaluator on remote
   * points.
   */
  CouplingOperator(
    const Inclusions<dim>           &inclusions,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints = AffineConstraints<number>(),
    const MappingQ<dim>             &mapping     = MappingQ1<dim>(),
    const FiniteElement<dim>        &fe          = FE_Q<dim>(1));

  /**
   * Initialize a vector with the layout expected by this operator.
   */
  void
  initialize_dof_vector(VectorType &vec) const;

  /**
   * Apply the coupling operator.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const;

  /**
   * Apply the transpose coupling operator.
   */
  void
  Tvmult(VectorType &dst, const VectorType &src) const;

  /**
   * Add the operator action to an existing destination vector.
   */
  void
  vmult_add(VectorType &dst, const VectorType &src) const;

  /**
   * Add the transpose operator action to an existing destination vector.
   */
  void
  Tvmult_add(VectorType &dst, const VectorType &src) const;


private:
  /**
   * Remote point-evaluation backend used to map points to cells.
   */
  Utilities::MPI::RemotePointEvaluation<dim> rpe;
  /**
   * Mapping used to evaluate basis functions at remote points.
   */
  ObserverPointer<const Mapping<dim>> mapping;
  /**
   * Finite element used by the coupled background field.
   */
  ObserverPointer<const FiniteElement<dim>> fe;
  /**
   * DoF handler associated with the background field.
   */
  ObserverPointer<const DoFHandler<dim>> dof_handler;
  /**
   * Inclusion geometry and reduced-basis data.
   */
  ObserverPointer<const Inclusions<dim>> inclusions;
  /**
   * Constraints applied after matrix-free operator evaluation.
   */
  ObserverPointer<const AffineConstraints<number>> constraints;
  /**
   * Number of reduced coefficients per inclusion.
   */
  const unsigned int n_coefficients;
};



#endif
