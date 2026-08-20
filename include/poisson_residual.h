// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception (the "License"); either version 3.0 of the
// License, or (at your option) any later version. The full text of the
// license can be found in the LICENSE.md file at the top level of the
// ImmersX distribution.
//
// ---------------------------------------------------------------------

#ifndef immersx_poisson_residual_h
#define immersx_poisson_residual_h

#include "poisson.h"
#include "residual.h"

namespace ImmersX
{
  /**
   * Adapt an assembled Poisson operator to the semantic residual interface.
   *
   * The state is read from the evaluation context, so this adapter does not
   * use PoissonSolver::solution() as the argument of the equation. Existing
   * assembled matrix and right-hand-side APIs remain unchanged. The supplied
   * PoissonSolver must outlive this contributor.
   */
  template <int dim, int spacedim = dim>
  class PoissonResidualContributor
  {
  public:
    using VectorType = typename PoissonSolver<dim, spacedim>::VectorType;

    PoissonResidualContributor(const PoissonSolver<dim, spacedim> &problem,
                               const FieldId                       field_id)
      : problem_(problem)
      , field_id_(field_id)
    {}

    void
    add_residual(const EvaluationContext<VectorType> &context,
                 ResidualAccumulator<VectorType>     &residual) const
    {
      VectorType contribution;
      contribution.reinit(residual.field(field_id_));
      problem_.system_matrix().vmult(contribution,
                                     context.state().field(field_id_,
                                                           context.time()));
      contribution -= problem_.system_rhs();
      residual.add(field_id_, contribution);
    }

    FieldId
    field_id() const
    {
      return field_id_;
    }

  private:
    const PoissonSolver<dim, spacedim> &problem_;
    FieldId                             field_id_;
  };
} // namespace ImmersX

#endif // immersx_poisson_residual_h
