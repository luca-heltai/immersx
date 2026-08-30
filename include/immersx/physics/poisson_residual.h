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

#include <immersx/algebra/local_preconditioner.h>
#include <immersx/core/contributor.h>
#include <immersx/physics/poisson.h>

namespace ImmersX
{
  template <int dim, int spacedim = dim>
  struct PoissonFields
  {
    FieldId                             solution;
    const PoissonSolver<dim, spacedim> *problem = nullptr;
  };

  /** Register an assembled Poisson problem directly with an execution adapter.
   */
  template <typename Builder, int dim, int spacedim = dim>
  PoissonFields<dim, spacedim>
  contribute(Builder &builder, const PoissonSolver<dim, spacedim> &problem)
  {
    using VectorType = typename PoissonSolver<dim, spacedim>::VectorType;
    const auto solution =
      builder.algebraic_field("solution",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    const auto matrix = builder.matrix_operator(problem.system_matrix());
    builder.preconditioner(
      solution, [](const auto &linearized_matrix, const auto &reinit_vector) {
        return make_amg_preconditioner(linearized_matrix, reinit_vector);
      });

    builder.term(solution, "poisson")
      .residual([solution, &problem](const auto &context) {
        const auto                           &state = context.state(solution);
        dealii::PackagedOperation<VectorType> result;
        result.reinit_vector = [state](VectorType &vector, const bool omit) {
          vector.reinit(state, omit);
        };
        result.apply = [&problem, &state](VectorType &vector) {
          problem.system_matrix().vmult(vector, state);
          vector -= problem.system_rhs();
        };
        result.apply_add = [&problem, &state](VectorType &vector) {
          VectorType contribution;
          contribution.reinit(state);
          problem.system_matrix().vmult(contribution, state);
          contribution -= problem.system_rhs();
          vector += contribution;
        };
        return result;
      })
      .state(solution, matrix);

    return {solution, &problem};
  }

} // namespace ImmersX

#endif // immersx_poisson_residual_h
