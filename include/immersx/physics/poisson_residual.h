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
#include <immersx/core/fe_space.h>
#include <immersx/physics/poisson.h>

#include <memory>

namespace ImmersX
{
  template <int dim, int spacedim = dim>
  struct PoissonFields
  {
    using Space = FiniteElementSpaceView<dim, spacedim>;
    using ScalarField =
      Field<dim, spacedim, dealii::FEValuesExtractors::Scalar>;

    ScalarField                  solution;
    std::shared_ptr<const Space> space;
  };

  /** Register an assembled Poisson problem directly with an execution adapter.
   */
  template <typename Builder, int dim, int spacedim = dim>
  PoissonFields<dim, spacedim>
  contribute(Builder &builder, const PoissonSolver<dim, spacedim> &problem)
  {
    using VectorType = typename PoissonSolver<dim, spacedim>::VectorType;
    const auto solution_id =
      builder.algebraic_field("solution",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    using Space         = typename PoissonFields<dim, spacedim>::Space;
    auto       space    = std::make_shared<Space>(problem.dof_handler(),
                                         problem.mapping(),
                                         problem.constraints(),
                                         &problem.locally_relevant_dofs());
    const auto solution = space->field(solution_id,
                                       "solution",
                                       dealii::FEValuesExtractors::Scalar(0));
    const auto matrix   = builder.matrix_operator(problem.system_matrix());
    builder.preconditioner(solution_id,
                           [](const auto &linearized_matrix,
                              const auto &reinit_vector) {
                             return make_amg_preconditioner(linearized_matrix,
                                                            reinit_vector);
                           });

    builder.term(solution_id, "poisson")
      .residual([solution_id, &problem](const auto &context) {
        const auto &state = context.state(solution_id);
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
      .state(solution_id, matrix);

    return {solution, std::move(space)};
  }

} // namespace ImmersX

#endif // immersx_poisson_residual_h
