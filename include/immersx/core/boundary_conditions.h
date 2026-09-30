// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_boundary_conditions_h
#define immersx_boundary_conditions_h

#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <immersx/core/fe_space.h>

namespace ImmersX
{
  /** Configure one constant scalar Dirichlet boundary on an owning FE space. */
  template <int dim, int spacedim = dim>
  void
  set_constant_dirichlet_boundary_condition(
    FiniteElementSpace<dim, spacedim> &space,
    const dealii::types::boundary_id   boundary_id,
    const double                       value)
  {
    auto &constraints = space.constraints();
    constraints.reinit(space.locally_owned_dofs(),
                       space.locally_relevant_dofs());
    dealii::DoFTools::make_hanging_node_constraints(space.dof_handler(),
                                                    constraints);
    const dealii::Functions::ConstantFunction<spacedim> boundary_value(value);
    dealii::VectorTools::interpolate_boundary_values(space.dof_handler(),
                                                     boundary_id,
                                                     boundary_value,
                                                     constraints);
    constraints.close();
  }
} // namespace ImmersX

#endif // immersx_boundary_conditions_h
