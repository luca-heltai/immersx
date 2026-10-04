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

#include <immersx/core/detail/function_time.h>
#include <immersx/core/fe_space.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

namespace ImmersX
{
  /** Persistent, aggregable recipes for an FE space's affine constraints. */
  template <int dim, int spacedim = dim>
  class BoundaryConditions
  {
  public:
    using Space       = FiniteElementSpace<dim, spacedim>;
    using DoFHandler  = dealii::DoFHandler<dim, spacedim>;
    using Constraints = dealii::AffineConstraints<double>;
    using BoundaryId  = dealii::types::boundary_id;
    using Function    = dealii::Function<spacedim>;

    struct Target
    {
      std::function<const DoFHandler &()>       dof_handler;
      std::function<const dealii::IndexSet &()> locally_owned_dofs;
      std::function<const dealii::IndexSet &()> locally_relevant_dofs;
      std::function<Constraints &()>            constraints;
      std::optional<dealii::ComponentMask>      component_mask;
      std::optional<unsigned int>               first_vector_component;
      Space                                    *space = nullptr;
    };

    using ConstraintRecipe =
      std::function<void(double, const Target &, Constraints &)>;

    explicit BoundaryConditions(Space &space)
      : state_(std::make_shared<State>())
    {
      state_->target = make_target(space);
    }

    template <typename Extractor>
    explicit BoundaryConditions(const Field<dim, spacedim, Extractor> &field)
      : state_(std::make_shared<State>())
    {
      field.ensure_constraints();
      state_->target = make_target(field);
    }

    /** Add one strong Dirichlet rule, optionally restricted by component. */
    template <typename FunctionType>
    void
    add_dirichlet(const BoundaryId                            boundary_id,
                  const FunctionType                         &function,
                  const std::optional<dealii::ComponentMask> &mask = {})
    {
      static_assert(std::is_base_of_v<Function, FunctionType>,
                    "A Dirichlet rule must use a deal.II Function.");
      validate_component_mask(mask);
      state_->dirichlet_rules.emplace_back([boundary_id,
                                            &function,
                                            mask](const double  time,
                                                  const Target &target,
                                                  Constraints  &constraints) {
        detail::set_function_time(function, time);
        const auto effective_mask =
          mask.has_value() ? mask : target.component_mask;
        if (effective_mask.has_value())
          dealii::VectorTools::interpolate_boundary_values(target.dof_handler(),
                                                           boundary_id,
                                                           function,
                                                           constraints,
                                                           *effective_mask);
        else
          dealii::VectorTools::interpolate_boundary_values(target.dof_handler(),
                                                           boundary_id,
                                                           function,
                                                           constraints);
      });
    }

    template <typename FunctionType>
    void
    add_dirichlet(const std::set<BoundaryId>                 &boundary_ids,
                  const FunctionType                         &function,
                  const std::optional<dealii::ComponentMask> &mask = {})
    {
      for (const auto boundary_id : boundary_ids)
        add_dirichlet(boundary_id, function, mask);
    }

    /** Add one possibly time-dependent nonzero normal-flux rule. */
    template <typename FunctionType>
    void
    add_nonzero_normal_flux(
      const BoundaryId                  boundary_id,
      const FunctionType               &function,
      const std::optional<unsigned int> first_vector_component = {})
    {
      static_assert(std::is_base_of_v<Function, FunctionType>,
                    "A normal-flux rule must use a deal.II Function.");
      state_->normal_flux_rules.emplace_back(
        [boundary_id, &function, first_vector_component](
          const double time, const Target &target, Constraints &constraints) {
          detail::set_function_time(function, time);
          const auto component = first_vector_component.value_or(
            target.first_vector_component.value_or(0u));
          const std::set<BoundaryId>                   ids{boundary_id};
          const std::map<BoundaryId, const Function *> functions{
            {boundary_id, &function}};
          dealii::VectorTools::compute_nonzero_normal_flux_constraints(
            target.dof_handler(), component, ids, functions, constraints);
        });
    }

    template <typename FunctionType>
    void
    add_nonzero_normal_flux(
      const std::set<BoundaryId>       &boundary_ids,
      const FunctionType               &function,
      const std::optional<unsigned int> first_vector_component = {})
    {
      for (const auto boundary_id : boundary_ids)
        add_nonzero_normal_flux(boundary_id, function, first_vector_component);
    }

    /** Remove recipes; the current constraints remain until update(). */
    void
    clear()
    {
      state_->dirichlet_rules.clear();
      state_->normal_flux_rules.clear();
    }

    /** Rebuild the same AffineConstraints object at the supplied time. */
    void
    update(const double time) const
    {
      update_state(*state_, time);
    }

    /** Register update(time) as a once-per-evaluation context hook. */
    template <typename Builder>
    void
    register_with(Builder &builder) const
    {
      const auto state = state_;
      builder.context_update(
        [state](const auto &context) { update_state(*state, context.time()); });
    }

    void
    include_hanging_node_constraints(const bool include = true)
    {
      state_->include_hanging_nodes = include;
    }

    bool
    includes_hanging_node_constraints() const
    {
      return state_->include_hanging_nodes;
    }

    std::size_t
    n_dirichlet_rules() const
    {
      return state_->dirichlet_rules.size();
    }

    std::size_t
    n_normal_flux_rules() const
    {
      return state_->normal_flux_rules.size();
    }

    std::size_t
    value_revision() const
    {
      return state_->value_revision;
    }

    std::size_t
    structure_revision() const
    {
      return state_->structure_revision;
    }

    Space &
    space() const
    {
      AssertThrow(state_->target.space != nullptr,
                  dealii::ExcMessage(
                    "A field-based BoundaryConditions has no owning space."));
      return *state_->target.space;
    }

  private:
    struct State
    {
      Target                        target;
      std::vector<ConstraintRecipe> dirichlet_rules;
      std::vector<ConstraintRecipe> normal_flux_rules;
      bool                          include_hanging_nodes = true;
      mutable std::size_t           value_revision        = 0;
      mutable std::size_t           structure_revision    = 0;
    };

    static Target
    make_target(Space &space)
    {
      Target result;
      result.dof_handler = [&space]() -> const DoFHandler & {
        return space.dof_handler();
      };
      result.locally_owned_dofs = [&space]() -> const dealii::IndexSet & {
        return space.locally_owned_dofs();
      };
      result.locally_relevant_dofs = [&space]() -> const dealii::IndexSet & {
        return space.locally_relevant_dofs();
      };
      result.constraints = [&space]() -> Constraints & {
        return space.constraints();
      };
      result.space = &space;
      return result;
    }

    template <typename Extractor>
    static Target
    make_target(const Field<dim, spacedim, Extractor> &input)
    {
      const auto field = input;
      Target     result;
      result.dof_handler = [field]() -> const DoFHandler & {
        return field.dof_handler();
      };
      result.locally_owned_dofs = [field]() -> const dealii::IndexSet & {
        return field.locally_owned_dofs();
      };
      result.locally_relevant_dofs = [field]() -> const dealii::IndexSet & {
        return field.locally_relevant_dofs();
      };
      result.constraints = [field]() -> Constraints & {
        return field.mutable_constraints();
      };
      result.component_mask = field.component_mask();
      if constexpr (std::is_same_v<Extractor,
                                   dealii::FEValuesExtractors::Vector>)
        result.first_vector_component =
          field.extractor().first_vector_component;
      return result;
    }

    void
    validate_component_mask(
      const std::optional<dealii::ComponentMask> &mask) const
    {
      if (!mask.has_value() || !state_->target.component_mask.has_value())
        return;
      const auto &field_mask = *state_->target.component_mask;
      AssertDimension(mask->size(), field_mask.size());
      for (unsigned int component = 0; component < mask->size(); ++component)
        AssertThrow(!(*mask)[component] || field_mask[component],
                    dealii::ExcMessage(
                      "A Dirichlet sub-mask must be contained in the field "
                      "component mask."));
    }

    static std::vector<std::pair<dealii::types::global_dof_index, std::size_t>>
    structure_signature(const Constraints &constraints)
    {
      std::vector<std::pair<dealii::types::global_dof_index, std::size_t>>
        result;
      for (const auto &line : constraints.get_lines())
        result.emplace_back(line.index, line.entries.size());
      return result;
    }

    static void
    update_state(State &state, const double time)
    {
      const auto previous_structure =
        structure_signature(state.target.constraints());
      auto &constraints = state.target.constraints();
      constraints.clear();
      constraints.reinit(state.target.locally_owned_dofs(),
                         state.target.locally_relevant_dofs());
      if (state.include_hanging_nodes)
        dealii::DoFTools::make_hanging_node_constraints(
          state.target.dof_handler(), constraints);
      for (const auto &recipe : state.dirichlet_rules)
        recipe(time, state.target, constraints);
      for (const auto &recipe : state.normal_flux_rules)
        recipe(time, state.target, constraints);
      constraints.close();
      ++state.value_revision;
      if (previous_structure != structure_signature(constraints))
        ++state.structure_revision;
    }

    std::shared_ptr<State> state_;
  };

} // namespace ImmersX

#endif // immersx_boundary_conditions_h
