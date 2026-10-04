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

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

namespace ImmersX
{
  namespace detail
  {
    template <typename FunctionType, typename = void>
    struct has_set_time : std::false_type
    {};

    template <typename FunctionType>
    struct has_set_time<
      FunctionType,
      std::void_t<decltype(std::declval<FunctionType &>().set_time(0.))>>
      : std::true_type
    {};

    template <typename FunctionType>
    void
    set_function_time(const FunctionType &function, const double time)
    {
      using MutableFunction = std::remove_const_t<FunctionType>;
      if constexpr (has_set_time<MutableFunction>::value)
        const_cast<MutableFunction &>(function).set_time(time);
      else
        (void)time;
    }
  } // namespace detail

  /** Persistent, aggregable recipes for an FE space's affine constraints. */
  template <int dim, int spacedim = dim>
  class BoundaryConditions
  {
  public:
    using Space            = FiniteElementSpace<dim, spacedim>;
    using Constraints      = dealii::AffineConstraints<double>;
    using BoundaryId       = dealii::types::boundary_id;
    using Function         = dealii::Function<spacedim>;
    using ConstraintRecipe = std::function<void(double, Constraints &)>;

    explicit BoundaryConditions(Space &space)
      : space_(&space)
    {}

    /** Add one strong Dirichlet rule, optionally restricted by component. */
    template <typename FunctionType>
    void
    add_dirichlet(const BoundaryId                            boundary_id,
                  const FunctionType                         &function,
                  const std::optional<dealii::ComponentMask> &mask = {})
    {
      static_assert(std::is_base_of_v<Function, FunctionType>,
                    "A Dirichlet rule must use a deal.II Function.");
      dirichlet_rules_.emplace_back(
        [this, boundary_id, &function, mask](const double time,
                                             Constraints &constraints) {
          detail::set_function_time(function, time);
          if (mask.has_value())
            dealii::VectorTools::interpolate_boundary_values(
              space_->dof_handler(), boundary_id, function, constraints, *mask);
          else
            dealii::VectorTools::interpolate_boundary_values(
              space_->dof_handler(), boundary_id, function, constraints);
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
    add_nonzero_normal_flux(const BoundaryId    boundary_id,
                            const FunctionType &function,
                            const unsigned int  first_vector_component = 0)
    {
      static_assert(std::is_base_of_v<Function, FunctionType>,
                    "A normal-flux rule must use a deal.II Function.");
      normal_flux_rules_.emplace_back(
        [this, boundary_id, &function, first_vector_component](
          const double time, Constraints &constraints) {
          detail::set_function_time(function, time);
          const std::set<BoundaryId>                   ids{boundary_id};
          const std::map<BoundaryId, const Function *> functions{
            {boundary_id, &function}};
          dealii::VectorTools::compute_nonzero_normal_flux_constraints(
            space_->dof_handler(),
            first_vector_component,
            ids,
            functions,
            constraints);
        });
    }

    template <typename FunctionType>
    void
    add_nonzero_normal_flux(const std::set<BoundaryId> &boundary_ids,
                            const FunctionType         &function,
                            const unsigned int first_vector_component = 0)
    {
      for (const auto boundary_id : boundary_ids)
        add_nonzero_normal_flux(boundary_id, function, first_vector_component);
    }

    /** Remove recipes; the current constraints remain until update(). */
    void
    clear()
    {
      dirichlet_rules_.clear();
      normal_flux_rules_.clear();
    }

    /** Rebuild the same AffineConstraints object at the supplied time. */
    void
    update(const double time) const
    {
      const auto previous_structure = structure_signature();
      auto      &constraints        = space_->constraints();
      constraints.clear();
      constraints.reinit(space_->locally_owned_dofs(),
                         space_->locally_relevant_dofs());
      if (include_hanging_nodes_)
        dealii::DoFTools::make_hanging_node_constraints(space_->dof_handler(),
                                                        constraints);
      for (const auto &recipe : dirichlet_rules_)
        recipe(time, constraints);
      for (const auto &recipe : normal_flux_rules_)
        recipe(time, constraints);
      constraints.close();
      ++value_revision_;
      if (previous_structure != structure_signature())
        ++structure_revision_;
    }

    /** Register update(time) as a once-per-evaluation context hook. */
    template <typename Builder>
    void
    register_with(Builder &builder) const
    {
      builder.context_update(
        [this](const auto &context) { update(context.time()); });
    }

    void
    include_hanging_node_constraints(const bool include = true)
    {
      include_hanging_nodes_ = include;
    }

    bool
    includes_hanging_node_constraints() const
    {
      return include_hanging_nodes_;
    }

    std::size_t
    n_dirichlet_rules() const
    {
      return dirichlet_rules_.size();
    }

    std::size_t
    n_normal_flux_rules() const
    {
      return normal_flux_rules_.size();
    }

    std::size_t
    value_revision() const
    {
      return value_revision_;
    }

    std::size_t
    structure_revision() const
    {
      return structure_revision_;
    }

    Space &
    space() const
    {
      return *space_;
    }

  private:
    std::vector<std::pair<dealii::types::global_dof_index, std::size_t>>
    structure_signature() const
    {
      std::vector<std::pair<dealii::types::global_dof_index, std::size_t>>
        result;
      for (const auto &line : space_->constraints().get_lines())
        result.emplace_back(line.index, line.entries.size());
      return result;
    }

    Space                        *space_;
    std::vector<ConstraintRecipe> dirichlet_rules_;
    std::vector<ConstraintRecipe> normal_flux_rules_;
    bool                          include_hanging_nodes_ = true;
    mutable std::size_t           value_revision_        = 0;
    mutable std::size_t           structure_revision_    = 0;
  };

} // namespace ImmersX

#endif // immersx_boundary_conditions_h
