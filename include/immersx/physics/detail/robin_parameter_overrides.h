// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_robin_parameter_overrides_h
#define immersx_robin_parameter_overrides_h

#include <deal.II/base/scope_exit.h>

#include <immersx/core/robin_boundary.h>
#include <immersx/physics/modulated_parsed_function.h>

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace ImmersX::robin_detail
{
  /** Register boundary-id overrides with entry-wise fallback on every parse.
   * The coefficient has no modulation parameters; its expression can depend
   * on time except in a single-frequency harmonic problem.
   */
  template <int spacedim, typename Ids>
  bool
  register_parameter_overrides(
    dealii::ParameterAcceptor                                 &owner,
    const Ids                                                 &ids,
    const std::string                                         &function_prefix,
    const ModulatedParsedFunction<spacedim>                   &base_load,
    std::map<dealii::types::boundary_id,
             std::shared_ptr<dealii::ParameterAcceptorProxy<
               dealii::Functions::ParsedFunction<spacedim>>>> &coefficients,
    std::map<dealii::types::boundary_id,
             std::shared_ptr<ModulatedParsedFunction<spacedim>>> &loads,
    std::map<dealii::types::boundary_id, std::array<bool, 3>> &explicit_entries,
    const bool time_independent_coefficient = false)
  {
    auto &prm = dealii::ParameterAcceptor::prm;
    owner.leave_my_subsection(prm);
    const dealii::ScopeExit restore_owner(
      [&]() { owner.enter_my_subsection(prm); });
    const auto enter = [&prm](const std::string &section) {
      for (const auto &part :
           dealii::Utilities::split_string_list(section, '/'))
        if (!part.empty())
          prm.enter_subsection(part);
    };
    const auto leave = [&prm](const std::string &section) {
      for (const auto &part :
           dealii::Utilities::split_string_list(section, '/'))
        if (!part.empty())
          prm.leave_subsection();
    };
    const auto copy_entry = [&](const std::string &from,
                                const std::string &to,
                                const std::string &entry) {
      enter(from);
      const auto value = prm.get(entry);
      leave(from);
      enter(to);
      prm.set(entry, value);
      leave(to);
    };
    const std::string coefficient_base = function_prefix + "Robin coefficient";
    const std::string load_base = function_prefix + "Robin boundary conditions";
    const std::array<std::string, 3> entries{
      {"Function constants", "Function expression", "Variable names"}};
    bool                                       created = false;
    const std::set<dealii::types::boundary_id> unique_ids(ids.begin(),
                                                          ids.end());
    for (const auto id : unique_ids)
      {
        const auto suffix              = " " + std::to_string(id);
        const auto coefficient_section = coefficient_base + suffix;
        if (coefficients.count(id) == 0)
          {
            auto function = std::make_shared<dealii::ParameterAcceptorProxy<
              dealii::Functions::ParsedFunction<spacedim>>>(coefficient_section,
                                                            1);
            auto declare_actions =
              [handler = &prm, flags = &explicit_entries, id, entries]() {
                for (unsigned int i = 0; i < entries.size(); ++i)
                  handler->add_action(
                    entries[i],
                    [flags, id, i](const std::string &) {
                      (*flags)[id][i] = true;
                    },
                    false);
              };
            function->declare_parameters_call_back.connect(declare_actions);
            function->enter_my_subsection(prm);
            function->declare_parameters(prm);
            declare_actions();
            function->leave_my_subsection(prm);
            coefficients[id] = std::move(function);
            created          = true;
          }
        auto &flags = explicit_entries[id];
        for (unsigned int i = 0; i < entries.size(); ++i)
          if (!flags[i])
            copy_entry(coefficient_base, coefficient_section, entries[i]);
        flags.fill(false);
        if (time_independent_coefficient)
          {
            enter(coefficient_section);
            const bool depends_on_time =
              expression_depends_on_time(prm, spacedim);
            leave(coefficient_section);
            AssertThrow(
              !depends_on_time,
              dealii::ExcMessage(
                "Harmonic Robin coefficients must be time independent."));
          }
        if (loads.count(id) == 0)
          {
            const auto                        section = load_base + suffix;
            const dealii::Function<spacedim> &function_view = base_load;
            auto function = std::make_shared<ModulatedParsedFunction<spacedim>>(
              section, function_view.n_components);
            function->set_fallback_configuration_source(&base_load);
            function->enter_my_subsection(prm);
            function->declare_parameters(prm);
            function->leave_my_subsection(prm);
            loads[id] = std::move(function);
            for (const auto &entry : {"Function constants",
                                      "Function expression",
                                      "Variable names",
                                      "Modulation frequency",
                                      "Phase shift"})
              copy_entry(load_base, section, entry);
            created = true;
          }
      }
    return created;
  }
} // namespace ImmersX::robin_detail
#endif
