// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_constraint_h
#define immersx_constraint_h

#include <immersx/core/constraint_contributor.h>
#include <immersx/core/weak_term.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ImmersX
{
  namespace detail
  {
    template <typename Type>
    struct is_weak_term : std::false_type
    {};

    template <typename ObservableType, typename TargetField>
    struct is_weak_term<WeakTerm<ObservableType, TargetField>> : std::true_type
    {};
  } // namespace detail

  /** A signed sum of weak terms defining one constraint row. */
  template <typename Term>
  class ConstraintSum
  {
  public:
    struct Entry
    {
      double coefficient;
      Term   term;
    };

    ConstraintSum() = default;

    explicit ConstraintSum(Term term, const double coefficient = 1.)
      : entries_({{coefficient, std::move(term)}})
    {}

    ConstraintSum(ConstraintSum lhs, const double sign, Term rhs)
      : entries_(std::move(lhs.entries_))
    {
      entries_.push_back({sign, std::move(rhs)});
    }

    const std::vector<Entry> &
    entries() const
    {
      return entries_;
    }

  private:
    std::vector<Entry> entries_;
  };

  template <typename Term,
            std::enable_if_t<detail::is_weak_term<Term>::value, int> = 0>
  ConstraintSum<Term>
  operator+(Term lhs, Term rhs)
  {
    return ConstraintSum<Term>(ConstraintSum<Term>(std::move(lhs)),
                               1.,
                               std::move(rhs));
  }

  template <typename Term,
            std::enable_if_t<detail::is_weak_term<Term>::value, int> = 0>
  ConstraintSum<Term>
  operator-(Term lhs, Term rhs)
  {
    return ConstraintSum<Term>(ConstraintSum<Term>(std::move(lhs)),
                               -1.,
                               std::move(rhs));
  }

  template <typename Term>
  ConstraintSum<Term>
  operator+(ConstraintSum<Term> lhs, Term rhs)
  {
    return ConstraintSum<Term>(std::move(lhs), 1., std::move(rhs));
  }

  template <typename Term>
  ConstraintSum<Term>
  operator-(ConstraintSum<Term> lhs, Term rhs)
  {
    return ConstraintSum<Term>(std::move(lhs), -1., std::move(rhs));
  }

  /** A generic Lagrange-multiplier constraint over weak terms. */
  template <typename Term, typename Rhs = std::monostate>
  class Constraint
  {
  public:
    using term_type = Term;
    using rhs_type  = Rhs;

    explicit Constraint(ConstraintSum<Term> terms, Rhs rhs = Rhs())
      : terms_(std::move(terms))
      , rhs_(std::move(rhs))
    {
      AssertThrow(!terms_.entries().empty(),
                  dealii::ExcMessage(
                    "A constraint must contain at least one weak term."));

      const auto &target = terms_.entries().front().term.target();
      AssertThrow(!target.is_registered(),
                  dealii::ExcMessage(
                    "A constraint multiplier must be an unregistered field."));
      for (const auto &entry : terms_.entries())
        {
          AssertThrow(!entry.term.target().is_registered(),
                      dealii::ExcMessage(
                        "A constraint multiplier must be an unregistered "
                        "field."));
          AssertThrow(&entry.term.target().space() == &target.space() &&
                        entry.term.target().name() == target.name(),
                      dealii::ExcMessage(
                        "All weak terms in a constraint must use the same "
                        "multiplier field."));
        }
    }

    const auto &
    terms() const
    {
      return terms_.entries();
    }

    const auto &
    multiplier() const
    {
      return terms_.entries().front().term.target();
    }

    bool
    has_rhs() const
    {
      return !std::is_same_v<Rhs, std::monostate>;
    }

    template <typename Builder>
    ConstraintFields
    add(Builder &builder) const
    {
      const auto &multiplier_field = multiplier();
      const auto  multiplier_id =
        builder.algebraic_field(multiplier_field.name(),
                                multiplier_field.locally_owned_dofs(),
                                multiplier_field.locally_relevant_dofs());

      std::vector<FieldId> participants;
      for (const auto &entry : terms())
        {
          const auto participant = entry.term.observable().source_field();
          if (std::find(participants.begin(),
                        participants.end(),
                        participant) == participants.end())
            participants.push_back(participant);
        }
      builder.saddle_point(multiplier_id, participants);

      for (std::size_t index = 0; index < terms().size(); ++index)
        terms()[index].term.add_constraint_terms(builder,
                                                 multiplier_id,
                                                 terms()[index].coefficient,
                                                 index);

      if constexpr (!std::is_same_v<Rhs, std::monostate>)
        {
          const auto rhs = rhs_;
          builder.term(multiplier_id, "constraint.rhs")
            .residual([rhs](const auto &) {
              typename Builder::Model::Operation result;
              result.reinit_vector = [rhs](auto &vector, const bool) {
                vector.reinit(rhs);
              };
              result.apply = [rhs](auto &vector) {
                vector = rhs;
                vector *= -1.;
              };
              result.apply_add = [rhs](auto &vector) { vector -= rhs; };
              return result;
            });
        }

      return {multiplier_id};
    }

  private:
    ConstraintSum<Term> terms_;
    Rhs                 rhs_;
  };

  template <typename Term, typename Rhs>
  Constraint<Term, std::decay_t<Rhs>>
  make_constraint(ConstraintSum<Term> terms, Rhs &&rhs)
  {
    return Constraint<Term, std::decay_t<Rhs>>(std::move(terms),
                                               std::forward<Rhs>(rhs));
  }

  template <typename Term>
  Constraint<Term>
  make_constraint(ConstraintSum<Term> terms)
  {
    return Constraint<Term>(std::move(terms));
  }

  template <typename Term,
            std::enable_if_t<detail::is_weak_term<Term>::value, int> = 0>
  Constraint<Term>
  make_constraint(Term term)
  {
    return make_constraint(ConstraintSum<Term>(std::move(term)));
  }

  template <typename Term,
            typename Rhs,
            std::enable_if_t<detail::is_weak_term<Term>::value, int> = 0>
  Constraint<Term, std::decay_t<Rhs>>
  make_constraint(Term term, Rhs &&rhs)
  {
    return make_constraint(ConstraintSum<Term>(std::move(term)),
                           std::forward<Rhs>(rhs));
  }

  template <typename Builder, typename Term, typename Rhs>
  ConstraintFields
  contribute(Builder &builder, const Constraint<Term, Rhs> &constraint)
  {
    return constraint.add(builder);
  }
} // namespace ImmersX

#endif // immersx_constraint_h
