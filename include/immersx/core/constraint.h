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

#include <immersx/core/weak_term.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ImmersX
{
  struct ConstraintFields
  {
    FieldId multiplier;
  };

  namespace detail
  {
    template <typename Type>
    struct is_weak_term : std::false_type
    {};

    template <typename ObservableType, typename TargetField>
    struct is_weak_term<WeakTerm<ObservableType, TargetField>> : std::true_type
    {};

    template <typename Wanted,
              std::size_t Index,
              typename First,
              typename... Rest>
    constexpr std::size_t
    constraint_entry_index()
    {
      if constexpr (std::is_same_v<Wanted, First>)
        return Index;
      else
        return constraint_entry_index<Wanted, Index + 1, Rest...>();
    }

    /** Assemble the multiplier-space mass metric used by AL preconditioners. */
    template <typename VectorType, typename MatrixType, typename FieldType>
    typename SemiDiscreteModel<VectorType, MatrixType>::MatrixOperator
    make_multiplier_metric(SemidiscreteBuilder<VectorType, MatrixType> &builder,
                           const FieldType                             &field)
    {
      constexpr int dim      = FieldType::dimension();
      constexpr int spacedim = FieldType::spacedimension();
      using MatrixOperator =
        typename SemiDiscreteModel<VectorType, MatrixType>::MatrixOperator;

      const auto                degree = field.space().finite_element().degree;
      const dealii::QGauss<dim> quadrature(degree + 1);
      dealii::DynamicSparsityPattern sparsity(field.dof_handler().n_dofs(),
                                              field.dof_handler().n_dofs(),
                                              field.locally_owned_dofs());
      std::vector<dealii::types::global_dof_index> indices(
        field.dof_handler().get_fe().n_dofs_per_cell());
      for (const auto &cell : field.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            field.constraints().add_entries_local_to_global(indices,
                                                            indices,
                                                            sparsity,
                                                            false);
          }

      auto matrix = std::make_shared<MatrixType>();
      auto matrix_sparsity =
        initialize_weak_matrix(*matrix,
                               field.locally_owned_dofs(),
                               field.locally_owned_dofs(),
                               sparsity,
                               field.space().mpi_communicator());
      dealii::FEValues<dim, spacedim> values(field.mapping(),
                                             field.space().finite_element(),
                                             quadrature,
                                             dealii::update_values |
                                               dealii::update_JxW_values);
      dealii::FullMatrix<double>      local(indices.size(), indices.size());
      for (const auto &cell : field.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            values.reinit(cell);
            local = 0.;
            for (unsigned int i = 0; i < indices.size(); ++i)
              for (unsigned int j = 0; j < indices.size(); ++j)
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    const auto &view = values[field.extractor()];
                    local(i, j) += detail::natural_pairing(view.value(i, q),
                                                           view.value(j, q)) *
                                   values.JxW(q);
                  }
            field.constraints().distribute_local_to_global(local,
                                                           indices,
                                                           *matrix);
          }
      compress_weak_matrix(*matrix);

      MatrixOperator result   = builder.matrix_operator(*matrix);
      result.materialize      = [matrix, matrix_sparsity] { return matrix; };
      result.materialize_into = [matrix](MatrixType &destination) {
        destination.copy_from(*matrix);
      };
      return result;
    }
  } // namespace detail

  /** A signed sum of weak terms defining one constraint row. */
  template <typename... Terms>
  class ConstraintSum
  {
  public:
    template <std::size_t Index, typename Term>
    struct Entry
    {
      double coefficient;
      Term   term;
    };

    template <typename Sequence>
    struct MakeEntryVariant;

    template <std::size_t... Indices>
    struct MakeEntryVariant<std::index_sequence<Indices...>>
    {
      using type = std::variant<Entry<Indices, Terms>...>;
    };

    using EntryVariant = typename MakeEntryVariant<
      std::make_index_sequence<sizeof...(Terms)>>::type;

    ConstraintSum() = default;

    template <typename Term>
    void
    add(Term term, const double coefficient = 1.)
    {
      constexpr auto index =
        detail::constraint_entry_index<Term, 0, Terms...>();
      entries_.emplace_back(std::in_place_index<index>,
                            Entry<index, Term>{coefficient, std::move(term)});
    }

    template <typename Callback>
    void
    for_each(Callback &&callback) const
    {
      for (const auto &entry : entries_)
        std::visit(std::forward<Callback>(callback), entry);
    }

    bool
    empty() const
    {
      return entries_.empty();
    }

  private:
    template <typename, typename>
    friend class Constraint;

    std::vector<EntryVariant> entries_;
  };

  template <typename Lhs,
            typename Rhs,
            std::enable_if_t<detail::is_weak_term<Lhs>::value &&
                               detail::is_weak_term<Rhs>::value,
                             int> = 0>
  ConstraintSum<Lhs, Rhs>
  operator+(Lhs lhs, Rhs rhs)
  {
    ConstraintSum<Lhs, Rhs> result;
    result.add(std::move(lhs));
    result.add(std::move(rhs));
    return result;
  }

  template <typename Lhs,
            typename Rhs,
            std::enable_if_t<detail::is_weak_term<Lhs>::value &&
                               detail::is_weak_term<Rhs>::value,
                             int> = 0>
  ConstraintSum<Lhs, Rhs>
  operator-(Lhs lhs, Rhs rhs)
  {
    ConstraintSum<Lhs, Rhs> result;
    result.add(std::move(lhs));
    result.add(std::move(rhs), -1.);
    return result;
  }

  template <typename... LhsTerms, typename... RhsTerms>
  ConstraintSum<LhsTerms..., RhsTerms...>
  operator+(ConstraintSum<LhsTerms...> lhs, ConstraintSum<RhsTerms...> rhs)
  {
    ConstraintSum<LhsTerms..., RhsTerms...> result;
    lhs.for_each([&result](const auto &entry) {
      result.add(entry.term, entry.coefficient);
    });
    rhs.for_each([&result](const auto &entry) {
      result.add(entry.term, entry.coefficient);
    });
    return result;
  }

  template <typename... LhsTerms, typename... RhsTerms>
  ConstraintSum<LhsTerms..., RhsTerms...>
  operator-(ConstraintSum<LhsTerms...> lhs, ConstraintSum<RhsTerms...> rhs)
  {
    ConstraintSum<LhsTerms..., RhsTerms...> result;
    lhs.for_each([&result](const auto &entry) {
      result.add(entry.term, entry.coefficient);
    });
    rhs.for_each([&result](const auto &entry) {
      result.add(entry.term, -entry.coefficient);
    });
    return result;
  }

  template <typename... Terms,
            typename NewTerm,
            std::enable_if_t<detail::is_weak_term<NewTerm>::value, int> = 0>
  ConstraintSum<Terms..., NewTerm>
  operator+(ConstraintSum<Terms...> lhs, NewTerm rhs)
  {
    ConstraintSum<Terms..., NewTerm> result;
    lhs.for_each([&result](const auto &entry) {
      result.add(entry.term, entry.coefficient);
    });
    result.add(std::move(rhs));
    return result;
  }

  template <typename... Terms,
            typename NewTerm,
            std::enable_if_t<detail::is_weak_term<NewTerm>::value, int> = 0>
  ConstraintSum<Terms..., NewTerm>
  operator-(ConstraintSum<Terms...> lhs, NewTerm rhs)
  {
    ConstraintSum<Terms..., NewTerm> result;
    lhs.for_each([&result](const auto &entry) {
      result.add(entry.term, entry.coefficient);
    });
    result.add(std::move(rhs), -1.);
    return result;
  }

  /** A generic Lagrange-multiplier constraint over weak terms. */
  template <typename Term, typename Rhs = std::monostate>
  class Constraint
  {
  public:
    using term_type = Term;
    using rhs_type  = Rhs;

    explicit Constraint(Term terms, Rhs rhs = Rhs())
      : terms_(std::move(terms))
      , rhs_(std::move(rhs))
    {
      AssertThrow(!terms_.empty(),
                  dealii::ExcMessage(
                    "A constraint must contain at least one weak term."));

      const auto &target = multiplier();
      AssertThrow(!target.is_registered(),
                  dealii::ExcMessage(
                    "A constraint multiplier must be an unregistered field."));
      terms_.for_each([&](const auto &entry) {
        AssertThrow(!entry.term.observable().is_frozen(),
                    dealii::ExcMessage(
                      "Constraint weak terms must use active participant "
                      "fields; put prescribed data in the constraint rhs."));
        AssertThrow(!entry.term.target().source().is_registered(),
                    dealii::ExcMessage(
                      "A constraint multiplier must be an unregistered "
                      "field."));
        AssertThrow(
          static_cast<const void *>(&entry.term.target().source().space()) ==
              static_cast<const void *>(&target.space()) &&
            entry.term.target().source().name() == target.name(),
          dealii::ExcMessage("All weak terms in a constraint must use the same "
                             "multiplier field."));
      });
    }

    const auto &
    terms() const
    {
      return terms_;
    }

    const auto &
    multiplier() const
    {
      return std::get<0>(terms_.entries_.front()).term.target().source();
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
      terms_.for_each([&](const auto &entry) {
        const auto participant = entry.term.observable().source_field();
        if (std::find(participants.begin(), participants.end(), participant) ==
            participants.end())
          participants.push_back(participant);
      });
      const auto metric =
        detail::make_multiplier_metric(builder, multiplier_field);
      builder.saddle_point(multiplier_id, participants, metric);

      std::size_t index = 0;
      terms_.for_each([&](const auto &entry) {
        entry.term.add_constraint_terms(builder,
                                        multiplier_id,
                                        entry.coefficient,
                                        index++);
      });

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
    Term terms_;
    Rhs  rhs_;
  };

  template <typename... Terms, typename Rhs>
  Constraint<ConstraintSum<Terms...>, std::decay_t<Rhs>>
  make_constraint(ConstraintSum<Terms...> terms, Rhs &&rhs)
  {
    return Constraint<ConstraintSum<Terms...>, std::decay_t<Rhs>>(
      std::move(terms), std::forward<Rhs>(rhs));
  }

  template <typename... Terms>
  Constraint<ConstraintSum<Terms...>>
  make_constraint(ConstraintSum<Terms...> terms)
  {
    return Constraint<ConstraintSum<Terms...>>(std::move(terms));
  }

  template <typename Term,
            std::enable_if_t<detail::is_weak_term<Term>::value, int> = 0>
  Constraint<ConstraintSum<Term>>
  make_constraint(Term term)
  {
    ConstraintSum<Term> terms;
    terms.add(std::move(term));
    return make_constraint(std::move(terms));
  }

  template <typename Term,
            typename Rhs,
            std::enable_if_t<detail::is_weak_term<Term>::value, int> = 0>
  Constraint<ConstraintSum<Term>, std::decay_t<Rhs>>
  make_constraint(Term term, Rhs &&rhs)
  {
    ConstraintSum<Term> terms;
    terms.add(std::move(term));
    return make_constraint(std::move(terms), std::forward<Rhs>(rhs));
  }

  template <typename Builder, typename Term, typename Rhs>
  ConstraintFields
  contribute(Builder &builder, const Constraint<Term, Rhs> &constraint)
  {
    return constraint.add(builder);
  }
} // namespace ImmersX

#endif // immersx_constraint_h
