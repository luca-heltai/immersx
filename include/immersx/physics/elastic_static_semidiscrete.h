// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_elastic_static_semidiscrete_h
#define immersx_elastic_static_semidiscrete_h

#include <immersx/core/semidiscrete_pde_models.h>
#include <immersx/physics/elastic_static.h>

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace ImmersX
{
  /** A time-dependent equilibrium residual with algebraic displacement only. */
  template <int dim, int spacedim = dim>
  class ElasticStaticSemidiscrete
  {
    using Problem  = ElasticStaticProblem<dim, spacedim>;
    using Vector   = typename Problem::VectorType;
    using Matrix   = typename Problem::MatrixType;
    using Operator = dealii::LinearOperator<Vector>;
    using Entry    = std::tuple<dealii::types::global_dof_index,
                             dealii::types::global_dof_index,
                             double>;
    struct PreconditionerCache
    {
      std::vector<Entry>      entries;
      dealii::IndexSet        rows, columns;
      std::optional<Operator> inverse;
      unsigned int            builds         = 0;
      unsigned int            fixed_assembly = 0;
    };
    std::shared_ptr<PreconditionerCache> cache_ =
      std::make_shared<PreconditionerCache>();

  public:
    explicit ElasticStaticSemidiscrete(Problem &problem)
      : problem(problem)
    {}

    Problem &problem;

    /** Number of AMG setups shared by copies of this contributor. */
    unsigned int
    preconditioner_builds() const
    {
      return cache_->builds;
    }

    Operator
    preconditioner(const Matrix                              &matrix,
                   const std::function<void(Vector &, bool)> &reinit) const
    {
      if (problem.uses_fixed_stiffness())
        {
          if (!cache_->inverse ||
              cache_->fixed_assembly != problem.stiffness_assemblies())
            {
              // The immutable native matrix and its vector layout belong to
              // the Problem, whose lifetime already bounds this contributor.
              cache_->inverse = make_amg_preconditioner<Vector>(
                problem.stiffness_operator(),
                [native_problem = &problem](Vector &vector, const bool omit) {
                  vector.reinit(native_problem->solution(), omit);
                });
              cache_->fixed_assembly = problem.stiffness_assemblies();
              ++cache_->builds;
            }
          return *cache_->inverse;
        }
      std::vector<Entry> entries;
      for (const auto row : matrix.locally_owned_range_indices())
        for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
          entries.emplace_back(row, entry->column(), entry->value());
      const auto         rows    = matrix.locally_owned_range_indices();
      const auto         columns = matrix.locally_owned_domain_indices();
      const unsigned int changed = !cache_->inverse || cache_->rows != rows ||
                                   cache_->columns != columns ||
                                   cache_->entries != entries;
      if (dealii::Utilities::MPI::max(changed, matrix.get_mpi_communicator()) !=
          0)
        {
          // IDA's linearization matrix is transient. ML AMG may retain its
          // source, so each cached inverse owns an immutable matrix copy.
          auto retained_matrix = std::make_shared<Matrix>();
          retained_matrix->copy_from(matrix);
          const std::function<void(Vector &, bool)> retained_reinit =
            [retained_matrix, reinit](Vector &vector, const bool omit) {
              reinit(vector, omit);
            };
          cache_->inverse =
            make_amg_preconditioner<Vector>(*retained_matrix, retained_reinit);
          cache_->entries = std::move(entries);
          cache_->rows    = rows;
          cache_->columns = columns;
          ++cache_->builds;
        }
      return *cache_->inverse;
    }
  };

  /** Select parameter-driven loads and boundary conditions at candidate time.
   */
  template <int dim, int spacedim>
  ElasticStaticSemidiscrete<dim, spacedim>
  elastic_static_semidiscrete(ElasticStaticProblem<dim, spacedim> &problem)
  {
    return ElasticStaticSemidiscrete<dim, spacedim>(problem);
  }

  template <typename Builder, int dim, int spacedim>
  ElasticStaticFields<dim, spacedim>
  contribute(Builder                                        &builder,
             const ElasticStaticSemidiscrete<dim, spacedim> &description)
  {
    auto      &problem = description.problem;
    const auto id      = builder.algebraic_field("displacement",
                                            problem.locally_owned_dofs(),
                                            problem.locally_relevant_dofs());
    using Space        = typename ElasticStaticFields<dim, spacedim>::Space;
    auto space = std::make_shared<Space>(finite_element_space_view(problem));
    const auto displacement =
      space->field(id, "displacement", dealii::FEValuesExtractors::Vector(0));
    const auto stiffness =
      builder.matrix_operator(problem.stiffness_operator());
    builder.context_update(
      [&problem](const auto &context) { problem.update_time(context.time()); });
    builder.preconditioner(
      id, [description](const auto &matrix, const auto &prototype) {
        return description.preconditioner(matrix, prototype);
      });
    auto tangent = stiffness;
    if (dealii::Utilities::MPI::sum(
          problem.constraints().n_constraints(),
          problem.stiffness_operator().get_mpi_communicator()) != 0)
      tangent = semidiscrete_detail::constrained_matrix_operator_with_identity(
        stiffness, problem.constraints());
    if (problem.uses_fixed_stiffness())
      {
        // IDA uses identity rows for constraints. Retain that tangent once;
        // shared matrix provenance avoids copying it for later snapshots.
        using Vector = typename ElasticStaticProblem<dim, spacedim>::VectorType;
        tangent      = ImmersX::matrix_operator<Vector>(tangent.matrix());
      }
    builder.term(id, "elastic-equilibrium")
      .residual([id, &problem, stiffness](const auto &context) {
        return semidiscrete_detail::constrained_residual(stiffness.view *
                                                             context.state(id) -
                                                           problem.forcing(),
                                                         context.state(id),
                                                         problem.constraints());
      })
      .state(id, tangent);
    return {displacement, std::move(space)};
  }
} // namespace ImmersX
#endif
