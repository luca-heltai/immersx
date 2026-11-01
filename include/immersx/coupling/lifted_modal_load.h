// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of the ImmersX application, based on deal.II.
// ---------------------------------------------------------------------
#ifndef immersx_lifted_modal_load_h
#define immersx_lifted_modal_load_h

#include <immersx/core/constraint.h>
#include <immersx/core/observable_lift.h>
#include <immersx/core/weak_term.h>

#include <type_traits>

namespace ImmersX
{
  /** Assemble a physical modal mass and the load of a modal datum.
   * The datum is a modal coefficient on the representative curve, matching
   * ReducedCoupling's static right-hand-side convention, not a physical
   * vector evaluated around the cross section. The modes have maximum one, so
   * the datum is the amplitude of the mode, and the load is the integral of
   * (datum phi) . phi, weighted like the mass matrix. Accept either a deal.II
   * Function or a callback taking the retained point and component; the latter
   * can evaluate imported fields on the known source cell without point search.
   */
  template <typename Lifted, typename Function>
  auto
  lifted_modal_load(const Lifted      &lifted,
                    const Function    &function,
                    const unsigned int component)
  {
    using Matrix                          = ImmersXLA::MPI::SparseMatrix;
    using Vector                          = ImmersXLA::MPI::Vector;
    const auto                    &field  = lifted.source();
    auto                           matrix = std::make_shared<Matrix>();
    auto                           rhs    = std::make_shared<Vector>();
    dealii::DynamicSparsityPattern sparsity(field.locally_relevant_dofs());
    dealii::DoFTools::make_sparsity_pattern(field.dof_handler(),
                                            sparsity,
                                            field.constraints(),
                                            true);
    dealii::SparsityTools::distribute_sparsity_pattern(
      sparsity,
      field.locally_owned_dofs(),
      field.space().mpi_communicator(),
      field.locally_relevant_dofs());
    detail::initialize_weak_matrix(*matrix,
                                   field.locally_owned_dofs(),
                                   field.locally_owned_dofs(),
                                   sparsity,
                                   field.space().mpi_communicator());
    rhs->reinit(field.locally_owned_dofs(), field.space().mpi_communicator());
    for (const auto &point : lifted.lifted_points())
      {
        const auto                 n = point.dof_indices.size();
        dealii::FullMatrix<double> local(n, n);
        dealii::Vector<double>     load(n);
        const double mode_squared = point.mode_vector * point.mode_vector;
        const double datum        = [&]() {
          if constexpr (std::is_invocable_r_v<double,
                                              const Function &,
                                              const typename Lifted::Point &,
                                              unsigned int>)
            return function(point, component);
          else
            return function.value(point.representative_point, component);
        }();
        for (unsigned int i = 0; i < n; ++i)
          {
            load[i] = point.weight * mode_squared *
                      point.source_basis_values[i] * datum;
            for (unsigned int j = 0; j < n; ++j)
              local(i, j) = point.weight * mode_squared *
                            point.source_basis_values[i] *
                            point.source_basis_values[j];
          }
        field.constraints().distribute_local_to_global(
          local, load, point.dof_indices, *matrix, *rhs);
      }
    matrix->compress(dealii::VectorOperation::add);
    rhs->compress(dealii::VectorOperation::add);
    return std::make_pair(matrix, rhs);
  }

  /** A unified modal constraint with its physical section-weighted metric. */
  template <typename Weak>
  struct LiftedModalInteraction
  {
    using Matrix = ImmersXLA::MPI::SparseMatrix;
    using Vector = ImmersXLA::MPI::Vector;
    Constraint<ConstraintSum<Weak>, Vector> constraint;
    std::shared_ptr<Matrix>                 metric;

    template <typename Builder>
    auto
    operator()(Builder &builder) const
    {
      auto op             = builder.matrix_operator(*metric);
      op.materialize      = [metric = metric] { return metric; };
      op.materialize_into = [metric = metric](auto &destination) {
        destination.copy_from(*metric);
      };
      return constraint.add(builder, &op);
    }
  };

  template <typename Weak>
  LiftedModalInteraction<Weak>
  lifted_modal_constraint(ConstraintSum<Weak>                           terms,
                          std::shared_ptr<ImmersXLA::MPI::SparseMatrix> metric,
                          ImmersXLA::MPI::Vector                        rhs)
  {
    return {make_constraint(std::move(terms), std::move(rhs)),
            std::move(metric)};
  }

  /** Add an owned affine FE operator without exposing execution block numbers.
   */
  template <typename Field, typename Matrix, typename Vector>
  auto
  affine_field_term(Field                   field,
                    std::shared_ptr<Matrix> matrix,
                    std::shared_ptr<Vector> rhs)
  {
    return [field, matrix, rhs](auto &builder) {
      using Builder       = std::decay_t<decltype(builder)>;
      auto op             = builder.matrix_operator(*matrix);
      op.materialize      = [matrix] { return matrix; };
      op.materialize_into = [matrix](Matrix &destination) {
        destination.copy_from(*matrix);
      };
      builder.term(field.id(), "affine")
        .residual([field, matrix, rhs](const auto &context) {
          const auto                        &state = context.state(field.id());
          typename Builder::Model::Operation result;
          result.reinit_vector = [rhs](auto &v, const bool) { v.reinit(*rhs); };
          result.apply         = [matrix, rhs, &state](auto &v) {
            matrix->vmult(v, state);
            v -= *rhs;
          };
          result.apply_add = [matrix, rhs, &state](auto &v) {
            Vector temporary;
            temporary.reinit(*rhs);
            matrix->vmult(temporary, state);
            temporary -= *rhs;
            v += temporary;
          };
          return result;
        })
        .state(field.id(), op);
      return field;
    };
  }
} // namespace ImmersX
#endif
