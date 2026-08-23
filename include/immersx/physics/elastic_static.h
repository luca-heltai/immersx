// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_elastic_static_h
#define immersx_elastic_static_h

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/contributor.h>

#include <utility>
#include <vector>

namespace ImmersX
{
  using namespace dealii;

  /** Isotropic material data for a static elasticity Problem. */
  struct ElasticStaticMaterial
  {
    double lame_mu     = 1.;
    double lame_lambda = 1.;
  };

  /**
   * Minimal assembled linear elasticity Problem for a static solve.
   *
   * The Problem owns the mesh, displacement DoFs, constraints, stiffness
   * matrix, forcing, and accepted solution.  It deliberately does not own a
   * solver or any coupling state.
   */
  template <int dim, int spacedim = dim>
  class ElasticStaticProblem
  {
    static_assert(dim == spacedim,
                  "ElasticStaticProblem currently requires dim == spacedim.");

  public:
    using VectorType = ImmersXLA::MPI::Vector;
    using MatrixType = ImmersXLA::MPI::SparseMatrix;

    ElasticStaticProblem(const MPI_Comm     communicator       = MPI_COMM_WORLD,
                         const unsigned int initial_refinement = 2,
                         const unsigned int fe_degree          = 1,
                         const ElasticStaticMaterial material  = {})
      : communicator_(communicator)
      , triangulation_(communicator_)
      , dof_handler_(triangulation_)
      , fe_(FE_Q<spacedim>(fe_degree), spacedim)
      , initial_refinement_(initial_refinement)
      , material_(material)
    {}

    /** Create the mesh, DoFs, constraints, stiffness matrix, and vectors. */
    void
    setup()
    {
      AssertThrow(dof_handler_.n_dofs() == 0,
                  dealii::ExcMessage(
                    "ElasticStaticProblem::setup() may only be called once."));

      GridGenerator::hyper_cube(triangulation_, 0., 1.);
      triangulation_.refine_global(initial_refinement_);
      dof_handler_.distribute_dofs(fe_);

      locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
      locally_relevant_dofs_ =
        DoFTools::extract_locally_relevant_dofs(dof_handler_);

      constraints_.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);
      Functions::ZeroFunction<spacedim> zero(spacedim);
      VectorTools::interpolate_boundary_values(dof_handler_,
                                               0,
                                               zero,
                                               constraints_);
      constraints_.close();

      DynamicSparsityPattern sparsity(locally_relevant_dofs_);
      DoFTools::make_sparsity_pattern(dof_handler_,
                                      sparsity,
                                      constraints_,
                                      true);
      SparsityTools::distribute_sparsity_pattern(sparsity,
                                                 locally_owned_dofs_,
                                                 communicator_,
                                                 locally_relevant_dofs_);

      stiffness_matrix_.reinit(locally_owned_dofs_,
                               locally_owned_dofs_,
                               sparsity,
                               communicator_);
      forcing_.reinit(locally_owned_dofs_, communicator_);
      solution_.reinit(locally_owned_dofs_, communicator_);
      forcing_  = 0.;
      solution_ = 0.;

      assemble_stiffness();
    }

    const dealii::IndexSet &
    locally_owned_dofs() const
    {
      return locally_owned_dofs_;
    }

    const dealii::IndexSet &
    locally_relevant_dofs() const
    {
      return locally_relevant_dofs_;
    }

    const dealii::DoFHandler<spacedim> &
    dof_handler() const
    {
      return dof_handler_;
    }

    const dealii::AffineConstraints<double> &
    constraints() const
    {
      return constraints_;
    }

    const MatrixType &
    stiffness_operator() const
    {
      return stiffness_matrix_;
    }

    const VectorType &
    forcing() const
    {
      return forcing_;
    }

    const VectorType &
    solution() const
    {
      return solution_;
    }

    /** Assemble a body force into the Problem-owned forcing vector. */
    void
    set_forcing(const dealii::Function<spacedim> &function)
    {
      AssertThrow(function.n_components == spacedim,
                  dealii::ExcDimensionMismatch(function.n_components,
                                               spacedim));
      AssertThrow(dof_handler_.n_dofs() != 0,
                  dealii::ExcMessage(
                    "Call setup() before setting the elasticity forcing."));

      forcing_ = 0.;
      const QGauss<spacedim>      quadrature(fe_.degree + 1);
      FEValues<spacedim>          fe_values(fe_,
                                   quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);
      const unsigned int          dofs_per_cell = fe_.n_dofs_per_cell();
      const unsigned int          n_q_points    = quadrature.size();
      Vector<double>              cell_rhs(dofs_per_cell);
      std::vector<Vector<double>> values(n_q_points, Vector<double>(spacedim));
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell_rhs = 0.;
            fe_values.reinit(cell);
            function.vector_value_list(fe_values.get_quadrature_points(),
                                       values);
            for (unsigned int q = 0; q < n_q_points; ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const auto component = fe_.system_to_component_index(i).first;
                  cell_rhs(i) +=
                    fe_values[displacement_].value(i, q)[component] *
                    values[q][component] * fe_values.JxW(q);
                }

            cell->get_dof_indices(local_dof_indices);
            constraints_.distribute_local_to_global(cell_rhs,
                                                    local_dof_indices,
                                                    forcing_);
          }
      forcing_.compress(dealii::VectorOperation::add);
    }

    void
    set_solution(const VectorType &new_solution)
    {
      solution_ = new_solution;
    }

  private:
    void
    assemble_stiffness()
    {
      const QGauss<spacedim> quadrature(fe_.degree + 1);
      FEValues<spacedim>     fe_values(fe_,
                                   quadrature,
                                   update_gradients | update_JxW_values);
      const unsigned int     dofs_per_cell = fe_.n_dofs_per_cell();
      const unsigned int     n_q_points    = quadrature.size();
      FullMatrix<double>     cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>         cell_rhs(dofs_per_cell);
      std::vector<Tensor<2, spacedim>>     symmetric_gradients(dofs_per_cell);
      std::vector<double>                  divergences(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      stiffness_matrix_ = 0.;
      for (const auto &cell : dof_handler_.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell_matrix = 0.;
            cell_rhs    = 0.;
            fe_values.reinit(cell);
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    symmetric_gradients[k] =
                      fe_values[displacement_].symmetric_gradient(k, q);
                    divergences[k] = fe_values[displacement_].divergence(k, q);
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                      (2. * material_.lame_mu *
                         scalar_product(symmetric_gradients[i],
                                        symmetric_gradients[j]) +
                       material_.lame_lambda * divergences[i] *
                         divergences[j]) *
                      fe_values.JxW(q);
              }

            cell->get_dof_indices(local_dof_indices);
            constraints_.distribute_local_to_global(cell_matrix,
                                                    cell_rhs,
                                                    local_dof_indices,
                                                    stiffness_matrix_,
                                                    forcing_);
          }

      stiffness_matrix_.compress(dealii::VectorOperation::add);
      forcing_.compress(dealii::VectorOperation::add);
    }

    MPI_Comm communicator_;

    parallel::distributed::Triangulation<spacedim> triangulation_;
    DoFHandler<spacedim>                           dof_handler_;
    FESystem<spacedim>                             fe_;
    AffineConstraints<double>                      constraints_;

    IndexSet locally_owned_dofs_;
    IndexSet locally_relevant_dofs_;

    MatrixType stiffness_matrix_;
    VectorType forcing_;
    VectorType solution_;

    unsigned int               initial_refinement_;
    ElasticStaticMaterial      material_;
    FEValuesExtractors::Vector displacement_{0};
  };

  struct ElasticStaticFields
  {
    FieldId displacement;
  };

  /** Register the static elasticity residual with an execution adapter. */
  template <typename Builder, int dim, int spacedim>
  ElasticStaticFields
  contribute(Builder                                   &builder,
             const ElasticStaticProblem<dim, spacedim> &problem)
  {
    using VectorType = typename ElasticStaticProblem<dim, spacedim>::VectorType;
    const auto displacement =
      builder.algebraic_field("displacement",
                              problem.locally_owned_dofs(),
                              problem.locally_relevant_dofs());
    const auto stiffness =
      ImmersX::payload_free(dealii::linear_operator<VectorType, VectorType>(
        problem.stiffness_operator()));

    builder.term(displacement, "elastic-static")
      .residual([displacement, &problem](const auto &context) {
        const auto &state = context.state(displacement);
        dealii::PackagedOperation<VectorType> result;
        result.reinit_vector = [state](VectorType &vector, const bool omit) {
          vector.reinit(state, omit);
        };
        result.apply = [&problem, &state](VectorType &vector) {
          problem.stiffness_operator().vmult(vector, state);
          vector -= problem.forcing();
        };
        result.apply_add = [&problem, &state](VectorType &vector) {
          VectorType contribution;
          contribution.reinit(state);
          problem.stiffness_operator().vmult(contribution, state);
          contribution -= problem.forcing();
          vector += contribution;
        };
        return result;
      })
      .state(displacement, stiffness);

    return {displacement};
  }
} // namespace ImmersX

#endif // immersx_elastic_static_h
