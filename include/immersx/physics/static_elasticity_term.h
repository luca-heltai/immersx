// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------
#ifndef immersx_static_elasticity_term_h
#define immersx_static_elasticity_term_h

#include <deal.II/base/mpi.h>

#include <boost/serialization/set.hpp>

#include <immersx/core/known_term.h>
#include <immersx/core/weak_term.h>
#include <immersx/physics/material_properties.h>

#include <set>

namespace ImmersX
{
  /** Isotropic small-strain constitutive law, with its exact derivative. */
  template <int spacedim>
  struct IsotropicElasticStress
  {
    double mu;
    double lambda;

    dealii::SymmetricTensor<2, spacedim>
    evaluate(const dealii::SymmetricTensor<2, spacedim> &strain,
             const double                                divergence) const
    {
      return 2. * mu * strain +
             lambda * divergence * dealii::unit_symmetric_tensor<spacedim>();
    }

    dealii::SymmetricTensor<2, spacedim>
    linearize(const dealii::SymmetricTensor<2, spacedim> &,
              const double,
              const dealii::SymmetricTensor<2, spacedim> &strain,
              const double                                divergence) const
    {
      return evaluate(strain, divergence);
    }
  };

  /** The affine elasticity energy for one constant constitutive law. */
  template <typename FieldType>
  struct IsotropicElasticityTerm
  {
    FieldType                                           field;
    IsotropicElasticStress<FieldType::spacedimension()> law;

    template <typename Builder>
    FieldType
    operator()(Builder &builder) const
    {
      weak_term(2. * law.mu * symmetric_gradient(field),
                symmetric_gradient(test(field)))
        .add(builder);
      weak_term(law.lambda * divergence(field), divergence(test(field)))
        .add(builder);
      return field;
    }
  };

  template <typename FieldType>
  IsotropicElasticityTerm<FieldType>
  isotropic_elasticity_term(
    FieldType                                           field,
    IsotropicElasticStress<FieldType::spacedimension()> law)
  {
    return {std::move(field), law};
  }

  /** Add static elasticity, body loads, and vector tractions using FE
   * expressions. The parameter object supplies material and boundary function
   * lookups.
   */
  template <typename FieldType, typename Parameters>
  auto
  static_elasticity_term(FieldType displacement, const Parameters &parameters)
  {
    return [displacement, &parameters](auto &builder) {
      constexpr int dim      = FieldType::dimension();
      constexpr int spacedim = FieldType::spacedimension();
      using namespace dealii;
      std::set<types::material_id> local_ids;
      for (const auto &cell :
           displacement.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          local_ids.insert(cell->material_id());
      const auto all_ids =
        Utilities::MPI::all_gather(displacement.space().mpi_communicator(),
                                   local_ids);
      std::set<types::material_id> ids;
      for (const auto &part : all_ids)
        ids.insert(part.begin(), part.end());
      for (const auto id : ids)
        {
          const auto &material = parameters.get_material_properties(id);
          const IntegrationSupport<dim, spacedim> support(
            displacement.triangulation(), id);
          (weak_term(2. * material.Lame_mu * symmetric_gradient(displacement),
                     symmetric_gradient(test(displacement))) *
           support)
            .add(builder);
          (weak_term(material.Lame_lambda * divergence(displacement),
                     divergence(test(displacement))) *
           support)
            .add(builder);
        }
      KnownSource<Tensor<1, spacedim>, spacedim> force(
        [&parameters](const auto &context) {
          constexpr int spacedim = FieldType::spacedimension();
          const auto   &function = parameters.get_rhs(context.material_id);
          Tensor<1, spacedim> result;
          for (unsigned int d = 0; d < spacedim; ++d)
            result[d] = -function.value(context.point, d);
          return result;
        });
      known_term(force, test(displacement)).add(builder);
      for (const auto id : parameters.neumann_ids)
        {
          const auto &function = parameters.get_neumann_bc(id);
          KnownSource<Tensor<1, spacedim>, spacedim> traction(
            [&function](const auto &context) {
              constexpr int       spacedim = FieldType::spacedimension();
              Tensor<1, spacedim> result;
              for (unsigned int d = 0; d < spacedim; ++d)
                result[d] = -function.value(context.point, d);
              return result;
            });
          known_term(traction, test(displacement)).on_boundary(id).add(builder);
        }
      return displacement;
    };
  }

  /** Symmetric Nitsche boundary contribution to static elasticity.
   * Uses the full vector datum in the consistency load, including tangential
   * components, and the adjacent cell's material and diameter.
   */
  template <typename FieldType, typename Parameters>
  auto
  static_elasticity_nitsche(FieldType         displacement,
                            const Parameters &parameters)
  {
    return [displacement, &parameters](auto &builder) {
      using namespace dealii;
      constexpr int dim        = FieldType::dimension();
      constexpr int spacedim   = FieldType::spacedimension();
      using Builder            = std::decay_t<decltype(builder)>;
      using Matrix             = ImmersXLA::MPI::SparseMatrix;
      using Vector             = ImmersXLA::MPI::Vector;
      auto        matrix       = std::make_shared<Matrix>();
      auto        rhs          = std::make_shared<Vector>();
      const auto &owned        = displacement.locally_owned_dofs();
      const auto &relevant     = displacement.locally_relevant_dofs();
      const auto  communicator = displacement.space().mpi_communicator();
      DynamicSparsityPattern sparsity(relevant);
      DoFTools::make_sparsity_pattern(displacement.dof_handler(),
                                      sparsity,
                                      displacement.constraints(),
                                      true);
      SparsityTools::distribute_sparsity_pattern(sparsity,
                                                 owned,
                                                 communicator,
                                                 relevant);
      auto matrix_sparsity = detail::initialize_weak_matrix(
        *matrix, owned, owned, sparsity, communicator);
      rhs->reinit(owned, communicator);
      const auto                 &fe = displacement.finite_element();
      FEFaceValues<dim, spacedim> values(
        displacement.mapping(),
        fe,
        compatibility::gauss_quadrature<dim - 1>(
          fe.reference_cell().face_reference_cell(0), fe.degree + 1),
        update_values | update_gradients | update_normal_vectors |
          update_quadrature_points | update_JxW_values);
      const auto                           n = fe.n_dofs_per_cell();
      FullMatrix<double>                   local(n, n);
      dealii::Vector<double>               load(n);
      std::vector<types::global_dof_index> indices(n);
      for (const auto &cell :
           displacement.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            local = 0.;
            load  = 0.;
            const auto &material =
              parameters.get_material_properties(cell->material_id());
            const IsotropicElasticStress<spacedim> law{material.Lame_mu,
                                                       material.Lame_lambda};
            for (const auto face : cell->face_indices())
              if (cell->face(face)->at_boundary() &&
                  parameters.weak_dirichlet_ids.count(
                    cell->face(face)->boundary_id()))
                {
                  values.reinit(cell, face);
                  const auto &function = parameters.get_dirichlet_bc(
                    cell->face(face)->boundary_id());
                  const auto &view = values[displacement.extractor()];
                  for (const auto q : values.quadrature_point_indices())
                    {
                      const auto   normal = values.normal_vector(q);
                      const double penalty =
                        parameters.penalty_term / cell->diameter();
                      Tensor<1, spacedim> g;
                      for (unsigned int d = 0; d < spacedim; ++d)
                        g[d] = function.value(values.quadrature_point(q), d);
                      for (unsigned int i = 0; i < n; ++i)
                        {
                          const auto vi = view.value(i, q);
                          const auto ti =
                            law.evaluate(view.symmetric_gradient(i, q),
                                         view.divergence(i, q)) *
                            normal;
                          load[i] +=
                            (penalty * (vi * g) - ti * g) * values.JxW(q);
                          for (unsigned int j = 0; j < n; ++j)
                            {
                              const auto vj = view.value(j, q);
                              const auto tj =
                                law.evaluate(view.symmetric_gradient(j, q),
                                             view.divergence(j, q)) *
                                normal;
                              local(i, j) +=
                                (penalty * (vi * vj) - ti * vj - tj * vi) *
                                values.JxW(q);
                            }
                        }
                    }
                }
            cell->get_dof_indices(indices);
            displacement.constraints().distribute_local_to_global(
              local, load, indices, *matrix, *rhs);
          }
      matrix->compress(VectorOperation::add);
      rhs->compress(VectorOperation::add);
      auto op             = builder.matrix_operator(*matrix);
      op.materialize      = [matrix, matrix_sparsity] { return matrix; };
      op.materialize_into = [matrix](Matrix &destination) {
        destination.copy_from(*matrix);
      };
      builder.term(displacement.id(), "nitsche")
        .residual([displacement, matrix, rhs](const auto &context) {
          const auto &state = context.state(displacement.id());
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
        .state(displacement.id(), op);
      return displacement;
    };
  }
} // namespace ImmersX
#endif
