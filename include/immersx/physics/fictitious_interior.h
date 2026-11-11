// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II library.
//
// ---------------------------------------------------------------------
#ifndef immersx_fictitious_interior_h
#define immersx_fictitious_interior_h

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <immersx/compatibility/reference_cell.h>
#include <immersx/core/matrix_operator.h>
#include <immersx/physics/elastodynamics.h>

#include <array>
#include <cmath>
#include <map>

namespace ImmersX
{
  /** Generalized forces conjugate to the two strains are (2 q_r, q_z). */
  struct FictitiousInteriorModes
  {
    static dealii::FullMatrix<double>
    elastic_hessian(const ElastodynamicsMaterialCoefficients &m)
    {
      dealii::FullMatrix<double> h(2, 2);
      h(0, 0) = 4 * (m.lame_lambda + m.lame_mu);
      h(0, 1) = h(1, 0) = 2 * m.lame_lambda;
      h(1, 1)           = m.lame_lambda + 2 * m.lame_mu;
      return h;
    }

    static dealii::FullMatrix<double>
    viscous_hessian(const ElastodynamicsMaterialCoefficients &m)
    {
      auto h = elastic_hessian(m);
      h *= m.rayleigh_beta;
      h(0, 0) += 2 * m.viscosity_eta;
      h(1, 1) += m.viscosity_eta;
      return h;
    }
  };

  /** Fixed, fully contained straight circular cylinder. */
  struct FictitiousInteriorCylinder
  {
    dealii::Point<3> begin, end;
    double           radius = 0.;
  };

  /**
   * Remove two spatially constant affine strain modes using native volume
   * Gauss weights. The cut volume is sampled, not replaced by pi r^2 L.
   * With homogeneous isotropic coefficients this is an energy projection
   * under the SAME quadrature as the native operators. No mass is removed.
   *
   * Matrices retain physical boundary columns, like the native raw operators;
   * constrained momentum rows are zero. Mesh slaves are eliminated with the
   * same hanging-node map as the native matrices. Prescribed boundary columns
   * remain, so candidates include boundary data without an affine-force
   * estimate.
   */
  template <typename VectorType, typename MatrixType>
  class FictitiousInteriorCorrection
  {
  public:
    template <typename Problem>
    FictitiousInteriorCorrection(const Problem                     &problem,
                                 const ElastodynamicsParameters<3> &parameters,
                                 const FictitiousInteriorCylinder  &cylinder)
      : material(parameters.default_material.coefficients)
    {
      using namespace dealii;
      const auto communicator = problem.triangulation().get_mpi_communicator();
      auto       tangent      = cylinder.end - cylinder.begin;
      const double length     = tangent.norm();
      AssertThrow(
        length > 0 && cylinder.radius > 0,
        ExcMessage("Modal correction requires a positive reference cylinder."));
      tangent /= length;
      const auto &mesh_constraints   = problem.hanging_node_constraints();
      bool        mesh_or_prescribed = true;
      for (const auto *constraints :
           {&problem.constraints(), &problem.velocity_constraints()})
        for (const auto &line : constraints->get_lines())
          if (!line.entries.empty())
            mesh_or_prescribed =
              mesh_or_prescribed && mesh_constraints.is_constrained(line.index);
      AssertThrow(
        Utilities::MPI::min(mesh_or_prescribed ? 1u : 0u, communicator) == 1,
        ExcMessage(
          "Modal correction supports hanging nodes and prescribed values only."));
      const auto &owned = problem.locally_owned_dofs();
      for (auto &b : *integrals)
        {
          b.reinit(owned, communicator);
          b = 0.;
        }
      const auto quadrature =
        compatibility::gauss_quadrature<3>(problem.fe().reference_cell(),
                                           problem.fe().degree + 1);
      FEValues<3>                          values(problem.mapping(),
                         problem.fe(),
                         quadrature,
                         update_gradients | update_quadrature_points |
                           update_JxW_values);
      const FEValuesExtractors::Vector     displacement(0);
      std::vector<types::global_dof_index> indices(
        problem.fe().n_dofs_per_cell());
      bool   homogeneous  = true;
      double local_volume = 0.;
      for (const auto &cell : problem.dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            values.reinit(cell);
            cell->get_dof_indices(indices);
            const auto m =
              parameters.get_material_properties(cell->material_id());
            std::array<dealii::Vector<double>, 2> local_integrals;
            for (auto &b : local_integrals)
              b.reinit(indices.size());
            for (const auto q : values.quadrature_point_indices())
              {
                const auto delta = values.quadrature_point(q) - cylinder.begin;
                const double axial = delta * tangent;
                if (axial < 0 || axial > length ||
                    (delta - axial * tangent).norm_square() >
                      cylinder.radius * cylinder.radius)
                  continue;
                homogeneous = homogeneous && m.lame_mu == material.lame_mu &&
                              m.lame_lambda == material.lame_lambda &&
                              m.viscosity_eta == material.viscosity_eta &&
                              m.rayleigh_beta == material.rayleigh_beta;
                const double weight = values.JxW(q);
                local_volume += weight;
                for (unsigned int i = 0; i < indices.size(); ++i)
                  {
                    const auto eps =
                      values[displacement].symmetric_gradient(i, q);
                    const double h = tangent * (eps * tangent);
                    local_integrals[0][i] += weight * .5 * (trace(eps) - h);
                    local_integrals[1][i] += weight * h;
                  }
              }
            for (unsigned int mode = 0; mode < 2; ++mode)
              mesh_constraints.distribute_local_to_global(local_integrals[mode],
                                                          indices,
                                                          (*integrals)[mode]);
          }
      AssertThrow(
        Utilities::MPI::min(homogeneous ? 1u : 0u, communicator) == 1,
        ExcMessage(
          "Modal correction requires homogeneous interior elastic and viscous coefficients matching the default material."));
      AssertThrow(
        material.lame_mu > 0 &&
          3 * material.lame_lambda + 2 * material.lame_mu > 0 &&
          material.viscosity_eta >= 0 && material.rayleigh_beta >= 0,
        ExcMessage(
          "Modal correction requires positive elastic and nonnegative viscous energy."));
      sampled_volume = Utilities::MPI::sum(local_volume, communicator);
      AssertThrow(
        sampled_volume > 0,
        ExcMessage(
          "No solid Gauss points resolve the reference lumen; refine the solid mesh."));
      exact_volume = numbers::PI * cylinder.radius * cylinder.radius * length;
      for (auto &b : *integrals)
        b.compress(VectorOperation::add);

      constrained_rows = dealii::IndexSet(owned.size());
      for (const auto &line : problem.velocity_constraints().get_lines())
        constrained_rows.add_index(line.index);
      constrained_rows.compress();
    }

    MaterializedOperator<VectorType, MatrixType>
    elastic_operator() const
    {
      return make_operator(FictitiousInteriorModes::elastic_hessian(material));
    }

    MaterializedOperator<VectorType, MatrixType>
    viscous_operator() const
    {
      return make_operator(FictitiousInteriorModes::viscous_hessian(material));
    }

    std::array<double, 2>
    modes(const VectorType &state) const
    {
      std::array<double, 2> result{};
      for (const auto i : (*integrals)[0].locally_owned_elements())
        for (unsigned int mode = 0; mode < 2; ++mode)
          result[mode] += (*integrals)[mode][i] * state[i];
      for (auto &entry : result)
        entry =
          dealii::Utilities::MPI::sum(entry,
                                      (*integrals)[0].get_mpi_communicator()) /
          sampled_volume;
      return result;
    }

    double
    removed_elastic_energy(const VectorType &state) const
    {
      const auto x = modes(state);
      const auto h = FictitiousInteriorModes::elastic_hessian(material);
      return .5 * sampled_volume *
             (x[0] * (h(0, 0) * x[0] + h(0, 1) * x[1]) +
              x[1] * (h(1, 0) * x[0] + h(1, 1) * x[1]));
    }

    double
    removed_viscous_dissipation(const VectorType &velocity) const
    {
      const auto x = modes(velocity);
      const auto h = FictitiousInteriorModes::viscous_hessian(material);
      return sampled_volume * (x[0] * (h(0, 0) * x[0] + h(0, 1) * x[1]) +
                               x[1] * (h(1, 0) * x[0] + h(1, 1) * x[1]));
    }

    double
    volume() const
    {
      return sampled_volume;
    }
    double
    reference_volume() const
    {
      return exact_volume;
    }

  private:
    MaterializedOperator<VectorType, MatrixType>
    make_operator(const dealii::FullMatrix<double> &h) const
    {
      const auto   b           = integrals;
      const auto   constrained = constrained_rows;
      const double volume      = sampled_volume;
      const auto   apply = [b, h, constrained, volume](VectorType       &dst,
                                                     const VectorType &src,
                                                     const bool transpose) {
        std::array<double, 2> moment{};
        for (const auto i : (*b)[0].locally_owned_elements())
          if (!transpose || !constrained.is_element(i))
            for (unsigned int mode = 0; mode < 2; ++mode)
              moment[mode] += (*b)[mode][i] * src[i];
        for (auto &entry : moment)
          entry =
            dealii::Utilities::MPI::sum(entry, (*b)[0].get_mpi_communicator());
        dst.equ((h(0, 0) * moment[0] + h(0, 1) * moment[1]) / volume, (*b)[0]);
        dst.add((h(1, 0) * moment[0] + h(1, 1) * moment[1]) / volume, (*b)[1]);
        if (!transpose)
          for (const auto i : dst.locally_owned_elements())
            if (constrained.is_element(i))
              dst[i] = 0.;
      };
      const auto action = std::make_shared<const decltype(apply)>(apply);
      MaterializedOperator<VectorType, MatrixType> result;
      auto                                        &op = result.view;
      op.is_null_operator                             = false;
      op.reinit_domain_vector                         = op.reinit_range_vector =
        [b](VectorType &v, const bool omit) { v.reinit((*b)[0], omit); };
      op.vmult = [action](VectorType &dst, const VectorType &src) {
        (*action)(dst, src, false);
      };
      op.Tvmult = [action](VectorType &dst, const VectorType &src) {
        (*action)(dst, src, true);
      };
      op.vmult_add = [action, b](VectorType &dst, const VectorType &src) {
        VectorType value;
        value.reinit((*b)[0]);
        (*action)(value, src, false);
        dst += value;
      };
      op.Tvmult_add = [action, b](VectorType &dst, const VectorType &src) {
        VectorType value;
        value.reinit((*b)[0]);
        (*action)(value, src, true);
        dst += value;
      };

      // Iterative actions above store only two distributed vectors. Explicit
      // direct solves request this outer-product matrix lazily.
      auto       cached = std::make_shared<std::shared_ptr<MatrixType>>();
      const auto build  = [b, h, constrained, volume, cached]() {
        if (*cached)
          return *cached;
        using namespace dealii;
        const auto &owned = (*b)[0].locally_owned_elements();
        const auto  comm  = (*b)[0].get_mpi_communicator();
        std::map<types::global_dof_index, std::array<double, 2>> local, global;
        for (const auto i : owned)
          if ((*b)[0][i] != 0 || (*b)[1][i] != 0)
            local[i] = std::array<double, 2>{{(*b)[0][i], (*b)[1][i]}};
        for (const auto &part : Utilities::MPI::all_gather(comm, local))
          global.insert(part.begin(), part.end());
        DynamicSparsityPattern sparsity(owned.size(), owned.size(), owned);
        for (const auto &[i, bi] : local)
          for (const auto &[j, bj] : global)
            sparsity.add(i, j);
        auto matrix = std::make_shared<MatrixType>();
        matrix->reinit(owned, owned, sparsity, comm, false);
        for (const auto &[i, bi] : local)
          if (!constrained.is_element(i))
            for (const auto &[j, bj] : global)
              matrix->set(i,
                          j,
                          (bi[0] * (h(0, 0) * bj[0] + h(0, 1) * bj[1]) +
                           bi[1] * (h(1, 0) * bj[0] + h(1, 1) * bj[1])) /
                            volume);
        matrix->compress(VectorOperation::insert);
        *cached = matrix;
        return matrix;
      };
      result.materialize      = build;
      result.materialize_into = [build](MatrixType &dst) {
        dst.copy_from(*build());
      };
      return result;
    }

    ElastodynamicsMaterialCoefficients         material;
    std::shared_ptr<std::array<VectorType, 2>> integrals =
      std::make_shared<std::array<VectorType, 2>>();
    dealii::IndexSet constrained_rows;
    double           sampled_volume = 0., exact_volume = 0.;
  };
} // namespace ImmersX
#endif
