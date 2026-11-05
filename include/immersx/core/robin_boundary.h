// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_robin_boundary_h
#define immersx_robin_boundary_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <string>
#include <type_traits>

namespace ImmersX::robin_detail
{
  /** Detect the configured time variable as a complete expression token. */
  inline bool
  expression_depends_on_time(const dealii::ParameterHandler &prm,
                             const unsigned int              dimension)
  {
    const auto variables =
      dealii::Utilities::split_string_list(prm.get("Variable names"));
    if (variables.size() <= dimension)
      return false;
    const auto &time_variable = variables[dimension];
    const auto  expression    = prm.get("Function expression");
    for (std::size_t i = 0; i < expression.size();)
      if (std::isalpha(static_cast<unsigned char>(expression[i])) ||
          expression[i] == '_')
        {
          const auto begin = i++;
          while (i < expression.size() &&
                 (std::isalnum(static_cast<unsigned char>(expression[i])) ||
                  expression[i] == '_'))
            ++i;
          if (expression.substr(begin, i - begin) == time_variable)
            return true;
        }
      else
        ++i;
    return false;
  }

  /** Call once collectively after the local boundary assembly loop. */
  inline void
  assert_valid_boundary_data(const unsigned int local_invalid,
                             const MPI_Comm     communicator)
  {
    AssertThrow(dealii::Utilities::MPI::max(local_invalid, communicator) == 0,
                dealii::ExcMessage(
                  "Robin coefficients must be finite and nonnegative; Robin "
                  "boundary data must be finite and evaluable."));
  }

  /** Essential ids may also contain weak Dirichlet and normal constraints. */
  template <typename RobinIds, typename EssentialIds, typename NeumannIds>
  void
  assert_disjoint_boundary_ids(const RobinIds     &robin_ids,
                               const EssentialIds &essential_ids,
                               const NeumannIds   &neumann_ids,
                               const MPI_Comm      communicator)
  {
    unsigned int invalid = 0;
    for (const auto id : robin_ids)
      if (std::find(essential_ids.begin(), essential_ids.end(), id) !=
            essential_ids.end() ||
          std::find(neumann_ids.begin(), neumann_ids.end(), id) !=
            neumann_ids.end())
        invalid = 1;
    AssertThrow(dealii::Utilities::MPI::max(invalid, communicator) == 0,
                dealii::ExcMessage(
                  "Robin boundary ids must be disjoint from Dirichlet and "
                  "Neumann boundary ids."));
  }

  /** Add beta*u*v and g*v on one already reinitialized boundary face.
   * The caller sets Function times and later checks the returned invalid flag
   * collectively. Evaluation failures are recorded locally, so a rank without
   * the offending face still reaches that collective. Vector views allow a
   * mixed finite element: basis functions outside the selected vector vanish.
   */
  template <int dim, int spacedim, typename Extractor>
  unsigned int
  add_face_terms(const dealii::FEFaceValues<dim, spacedim> &values,
                 const Extractor                           &extractor,
                 const dealii::Function<spacedim>          &beta,
                 const dealii::Function<spacedim>          &g,
                 dealii::FullMatrix<double>                &matrix,
                 dealii::Vector<double>                    &rhs,
                 const double                               matrix_scale = 1.,
                 const double                               load_scale   = 1.)
  {
    const auto &view          = values[extractor];
    using Value               = std::decay_t<decltype(view.value(0u, 0u))>;
    constexpr bool     scalar = std::is_arithmetic_v<Value>;
    const unsigned int dofs   = values.get_fe().n_dofs_per_cell();
    AssertDimension(matrix.m(), dofs);
    AssertDimension(matrix.n(), dofs);
    AssertDimension(rhs.size(), dofs);
    AssertDimension(beta.n_components, 1);
    AssertDimension(g.n_components, scalar ? 1 : spacedim);
    unsigned int invalid = 0;
    for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
      {
        const auto &point       = values.quadrature_point(q);
        double      coefficient = 0.;
        Value       datum{};
        bool        valid = true;
        try
          {
            coefficient = beta.value(point);
            valid       = std::isfinite(coefficient) && coefficient >= 0.;
            if constexpr (scalar)
              {
                datum = g.value(point);
                valid = valid && std::isfinite(datum);
              }
            else
              for (unsigned int d = 0; d < spacedim; ++d)
                {
                  datum[d] = g.value(point, d);
                  valid    = valid && std::isfinite(datum[d]);
                }
          }
        catch (const std::exception &)
          {
            valid = false;
          }
        if (!valid)
          {
            invalid = 1;
            continue;
          }
        for (unsigned int i = 0; i < dofs; ++i)
          {
            const auto vi = view.value(i, q);
            rhs(i) += load_scale * (vi * datum) * values.JxW(q);
            for (unsigned int j = 0; j < dofs; ++j)
              matrix(i, j) += matrix_scale * coefficient *
                              (vi * view.value(j, q)) * values.JxW(q);
          }
      }
    return invalid;
  }
} // namespace ImmersX::robin_detail
#endif
