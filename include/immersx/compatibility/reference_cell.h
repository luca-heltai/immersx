// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0
#ifndef immersx_compatibility_reference_cell_h
#define immersx_compatibility_reference_cell_h

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/reference_cell.h>

namespace ImmersX::compatibility
{
  namespace detail
  {
    // deal.II 9.7 takes the dimension as a method template argument;
    // dimension-typed ReferenceCell versions carry it in the cell type.
    template <int dim, typename ReferenceCell>
    auto
    gauss_quadrature(const ReferenceCell &cell, const unsigned int order, int)
      -> decltype(cell.template get_gauss_type_quadrature<dim>(order))
    {
      return cell.template get_gauss_type_quadrature<dim>(order);
    }

    template <int dim, typename ReferenceCell>
    auto
    gauss_quadrature(const ReferenceCell &cell, const unsigned int order, long)
      -> decltype(cell.get_gauss_type_quadrature(order))
    {
      return cell.get_gauss_type_quadrature(order);
    }

    template <int dim, int spacedim, typename ReferenceCell>
    auto
    linear_mapping(const ReferenceCell &cell, int)
      -> decltype(cell.template get_default_linear_mapping<dim, spacedim>())
    {
      return cell.template get_default_linear_mapping<dim, spacedim>();
    }

    template <int dim, int spacedim, typename ReferenceCell>
    auto
    linear_mapping(const ReferenceCell &cell, long)
      -> decltype(cell.template get_default_linear_mapping<spacedim>())
    {
      return cell.template get_default_linear_mapping<spacedim>();
    }
    template <int dim, int spacedim, typename ReferenceCell>
    auto
    default_mapping(const ReferenceCell &cell, const unsigned int degree, int)
      -> decltype(cell.template get_default_mapping<dim, spacedim>(degree))
    {
      return cell.template get_default_mapping<dim, spacedim>(degree);
    }

    template <int dim, int spacedim, typename ReferenceCell>
    auto
    default_mapping(const ReferenceCell &cell, const unsigned int degree, long)
      -> decltype(cell.template get_default_mapping<spacedim>(degree))
    {
      return cell.template get_default_mapping<spacedim>(degree);
    }
  } // namespace detail

  /** Call ReferenceCell's quadrature API across deal.II versions. */
  template <int dim, typename ReferenceCell>
  auto
  gauss_quadrature(const ReferenceCell &cell, const unsigned int order)
  {
    return detail::gauss_quadrature<dim>(cell, order, 0);
  }

  /** Return deal.II's static linear mapping for the reference cell. */
  template <int dim, int spacedim = dim, typename ReferenceCell>
  const dealii::Mapping<dim, spacedim> &
  linear_mapping(const ReferenceCell &cell)
  {
    return detail::linear_mapping<dim, spacedim>(cell, 0);
  }
  /** Call ReferenceCell's degree-dependent mapping API across versions. */
  template <int dim, int spacedim = dim, typename ReferenceCell>
  auto
  default_mapping(const ReferenceCell &cell, const unsigned int degree)
  {
    return detail::default_mapping<dim, spacedim>(cell, degree, 0);
  }
} // namespace ImmersX::compatibility

#endif
