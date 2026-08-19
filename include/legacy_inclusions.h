// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// ---------------------------------------------------------------------

#ifndef immersx_legacy_inclusions_h
#define immersx_legacy_inclusions_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include "point_cloud.h"
#include "reduced_field_catalog.h"

namespace LegacyInclusions
{
  /** Convert a legacy Fourier coefficient to the normalized reference basis.
   *
   * The constant mode is already normalized. Every non-constant sine/cosine
   * mode has L2 norm sqrt(pi) on the unit circle, while the reference basis is
   * normalized to have norm sqrt(2*pi). Therefore a legacy coefficient is
   * divided by sqrt(2) for all non-constant modes.
   */
  double
  fourier_to_reference_scale(const unsigned int mode);

  /** Return canonical basis indices in legacy Fourier order.
   *
   * The reference basis is generated from PolynomialsP monomials. This helper
   * maps the legacy sequence [1, cos(theta), sin(theta), ...] to the
   * corresponding canonical indices without introducing a Fourier basis.
   */
  std::vector<unsigned int>
  fourier_to_reference_indices(const unsigned int n_coefficients,
                               const unsigned int n_components,
                               const unsigned int polynomial_degree,
                               const unsigned int spacedim);

  /** Read the 2D legacy point format into the canonical point-cloud input. */
  void
  read_2d(const std::string         &inclusions_file,
          const std::string         &data_file,
          const unsigned int         n_coefficients,
          const unsigned int         n_components,
          const std::vector<double> &reference_data,
          PointCloud<2>             &point_cloud);

  /** Read the 3D legacy segment format into a canonical 1D mesh and fields. */
  void
  read_3d(const std::string           &inclusions_file,
          const std::string           &data_file,
          const unsigned int           n_coefficients,
          const unsigned int           n_components,
          const std::vector<double>   &reference_data,
          dealii::Triangulation<1, 3> &tria,
          dealii::DoFHandler<1, 3>    &properties_dh,
          dealii::Vector<double>      &properties,
          FieldCatalog                &catalog);
} // namespace LegacyInclusions

#endif // immersx_legacy_inclusions_h
