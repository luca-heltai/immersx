// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_reference_frame_h
#define immersx_reference_frame_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#include <cmath>

namespace ImmersX
{
  namespace detail
  {
    /**
     * Return the proper orthogonal map from the reference frame to a frame
     * whose last coordinate direction is @p tangent.
     *
     * The reference vertical is e_y in two dimensions and e_z in three
     * dimensions.  The anti-parallel three-dimensional case is assigned the
     * deterministic half-turn about e_x.
     */
    template <int spacedim>
    dealii::Tensor<2, spacedim>
    reference_to_physical_rotation(const dealii::Tensor<1, spacedim> &tangent)
    {
      static_assert(spacedim == 2 || spacedim == 3,
                    "Reference-frame rotations support 2D and 3D only.");

      AssertThrow(tangent.norm() > 0.,
                  dealii::ExcMessage(
                    "The physical frame direction must be non-zero."));

      dealii::Tensor<1, spacedim> reference_vertical;
      reference_vertical[spacedim - 1] = 1.;
      const auto unit_tangent          = tangent / tangent.norm();

      dealii::Tensor<2, spacedim> rotation;

      if constexpr (spacedim == 2)
        {
          // R(theta) e_y = (-sin(theta), cos(theta)).
          const double cosine = reference_vertical * unit_tangent;
          const double sine   = -unit_tangent[0];
          rotation[0][0]      = cosine;
          rotation[0][1]      = -sine;
          rotation[1][0]      = sine;
          rotation[1][1]      = cosine;
        }
      else
        {
          auto axis =
            dealii::cross_product_3d(reference_vertical, unit_tangent);
          const auto       sine      = axis.norm();
          const auto       cosine    = reference_vertical * unit_tangent;
          constexpr double tolerance = 1.e-14;

          if (sine > tolerance)
            axis /= sine;
          else if (cosine > 0.)
            {
              axis = 0.;
            }
          else
            {
              // e_z -> -e_z is a pi rotation about e_x.  This choice is
              // deterministic and keeps the transformation proper.
              axis    = 0.;
              axis[0] = 1.;
            }

          for (unsigned int i = 0; i < spacedim; ++i)
            for (unsigned int j = 0; j < spacedim; ++j)
              rotation[i][j] =
                (i == j ? cosine : 0.) + (1. - cosine) * axis[i] * axis[j];

          if (sine > tolerance)
            {
              rotation[0][1] -= sine * axis[2];
              rotation[0][2] += sine * axis[1];
              rotation[1][0] += sine * axis[2];
              rotation[1][2] -= sine * axis[0];
              rotation[2][0] -= sine * axis[1];
              rotation[2][1] += sine * axis[0];
            }
        }

      return rotation;
    }
  } // namespace detail
} // namespace ImmersX

#endif // immersx_reference_frame_h
