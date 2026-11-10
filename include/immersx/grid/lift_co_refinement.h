// Copyright (C) 2026 by Luca Heltai
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef immersx_lift_co_refinement_h
#define immersx_lift_co_refinement_h

#include <immersx/core/domain.h>
#include <immersx/coupling/tensor_product_lift.h>
#include <immersx/grid/local_co_refinement_parameters.h>

#include <array>
#include <functional>
#include <limits>

namespace ImmersX
{
  /** Counts refer to physical point/cell pairs, including shared-face ties.
   * Index zero denotes the axial scale; index one the cross-section scale.
   * The ratios use the solid cell's bounding-box diagonal.
   */
  struct LiftCoRefinementReport
  {
    unsigned long long initial_solid_cells = 0, initial_fluid_cells = 0;
    unsigned long long solid_cells = 0, fluid_cells = 0;
    unsigned long long point_cell_pairs = 0, outside_points = 0;
    unsigned int       solid_passes = 0, fluid_passes = 0, section_passes = 0;
    unsigned int       iterations = 0, section_level = 0;
    std::array<unsigned long long, 2> lower_violations{}, upper_violations{};
    std::array<double, 2>             minimum_ratio{
                  {std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()}};
    std::array<double, 2> maximum_ratio{};

    bool
    satisfied() const
    {
      return outside_points == 0 && lower_violations[0] == 0 &&
             lower_violations[1] == 0 && upper_violations[0] == 0 &&
             upper_violations[1] == 0;
    }
  };

  /** Co-refine a 3D solid and the curve-times-section physical lift.
   *
   * Axial scale is fluid-cell length. Circumferential scale is the positive
   * physical quadrature length of each section point: lifted JxW / axial JxW.
   * Both scales must lie in the configured open interval relative to the
   * solid cell's bounding-box diagonal, on every point-containing cell.
   * A low ratio refines the solid; a high axial ratio refines that fluid cell;
   * a high section ratio refines the section globally. Strategies space,
   * embedded and both select solid, lift, or both. The meshes and support
   * locations are re-queried after every pass. No coarsening is performed.
   *
   * Max refinement level limits the adaptive solid/fluid passes; the separate
   * section cap limits adaptive section refinement. Explicit global pre/post
   * and local solid post passes follow RefinementParameters conventions.
   * The returned report retains violations blocked by strategy or caps.
   * Coverage refers to the quadrature points, not the whole swept surface.
   * This compatibility criterion does not prove an inf-sup condition.
   *
   * Domains remain in their preparation lifecycle, transferring imported
   * fields with every mesh change. The radius callback evaluates retained data
   * at (CellId, unit point). Final application spaces must be created
   * afterwards.
   */
  LiftCoRefinementReport
  co_refine_lifted_domains(
    Domain<3>                                             &solid,
    Domain<1, 3>                                          &fluid,
    TensorProductLift<1, 2, 3, 3>                         &lift,
    const RefinementParameters<1>                         &parameters,
    const std::function<double(const dealii::CellId &,
                               const dealii::Point<1> &)> &radius,
    unsigned int maximum_section_level = 10);
} // namespace ImmersX
#endif
