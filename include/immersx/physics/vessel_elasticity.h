#ifndef immersx_vessel_elasticity_h
#define immersx_vessel_elasticity_h

#include <immersx/physics/coupled_elasticity.h>

namespace ImmersX
{
  /** Elasticity with vessel wall pressure and radius postprocessing.
   * This problem can run independently or exchange vessel data with an external
   * flow solver. It is the same type as the historical
   * CoupledElasticityProblem.
   */
  template <int dim, int spacedim = dim>
  using VesselElasticityProblem = CoupledElasticityProblem<dim, spacedim>;
} // namespace ImmersX

#endif
