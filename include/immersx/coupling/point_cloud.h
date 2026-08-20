// ---------------------------------------------------------------------
// Generic point-cloud input for zero-dimensional reduced domains.
// ---------------------------------------------------------------------

#ifndef immersx_point_cloud_h
#define immersx_point_cloud_h

#include <deal.II/base/point.h>

#include <immersx/core/reduced_field_catalog.h>

#include <vector>

namespace ImmersX
{
  /** Distribution contract for programmatic and imported point clouds. */
  enum class PointCloudDistribution
  {
    /** Every rank provides the same global cloud; rank zero is the source. */
    replicated,
    /** Every rank provides only its own source subset. */
    rank_local
  };

  /** Neutral point-cloud representation used by zero-dimensional domains.
   * Property arrays are stored point-major within each catalog entry. Data
   * attached to vertices is reordered to the referenced point by file readers.
   * A programmatic cloud may contain only points; omitted fields use defaults.
   */
  template <int spacedim>
  struct PointCloud
  {
    std::vector<dealii::Point<spacedim>> points;
    std::vector<std::vector<double>>     properties;
    FieldCatalog                         catalog;
    std::vector<std::string>             property_names;
    PointCloudDistribution distribution = PointCloudDistribution::replicated;
  };

} // namespace ImmersX

#endif // immersx_point_cloud_h
