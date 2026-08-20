// ---------------------------------------------------------------------
// Generic metadata for reduced-space fields.
// ---------------------------------------------------------------------

#ifndef immersx_reduced_field_catalog_h
#define immersx_reduced_field_catalog_h

#include <string>
#include <vector>

namespace ImmersX
{
  /** Where a field is attached to a reduced mesh. */
  enum class FieldAssociation
  {
    point_data,
    cell_data
  };

  /** Metadata for one scalar or vector field. */
  struct ReducedFieldDescriptor
  {
    std::string      name;
    FieldAssociation association        = FieldAssociation::point_data;
    unsigned int     n_components       = 0;
    unsigned int     first_fe_component = 0;
    unsigned int     block_index        = 0;
  };

  using FieldCatalog = std::vector<ReducedFieldDescriptor>;

} // namespace ImmersX

#endif // immersx_reduced_field_catalog_h
