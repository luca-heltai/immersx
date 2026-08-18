// ---------------------------------------------------------------------
// Generic reduced-field finite-element and transfer utilities.
// ---------------------------------------------------------------------

#ifndef immersx_reduced_field_utils_h
#define immersx_reduced_field_utils_h

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <memory>
#include <vector>

#include "reduced_field_catalog.h"

namespace ReducedFieldUtils
{
  /** Build the FE representation implied by a reduced-field catalog.
   * Point fields use FE_Q(1), while cell fields use FE_DGQ(0). Each catalog
   * entry becomes one FE block, preserving the catalog component ordering.
   */
  template <int dim, int spacedim>
  std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
  field_catalog_to_finite_element(const FieldCatalog &catalog);

  /** Return the component sizes of the blocks of a finite element system. */
  template <int dim, int spacedim>
  dealii::BlockIndices
  get_block_indices(const dealii::FiniteElement<dim, spacedim> &fe)
  {
    dealii::BlockIndices block_indices;
    for (unsigned int i = 0; i < fe.n_blocks(); ++i)
      block_indices.push_back(fe.base_element(i).n_components());
    return block_indices;
  }

  /** Transfer serial reduced-field values to the distributed DoF vector. */
  template <int dim, int spacedim>
  void
  serial_vector_to_distributed_vector(
    const dealii::DoFHandler<dim, spacedim>            &serial_dh,
    const dealii::DoFHandler<dim, spacedim>            &parallel_dh,
    const dealii::Vector<double>                       &serial_vec,
    dealii::LinearAlgebra::distributed::Vector<double> &parallel_vec);

  /** Map distributed locally owned vertices back to serial vertex indices. */
  template <int dim, int spacedim>
  std::vector<dealii::types::global_vertex_index>
  distributed_to_serial_vertex_indices(
    const dealii::Triangulation<dim, spacedim> &serial_tria,
    const dealii::Triangulation<dim, spacedim> &parallel_tria);

  /** Convert serialized field-major values into a DoF vector. */
  template <int dim, int spacedim, typename VectorType>
  void
  data_to_dealii_vector(const dealii::Triangulation<dim, spacedim> &serial_tria,
                        const dealii::Vector<double>               &data,
                        const dealii::DoFHandler<dim, spacedim>    &dh,
                        VectorType &output_vector)
  {
    AssertDimension(dh.n_dofs(), output_vector.size());
    const auto &fe = dh.get_fe();

    const auto dist_to_serial_vertices =
      distributed_to_serial_vertex_indices(serial_tria, dh.get_triangulation());
    const auto &locally_owned_dofs = dh.locally_owned_dofs();

    dealii::types::global_dof_index dofs_offset        = 0;
    unsigned int                    vertex_comp_offset = 0;
    unsigned int                    cell_comp_offset   = 0;
    for (unsigned int field = 0; field < fe.n_blocks(); ++field)
      {
        const auto        &field_fe = fe.base_element(field);
        const unsigned int n_comps  = field_fe.n_components();
        if (field_fe.n_dofs_per_vertex() > 0)
          {
            const dealii::types::global_dof_index n_local_dofs =
              n_comps * serial_tria.n_vertices();
            for (const auto &cell : dh.active_cell_iterators())
              if (cell->is_locally_owned())
                for (const auto v : cell->vertex_indices())
                  {
                    const auto serial_vertex_index =
                      dist_to_serial_vertices[cell->vertex_index(v)];
                    if (serial_vertex_index !=
                        dealii::numbers::invalid_unsigned_int)
                      for (unsigned int c = 0; c < n_comps; ++c)
                        {
                          const auto dof_index =
                            cell->vertex_dof_index(v, vertex_comp_offset + c);
                          Assert(locally_owned_dofs.is_element(dof_index),
                                 dealii::ExcInternalError());
                          output_vector[dof_index] =
                            data[dofs_offset + n_comps * serial_vertex_index +
                                 c];
                        }
                  }
            dofs_offset += n_local_dofs;
            vertex_comp_offset += n_comps;
          }
        else if (field_fe.template n_dofs_per_object<dim>() > 0)
          {
            const dealii::types::global_dof_index n_local_dofs =
              n_comps * serial_tria.n_global_active_cells();
            auto serial_cell   = serial_tria.begin_active();
            auto parallel_cell = dh.begin_active();
            for (; parallel_cell != dh.end(); ++parallel_cell)
              if (parallel_cell->is_locally_owned())
                {
                  while (serial_cell->id() < parallel_cell->id())
                    ++serial_cell;
                  const auto serial_cell_index =
                    serial_cell->global_active_cell_index();
                  for (unsigned int c = 0; c < n_comps; ++c)
                    {
                      const auto dof_index =
                        parallel_cell->dof_index(cell_comp_offset + c);
                      Assert(locally_owned_dofs.is_element(dof_index),
                             dealii::ExcInternalError());
                      output_vector[dof_index] =
                        data[dofs_offset + n_comps * serial_cell_index + c];
                    }
                }
            dofs_offset += n_local_dofs;
            cell_comp_offset += n_comps;
          }
      }
  }
} // namespace ReducedFieldUtils

#endif // immersx_reduced_field_utils_h
