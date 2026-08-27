// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <immersx/core/reduced_field_utils.h>
#include <immersx/io/imported_finite_element_fields.h>
#include <immersx/io/vtk_utils.h>

#include <algorithm>
#include <cmath>

namespace ImmersX
{
  namespace
  {
    template <int dim, int spacedim>
    void
    assert_same_mesh(
      const dealii::Triangulation<dim, spacedim> &serial_triangulation,
      const dealii::parallel::TriangulationBase<dim, spacedim>
        &target_triangulation)
    {
      AssertThrow(serial_triangulation.n_global_active_cells() ==
                    target_triangulation.n_global_active_cells(),
                  dealii::ExcMessage(
                    "Imported fields require a target triangulation with the "
                    "same number of active cells as the VTK mesh."));
      for (const auto &target_cell :
           target_triangulation.active_cell_iterators())
        if (target_cell->is_locally_owned())
          {
            auto serial_cell = serial_triangulation.begin_active();
            while (serial_cell != serial_triangulation.end() &&
                   serial_cell->id() != target_cell->id())
              ++serial_cell;
            AssertThrow(serial_cell != serial_triangulation.end(),
                        dealii::ExcMessage(
                          "Imported fields require matching active-cell "
                          "identifiers on the target triangulation."));
            for (const auto vertex : serial_cell->vertex_indices())
              AssertThrow(
                serial_cell->vertex(vertex).distance(
                  target_cell->vertex(vertex)) < 1.e-12,
                dealii::ExcMessage(
                  "Imported fields require the VTK and target meshes to "
                  "have matching vertex coordinates."));
          }
    }
  } // namespace

  template <int dim, int spacedim>
  ImportedFiniteElementFields<dim, spacedim>::ImportedFiniteElementFields(
    const std::string       &vtk_filename,
    const TriangulationType &triangulation,
    const MPI_Comm           communicator)
    : triangulation_(&triangulation)
    , dof_handler_(const_cast<TriangulationType &>(triangulation))
    , constraints_()
    , coefficients_(std::make_shared<VectorType>())
    , communicator_(communicator)
    , representation_(triangulation,
                      dof_handler_,
                      locally_owned_dofs_,
                      locally_relevant_dofs_,
                      constraints_)
  {
    constraints_.close();
#ifdef DEAL_II_WITH_VTK
    AssertThrow(triangulation.n_global_active_cells() > 0,
                dealii::ExcMessage(
                  "The target triangulation must be prepared before importing "
                  "finite-element fields."));

    dealii::Triangulation<dim, spacedim> serial_triangulation;
    dealii::DoFHandler<dim, spacedim> serial_dof_handler(serial_triangulation);
    dealii::Vector<double>            serial_coefficients;
    VTKUtils::read_vtk(vtk_filename,
                       serial_dof_handler,
                       serial_coefficients,
                       catalog_);
    assert_same_mesh(serial_triangulation, triangulation);

    finite_element_ = serial_dof_handler.get_fe().clone();
    dof_handler_.distribute_dofs(*finite_element_);
    locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
    coefficients_->reinit(locally_owned_dofs_, communicator_);
    AssertDimension(serial_coefficients.size(), serial_dof_handler.n_dofs());
    AssertDimension(dof_handler_.n_dofs(), serial_dof_handler.n_dofs());
    std::vector<dealii::types::global_dof_index> serial_indices(
      serial_dof_handler.get_fe().n_dofs_per_cell());
    std::vector<dealii::types::global_dof_index> target_indices(
      dof_handler_.get_fe().n_dofs_per_cell());
    auto serial_cell = serial_dof_handler.begin_active();
    for (const auto &target_cell : dof_handler_.active_cell_iterators())
      if (target_cell->is_locally_owned())
        {
          while (serial_cell->id() < target_cell->id())
            ++serial_cell;
          AssertThrow(serial_cell->id() == target_cell->id(),
                      dealii::ExcMessage(
                        "Imported field DoF transfer encountered a missing "
                        "active cell."));
          serial_cell->get_dof_indices(serial_indices);
          target_cell->get_dof_indices(target_indices);
          for (unsigned int i = 0; i < target_indices.size(); ++i)
            if (coefficients_->locally_owned_elements().is_element(
                  target_indices[i]))
              (*coefficients_)[target_indices[i]] =
                serial_coefficients[serial_indices[i]];
        }
    coefficients_->compress(dealii::VectorOperation::insert);

#else
    (void)vtk_filename;
    AssertThrow(false,
                dealii::ExcMessage(
                  "Importing finite-element fields requires VTK support."));
#endif
  }

  template <int dim, int spacedim>
  typename ImportedFiniteElementFields<dim, spacedim>::FieldView
  ImportedFiniteElementFields<dim, spacedim>::field(
    const std::string &name,
    const unsigned int component) const
  {
    const auto it = std::find_if(catalog_.begin(),
                                 catalog_.end(),
                                 [&name](const auto &descriptor) {
                                   return descriptor.name == name;
                                 });
    AssertThrow(it != catalog_.end(),
                dealii::ExcMessage("Unknown imported field '" + name + "'."));
    AssertThrow(component < it->n_components,
                dealii::ExcMessage("Invalid component for imported field '" +
                                   name + "'."));

    const auto           component_offset = it->first_fe_component + component;
    const Representation component_representation(
      *triangulation_,
      dof_handler_,
      locally_owned_dofs_,
      locally_relevant_dofs_,
      constraints_,
      dealii::StaticMappingQ1<dim, spacedim>::mapping,
      dealii::FEValuesExtractors::Scalar(component_offset));
    return FieldView(component_representation, coefficients_, *it, component);
  }

  /// @cond DOXYGEN_IGNORE_EXPLICIT_INSTANTIATIONS
  template class ImportedFiniteElementFields<1, 1>;
  template class ImportedFiniteElementFields<1, 2>;
  template class ImportedFiniteElementFields<1, 3>;
  template class ImportedFiniteElementFields<2, 2>;
  template class ImportedFiniteElementFields<2, 3>;
  template class ImportedFiniteElementFields<3, 3>;
  /// @endcond

} // namespace ImmersX
