// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/tria.h>

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
  template <int dim, int spacedim>
  void
  ImportedFiniteElementFields<dim, spacedim>::Storage::prepare_for_refinement()
  {
    using DistributedTriangulation =
      dealii::parallel::distributed::Triangulation<dim, spacedim>;

    AssertThrow(dynamic_cast<const DistributedTriangulation *>(triangulation) !=
                  nullptr,
                dealii::ExcMessage(
                  "Imported finite-element fields support refinement only on a "
                  "parallel::distributed::Triangulation."));
    AssertThrow(refinement_transfer == nullptr,
                dealii::ExcMessage(
                  "Imported finite-element field refinement transfer is "
                  "already pending."));
    AssertThrow(finite_element != nullptr && coefficients != nullptr,
                dealii::ExcMessage(
                  "Imported finite-element field storage is not initialized "
                  "for refinement."));

    refinement_input = std::make_shared<VectorType>();
    refinement_input->reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             communicator);
    *refinement_input = *coefficients;
    refinement_input->update_ghost_values();

    refinement_transfer = std::make_shared<SolutionTransferType>(dof_handler);
    refinement_transfer->prepare_for_coarsening_and_refinement(
      *refinement_input);
  }

  template <int dim, int spacedim>
  void
  ImportedFiniteElementFields<dim, spacedim>::Storage::complete_refinement()
  {
    AssertThrow(refinement_transfer != nullptr,
                dealii::ExcMessage(
                  "Imported finite-element field refinement transfer was not "
                  "prepared."));
    AssertThrow(finite_element != nullptr && coefficients != nullptr,
                dealii::ExcMessage(
                  "Imported finite-element field storage is not initialized "
                  "for refinement."));

    dof_handler.distribute_dofs(*finite_element);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.clear();
    constraints.close();

    VectorType transferred;
    transferred.reinit(locally_owned_dofs, communicator);
    refinement_transfer->interpolate(transferred);
    transferred.compress(dealii::VectorOperation::insert);
    *coefficients = transferred;

    refinement_transfer.reset();
    refinement_input.reset();
  }

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
    : storage_(std::make_shared<Storage>(triangulation, communicator))
  {
    const auto storage      = std::const_pointer_cast<Storage>(storage_);
    auto       coefficients = std::make_shared<VectorType>();
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
                       storage->catalog);
    assert_same_mesh(serial_triangulation, triangulation);

    storage->finite_element = serial_dof_handler.get_fe().clone();
    storage->dof_handler.distribute_dofs(*storage->finite_element);
    storage->locally_owned_dofs = storage->dof_handler.locally_owned_dofs();
    storage->locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(storage->dof_handler);
    coefficients->reinit(storage->locally_owned_dofs, communicator);
    AssertDimension(serial_coefficients.size(), serial_dof_handler.n_dofs());
    AssertDimension(storage->dof_handler.n_dofs(), serial_dof_handler.n_dofs());
    std::vector<dealii::types::global_dof_index> serial_indices(
      serial_dof_handler.get_fe().n_dofs_per_cell());
    std::vector<dealii::types::global_dof_index> target_indices(
      storage->dof_handler.get_fe().n_dofs_per_cell());
    auto serial_cell = serial_dof_handler.begin_active();
    for (const auto &target_cell : storage->dof_handler.active_cell_iterators())
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
            if (coefficients->locally_owned_elements().is_element(
                  target_indices[i]))
              (*coefficients)[target_indices[i]] =
                serial_coefficients[serial_indices[i]];
        }
    coefficients->compress(dealii::VectorOperation::insert);
    storage->coefficients = std::move(coefficients);

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
    const auto it = std::find_if(storage_->catalog.begin(),
                                 storage_->catalog.end(),
                                 [&name](const auto &descriptor) {
                                   return descriptor.name == name;
                                 });
    AssertThrow(it != storage_->catalog.end(),
                dealii::ExcMessage("Unknown imported field '" + name + "'."));
    AssertThrow(component < it->n_components,
                dealii::ExcMessage("Invalid component for imported field '" +
                                   name + "'."));

    return FieldView(storage_, *it, component);
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
