// ---------------------------------------------------------------------
// Generic reduced-field finite-element and transfer utilities.
// ---------------------------------------------------------------------

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/reduced_field_utils.h>

#include <map>

namespace ImmersX
{
  namespace ReducedFieldUtils
  {
    template <int dim, int spacedim>
    std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
    field_catalog_to_finite_element(const FieldCatalog &catalog,
                                    const bool          simplex_mesh)
    {
      std::vector<std::shared_ptr<dealii::FiniteElement<dim, spacedim>>>
        field_fes;
      field_fes.reserve(catalog.size());
      for (const auto &field : catalog)
        {
          AssertThrow(field.n_components > 0,
                      dealii::ExcMessage(
                        "Reduced field components must be positive."));
          std::unique_ptr<dealii::FiniteElement<dim, spacedim>> scalar;
          if (field.association == FieldAssociation::point_data)
            {
              if (simplex_mesh)
                scalar =
                  std::make_unique<dealii::FE_SimplexP<dim, spacedim>>(1);
              else
                scalar = std::make_unique<dealii::FE_Q<dim, spacedim>>(1);
            }
          else if (field.association == FieldAssociation::cell_data)
            {
              if (simplex_mesh)
                scalar =
                  std::make_unique<dealii::FE_SimplexDGP<dim, spacedim>>(0);
              else
                scalar = std::make_unique<dealii::FE_DGQ<dim, spacedim>>(0);
            }
          AssertThrow(scalar != nullptr, dealii::ExcNotImplemented());
          if (field.n_components == 1)
            field_fes.push_back(std::move(scalar));
          else
            field_fes.push_back(
              std::make_shared<dealii::FESystem<dim, spacedim>>(
                *scalar, field.n_components));
        }

      if (field_fes.empty())
        return std::make_unique<dealii::FE_Nothing<dim, spacedim>>(
          simplex_mesh ? dealii::ReferenceCells::get_simplex<dim>() :
                         dealii::ReferenceCells::get_hypercube<dim>());

      std::vector<const dealii::FiniteElement<dim, spacedim> *> field_fe_ptrs;
      field_fe_ptrs.reserve(field_fes.size());
      for (const auto &field_fe : field_fes)
        field_fe_ptrs.push_back(field_fe.get());
      return std::make_unique<dealii::FESystem<dim, spacedim>>(
        field_fe_ptrs, std::vector<unsigned int>(field_fe_ptrs.size(), 1));
    }

    namespace internal
    {
      template <int dim, int spacedim, typename ParallelVectorType>
      void
      serial_vector_to_distributed_vector(
        const dealii::DoFHandler<dim, spacedim> &serial_dh,
        const dealii::DoFHandler<dim, spacedim> &parallel_dh,
        const dealii::Vector<double>            &serial_vec,
        ParallelVectorType                      &parallel_vec)
      {
        AssertDimension(serial_vec.size(), serial_dh.n_dofs());
        AssertDimension(parallel_vec.size(), parallel_dh.n_dofs());
        AssertDimension(parallel_dh.n_dofs(), serial_dh.n_dofs());
        AssertThrow(serial_dh.get_fe() == parallel_dh.get_fe(),
                    dealii::ExcMessage(
                      "The finite element systems of the serial and "
                      "parallel DoFHandlers must be the same."));

        std::vector<dealii::types::global_dof_index> serial_dof_indices(
          serial_dh.get_fe().n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> parallel_dof_indices(
          parallel_dh.get_fe().n_dofs_per_cell());

        std::map<
          dealii::CellId,
          typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator>
          serial_cells;
        for (const auto &cell : serial_dh.active_cell_iterators())
          serial_cells.emplace(cell->id(), cell);
        auto parallel_cell = parallel_dh.begin_active();
        for (; parallel_cell != parallel_dh.end(); ++parallel_cell)
          if (parallel_cell->is_locally_owned())
            {
              // Active iteration is level-wise, not ordered by CellId on a
              // locally refined mesh. Also retain a checked failure for input
              // meshes with missing cells instead of a debug iterator
              // assertion.
              const auto found = serial_cells.find(parallel_cell->id());
              AssertThrow(
                found != serial_cells.end(),
                dealii::ExcMessage(
                  "The serial DoFHandler has no active cell with CellId " +
                  parallel_cell->id().to_string() + "."));
              const auto serial_cell = found->second;
              AssertThrow(
                serial_cell->id() == parallel_cell->id(),
                dealii::ExcMessage(
                  "Serial and parallel DoFHandlers have different active "
                  "CellIds: serial " +
                  serial_cell->id().to_string() + ", parallel " +
                  parallel_cell->id().to_string() + "."));

              serial_cell->get_dof_indices(serial_dof_indices);
              parallel_cell->get_dof_indices(parallel_dof_indices);
              unsigned int serial_index = 0;
              for (const auto i : parallel_dof_indices)
                {
                  if (parallel_vec.in_local_range(i))
                    parallel_vec[i] =
                      serial_vec[serial_dof_indices[serial_index]];
                  ++serial_index;
                }
            }
        parallel_vec.compress(dealii::VectorOperation::insert);
      }
    } // namespace internal

    template <int dim, int spacedim>
    void
    serial_vector_to_distributed_vector(
      const dealii::DoFHandler<dim, spacedim>            &serial_dh,
      const dealii::DoFHandler<dim, spacedim>            &parallel_dh,
      const dealii::Vector<double>                       &serial_vec,
      dealii::LinearAlgebra::distributed::Vector<double> &parallel_vec)
    {
      internal::serial_vector_to_distributed_vector(serial_dh,
                                                    parallel_dh,
                                                    serial_vec,
                                                    parallel_vec);
    }

    template <int dim, int spacedim>
    void
    serial_vector_to_distributed_vector(
      const dealii::DoFHandler<dim, spacedim> &serial_dh,
      const dealii::DoFHandler<dim, spacedim> &parallel_dh,
      const dealii::Vector<double>            &serial_vec,
      ImmersXLA::MPI::Vector                  &parallel_vec)
    {
      internal::serial_vector_to_distributed_vector(serial_dh,
                                                    parallel_dh,
                                                    serial_vec,
                                                    parallel_vec);
    }

    template <int dim, int spacedim>
    std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices(
      const dealii::Triangulation<dim, spacedim> &serial_tria,
      const dealii::Triangulation<dim, spacedim> &parallel_tria)
    {
      const auto locally_owned_indices =
        dealii::GridTools::get_locally_owned_vertices(parallel_tria);
      std::vector<dealii::types::global_vertex_index> result(
        parallel_tria.n_vertices(), dealii::numbers::invalid_unsigned_int);

      auto serial_cell   = serial_tria.begin_active();
      auto parallel_cell = parallel_tria.begin_active();
      for (; parallel_cell != parallel_tria.end(); ++parallel_cell)
        if (parallel_cell->is_locally_owned())
          {
            while (serial_cell != serial_tria.end() &&
                   serial_cell->id() < parallel_cell->id())
              ++serial_cell;
            AssertThrow(
              serial_cell != serial_tria.end(),
              dealii::ExcMessage(
                "The serial triangulation has no active cell with CellId " +
                parallel_cell->id().to_string() + "."));
            AssertThrow(
              serial_cell->id() == parallel_cell->id(),
              dealii::ExcMessage(
                "Serial and parallel triangulations have different active "
                "CellIds: serial " +
                serial_cell->id().to_string() + ", parallel " +
                parallel_cell->id().to_string() + "."));
            for (const unsigned int v : serial_cell->vertex_indices())
              {
                const auto serial_index   = serial_cell->vertex_index(v);
                const auto parallel_index = parallel_cell->vertex_index(v);
                if (locally_owned_indices[parallel_index])
                  result[parallel_index] = serial_index;
              }
          }
      return result;
    }

    /// @cond DOXYGEN_IGNORE_EXPLICIT_INSTANTIATIONS
    template std::unique_ptr<dealii::FiniteElement<1, 1>>
    field_catalog_to_finite_element<1, 1>(const FieldCatalog &, const bool);
    template std::unique_ptr<dealii::FiniteElement<1, 2>>
    field_catalog_to_finite_element<1, 2>(const FieldCatalog &, const bool);
    template std::unique_ptr<dealii::FiniteElement<1, 3>>
    field_catalog_to_finite_element<1, 3>(const FieldCatalog &, const bool);
    template std::unique_ptr<dealii::FiniteElement<2, 2>>
    field_catalog_to_finite_element<2, 2>(const FieldCatalog &, const bool);
    template std::unique_ptr<dealii::FiniteElement<2, 3>>
    field_catalog_to_finite_element<2, 3>(const FieldCatalog &, const bool);
    template std::unique_ptr<dealii::FiniteElement<3, 3>>
    field_catalog_to_finite_element<3, 3>(const FieldCatalog &, const bool);

    template void
    serial_vector_to_distributed_vector<1, 1>(
      const dealii::DoFHandler<1, 1> &,
      const dealii::DoFHandler<1, 1> &,
      const dealii::Vector<double> &,
      dealii::LinearAlgebra::distributed::Vector<double> &);
    template void
    serial_vector_to_distributed_vector<1, 2>(
      const dealii::DoFHandler<1, 2> &,
      const dealii::DoFHandler<1, 2> &,
      const dealii::Vector<double> &,
      dealii::LinearAlgebra::distributed::Vector<double> &);
    template void
    serial_vector_to_distributed_vector<1, 3>(
      const dealii::DoFHandler<1, 3> &,
      const dealii::DoFHandler<1, 3> &,
      const dealii::Vector<double> &,
      dealii::LinearAlgebra::distributed::Vector<double> &);
    template void
    serial_vector_to_distributed_vector<2, 2>(
      const dealii::DoFHandler<2, 2> &,
      const dealii::DoFHandler<2, 2> &,
      const dealii::Vector<double> &,
      dealii::LinearAlgebra::distributed::Vector<double> &);
    template void
    serial_vector_to_distributed_vector<2, 3>(
      const dealii::DoFHandler<2, 3> &,
      const dealii::DoFHandler<2, 3> &,
      const dealii::Vector<double> &,
      dealii::LinearAlgebra::distributed::Vector<double> &);
    template void
    serial_vector_to_distributed_vector<3, 3>(
      const dealii::DoFHandler<3, 3> &,
      const dealii::DoFHandler<3, 3> &,
      const dealii::Vector<double> &,
      dealii::LinearAlgebra::distributed::Vector<double> &);

    template void
    serial_vector_to_distributed_vector<1, 1>(const dealii::DoFHandler<1, 1> &,
                                              const dealii::DoFHandler<1, 1> &,
                                              const dealii::Vector<double> &,
                                              ImmersXLA::MPI::Vector &);
    template void
    serial_vector_to_distributed_vector<1, 2>(const dealii::DoFHandler<1, 2> &,
                                              const dealii::DoFHandler<1, 2> &,
                                              const dealii::Vector<double> &,
                                              ImmersXLA::MPI::Vector &);
    template void
    serial_vector_to_distributed_vector<1, 3>(const dealii::DoFHandler<1, 3> &,
                                              const dealii::DoFHandler<1, 3> &,
                                              const dealii::Vector<double> &,
                                              ImmersXLA::MPI::Vector &);
    template void
    serial_vector_to_distributed_vector<2, 2>(const dealii::DoFHandler<2, 2> &,
                                              const dealii::DoFHandler<2, 2> &,
                                              const dealii::Vector<double> &,
                                              ImmersXLA::MPI::Vector &);
    template void
    serial_vector_to_distributed_vector<2, 3>(const dealii::DoFHandler<2, 3> &,
                                              const dealii::DoFHandler<2, 3> &,
                                              const dealii::Vector<double> &,
                                              ImmersXLA::MPI::Vector &);
    template void
    serial_vector_to_distributed_vector<3, 3>(const dealii::DoFHandler<3, 3> &,
                                              const dealii::DoFHandler<3, 3> &,
                                              const dealii::Vector<double> &,
                                              ImmersXLA::MPI::Vector &);

    template std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices<1, 1>(
      const dealii::Triangulation<1, 1> &,
      const dealii::Triangulation<1, 1> &);
    template std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices<1, 2>(
      const dealii::Triangulation<1, 2> &,
      const dealii::Triangulation<1, 2> &);
    template std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices<1, 3>(
      const dealii::Triangulation<1, 3> &,
      const dealii::Triangulation<1, 3> &);
    template std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices<2, 2>(
      const dealii::Triangulation<2, 2> &,
      const dealii::Triangulation<2, 2> &);
    template std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices<2, 3>(
      const dealii::Triangulation<2, 3> &,
      const dealii::Triangulation<2, 3> &);
    template std::vector<dealii::types::global_vertex_index>
    distributed_to_serial_vertex_indices<3, 3>(
      const dealii::Triangulation<3, 3> &,
      const dealii::Triangulation<3, 3> &);
    /// @endcond
  } // namespace ReducedFieldUtils
} // namespace ImmersX
