#include "vtk_utils.h"

#include <string>
#include <vector>

#ifdef DEAL_II_WITH_VTK
#  include <deal.II/dofs/dof_renumbering.h>

#  include <deal.II/fe/fe.h>
#  include <deal.II/fe/fe_dgq.h>
#  include <deal.II/fe/fe_q.h>
#  include <deal.II/fe/fe_system.h>

#  include <deal.II/grid/cell_data.h>
#  include <deal.II/grid/grid_in.h>
#  include <deal.II/grid/grid_out.h>
#  include <deal.II/grid/tria.h>
#  include <deal.II/grid/tria_accessor.h>
#  include <deal.II/grid/tria_iterator.h>

#  include <vtkCellData.h>
#  include <vtkCleanUnstructuredGrid.h>
#  include <vtkDataArray.h>
#  include <vtkPointData.h>
#  include <vtkSmartPointer.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkUnstructuredGridReader.h>

#  include <stdexcept>

namespace VTKUtils
{
  template <int dim, int spacedim>
  void
  read_vtk(const std::string            &vtk_filename,
           Triangulation<dim, spacedim> &tria,
           const bool                    cleanup)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));

    auto cleaner = vtkSmartPointer<vtkCleanUnstructuredGrid>::New();

    // Cleanup the triangulation if requested
    if (cleanup)
      {
        cleaner->SetInputData(grid);
        cleaner->Update();
        grid = cleaner->GetOutput();
      }

    // Read points
    vtkPoints                   *vtk_points = grid->GetPoints();
    const vtkIdType              n_points   = vtk_points->GetNumberOfPoints();
    std::vector<Point<spacedim>> points(n_points);
    for (vtkIdType i = 0; i < n_points; ++i)
      {
        std::array<double, 3> coords = {0, 0, 0};
        vtk_points->GetPoint(i, coords.data());
        for (unsigned int d = 0; d < spacedim; ++d)
          points[i][d] = coords[d];
      }

    // Read cells
    std::vector<CellData<dim>> cells;
    const vtkIdType            n_cells = grid->GetNumberOfCells();
    for (vtkIdType i = 0; i < n_cells; ++i)
      {
        vtkCell *cell = grid->GetCell(i);
        if constexpr (dim == 1)
          {
            if (cell->GetCellType() != VTK_LINE)
              AssertThrow(false,
                          ExcMessage(
                            "Unsupported cell type in 1D VTK file: only "
                            "VTK_LINE is supported."));
            AssertThrow(cell->GetNumberOfPoints() == 2,
                        ExcMessage(
                          "Only line cells with 2 points are supported."));
            CellData<1> cell_data;
            for (unsigned int j = 0; j < 2; ++j)
              cell_data.vertices[j] = cell->GetPointId(j);
            cell_data.material_id = 0;
            cells.push_back(cell_data);
          }
        else if constexpr (dim == 2)
          {
            if (cell->GetCellType() == VTK_QUAD)
              {
                AssertThrow(cell->GetNumberOfPoints() == 4,
                            ExcMessage(
                              "Only quad cells with 4 points are supported."));
                CellData<2> cell_data;
                for (unsigned int j = 0; j < 4; ++j)
                  cell_data.vertices[j] = cell->GetPointId(j);
                cell_data.material_id = 0;
                cells.push_back(cell_data);
              }
            else if (cell->GetCellType() == VTK_TRIANGLE)
              {
                AssertThrow(
                  cell->GetNumberOfPoints() == 3,
                  ExcMessage(
                    "Only triangle cells with 3 points are supported."));
                CellData<2> cell_data;
                for (unsigned int j = 0; j < 3; ++j)
                  cell_data.vertices[j] = cell->GetPointId(j);
                cell_data.material_id = 0;
                cells.push_back(cell_data);
              }
            else
              AssertThrow(false,
                          ExcMessage(
                            "Unsupported cell type in 2D VTK file: only "
                            "VTK_QUAD and VTK_TRIANGLE are supported."));
          }
        else if constexpr (dim == 3)
          {
            if (cell->GetCellType() == VTK_HEXAHEDRON)
              {
                AssertThrow(cell->GetNumberOfPoints() == 8,
                            ExcMessage(
                              "Only hex cells with 8 points are supported."));
                CellData<3> cell_data;
                for (unsigned int j = 0; j < 8; ++j)
                  cell_data.vertices[j] = cell->GetPointId(j);
                cell_data.material_id = 0;
                // Numbering of vertices in VTK files is different from deal.II
                std::swap(cell_data.vertices[2], cell_data.vertices[3]);
                std::swap(cell_data.vertices[6], cell_data.vertices[7]);
                cells.push_back(cell_data);
              }
            else if (cell->GetCellType() == VTK_TETRA)
              {
                AssertThrow(
                  cell->GetNumberOfPoints() == 4,
                  ExcMessage(
                    "Only tetrahedron cells with 4 points are supported."));
                CellData<3> cell_data;
                for (unsigned int j = 0; j < 4; ++j)
                  cell_data.vertices[j] = cell->GetPointId(j);
                cell_data.material_id = 0;
                cells.push_back(cell_data);
              }
            else if (cell->GetCellType() == VTK_WEDGE)
              {
                AssertThrow(cell->GetNumberOfPoints() == 6,
                            ExcMessage(
                              "Only prism cells with 6 points are supported."));
                CellData<3> cell_data;
                for (unsigned int j = 0; j < 6; ++j)
                  cell_data.vertices[j] = cell->GetPointId(j);
                cell_data.material_id = 0;
                cells.push_back(cell_data);
              }
            else if (cell->GetCellType() == VTK_PYRAMID)
              {
                AssertThrow(
                  cell->GetNumberOfPoints() == 5,
                  ExcMessage(
                    "Only pyramid cells with 5 points are supported."));
                CellData<3> cell_data;
                for (unsigned int j = 0; j < 5; ++j)
                  cell_data.vertices[j] = cell->GetPointId(j);
                cell_data.material_id = 0;
                cells.push_back(cell_data);
              }
            else
              AssertThrow(
                false,
                ExcMessage(
                  "Unsupported cell type in 3D VTK file: only "
                  "VTK_HEXAHEDRON, VTK_TETRA, VTK_WEDGE, and VTK_PYRAMID are supported."));
          }
        else
          {
            AssertThrow(false, ExcMessage("Unsupported dimension."));
          }
      }

    // Create triangulation
    tria.create_triangulation(points, cells, SubCellData());
  }

  void
  read_cell_data(const std::string &vtk_filename,
                 const std::string &cell_data_name,
                 Vector<double>    &output_vector)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));
    vtkDataArray *data_array =
      grid->GetCellData()->GetArray(cell_data_name.c_str());
    AssertThrow(data_array,
                ExcMessage("Cell data array '" + cell_data_name +
                           "' not found in VTK file: " + vtk_filename));
    vtkIdType n_tuples     = data_array->GetNumberOfTuples();
    int       n_components = data_array->GetNumberOfComponents();
    output_vector.reinit(n_tuples * n_components);
    for (vtkIdType i = 0; i < n_tuples; ++i)
      for (int j = 0; j < n_components; ++j)
        output_vector[i * n_components + j] = data_array->GetComponent(i, j);
  }

  void
  read_point_data(const std::string &vtk_filename,
                  const std::string &point_data_name,
                  Vector<double>    &output_vector)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));
    vtkDataArray *data_array =
      grid->GetPointData()->GetArray(point_data_name.c_str());
    AssertThrow(data_array,
                ExcMessage("Point data array '" + point_data_name +
                           "' not found in VTK file: " + vtk_filename));
    vtkIdType n_tuples     = data_array->GetNumberOfTuples();
    int       n_components = data_array->GetNumberOfComponents();
    output_vector.reinit(n_tuples * n_components);
    for (vtkIdType i = 0; i < n_tuples; ++i)
      for (int j = 0; j < n_components; ++j)
        output_vector[i * n_components + j] = data_array->GetComponent(i, j);
  }

  template <int dim, int spacedim>
  void
  read_vtk(const std::string         &vtk_filename,
           DoFHandler<dim, spacedim> &dof_handler,
           Vector<double>            &output_vector,
           std::vector<std::string>  &data_names)
  {
    // Get a non-const reference to the triangulation
    auto &tria = const_cast<Triangulation<dim, spacedim> &>(
      dof_handler.get_triangulation());
    // Read the mesh from the VTK file
    read_vtk(vtk_filename, tria, /*cleanup=*/true);

    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));

    vtkCellData  *cell_data  = grid->GetCellData();
    vtkPointData *point_data = grid->GetPointData();

    std::vector<std::shared_ptr<FiniteElement<dim, spacedim>>> fe_collection;
    std::vector<unsigned int> n_components_collection;
    data_names.clear();

    // Query cell data fields
    for (int i = 0; i < cell_data->GetNumberOfArrays(); ++i)
      {
        vtkDataArray *arr = cell_data->GetArray(i);
        if (!arr)
          continue;
        std::string name   = arr->GetName();
        int         n_comp = arr->GetNumberOfComponents();
        fe_collection.push_back(
          std::make_shared<FESystem<dim, spacedim>>(FE_DGQ<dim, spacedim>(0),
                                                    n_comp));
        n_components_collection.push_back(n_comp);
        data_names.push_back(name);
      }
    // Query point data fields
    for (int i = 0; i < point_data->GetNumberOfArrays(); ++i)
      {
        vtkDataArray *arr = point_data->GetArray(i);
        if (!arr)
          continue;
        std::string name   = arr->GetName();
        int         n_comp = arr->GetNumberOfComponents();
        fe_collection.push_back(
          std::make_shared<FESystem<dim, spacedim>>(FE_Q<dim, spacedim>(1),
                                                    n_comp));
        n_components_collection.push_back(n_comp);
        data_names.push_back(name);
      }

    // Build a FESystem with all fields
    std::vector<const FiniteElement<dim, spacedim> *> fe_ptrs;
    std::vector<unsigned int>                         multiplicities;
    for (const auto &fe : fe_collection)
      {
        fe_ptrs.push_back(fe.get());
        multiplicities.push_back(1);
      }
    FESystem<dim, spacedim> fe_system(fe_ptrs, multiplicities);
    dof_handler.distribute_dofs(fe_system);
    DoFRenumbering::block_wise(dof_handler);

    // Read all data into output_vector
    output_vector.reinit(dof_handler.n_dofs());
    unsigned int dof_offset = 0;
    unsigned int field_idx  = 0;
    // Cell data
    for (int i = 0; i < cell_data->GetNumberOfArrays(); ++i, ++field_idx)
      {
        vtkDataArray *arr = cell_data->GetArray(i);
        if (!arr)
          continue;
        vtkIdType n_tuples = arr->GetNumberOfTuples();
        int       n_comp   = arr->GetNumberOfComponents();
        for (vtkIdType j = 0; j < n_tuples; ++j)
          for (int k = 0; k < n_comp; ++k)
            output_vector[dof_offset + j * n_comp + k] =
              arr->GetComponent(j, k);
        dof_offset += n_tuples * n_comp;
      }
    // Point data
    for (int i = 0; i < point_data->GetNumberOfArrays(); ++i, ++field_idx)
      {
        vtkDataArray *arr = point_data->GetArray(i);
        if (!arr)
          continue;
        vtkIdType n_tuples = arr->GetNumberOfTuples();
        int       n_comp   = arr->GetNumberOfComponents();
        for (vtkIdType j = 0; j < n_tuples; ++j)
          for (int k = 0; k < n_comp; ++k)
            output_vector[dof_offset + j * n_comp + k] =
              arr->GetComponent(j, k);
        dof_offset += n_tuples * n_comp;
      }
  }

  /**
   * Distribute the output_vector to all processors in the triangulation using
   * the partitioning of the tria.
   */
  template <int dim, int spacedim>
  void
  update_vector_layout(
    const Triangulation<dim, spacedim>         &tria,
    LinearAlgebra::distributed::Vector<double> &output_vector,
    const Vector<double>                       &local_vector)
  {
    Assert(tria.get_communicator() == output_vector.get_mpi_communicator(),
           ExcMessage(
             "The communicator of the triangulation and the vector must be "
             "the same."));
    Assert(local_vector.size() > 0,
           ExcMessage("The output vector must be empty before calling this "
                      "function."));

    // Distribute the output_vector to all processors in the triangulation
    // using the partitioning of the tria.
    const auto &partitioner =
      tria.global_active_cell_index_partitioner().lock();
    output_vector.reinit(partitioner, tria.get_communicator());

    // Set the locally owned elements to the values from the local_vector
    // and update the ghost values.
    for (unsigned int i = 0; i < output_vector.locally_owned_size(); ++i)
      output_vector.local_element(i) = local_vector(i);

    output_vector.compress(VectorOperation::insert);
  }

} // namespace VTKUtils

// Explicit instantiation for 1D, 2D and 3D

template void
VTKUtils::read_vtk(const std::string &, Triangulation<1, 1> &, const bool);
template void
VTKUtils::read_vtk(const std::string &, Triangulation<1, 2> &, const bool);
template void
VTKUtils::read_vtk(const std::string &, Triangulation<1, 3> &, const bool);
template void
VTKUtils::read_vtk(const std::string &, Triangulation<2, 2> &, const bool);
template void
VTKUtils::read_vtk(const std::string &, Triangulation<2, 3> &, const bool);
template void
VTKUtils::read_vtk(const std::string &, Triangulation<3, 3> &, const bool);

template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<1, 1> &,
                   Vector<double> &,
                   std::vector<std::string> &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<1, 2> &,
                   Vector<double> &,
                   std::vector<std::string> &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<1, 3> &,
                   Vector<double> &,
                   std::vector<std::string> &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<2, 2> &,
                   Vector<double> &,
                   std::vector<std::string> &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<2, 3> &,
                   Vector<double> &,
                   std::vector<std::string> &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<3, 3> &,
                   Vector<double> &,
                   std::vector<std::string> &);

#endif // DEAL_II_WITH_VTK