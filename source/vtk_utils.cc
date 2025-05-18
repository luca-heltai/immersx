#include "vtk_utils.h"

#include <string>
#include <vector>

#ifdef DEAL_II_WITH_VTK
#  include <deal.II/distributed/fully_distributed_tria.h>

#  include <deal.II/dofs/dof_renumbering.h>

#  include <deal.II/fe/fe.h>
#  include <deal.II/fe/fe_dgq.h>
#  include <deal.II/fe/fe_nothing.h>
#  include <deal.II/fe/fe_q.h>
#  include <deal.II/fe/fe_system.h>

#  include <deal.II/grid/grid_in.h>
#  include <deal.II/grid/grid_out.h>
#  include <deal.II/grid/grid_tools.h>
#  include <deal.II/grid/tria.h>
#  include <deal.II/grid/tria_accessor.h>
#  include <deal.II/grid/tria_description.h>
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
        AssertThrow(grid,
                    ExcMessage("Failed to clean VTK file: " + vtk_filename));
      }

    // Read points
    vtkPoints                   *vtk_points = grid->GetPoints();
    const vtkIdType              n_points   = vtk_points->GetNumberOfPoints();
    std::vector<Point<spacedim>> points(n_points);
    for (vtkIdType i = 0; i < n_points; ++i)
      {
        std::array<double, 3> coords = {{0, 0, 0}};
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
  read_vertex_data(const std::string &vtk_filename,
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

  void
  read_data(const std::string &vtk_filename, Vector<double> &output_vector)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));

    std::vector<double> data;

    vtkPointData *point_data = grid->GetPointData();
    if (point_data)
      {
        for (int i = 0; i < point_data->GetNumberOfArrays(); ++i)
          {
            vtkDataArray *data_array = point_data->GetArray(i);
            if (!data_array)
              continue;
            vtkIdType    n_tuples     = data_array->GetNumberOfTuples();
            int          n_components = data_array->GetNumberOfComponents();
            unsigned int current_size = data.size();
            data.resize(current_size + n_tuples * n_components, 0.0);
            for (vtkIdType tuple_idx = 0; tuple_idx < n_tuples; ++tuple_idx)
              for (int comp_idx = 0; comp_idx < n_components; ++comp_idx)
                data[current_size + tuple_idx * n_components + comp_idx] =
                  data_array->GetComponent(tuple_idx, comp_idx);
          }
      }

    vtkCellData *cell_data = grid->GetCellData();
    if (cell_data)
      {
        for (int i = 0; i < cell_data->GetNumberOfArrays(); ++i)
          {
            vtkDataArray *data_array = cell_data->GetArray(i);
            if (!data_array)
              continue;
            vtkIdType    n_tuples     = data_array->GetNumberOfTuples();
            int          n_components = data_array->GetNumberOfComponents();
            unsigned int current_size = data.size();
            data.resize(current_size + n_tuples * n_components, true);
            for (vtkIdType tuple_idx = 0; tuple_idx < n_tuples; ++tuple_idx)
              for (int comp_idx = 0; comp_idx < n_components; ++comp_idx)
                data[current_size + tuple_idx * n_components + comp_idx] =
                  data_array->GetComponent(tuple_idx, comp_idx);
          }
      }
    output_vector.reinit(data.size());
    std::copy(data.begin(), data.end(), output_vector.begin());
  }

  template <int dim, int spacedim>
  std::pair<std::unique_ptr<FiniteElement<dim, spacedim>>,
            std::vector<std::string>>
  vtk_to_finite_element(const std::string &vtk_filename)
  {
    std::vector<std::string> data_names;
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));

    vtkCellData  *cell_data  = grid->GetCellData();
    vtkPointData *point_data = grid->GetPointData();

    std::vector<std::shared_ptr<FiniteElement<dim, spacedim>>> fe_collection;
    std::vector<unsigned int> n_components_collection;

    // Query point data fields
    for (int i = 0; i < point_data->GetNumberOfArrays(); ++i)
      {
        vtkDataArray *arr = point_data->GetArray(i);
        if (!arr)
          continue;
        std::string name   = arr->GetName();
        int         n_comp = arr->GetNumberOfComponents();
        if (n_comp == 1)
          fe_collection.push_back(std::make_shared<FE_Q<dim, spacedim>>(1));
        else
          // Use FESystem for vector fields
          fe_collection.push_back(
            std::make_shared<FESystem<dim, spacedim>>(FE_Q<dim, spacedim>(1),
                                                      n_comp));
        n_components_collection.push_back(n_comp);
        data_names.push_back(name);
      }

    // Query cell data fields
    for (int i = 0; i < cell_data->GetNumberOfArrays(); ++i)
      {
        vtkDataArray *arr = cell_data->GetArray(i);
        if (!arr)
          continue;
        std::string name   = arr->GetName();
        int         n_comp = arr->GetNumberOfComponents();
        if (n_comp == 1)
          fe_collection.push_back(std::make_shared<FE_DGQ<dim, spacedim>>(0));
        else
          // Use FESystem for vector fields
          fe_collection.push_back(
            std::make_shared<FESystem<dim, spacedim>>(FE_DGQ<dim, spacedim>(0),
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
    if (fe_ptrs.empty())
      return std::make_pair(std::make_unique<FE_Nothing<dim, spacedim>>(),
                            std::vector<std::string>());
    else
      {
        return std::make_pair(
          std::make_unique<FESystem<dim, spacedim>>(fe_ptrs, multiplicities),
          data_names);
      }
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

    // Make sure the triangulation is actually a serial triangulation
    auto parallel_tria =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria);
    AssertThrow(parallel_tria == nullptr,
                ExcMessage(
                  "The input triangulation must be a serial triangulation."));

    // Clear the triangulation to ensure it is empty before reading
    tria.clear();
    // Read the mesh from the VTK file
    read_vtk(vtk_filename, tria, /*cleanup=*/true);

    Vector<double> raw_data_vector;
    read_data(vtk_filename, raw_data_vector);

    auto [fe, data_names_from_fe] =
      vtk_to_finite_element<dim, spacedim>(vtk_filename);

    dof_handler.distribute_dofs(*fe);
    output_vector.reinit(dof_handler.n_dofs());
    data_to_dealii_vector(tria, raw_data_vector, dof_handler, output_vector);

    AssertDimension(dof_handler.n_dofs(), output_vector.size());
    AssertDimension(dof_handler.get_fe().n_blocks(), data_names_from_fe.size());
    data_names = data_names_from_fe;
  }

  template <int dim, int spacedim>
  void
  serial_vector_to_distributed_vector(
    const DoFHandler<dim, spacedim>            &serial_dh,
    const DoFHandler<dim, spacedim>            &parallel_dh,
    const Vector<double>                       &serial_vec,
    LinearAlgebra::distributed::Vector<double> &parallel_vec)
  {
    AssertDimension(serial_vec.size(), serial_dh.n_dofs());
    AssertDimension(parallel_vec.size(), parallel_dh.n_dofs());
    AssertDimension(parallel_dh.n_dofs(), serial_dh.n_dofs());

    // Check that the two fe are the same
    AssertThrow(serial_dh.get_fe() == parallel_dh.get_fe(),
                ExcMessage("The finite element systems of the serial and "
                           "parallel DoFHandlers must be the same."));

    std::vector<types::global_dof_index> serial_dof_indices(
      serial_dh.get_fe().n_dofs_per_cell());
    std::vector<types::global_dof_index> parallel_dof_indices(
      parallel_dh.get_fe().n_dofs_per_cell());

    // Assumption: serial and parallel meshes have the same ordering of cells.
    auto serial_cell   = serial_dh.begin_active();
    auto parallel_cell = parallel_dh.begin_active();
    for (; parallel_cell != parallel_dh.end(); ++parallel_cell)
      if (parallel_cell->is_locally_owned())
        {
          // Advanced serial cell until we reach the same cell index of the
          // parallel cell
          while (serial_cell->id() < parallel_cell->id())
            ++serial_cell;
          serial_cell->get_dof_indices(serial_dof_indices);
          parallel_cell->get_dof_indices(parallel_dof_indices);
          unsigned int serial_index = 0;
          for (const auto &i : parallel_dof_indices)
            {
              if (parallel_vec.in_local_range(i))
                {
                  parallel_vec[i] =
                    serial_vec[serial_dof_indices[serial_index]];
                }
              ++serial_index;
            }
        }
    parallel_vec.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  std::vector<types::global_vertex_index>
  distributed_to_serial_vertex_indices(
    const Triangulation<dim, spacedim> &serial_tria,
    const Triangulation<dim, spacedim> &parallel_tria)
  {
    const auto locally_owned_indices =
      GridTools::get_locally_owned_vertices(parallel_tria);
    std::vector<types::global_vertex_index>
      distributed_to_serial_vertex_indices(parallel_tria.n_vertices(),
                                           numbers::invalid_unsigned_int);

    // Assumption: serial and parallel meshes have the same ordering of cells.
    auto serial_cell   = serial_tria.begin_active();
    auto parallel_cell = parallel_tria.begin_active();
    for (; parallel_cell != parallel_tria.end(); ++parallel_cell)
      if (parallel_cell->is_locally_owned())
        {
          // Advanced serial cell until we reach the same cell index of the
          // parallel cell
          while (serial_cell->id() < parallel_cell->id())
            ++serial_cell;
          for (const unsigned int &v : serial_cell->vertex_indices())
            {
              const auto serial_index   = serial_cell->vertex_index(v);
              const auto parallel_index = parallel_cell->vertex_index(v);
              if (locally_owned_indices[parallel_index])
                distributed_to_serial_vertex_indices[parallel_index] =
                  serial_index;
            }
        }
    return distributed_to_serial_vertex_indices;
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

template std::pair<std::unique_ptr<FiniteElement<1, 1>>,
                   std::vector<std::string>>
VTKUtils::vtk_to_finite_element(const std::string &);
template std::pair<std::unique_ptr<FiniteElement<1, 2>>,
                   std::vector<std::string>>
VTKUtils::vtk_to_finite_element(const std::string &);
template std::pair<std::unique_ptr<FiniteElement<1, 3>>,
                   std::vector<std::string>>
VTKUtils::vtk_to_finite_element(const std::string &);
template std::pair<std::unique_ptr<FiniteElement<2, 2>>,
                   std::vector<std::string>>
VTKUtils::vtk_to_finite_element(const std::string &);
template std::pair<std::unique_ptr<FiniteElement<2, 3>>,
                   std::vector<std::string>>
VTKUtils::vtk_to_finite_element(const std::string &);
template std::pair<std::unique_ptr<FiniteElement<3, 3>>,
                   std::vector<std::string>>
VTKUtils::vtk_to_finite_element(const std::string &);

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

template void
VTKUtils::serial_vector_to_distributed_vector(
  const DoFHandler<1, 1> &,
  const DoFHandler<1, 1> &,
  const Vector<double> &,
  LinearAlgebra::distributed::Vector<double> &);
template void
VTKUtils::serial_vector_to_distributed_vector(
  const DoFHandler<1, 2> &,
  const DoFHandler<1, 2> &,
  const Vector<double> &,
  LinearAlgebra::distributed::Vector<double> &);
template void
VTKUtils::serial_vector_to_distributed_vector(
  const DoFHandler<1, 3> &,
  const DoFHandler<1, 3> &,
  const Vector<double> &,
  LinearAlgebra::distributed::Vector<double> &);
template void
VTKUtils::serial_vector_to_distributed_vector(
  const DoFHandler<2, 2> &,
  const DoFHandler<2, 2> &,
  const Vector<double> &,
  LinearAlgebra::distributed::Vector<double> &);
template void
VTKUtils::serial_vector_to_distributed_vector(
  const DoFHandler<2, 3> &,
  const DoFHandler<2, 3> &,
  const Vector<double> &,
  LinearAlgebra::distributed::Vector<double> &);
template void
VTKUtils::serial_vector_to_distributed_vector(
  const DoFHandler<3, 3> &,
  const DoFHandler<3, 3> &,
  const Vector<double> &,
  LinearAlgebra::distributed::Vector<double> &);

template std::vector<types::global_vertex_index>
VTKUtils::distributed_to_serial_vertex_indices(const Triangulation<1, 1> &,
                                               const Triangulation<1, 1> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_to_serial_vertex_indices(const Triangulation<1, 2> &,
                                               const Triangulation<1, 2> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_to_serial_vertex_indices(const Triangulation<1, 3> &,
                                               const Triangulation<1, 3> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_to_serial_vertex_indices(const Triangulation<2, 2> &,
                                               const Triangulation<2, 2> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_to_serial_vertex_indices(const Triangulation<2, 3> &,
                                               const Triangulation<2, 3> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_to_serial_vertex_indices(const Triangulation<3, 3> &,
                                               const Triangulation<3, 3> &);


#endif // DEAL_II_WITH_VTK