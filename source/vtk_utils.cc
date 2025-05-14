#include "vtk_utils.h"

#include <string>
#include <vector>

#ifdef DEAL_II_WITH_VTK
#  include <deal.II/distributed/fully_distributed_tria.h>

#  include <deal.II/dofs/dof_renumbering.h>

#  include <deal.II/fe/fe.h>
#  include <deal.II/fe/fe_dgq.h>
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
        AssertIndexRange(i, fe_system.n_blocks());

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
  distributed_vertex_indices_to_serial_vertex_indices(
    const Triangulation<dim, spacedim>               &serial_tria,
    const parallel::TriangulationBase<dim, spacedim> &parallel_tria)
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


  template <int dim>
  void
  fill_distributed_vector_from_serial(
    const IndexSet       &owned_dofs,
    const Vector<double> &serial_vec,
    const std::map<Point<dim>, types::global_dof_index, PointComparator<dim>>
                                               &serial_map,
    LinearAlgebra::distributed::Vector<double> &parallel_vec,
    const std::map<Point<dim>, types::global_dof_index, PointComparator<dim>>
            &parallel_map,
    MPI_Comm comm)
  {
    Assert(parallel_vec.size() == 0,
           ExcMessage("The parallel vector must be empty before filling it."));
    AssertThrow(owned_dofs.n_elements() > 0,
                ExcMessage("The owned DoF index set must not be empty."));
    // Initialize parallel layout of the vector using DoFHandler
    parallel_vec.reinit(owned_dofs, comm);

    // Transfer data from serial to parallel vector
    for (const auto &p_pair : parallel_map)
      {
        const auto &pt             = p_pair.first;
        const auto &parallel_index = p_pair.second;

        if (!owned_dofs.is_element(parallel_index))
          continue;

        auto it = serial_map.find(pt);
        if (it != serial_map.end())
          {
            types::global_dof_index serial_index = it->second;
            parallel_vec[parallel_index]         = serial_vec[serial_index];
          }
        else
          {
            std::cerr << "No match found for point: " << pt << std::endl;
            AssertThrow(false, ExcInternalError());
          }
      }

    parallel_vec.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  std::map<unsigned int, unsigned int>
  create_vertex_mapping(
    const Triangulation<dim, spacedim>                             &serial_tria,
    const parallel::fullydistributed::Triangulation<dim, spacedim> &dist_tria)
  {
    // Define a point comparator with tolerance for floating point comparison
    struct PointComparator
    {
      bool
      operator()(const Point<spacedim> &p1, const Point<spacedim> &p2) const
      {
        const double tolerance = 1e-12;
        for (unsigned int d = 0; d < spacedim; ++d)
          {
            if (std::abs(p1[d] - p2[d]) > tolerance)
              return p1[d] < p2[d];
          }
        return false; // Points are considered equal
      }
    };

    // Create maps from coordinates to indices
    std::map<Point<spacedim>, unsigned int, PointComparator>
      serial_point_to_index;
    std::map<Point<spacedim>, unsigned int, PointComparator>
      dist_point_to_index;

    // Fill serial map
    for (unsigned int i = 0; i < serial_tria.n_vertices(); ++i)
      {
        serial_point_to_index[serial_tria.get_vertices()[i]] = i;
      }

    // Get locally owned vertices
    std::vector<bool> locally_owned_vertices =
      GridTools::get_locally_owned_vertices(dist_tria);

    // Fill distributed map with only locally owned vertices
    const std::vector<Point<spacedim>> &distributed_vertices =
      dist_tria.get_vertices();
    for (unsigned int i = 0; i < dist_tria.n_vertices(); ++i)
      if (i < locally_owned_vertices.size() && locally_owned_vertices[i])
        dist_point_to_index[distributed_vertices[i]] = i;


    // Create the mapping from distributed vertex index to serial vertex index
    std::map<unsigned int, unsigned int> dist_to_serial_mapping;
    for (const auto &pair : dist_point_to_index)
      {
        const Point<spacedim> &point    = pair.first;
        const unsigned int     dist_idx = pair.second;

        auto it = serial_point_to_index.find(point);
        if (it != serial_point_to_index.end())
          {
            dist_to_serial_mapping[dist_idx] = it->second;
          }
      }

    return dist_to_serial_mapping;
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

template void
VTKUtils::fill_distributed_vector_from_serial<1>(
  const IndexSet &,
  const Vector<double> &,
  const std::map<Point<1>, types::global_dof_index, PointComparator<1>> &,
  LinearAlgebra::distributed::Vector<double> &,
  const std::map<Point<1>, types::global_dof_index, PointComparator<1>> &,
  MPI_Comm);

template void
VTKUtils::fill_distributed_vector_from_serial<2>(
  const IndexSet &,
  const Vector<double> &,
  const std::map<Point<2>, types::global_dof_index, PointComparator<2>> &,
  LinearAlgebra::distributed::Vector<double> &,
  const std::map<Point<2>, types::global_dof_index, PointComparator<2>> &,
  MPI_Comm);

template void
VTKUtils::fill_distributed_vector_from_serial<3>(
  const IndexSet &,
  const Vector<double> &,
  const std::map<Point<3>, types::global_dof_index, PointComparator<3>> &,
  LinearAlgebra::distributed::Vector<double> &,
  const std::map<Point<3>, types::global_dof_index, PointComparator<3>> &,
  MPI_Comm);

template std::map<unsigned int, unsigned int>
VTKUtils::create_vertex_mapping(
  const Triangulation<1, 1> &,
  const parallel::fullydistributed::Triangulation<1, 1> &);
template std::map<unsigned int, unsigned int>
VTKUtils::create_vertex_mapping(
  const Triangulation<1, 2> &,
  const parallel::fullydistributed::Triangulation<1, 2> &);
template std::map<unsigned int, unsigned int>
VTKUtils::create_vertex_mapping(
  const Triangulation<1, 3> &,
  const parallel::fullydistributed::Triangulation<1, 3> &);
template std::map<unsigned int, unsigned int>
VTKUtils::create_vertex_mapping(
  const Triangulation<2, 2> &,
  const parallel::fullydistributed::Triangulation<2, 2> &);
template std::map<unsigned int, unsigned int>
VTKUtils::create_vertex_mapping(
  const Triangulation<2, 3> &,
  const parallel::fullydistributed::Triangulation<2, 3> &);
template std::map<unsigned int, unsigned int>
VTKUtils::create_vertex_mapping(
  const Triangulation<3, 3> &,
  const parallel::fullydistributed::Triangulation<3, 3> &);

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
VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
  const Triangulation<1, 1> &,
  const parallel::TriangulationBase<1, 1> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
  const Triangulation<1, 2> &,
  const parallel::TriangulationBase<1, 2> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
  const Triangulation<1, 3> &,
  const parallel::TriangulationBase<1, 3> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
  const Triangulation<2, 2> &,
  const parallel::TriangulationBase<2, 2> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
  const Triangulation<2, 3> &,
  const parallel::TriangulationBase<2, 3> &);
template std::vector<types::global_vertex_index>
VTKUtils::distributed_vertex_indices_to_serial_vertex_indices(
  const Triangulation<3, 3> &,
  const parallel::TriangulationBase<3, 3> &);


#endif // DEAL_II_WITH_VTK