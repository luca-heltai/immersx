#include "vtk_utils.h"

#include <set>
#include <string>
#include <vector>

#include "reduced_field_utils.h"

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

#  include <vtkCell.h>
#  include <vtkCellData.h>
#  include <vtkDataArray.h>
#  include <vtkInformation.h>
#  include <vtkPointData.h>
#  include <vtkSmartPointer.h>
#  include <vtkStreamingDemandDrivenPipeline.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkUnstructuredGridReader.h>
#  include <vtkXMLPUnstructuredGridReader.h>
#  include <vtkXMLUnstructuredGridReader.h>

#  include <algorithm>
#  include <cctype>
#  include <limits>
#  include <stdexcept>
#  include <utility>

namespace VTKUtils
{
  namespace
  {
    vtkSmartPointer<vtkUnstructuredGrid>
    read_unstructured_grid(const std::string &filename,
                           const unsigned int requested_piece =
                             std::numeric_limits<unsigned int>::max(),
                           const unsigned int n_requested_pieces = 1)
    {
      std::string extension;
      const auto  dot = filename.find_last_of('.');
      if (dot != std::string::npos)
        extension = filename.substr(dot + 1);
      std::transform(extension.begin(),
                     extension.end(),
                     extension.begin(),
                     [](const unsigned char c) { return std::tolower(c); });

      auto copy_output = [](vtkUnstructuredGrid *output) {
        AssertThrow(output != nullptr,
                    ExcMessage("Failed to read VTK unstructured grid."));
        auto result = vtkSmartPointer<vtkUnstructuredGrid>::New();
        result->ShallowCopy(output);
        return result;
      };
      if (extension == "vtu")
        {
          auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
          reader->SetFileName(filename.c_str());
          reader->Update();
          return copy_output(reader->GetOutput());
        }
      if (extension == "pvtu")
        {
          auto reader = vtkSmartPointer<vtkXMLPUnstructuredGridReader>::New();
          reader->SetFileName(filename.c_str());
          if (requested_piece != std::numeric_limits<unsigned int>::max())
            {
              AssertThrow(n_requested_pieces > 0,
                          ExcMessage("PVTU piece count must be positive."));
              reader->UpdateInformation();
              auto *information = reader->GetOutputInformation(0);
              information->Set(
                vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER(),
                static_cast<int>(requested_piece));
              information->Set(
                vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES(),
                static_cast<int>(n_requested_pieces));
            }
          reader->Update();
          return copy_output(reader->GetOutput());
        }
      if (extension == "vtk")
        {
          auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
          reader->SetFileName(filename.c_str());
          reader->Update();
          return copy_output(reader->GetOutput());
        }
      AssertThrow(false,
                  ExcMessage(
                    "Unsupported VTK unstructured-grid extension in '" +
                    filename + "'; expected .vtk, .vtu, or .pvtu."));
      return nullptr;
    }

    void
    append_point_cloud_array(vtkDataArray                     *array,
                             const FieldAssociation            association,
                             const vtkIdType                   expected_tuples,
                             FieldCatalog                     &catalog,
                             std::vector<std::vector<double>> &properties,
                             std::vector<std::string>         &names)
    {
      AssertThrow(array != nullptr, ExcMessage("Null VTK data array."));
      const char *raw_name = array->GetName();
      AssertThrow(raw_name != nullptr && *raw_name != '\0',
                  ExcMessage("VTK data arrays must have non-empty names."));
      const std::string name(raw_name);
      for (const auto &field : catalog)
        AssertThrow(field.name != name || field.association != association,
                    ExcMessage("Duplicate VTK array name '" + name + "'."));
      const int n_components = array->GetNumberOfComponents();
      AssertThrow(n_components > 0,
                  ExcMessage("VTK array '" + name + "' has no components."));
      AssertThrow(array->GetNumberOfTuples() == expected_tuples,
                  ExcMessage("VTK array '" + name +
                             "' has an invalid tuple count."));
      FieldDescriptor descriptor;
      descriptor.name         = name;
      descriptor.association  = association;
      descriptor.n_components = static_cast<unsigned int>(n_components);
      descriptor.first_fe_component =
        catalog.empty() ?
          0 :
          catalog.back().first_fe_component + catalog.back().n_components;
      descriptor.block_index = static_cast<unsigned int>(catalog.size());
      catalog.push_back(descriptor);
      names.push_back(name);
      properties.emplace_back(static_cast<std::size_t>(expected_tuples) *
                              static_cast<std::size_t>(n_components));
    }
  } // namespace

  template <int spacedim>
  void
  read_vtk_point_cloud(const std::string    &vtk_filename,
                       PointCloud<spacedim> &point_cloud,
                       const unsigned int    requested_piece,
                       const unsigned int    n_requested_pieces)
  {
    const auto grid =
      read_unstructured_grid(vtk_filename, requested_piece, n_requested_pieces);
    AssertThrow(grid->GetPoints() != nullptr,
                ExcMessage("VTK particle file has no points: " + vtk_filename));
    AssertThrow(spacedim <= 3,
                ExcMessage(
                  "VTK point coordinates support at most 3 dimensions."));
    const vtkIdType n_points = grid->GetNumberOfPoints();
    const vtkIdType n_cells  = grid->GetNumberOfCells();
    AssertThrow(
      n_points > 0 && n_cells > 0,
      ExcMessage(
        "VTK particle file must contain points and VTK_VERTEX cells."));
    std::vector<vtkIdType> point_to_cell(n_points, -1);
    for (vtkIdType cell_index = 0; cell_index < n_cells; ++cell_index)
      {
        vtkCell *cell = grid->GetCell(cell_index);
        AssertThrow(
          cell->GetCellType() == VTK_VERTEX && cell->GetNumberOfPoints() == 1,
          ExcMessage("VTK particle input may contain only VTK_VERTEX cells; "
                     "cell " +
                     std::to_string(cell_index) + " is unsupported."));
        const vtkIdType point = cell->GetPointId(0);
        AssertThrow(point >= 0 && point < n_points,
                    ExcMessage("VTK_VERTEX references an invalid point."));
        AssertThrow(point_to_cell[point] == -1,
                    ExcMessage(
                      "Each particle point must be referenced by exactly one "
                      "VTK_VERTEX cell."));
        point_to_cell[point] = cell_index;
      }
    for (vtkIdType point = 0; point < n_points; ++point)
      AssertThrow(
        point_to_cell[point] != -1,
        ExcMessage(
          "Every particle point must be referenced by a VTK_VERTEX cell."));

    point_cloud = {};
    point_cloud.points.resize(n_points);
    for (vtkIdType point = 0; point < n_points; ++point)
      {
        double coordinates[3] = {0., 0., 0.};
        grid->GetPoints()->GetPoint(point, coordinates);
        for (unsigned int d = 0; d < spacedim; ++d)
          point_cloud.points[point][d] = coordinates[d];
      }
    auto append = [&](vtkDataArray          *array,
                      const FieldAssociation association,
                      const vtkIdType        expected) {
      append_point_cloud_array(array,
                               association,
                               expected,
                               point_cloud.catalog,
                               point_cloud.properties,
                               point_cloud.property_names);
    };
    if (auto *data = grid->GetPointData())
      for (int i = 0; i < data->GetNumberOfArrays(); ++i)
        append(data->GetArray(i), FieldAssociation::point_data, n_points);
    const std::size_t point_field_count = point_cloud.properties.size();
    if (auto *data = grid->GetCellData())
      for (int i = 0; i < data->GetNumberOfArrays(); ++i)
        append(data->GetArray(i), FieldAssociation::cell_data, n_cells);

    for (std::size_t field = 0; field < point_field_count; ++field)
      {
        vtkDataArray *array = grid->GetPointData()->GetArray(
          point_cloud.catalog[field].name.c_str());
        const unsigned int components = point_cloud.catalog[field].n_components;
        for (vtkIdType point = 0; point < n_points; ++point)
          for (unsigned int component = 0; component < components; ++component)
            point_cloud.properties[field][point * components + component] =
              array->GetComponent(point, component);
      }
    for (std::size_t field = point_field_count;
         field < point_cloud.properties.size();
         ++field)
      {
        const auto   &descriptor = point_cloud.catalog[field];
        vtkDataArray *array =
          grid->GetCellData()->GetArray(descriptor.name.c_str());
        for (vtkIdType point = 0; point < n_points; ++point)
          {
            const vtkIdType cell = point_to_cell[point];
            for (unsigned int component = 0;
                 component < descriptor.n_components;
                 ++component)
              point_cloud.properties[field][point * descriptor.n_components +
                                            component] =
                array->GetComponent(cell, component);
          }
      }
  }

  template <int spacedim>
  void
  read_vtk_point_cloud(const std::string                &vtk_filename,
                       std::vector<Point<spacedim>>     &points,
                       std::vector<std::vector<double>> &properties,
                       FieldCatalog                     &catalog,
                       std::vector<std::string>         &property_names)
  {
    PointCloud<spacedim> data;
    read_vtk_point_cloud(vtk_filename, data);
    points         = std::move(data.points);
    properties     = std::move(data.properties);
    catalog        = std::move(data.catalog);
    property_names = std::move(data.property_names);
  }

  template <int spacedim>
  void
  read_vtk_point_cloud(const std::string            &vtk_filename,
                       std::vector<Point<spacedim>> &points,
                       Vector<double>               &properties,
                       FieldCatalog                 &catalog,
                       std::vector<std::string>     &property_names)
  {
    std::vector<std::vector<double>> field_properties;
    read_vtk_point_cloud(
      vtk_filename, points, field_properties, catalog, property_names);
    std::size_t total = 0;
    for (const auto &field : field_properties)
      total += field.size();
    properties.reinit(total);
    std::size_t offset = 0;
    for (const auto &field : field_properties)
      {
        std::copy(field.begin(), field.end(), properties.begin() + offset);
        offset += field.size();
      }
  }

  template <int dim, int spacedim>
  void
  read_vtk(const std::string            &vtk_filename,
           Triangulation<dim, spacedim> &tria,
           const bool)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));

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
  std::unique_ptr<FiniteElement<dim, spacedim>>
  vtk_to_finite_element(const std::string &vtk_filename, FieldCatalog &catalog)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));
    catalog.clear();
    std::set<std::string> names[2];
    unsigned int          first_component = 0;
    const auto            append          = [&](vtkDataArray          *array,
                            const FieldAssociation association,
                            const vtkIdType        expected_tuples) {
      AssertThrow(array != nullptr, ExcMessage("Null VTK data array."));
      const char *raw_name = array->GetName();
      AssertThrow(raw_name != nullptr && *raw_name != '\0',
                  ExcMessage("VTK data arrays must have non-empty names."));
      const std::string  name(raw_name);
      const unsigned int association_index =
        association == FieldAssociation::point_data ? 0 : 1;
      AssertThrow(names[association_index].insert(name).second,
                  ExcMessage("Duplicate VTK array name '" + name + "'."));
      const int n_components = array->GetNumberOfComponents();
      AssertThrow(n_components > 0,
                  ExcMessage("VTK array '" + name + "' has no components."));
      AssertThrow(array->GetNumberOfTuples() == expected_tuples,
                  ExcMessage("VTK array '" + name +
                             "' has an invalid tuple count."));
      FieldDescriptor descriptor;
      descriptor.name        = name;
      descriptor.association = association;
      descriptor.n_components = static_cast<unsigned int>(n_components);
      descriptor.first_fe_component = first_component;
      descriptor.block_index = static_cast<unsigned int>(catalog.size());
      catalog.push_back(descriptor);
      first_component += descriptor.n_components;
    };
    if (vtkPointData *point_data = grid->GetPointData())
      for (int i = 0; i < point_data->GetNumberOfArrays(); ++i)
        append(point_data->GetArray(i),
               FieldAssociation::point_data,
               grid->GetNumberOfPoints());
    if (vtkCellData *cell_data = grid->GetCellData())
      for (int i = 0; i < cell_data->GetNumberOfArrays(); ++i)
        append(cell_data->GetArray(i),
               FieldAssociation::cell_data,
               grid->GetNumberOfCells());
    return ReducedFieldUtils::field_catalog_to_finite_element<dim, spacedim>(
      catalog);
  }

  /** Read coefficients in exactly the order established by a catalogue. */
  void
  read_catalogued_data(const std::string  &vtk_filename,
                       const FieldCatalog &catalog,
                       Vector<double>     &output_vector)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));
    std::vector<double> data;
    for (const auto &field : catalog)
      {
        vtkDataArray *array =
          field.association == FieldAssociation::point_data ?
            grid->GetPointData()->GetArray(field.name.c_str()) :
            grid->GetCellData()->GetArray(field.name.c_str());
        AssertThrow(array != nullptr,
                    ExcMessage("Catalogue field '" + field.name +
                               "' is missing from VTK file."));
        AssertThrow(array->GetNumberOfComponents() ==
                      static_cast<int>(field.n_components),
                    ExcMessage(
                      "VTK field '" + field.name +
                      "' changed component count after catalogue creation."));
        const vtkIdType expected_tuples =
          field.association == FieldAssociation::point_data ?
            grid->GetNumberOfPoints() :
            grid->GetNumberOfCells();
        AssertThrow(array->GetNumberOfTuples() == expected_tuples,
                    ExcMessage("VTK field '" + field.name +
                               "' has an invalid tuple count."));
        const auto first = data.size();
        data.resize(first + expected_tuples * field.n_components);
        for (vtkIdType tuple = 0; tuple < expected_tuples; ++tuple)
          for (unsigned int component = 0; component < field.n_components;
               ++component)
            data[first + tuple * field.n_components + component] =
              array->GetComponent(tuple, component);
      }
    output_vector.reinit(data.size());
    std::copy(data.begin(), data.end(), output_vector.begin());
  }

  template <int dim, int spacedim>
  std::pair<std::unique_ptr<FiniteElement<dim, spacedim>>,
            std::vector<std::string>>
  vtk_to_finite_element(const std::string &vtk_filename)
  {
    FieldCatalog catalog;
    auto fe = vtk_to_finite_element<dim, spacedim>(vtk_filename, catalog);
    std::vector<std::string> data_names;
    data_names.reserve(catalog.size());
    for (const auto &field : catalog)
      data_names.push_back(field.name);
    return std::make_pair(std::move(fe), std::move(data_names));
  }



  template <int dim, int spacedim>
  void
  read_vtk(const std::string         &vtk_filename,
           DoFHandler<dim, spacedim> &dof_handler,
           Vector<double>            &output_vector,
           std::vector<std::string>  &data_names)
  {
    FieldCatalog catalog;
    read_vtk(vtk_filename, dof_handler, output_vector, catalog);
    data_names.clear();
    data_names.reserve(catalog.size());
    for (const auto &field : catalog)
      data_names.push_back(field.name);
  }

  template <int dim, int spacedim>
  void
  read_vtk(const std::string         &vtk_filename,
           DoFHandler<dim, spacedim> &dof_handler,
           Vector<double>            &output_vector,
           FieldCatalog              &catalog)
  {
    auto &tria = const_cast<Triangulation<dim, spacedim> &>(
      dof_handler.get_triangulation());
    auto parallel_tria =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria);
    AssertThrow(parallel_tria == nullptr,
                ExcMessage(
                  "The input triangulation must be a serial triangulation."));
    tria.clear();
    read_vtk(vtk_filename, tria, /*cleanup=*/true);
    auto fe = vtk_to_finite_element<dim, spacedim>(vtk_filename, catalog);
    Vector<double> raw_data_vector;
    read_catalogued_data(vtk_filename, catalog, raw_data_vector);
    dof_handler.distribute_dofs(*fe);
    output_vector.reinit(dof_handler.n_dofs());
    ReducedFieldUtils::data_to_dealii_vector(tria,
                                             raw_data_vector,
                                             dof_handler,
                                             output_vector);
    AssertDimension(dof_handler.n_dofs(), output_vector.size());
    AssertDimension(dof_handler.get_fe().n_blocks(), catalog.size());
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

template std::unique_ptr<FiniteElement<1, 1>>
VTKUtils::vtk_to_finite_element(const std::string &, FieldCatalog &);
template std::unique_ptr<FiniteElement<1, 2>>
VTKUtils::vtk_to_finite_element(const std::string &, FieldCatalog &);
template std::unique_ptr<FiniteElement<1, 3>>
VTKUtils::vtk_to_finite_element(const std::string &, FieldCatalog &);
template std::unique_ptr<FiniteElement<2, 2>>
VTKUtils::vtk_to_finite_element(const std::string &, FieldCatalog &);
template std::unique_ptr<FiniteElement<2, 3>>
VTKUtils::vtk_to_finite_element(const std::string &, FieldCatalog &);
template std::unique_ptr<FiniteElement<3, 3>>
VTKUtils::vtk_to_finite_element(const std::string &, FieldCatalog &);

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
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<1, 1> &,
                   Vector<double> &,
                   FieldCatalog &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<1, 2> &,
                   Vector<double> &,
                   FieldCatalog &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<1, 3> &,
                   Vector<double> &,
                   FieldCatalog &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<2, 2> &,
                   Vector<double> &,
                   FieldCatalog &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<2, 3> &,
                   Vector<double> &,
                   FieldCatalog &);
template void
VTKUtils::read_vtk(const std::string &,
                   DoFHandler<3, 3> &,
                   Vector<double> &,
                   FieldCatalog &);

template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               PointCloud<1> &,
                               const unsigned int,
                               const unsigned int);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               PointCloud<2> &,
                               const unsigned int,
                               const unsigned int);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               PointCloud<3> &,
                               const unsigned int,
                               const unsigned int);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               std::vector<Point<1>> &,
                               std::vector<std::vector<double>> &,
                               FieldCatalog &,
                               std::vector<std::string> &);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               std::vector<Point<2>> &,
                               std::vector<std::vector<double>> &,
                               FieldCatalog &,
                               std::vector<std::string> &);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               std::vector<Point<3>> &,
                               std::vector<std::vector<double>> &,
                               FieldCatalog &,
                               std::vector<std::string> &);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               std::vector<Point<1>> &,
                               Vector<double> &,
                               FieldCatalog &,
                               std::vector<std::string> &);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               std::vector<Point<2>> &,
                               Vector<double> &,
                               FieldCatalog &,
                               std::vector<std::string> &);
template void
VTKUtils::read_vtk_point_cloud(const std::string &,
                               std::vector<Point<3>> &,
                               Vector<double> &,
                               FieldCatalog &,
                               std::vector<std::string> &);

#endif // DEAL_II_WITH_VTK
