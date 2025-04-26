#include "vtk_utils.h"

#ifdef DEAL_II_WITH_VTK
#  include <vtkCellData.h>
#  include <vtkDataArray.h>
#  include <vtkPointData.h>
#  include <vtkSmartPointer.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkUnstructuredGridReader.h>

#  include <stdexcept>

namespace VTKUtils
{
  void
  read_cell_data(const std::string &vtk_filename,
                 const std::string &cell_data_name,
                 Vector<double>    &output_vector)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid,
                dealii::ExcMessage("Failed to read VTK file: " + vtk_filename));
    vtkDataArray *data_array =
      grid->GetCellData()->GetArray(cell_data_name.c_str());
    AssertThrow(data_array,
                dealii::ExcMessage("Cell data array '" + cell_data_name +
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
    AssertThrow(grid,
                dealii::ExcMessage("Failed to read VTK file: " + vtk_filename));
    vtkDataArray *data_array =
      grid->GetPointData()->GetArray(point_data_name.c_str());
    AssertThrow(data_array,
                dealii::ExcMessage("Point data array '" + point_data_name +
                                   "' not found in VTK file: " + vtk_filename));
    vtkIdType n_tuples     = data_array->GetNumberOfTuples();
    int       n_components = data_array->GetNumberOfComponents();
    output_vector.reinit(n_tuples * n_components);
    for (vtkIdType i = 0; i < n_tuples; ++i)
      for (int j = 0; j < n_components; ++j)
        output_vector[i * n_components + j] = data_array->GetComponent(i, j);
  }
} // namespace VTKUtils

#endif // DEAL_II_WITH_VTK