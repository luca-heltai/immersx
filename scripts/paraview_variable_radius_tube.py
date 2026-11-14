# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ParaView plugin: generate tubes using a point or cell radius/area array."""

import math

from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkDataSet, vtkPolyData
from vtkmodules.vtkFiltersCore import (
    vtkArrayCalculator, vtkCellDataToPointData, vtkTubeFilter,
)
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter


@smproxy.filter(name="VariableRadiusTube", label="Variable Radius Tube")
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class VariableRadiusTube(VTKPythonAlgorithmBase):
    """Use radius directly, or calculate sqrt(area/pi) after cell averaging."""

    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=1,
                         inputType="vtkDataSet", outputType="vtkPolyData")
        self._area = "area"
        self._array = ""
        self._association = vtkDataObject.FIELD_ASSOCIATION_POINTS
        self._interpretation = 0
        self._sides = 24
        self._capping = True

    @smproperty.stringvector(name="AreaArray", default_values="area")
    def SetAreaArray(self, name):
        """Fallback area name when InputArray is empty and no radius exists."""
        self._area = name
        self.Modified()

    @smproperty.stringvector(name="InputArray", number_of_elements=5,
                             element_types="0 0 0 0 2",
                             default_values="0 0 0 0 ")
    @smdomain.xml('''<ArrayListDomain name="array_list" attribute_type="Scalars"
        input_domain_name="inputs" none_string="Automatic">
        <RequiredProperties><Property name="Input" function="Input"/>
        </RequiredProperties></ArrayListDomain>''')
    def SetInputArray(self, index, port, connection, association, name):
        """Explicit array name and Point/Cell Data association."""
        self._association = int(association)
        self._array = name or ""
        self.Modified()

    @smproperty.intvector(name="Interpretation", default_values=0)
    @smdomain.xml('''<EnumerationDomain name="enum">
        <Entry text="Automatic" value="0"/><Entry text="Radius" value="1"/>
        <Entry text="Area" value="2"/></EnumerationDomain>''')
    def SetInterpretation(self, value):
        if int(value) not in (0, 1, 2):
            raise ValueError("Interpretation must be Automatic, Radius or Area.")
        self._interpretation = int(value)
        self.Modified()

    @smproperty.intvector(name="NumberOfSides", default_values=24)
    @smdomain.intrange(min=3, max=256)
    def SetNumberOfSides(self, value):
        self._sides = max(3, min(256, int(value)))
        self.Modified()

    @smproperty.intvector(name="Capping", default_values=1)
    @smdomain.xml('<BooleanDomain name="bool"/>')
    def SetCapping(self, value):
        self._capping = bool(value)
        self.Modified()

    def _select_array(self, source):
        associations = (
            (vtkDataObject.FIELD_ASSOCIATION_POINTS, source.GetPointData(),
             source.GetNumberOfPoints()),
            (vtkDataObject.FIELD_ASSOCIATION_CELLS, source.GetCellData(),
             source.GetNumberOfCells()),
        )
        if self._array:
            for association, data, size in associations:
                if association == self._association:
                    return association, data.GetArray(self._array), size, self._array
            raise ValueError("InputArray must select Point Data or Cell Data.")
        for association, data, size in associations:
            for i in range(data.GetNumberOfArrays()):
                name = data.GetArrayName(i)
                if name and name.lower() == "radius":
                    return association, data.GetArray(i), size, name
        for association, data, size in associations:
            array = data.GetArray(self._area)
            if array is not None:
                return association, array, size, self._area
        raise ValueError("No radius or selected fallback area array exists.")

    def RequestData(self, request, inInfo, outInfo):
        source = vtkDataSet.GetData(inInfo[0])
        output = vtkPolyData.GetData(outInfo)
        output.Initialize()
        # Empty pieces are valid during parallel execution.
        if source.GetNumberOfPoints() == 0:
            return 1
        association, array, size, name = self._select_array(source)
        if array is None or array.GetNumberOfComponents() != 1:
            raise ValueError(f"'{name}' must be a scalar Point or Cell Data array.")
        if array.GetNumberOfTuples() != size:
            raise ValueError("Selected array size does not match its association.")
        # Validate before cell averaging, which could otherwise hide negatives.
        for i in range(size):
            value = array.GetTuple1(i)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Radius/area values must be finite and nonnegative.")

        current = source
        if association == vtkDataObject.FIELD_ASSOCIATION_CELLS:
            averaging = vtkCellDataToPointData()
            averaging.SetInputData(current)
            averaging.ProcessAllArraysOff()
            averaging.AddCellDataArray(name)
            averaging.Update()
            current = averaging.GetOutput()
        radius = (self._interpretation == 1 or
                  (self._interpretation == 0 and name.lower() == "radius"))
        if not radius:
            calculator = vtkArrayCalculator()
            calculator.SetInputData(current)
            calculator.SetAttributeTypeToPointData()
            calculator.AddScalarVariable("cross_section_area", name, 0)
            calculator.SetFunction(f"sqrt(cross_section_area/{math.pi:.17g})")
            calculator.SetResultArrayName("__immersx_tube_radius")
            calculator.Update()
            current = calculator.GetOutput()
            name = "__immersx_tube_radius"
        if not isinstance(current, vtkPolyData):
            surface = vtkDataSetSurfaceFilter()
            surface.SetInputData(current)
            surface.Update()
            current = surface.GetOutput()
        if current.GetNumberOfLines() == 0:
            raise ValueError("Input must contain line cells to generate tubes.")
        tube = vtkTubeFilter()
        tube.SetInputData(current)
        tube.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, name)
        tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tube.SetNumberOfSides(self._sides)
        tube.SetCapping(self._capping)
        tube.Update()
        output.ShallowCopy(tube.GetOutput())
        return 1
