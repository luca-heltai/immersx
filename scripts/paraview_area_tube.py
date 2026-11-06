# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ParaView plugin: point area -> radius calculator -> surface -> tubes."""

import math

from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkDataSet, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkArrayCalculator, vtkTubeFilter
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter


@smproxy.filter(name="AreaTube", label="Area Tube")
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class AreaTube(VTKPythonAlgorithmBase):
    """Generate tubes around lines using radius = sqrt(area / pi)."""

    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=1,
                         inputType="vtkDataSet", outputType="vtkPolyData")
        self._area = "area"
        self._sides = 24
        self._capping = True

    @smproperty.stringvector(name="AreaArray", default_values="area")
    def SetAreaArray(self, name):
        """Name of the scalar point-data array containing cross-sectional area."""
        self._area = name
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

    def RequestData(self, request, inInfo, outInfo):
        source = vtkDataSet.GetData(inInfo[0])
        output = vtkPolyData.GetData(outInfo)
        output.Initialize()
        # Empty pieces are valid during parallel execution.
        if source.GetNumberOfPoints() == 0:
            return 1
        area = source.GetPointData().GetArray(self._area)
        if area is None or area.GetNumberOfComponents() != 1:
            raise ValueError(
                f"'{self._area}' must be a scalar Point Data array. "
                "For Cell Data, apply Cell Data to Point Data first.")
        if area.GetNumberOfTuples() != source.GetNumberOfPoints():
            raise ValueError("Area array size does not match the input points.")
        for i in range(area.GetNumberOfTuples()):
            value = area.GetTuple1(i)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Area values must be finite and nonnegative.")

        calculator = vtkArrayCalculator()
        calculator.SetInputData(source)
        calculator.SetAttributeTypeToPointData()
        calculator.AddScalarVariable("cross_section_area", self._area, 0)
        calculator.SetFunction(f"sqrt(cross_section_area/{math.pi:.17g})")
        calculator.SetResultArrayName("radius")

        surface = vtkDataSetSurfaceFilter()
        surface.SetInputConnection(calculator.GetOutputPort())
        surface.Update()
        if surface.GetOutput().GetNumberOfLines() == 0:
            raise ValueError("Input must contain line cells to generate tubes.")

        tube = vtkTubeFilter()
        tube.SetInputConnection(surface.GetOutputPort())
        tube.SetInputArrayToProcess(
            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "radius")
        tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tube.SetNumberOfSides(self._sides)
        tube.SetCapping(self._capping)
        tube.Update()
        output.ShallowCopy(tube.GetOutput())
        return 1
