# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""VTK geometry regressions; proxy checks also run when invoked by pvpython."""
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

try:
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkInformationVector, vtkPoints
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray, vtkDataObject, vtkPolyData, vtkUnstructuredGrid,
    )
except ImportError:
    print('VTK Python modules unavailable')
    sys.exit(77)

try:
    import paraview.util.vtkAlgorithm
    HAVE_PARAVIEW = True
except ImportError:
    HAVE_PARAVIEW = False

    class Decorators:
        def __getattr__(self, name):
            def decorator(*args, **kwargs):
                # ParaView counts a string default as one element, even when
                # it contains spaces. Preserve that public decorator contract.
                if name in ('stringvector', 'intvector'):
                    count = kwargs.get('number_of_elements', 1)
                    defaults = kwargs.get('default_values')
                    size = len(defaults) if isinstance(defaults, (list, tuple)) else 1
                    if defaults is not None and count != size:
                        raise ValueError('Property defaults do not match its element count')
                return lambda target: target
            return decorator

    # Only proxy decorators are replaced. All numerical filters below use VTK.
    decorators = types.ModuleType('paraview.util.vtkAlgorithm')
    decorators.smdomain = decorators.smproperty = decorators.smproxy = Decorators()
    for name in ('paraview', 'paraview.util'):
        sys.modules[name] = types.ModuleType(name)
    sys.modules['paraview.util.vtkAlgorithm'] = decorators

PLUGIN = Path(__file__).resolve().parents[2] / 'scripts/paraview_variable_radius_tube.py'
spec = importlib.util.spec_from_file_location('radius_tube_plugin', PLUGIN)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


def scalar(data, association, name, values):
    array = vtkDoubleArray()
    array.SetName(name)
    for value in values:
        array.InsertNextValue(value)
    (data.GetPointData() if association == 0 else data.GetCellData()).AddArray(array)


def line(unstructured=False):
    points = vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    data = vtkUnstructuredGrid() if unstructured else vtkPolyData()
    data.SetPoints(points)
    if unstructured:
        data.InsertNextCell(3, 2, [0, 1])
    else:
        cells = vtkCellArray()
        cells.InsertNextCell(2, [0, 1])
        data.SetLines(cells)
    return data


def execute(tube, source):
    inputs, outputs = vtkInformationVector(), vtkInformationVector()
    inputs.SetNumberOfInformationObjects(1)
    inputs.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), source)
    outputs.SetNumberOfInformationObjects(1)
    output = vtkPolyData()
    outputs.GetInformationObject(0).Set(vtkDataObject.DATA_OBJECT(), output)
    tube.RequestData(None, [inputs], outputs)
    return output


class RadiusTube(unittest.TestCase):
    def radius(self, output, expected):
        self.assertGreater(output.GetNumberOfPoints(), 0)
        for i in range(output.GetNumberOfPoints()):
            _, y, z = output.GetPoint(i)
            self.assertAlmostEqual(math.hypot(y, z), expected, places=6)

    def test_input_types_associations_names_and_interpretations(self):
        for unstructured in (False, True):
            for association in (0, 1):
                for name in ('radius', 'Radius', 'RADIUS', 'rAdIuS', 'area'):
                    for explicit in (False, True):
                        for interpretation in (0, 1, 2):
                            with self.subTest(grid=unstructured, association=association,
                                              name=name, explicit=explicit,
                                              interpretation=interpretation):
                                source = line(unstructured)
                                scalar(source, association, name,
                                       [4.] * (2 if association == 0 else 1))
                                tube = plugin.VariableRadiusTube()
                                if explicit:
                                    tube.SetInputArray(0, 0, 0, association, name)
                                tube.SetInterpretation(interpretation)
                                is_radius = interpretation == 1 or (
                                    interpretation == 0 and name.lower() == 'radius')
                                with patch.object(plugin, 'vtkArrayCalculator',
                                                  wraps=plugin.vtkArrayCalculator) as calculator:
                                    with patch.object(plugin, 'vtkDataSetSurfaceFilter',
                                                      wraps=plugin.vtkDataSetSurfaceFilter) as surface:
                                        result = execute(tube, source)
                                self.radius(result, 4. if is_radius else math.sqrt(4./math.pi))
                                self.assertEqual(calculator.call_count, int(not is_radius))
                                self.assertEqual(surface.call_count, int(unstructured))

    def test_explicit_selection_and_association_precedence(self):
        source = line()
        scalar(source, 0, 'area', [math.pi, math.pi])
        scalar(source, 1, 'Radius', [2.])
        tube = plugin.VariableRadiusTube()
        self.radius(execute(tube, source), 2.)
        tube.SetInputArray(0, 0, 0, 0, 'area')
        self.radius(execute(tube, source), 1.)
        scalar(source, 0, 'RADIUS', [3., 3.])
        tube.SetInputArray(0, 0, 0, 0, '')
        self.radius(execute(tube, source), 3.)

    def test_shared_point_cell_averaging_precedes_area_conversion(self):
        for area in (False, True):
            source = vtkPolyData()
            points = vtkPoints()
            for x in (0., 1., 2.):
                points.InsertNextPoint(x, 0., 0.)
            source.SetPoints(points)
            cells = vtkCellArray()
            cells.InsertNextCell(2, [0, 1])
            cells.InsertNextCell(2, [1, 2])
            source.SetLines(cells)
            scalar(source, 1, 'area' if area else 'radius',
                   [math.pi, 9.*math.pi] if area else [1., 3.])
            result = execute(plugin.VariableRadiusTube(), source)
            self.assertGreater(result.GetNumberOfPoints(), 0)
            for i in range(result.GetNumberOfPoints()):
                x, y, z = result.GetPoint(i)
                expected = (1., math.sqrt(5.), 3.) if area else (1., 2., 3.)
                self.assertAlmostEqual(math.hypot(y, z), expected[round(x)], places=6)

    def test_invalid_values_are_rejected_before_averaging(self):
        for association in (0, 1):
            for value in (-1., float('nan'), float('inf')):
                with self.subTest(association=association, value=value):
                    source = line()
                    scalar(source, association, 'radius',
                           [value] * (2 if association == 0 else 1))
                    with self.assertRaises(ValueError):
                        execute(plugin.VariableRadiusTube(), source)

    def test_empty_missing_and_nonscalar_arrays(self):
        self.assertEqual(execute(plugin.VariableRadiusTube(), vtkPolyData()).GetNumberOfPoints(), 0)
        with self.assertRaises(ValueError):
            execute(plugin.VariableRadiusTube(), line())
        source = line()
        scalar(source, 0, 'radius', [1.])
        with self.assertRaises(ValueError):
            execute(plugin.VariableRadiusTube(), source)
        source.GetPointData().GetArray('radius').SetNumberOfComponents(2)
        with self.assertRaises(ValueError):
            execute(plugin.VariableRadiusTube(), source)

    def test_automatic_selection_label_uses_radius_or_area(self):
        for unstructured in (False, True):
            for association in (0, 1):
                for has_radius in (False, True):
                    with self.subTest(grid=unstructured, association=association,
                                      has_radius=has_radius):
                        source = line(unstructured)
                        scalar(source, 0, 'area', [math.pi, math.pi])
                        if has_radius:
                            scalar(source, association, 'Radius',
                                   [2.] * (2 if association == 0 else 1))
                        tube = plugin.VariableRadiusTube()
                        tube.SetInputArray(
                            0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS,
                            'Automatic')
                        self.radius(execute(tube, source),
                                    2. if has_radius else 1.)

    @unittest.skipUnless(HAVE_PARAVIEW, 'requires pvpython for actual proxy properties')
    def test_paraview_proxy_automatic_selection(self):
        from paraview.simple import LoadPlugin, TrivialProducer
        LoadPlugin(str(PLUGIN), remote=False)
        from paraview.simple import VariableRadiusTube
        for unstructured in (False, True):
            for association in (0, 1):
                for has_radius in (False, True):
                    with self.subTest(grid=unstructured, association=association,
                                      has_radius=has_radius):
                        source = line(unstructured)
                        scalar(source, 0, 'area', [math.pi, math.pi])
                        if has_radius:
                            scalar(source, association, 'Radius',
                                   [2.] * (2 if association == 0 else 1))
                        producer = TrivialProducer()
                        producer.GetClientSideObject().SetOutput(source)
                        proxy = VariableRadiusTube(Input=producer)
                        # Leave the real ArrayListDomain's Automatic selection
                        # untouched instead of overriding it from Python.
                        proxy.UpdatePipeline()
                        self.assertAlmostEqual(
                            proxy.GetDataInformation().GetBounds()[3],
                            2. if has_radius else 1., places=6)

    @unittest.skipUnless(HAVE_PARAVIEW, 'requires pvpython for actual proxy properties')
    def test_paraview_proxy(self):
        from paraview.simple import LoadPlugin, TrivialProducer
        LoadPlugin(str(PLUGIN), remote=False)
        from paraview.simple import VariableRadiusTube
        source = line()
        scalar(source, 0, 'RADIUS', [2., 2.])
        producer = TrivialProducer()
        producer.GetClientSideObject().SetOutput(source)
        proxy = VariableRadiusTube(Input=producer)
        proxy.InputArray = ['POINTS', 'RADIUS']
        proxy.Interpretation = 'Automatic'
        proxy.UpdatePipeline()
        self.assertAlmostEqual(proxy.GetDataInformation().GetBounds()[3], 2., places=6)


if __name__ == '__main__':
    unittest.main()
