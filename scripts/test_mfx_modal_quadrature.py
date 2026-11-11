#!/usr/bin/env python3
"""Analytical checks of the geometry used by the exported-field MMS report."""
import base64
import struct
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mfx_modal_convergence import cartesian_quadrature, hexahedron_quadrature, decode

CORNERS = ((0,0,0), (1,0,0), (1,1,0), (0,1,0),
           (0,0,1), (1,0,1), (1,1,1), (0,1,1))


class ExportedHexahedronQuadrature(unittest.TestCase):
    def test_sheared_affine_cell(self):
        vertices = [(2*x+y, 3*y, 4*z) for x,y,z in CORNERS]
        q = list(hexahedron_quadrature(vertices, 12))
        self.assertAlmostEqual(sum(w for p,w,s in q), 24.)
        self.assertAlmostEqual(sum(w*p[0] for p,w,s in q), 36.)
        for p,w,s in q:
            self.assertAlmostEqual(sum(s), 1.)
            # Nodal interpolation must reproduce the physical affine field.
            self.assertAlmostEqual(sum(s[i]*(vertices[i][0]+2*vertices[i][1])
                                       for i in range(8)), p[0]+2*p[1])

    def test_nonaffine_cell(self):
        a = .7
        vertices = [(x,y,z*(1+a*x*y)) for x,y,z in CORNERS]
        q = list(hexahedron_quadrature(vertices, 12))
        self.assertAlmostEqual(sum(w for p,w,s in q), 1+a/4)
        self.assertAlmostEqual(sum(w*p[2] for p,w,s in q), .5*(1+a/2+a*a/9))

    def test_cartesian_regression(self):
        vertices = [(2*x-1, 3*y+2, .5*z-2) for x,y,z in CORNERS]
        old = list(cartesian_quadrature(vertices, 3))
        new = list(hexahedron_quadrature(vertices, 12))
        for f in (lambda p: 1., lambda p: p[0]**2+p[1]**2+p[2]**2):
            self.assertAlmostEqual(sum(w*f(p) for p,w,s in old),
                                   sum(w*f(p) for p,w,s in new))

    def test_voxel_order(self):
        vertices = [(i&1, (i>>1)&1, (i>>2)&1) for i in range(8)]
        self.assertAlmostEqual(sum(w for p,w,s in hexahedron_quadrature(vertices, 11)), 1.)

    def test_compressed_vtk_cell_types(self):
        raw = bytes((11, 12))
        compressed = zlib.compress(raw)
        header = struct.pack('<IIII', 1, len(raw), len(raw), len(compressed))
        array = ET.Element('DataArray', type='UInt8', format='binary')
        array.text = (base64.b64encode(header)+base64.b64encode(compressed)).decode()
        tree = ET.Element('VTKFile', byte_order='LittleEndian',
                          compressor='vtkZLibDataCompressor')
        self.assertEqual(tuple(decode(array, tree)), (11, 12))

    def test_inverted_cell_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Nonpositive'):
            list(hexahedron_quadrature([(-x,y,z) for x,y,z in CORNERS], 12))


if __name__ == '__main__':
    unittest.main()
