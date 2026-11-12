#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check distributed lifted-wall VTK collections using only Python's standard library."""
import argparse
import base64
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

def data_array(array, tree):
    if array.attrib.get("format") == "ascii":
        return [float(value) for value in (array.text or "").split()]
    text = "".join((array.text or "").split())
    assert array.attrib["format"] == "binary"
    assert tree.attrib["compressor"] == "vtkZLibDataCompressor"
    assert tree.attrib["byte_order"] == "LittleEndian"
    header_kind = "Q" if tree.attrib.get("header_type") == "UInt64" else "I"
    size = struct.calcsize(header_kind)
    first = base64.b64decode(text[:4*((size+2)//3)])
    blocks = struct.unpack("<" + header_kind, first[:size])[0]
    header_length = 4 * (((3 + blocks)*size + 2)//3)
    header = struct.unpack("<" + header_kind*(3+blocks),
                           base64.b64decode(text[:header_length]))
    compressed = base64.b64decode(text[header_length:])
    parts, offset = [], 0
    for length in header[3:]:
        parts.append(zlib.decompress(compressed[offset:offset+length]))
        offset += length
    raw = b"".join(parts)
    kind = {"Float32": "f", "Float64": "d", "Int32": "i", "UInt32": "I", "UInt8": "B", "UInt64": "Q", "Int64": "q"}[array.attrib["type"]]
    values = struct.unpack("<" + kind*(len(raw)//struct.calcsize(kind)), raw)
    assert values and all(math.isfinite(value) for value in values)
    return values


def point_arrays(path):
    tree = ET.parse(path).getroot()
    arrays = {}
    for piece in tree.findall("./UnstructuredGrid/Piece"):
        # Immersed ownership can leave a rank with no vessel cells. deal.II
        # emits empty, unformatted arrays for that piece, not binary payloads.
        if int(piece.attrib["NumberOfPoints"]) == 0:
            continue
        for array in piece.findall("./PointData/DataArray"):
            arrays.setdefault(array.attrib["Name"], []).extend(data_array(array, tree))
    return arrays



def check_wall_particles(directory, times, properties, *, displacement=None,
                         target=None, disabled=False, nonzero_traction=False):
    directory = Path(directory)
    if disabled:
        assert not list(directory.glob("wall_particles*")), directory
        return
    entries = ET.parse(directory/"wall_particles.pvd").findall(".//DataSet")
    assert len(entries) == len(times), (len(entries), times)
    selected = set(properties)
    expected = {"id"}
    names = {"weight": "weight", "mode_values": "mode_radial",
             "modal_coefficients": "wall_traction_radial", "lifted_value": "wall_traction",
             "modal_data": "data_radial", "representative_point": "representative_point",
             "radius": "radius", "bulk_field": "displacement"}
    expected.update(names[key] for key in selected if key in names)
    if "indices" in selected:
        expected.update(("source_cell", "representative_qpoint", "section_qpoint"))
    maximum_traction = 0.
    counts = []
    for entry, time in zip(entries, times):
        assert math.isclose(float(entry.attrib["timestep"]), time, rel_tol=0., abs_tol=1.e-14)
        path = directory/entry.attrib["file"]
        arrays = point_arrays(path)
        assert set(arrays) == expected, (set(arrays), expected)
        tree = ET.parse(path).getroot()
        coordinates = []
        count = 0
        for piece in tree.findall("./UnstructuredGrid/Piece"):
            size = int(piece.attrib["NumberOfPoints"])
            count += size
            if size:
                coordinates.extend(data_array(piece.find("./Points/DataArray"), tree))
        assert count > 0, path
        counts.append(count)
        for name, values in arrays.items():
            width = 3 if name in {"wall_traction", "mode_radial", "displacement", "representative_point"} else 1
            assert len(values) == width*count, (name, len(values), count)
            assert all(math.isfinite(value) for value in values), name
        for q in range(count):
            if "weight" in selected:
                assert arrays["weight"][q] > 0.
            if "radius" in selected:
                radius = arrays["radius"][q]
                assert radius > 0.
                if "representative_point" in selected:
                    distance = math.sqrt(sum((coordinates[3*q+c]-arrays["representative_point"][3*q+c])**2 for c in range(3)))
                    assert math.isclose(distance, radius, rel_tol=2.e-5, abs_tol=1.e-8), (distance, radius)
            if "mode_values" in selected:
                direction = arrays["mode_radial"][3*q:3*q+3]
                assert math.isclose(sum(x*x for x in direction), 1., abs_tol=3.e-7)
            if "lifted_value" in selected:
                traction = arrays["wall_traction"][3*q:3*q+3]
                norm = math.sqrt(sum(x*x for x in traction))
                maximum_traction = max(maximum_traction, norm)
                if "modal_coefficients" in selected:
                    coefficient = arrays["wall_traction_radial"][q]
                    assert math.isclose(norm, abs(coefficient), rel_tol=3.e-7, abs_tol=1.e-7)
                    if "mode_values" in selected:
                        assert all(math.isclose(traction[c], coefficient*direction[c], rel_tol=3.e-7, abs_tol=1.e-7) for c in range(3))
            if displacement is not None and "bulk_field" in selected:
                assert all(abs(arrays["displacement"][3*q+c]-displacement[c]) < 1.e-9 for c in range(3))
            if target is not None and "modal_data" in selected:
                assert abs(arrays["data_radial"][q]-target) < 1.e-9
    assert len(set(counts)) == 1, counts
    if nonzero_traction:
        assert maximum_traction > 1.e-8, maximum_traction
    return {"states": len(entries), "particles_per_state": counts[0],
            "maximum_traction": maximum_traction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--times", default="0,0.001,0.002")
    parser.add_argument("--properties", default="lifted_value,weight")
    parser.add_argument("--displacement")
    parser.add_argument("--target", type=float)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--nonzero-traction", action="store_true")
    args = parser.parse_args()
    print(check_wall_particles(args.directory, [float(t) for t in args.times.split(",")],
          [name for name in args.properties.split(",") if name],
          displacement=[float(x) for x in args.displacement.split(",")] if args.displacement else None,
          target=args.target, disabled=args.disabled, nonzero_traction=args.nonzero_traction))


if __name__ == "__main__":
    main()
