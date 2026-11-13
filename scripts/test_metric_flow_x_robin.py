#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check the coupled Robin/RCR equilibrium, physical energies and flow fields."""
import argparse
import base64
import csv
import json
import math
from pathlib import Path
import struct
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib

from lifted_particle_output_checks import check_wall_particles


def data_array(array, tree):
    text = "".join((array.text or "").split())
    if array.attrib.get("format") == "ascii":
        return [float(value) for value in text.split()]
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
    kind = {"Float32": "f", "Float64": "d", "Int32": "i", "UInt32": "I", "UInt8": "B"}[array.attrib["type"]]
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


def outputs(directory, collection):
    entries = ET.parse(directory/collection).findall(".//DataSet")
    assert len(entries) == 3
    assert all(math.isclose(float(entry.attrib["timestep"]), expected,
                            rel_tol=0., abs_tol=1.e-14)
               for entry, expected in zip(entries, (0., .001, .002)))
    return [directory/entry.attrib["file"] for entry in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--disable-particles", action="store_true")
    parser.add_argument("--all-particle-properties", action="store_true")
    args = parser.parse_args()
    particle_properties = ([] if args.disable_particles else
        ["lifted_value", "weight", "mode_values", "modal_coefficients", "modal_data",
         "representative_point", "radius", "indices", "bulk_field"]
        if args.all_particle_properties else ["lifted_value", "weight"])
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robin-rcr-", dir=args.output) as temporary:
        scratch = Path(temporary)
        solid, flow = scratch/"solid", scratch/"flow"
        text = args.parameters.read_text() + f"""
subsection MetricFlowX coupling
  subsection Particle output
    set Properties = {','.join(particle_properties)}
  end
end
subsection Elastodynamics
  set Output directory = {solid}
end
subsection MetricFlowSystem<1, 3>
  set Output directory = {flow}
  subsection IDA parameters
    subsection Linear solver
      set Type = {'umfpack' if args.ranks == 1 else 'mumps'}
    end
  end
end
"""
        parameters = scratch/"case.prm"
        parameters.write_text(text)
        command = [str(args.executable.resolve()), str(parameters)]
        if args.ranks > 1:
            watchdog = shutil.which("timeout") or shutil.which("gtimeout")
            assert watchdog, "MPI tests require GNU timeout or gtimeout"
            command = [args.mpiexec, "-np", str(args.ranks),
                       watchdog, "--kill-after=2s", "105s"] + command
        process = subprocess.run(command, cwd=scratch, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=110)
        (args.output/"application.log").write_text(process.stdout)
        assert process.returncode == 0, process.stdout
        assert "MetricFlowX elastodynamics accepted IDA steps:" in process.stdout
        expected_u = struct.unpack("<f", struct.pack("<f", .01))[0]
        displacement_error = velocity_max = pressure_max = 0.
        for path in outputs(solid, "solid.pvd"):
            arrays = point_arrays(path)
            assert arrays, path
            displacement_error = max(displacement_error,
                max(abs(value - (expected_u if i % 3 == 0 else 0.))
                    for i, value in enumerate(arrays["displacement"])))
            velocity_max = max(velocity_max, max(abs(x) for x in arrays["velocity"]))
        assert displacement_error < 1.e-9, displacement_error
        assert velocity_max < 1.e-9, velocity_max
        for path in outputs(flow, "flow.pvd"):
            pieces = ET.parse(path).findall(".//Piece")
            assert len(pieces) == args.ranks
            vessel_ids = set()
            for piece in pieces:
                arrays = point_arrays(path.parent/piece.attrib["Source"])
                if not arrays:
                    continue
                vessel_ids.update(int(value) for value in arrays["vessel_id"])
                assert max(abs(x) for x in arrays["velocity"]) < 1.e-10
                for name in ("pressure", "Pc"):
                    value = max(abs(x) for x in arrays[name])
                    pressure_max = max(pressure_max, value)
                    assert value < 1.e-8, (name, value)
                for area, vessel in zip(arrays["area"], arrays["vessel_id"]):
                    expected = struct.unpack("<f", struct.pack("<f", {0: 1.e-4, 1: 2.e-4}[int(vessel)]))[0]
                    assert abs(area - expected) < 1.e-12
            assert vessel_ids == {0, 1}, vessel_ids
        check_wall_particles(flow, (0., .001, .002), particle_properties,
                             displacement=(expected_u, 0., 0.), target=0.,
                             disabled=args.disable_particles)
        with (solid/"energies.csv").open() as stream:
            energies = list(csv.DictReader(stream))
        assert len(energies) == 3
        for row, time in zip(energies, (0., .001, .002)):
            assert math.isclose(float(row["time_s"]), time, abs_tol=1.e-14)
            # Five Robin faces of a .4-sided cube; a constant .01 translation
            # has no elastic strain. This detects missing or frozen springs.
            boundary_norm_squared = 5*.4*.4*.01**2
            for name, factor in (("elastic_and_robin_energy_J", .5),
                                 ("applied_load_work_J", 1.)):
                assert math.isclose(float(row[name]), factor*(2+time)*boundary_norm_squared,
                                    rel_tol=1.e-9, abs_tol=1.e-14), row
        for first, last in zip(energies, energies[1:]):
            dt = float(last["time_s"]) - float(first["time_s"])
            for name, factor in (("elastic_and_robin_energy_J", .5),
                                 ("applied_load_work_J", 1.)):
                assert math.isclose(float(last[name])-float(first[name]),
                                    factor*dt*boundary_norm_squared,
                                    rel_tol=1.e-7, abs_tol=1.e-14)
        (args.output/"energies.csv").write_bytes((solid/"energies.csv").read_bytes())
        result = dict(ranks=args.ranks, states=3, displacement_error=displacement_error,
                      velocity_max=velocity_max, pressure_max=pressure_max,
                      time_dependent_robin_energy_and_work=True)
        (args.output/"validation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
