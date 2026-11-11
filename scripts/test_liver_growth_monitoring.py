#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exercise cooperative deadlines and bounded accepted-state VTK publication."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROLES = ("arterial", "portal", "venous")


def vessels(directory, role, ranks):
    """Read exact physical IDs and double-valued records without Python VTK."""
    pieces = ET.parse(directory / f"{role}.pvtu").findall(".//Piece")
    assert len(pieces) == ranks
    records = {}
    for piece in pieces:
        grid = ET.parse(directory / piece.attrib["Source"])
        cell_arrays = {a.attrib["Name"]: (a.text or "").split()
                       for a in grid.findall(".//CellData/DataArray")}
        point_arrays = {a.attrib["Name"]: (a.text or "").split()
                        for a in grid.findall(".//PointData/DataArray")}
        points = list(map(float, grid.find(".//Points/DataArray").text.split()))
        arrays = {a.attrib["Name"]: (a.text or "").split()
                  for a in grid.findall(".//Cells/DataArray")}
        nodes = list(map(int, point_arrays["PhysicalNodeId"]))
        radii = list(map(float, point_arrays["Radius"]))
        connectivity = list(map(int, arrays["connectivity"]))
        for i, text_id in enumerate(cell_arrays["VesselId"]):
            identifier = int(text_id)
            assert identifier not in records
            ends = connectivity[2*i:2*i+2]
            records[identifier] = (
                tuple(nodes[j] for j in ends),
                tuple(radii[j] for j in ends),
                tuple(tuple(points[3*j:3*j+3]) for j in ends),
                int(cell_arrays["System"][i]),
                int(cell_arrays["Generated"][i]),
                float(cell_arrays["DynamicViscosity"][i]))
            assert int(cell_arrays["DiagnosticOnly"][i]) == 1
    assert records, (directory, role)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="growth-monitoring-", dir=args.output))
    for stopped in (True, False):
        case = attempt / ("deadline" if stopped else "complete")
        case.mkdir()
        output = case / "result"
        text = re.sub(r"(?m)^  set Output directory = .*$",
                      f"  set Output directory = {output}", args.parameters.read_text())
        text += ("\nsubsection Liver\n  subsection Growth monitoring\n"
                 "    set Progress interval = 1e-12\n"
                 "    set Checkpoint interval = 1e-12\n"
                 f"    set Maximum generation wall time = {'1e-12' if stopped else '0'}\n"
                 "  end\nend\n")
        prm = case / "input.prm"
        prm.write_text(text)
        command = [str(args.executable.resolve()), str(prm.resolve())]
        if args.ranks > 1:
            command = [args.mpiexec, "-np", str(args.ranks)] + command
        with (case / "run.log").open("w") as log:
            run = subprocess.run(command, cwd=case, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=100)
        assert run.returncode == (2 if stopped else 0), (case, run.returncode)
        latest = output / "latest-checkpoint"
        metadata = json.loads((latest / "checkpoint.json").read_text())
        for key in ("generation_complete", "archive_adoptable", "resumable",
                    "hydraulic_fields_final"):
            assert metadata[key] is False, (key, metadata)
        assert metadata["diagnostic_only"] is True
        assert metadata["controlled_stop"] is stopped
        assert metadata["checkpoint_index"] >= (1 if stopped else 2)
        assert not (output / "checkpoint-staging").exists()
        assert not (output / "checkpoint-previous").exists()
        assert (output / "network").exists() is not stopped
        with (output / "progress.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(int(row["accepted_terminals"]) > 0 for row in rows)
        assert all(float(row["elapsed_s"]) >= 0 for row in rows)
        for role in ROLES:
            original = vessels(output / "initial-input", role, args.ranks)
            current = vessels(latest, role, args.ranks)
            assert all(current.get(identifier) == record
                       for identifier, record in original.items()), (case, role)
            assert len(current) == metadata["roles"][role]["physical_vessels"]
            assert metadata["roles"][role]["hydraulic_fields_final"] is False
            if stopped and role != "arterial":
                assert current == original
                assert metadata["roles"][role]["source"] == "original_input"
                assert metadata["roles"][role]["growth_accepted_terminals"] == 0
                assert metadata["roles"][role]["growth_phase_complete"] is False
                assert vessels(output / "invalid-result", role, args.ranks) == original
        if stopped:
            assert (output / "invalid-result" / "liver_0.pvtu").exists()
        print(f"Verified {'cooperative stop' if stopped else 'checkpoint rotation'}: {case}")


if __name__ == "__main__":
    main()
