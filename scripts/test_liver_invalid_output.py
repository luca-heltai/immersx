#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check that rejected liver growth still writes inspectable physical VTK."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--exception", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep failed and successful attempts available for inspection.
    attempt = Path(tempfile.mkdtemp(prefix="invalid-output-", dir=args.output))
    output = attempt / "result"
    parameters = args.parameters.read_text()
    parameters = re.sub(r"(?m)^  set Output directory = .*$",
                        f"  set Output directory = {output}", parameters)
    if args.exception:
        parameters += ("\nsubsection Liver\n  subsection Vascularization\n"
                       "    subsection Physiology\n      set Tissue demand = 0\n"
                       "    end\n  end\nend\n")
    else:
        parameters = parameters.replace("set Minimum vessel length = .0002",
                                        "set Minimum vessel length = 10")
        parameters = parameters.replace("set Sample budget per insertion = 2000",
                                        "set Sample budget per insertion = 1")
        parameters = parameters.replace("set Terminal count = 8",
                                        "set Terminal count = 1")
    prm = attempt / "input.prm"
    prm.write_text(parameters)
    command = [str(args.executable.resolve()), str(prm.resolve())]
    if args.ranks > 1:
        command = [args.mpiexec, "-np", str(args.ranks)] + command
    with (attempt / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=attempt, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=100)
    assert process.returncode != 0, "This fixture must be rejected"
    initial = output / "initial-input"
    initial_metadata = json.loads((initial / "diagnostic.json").read_text())
    assert initial_metadata["diagnostic_only"]
    assert not initial_metadata["generation_complete"]
    assert not initial_metadata["archive_adoptable"]
    for role in ("arterial", "portal", "venous"):
        pieces = ET.parse(initial / f"{role}.pvtu").findall(".//Piece")
        assert len(pieces) == args.ranks
        count = sum(int(record.attrib["NumberOfCells"])
                    for piece in pieces
                    for record in ET.parse(initial / piece.attrib["Source"]).findall(
                        ".//UnstructuredGrid/Piece"))
        assert count > 0, (role, count)
    assert not (output / "network").exists()
    if args.exception:
        log_text = (attempt / "run.log").read_text()
        assert "Tissue density and physical quadrature masses" in log_text
        assert not (output / "invalid-result").exists()
        print(f"Generation exception retained imported inputs only: {initial}")
        return
    diagnostics = output / "invalid-result"
    metadata = json.loads((diagnostics / "diagnostic.json").read_text())
    assert metadata["diagnostic_only"] and not metadata["validation_valid"]
    assert not metadata["generation_complete"] and not metadata["archive_adoptable"]
    assert metadata["omitted_vessels"] == 0
    for role in ("arterial", "portal", "venous"):
        pieces = ET.parse(diagnostics / f"{role}.pvtu").findall(".//Piece")
        assert len(pieces) == args.ranks
        cells = 0
        for piece in pieces:
            path = diagnostics / piece.attrib["Source"]
            grid = ET.parse(path)
            cells += sum(int(p.attrib["NumberOfCells"])
                         for p in grid.findall(".//UnstructuredGrid/Piece"))
            for array in grid.findall(".//CellData/DataArray"):
                if array.attrib["Name"] in ("DiagnosticOnly", "ValidationFailed"):
                    assert all(int(x) == 1 for x in (array.text or "").split())
        assert cells > 0, (role, cells)
    assert (diagnostics / "liver_0.pvtu").is_file()
    print(f"Rejected growth retained three diagnostic networks: {diagnostics}")


if __name__ == "__main__":
    main()
