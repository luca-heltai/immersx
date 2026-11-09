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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep failed and successful attempts available for inspection.
    attempt = Path(tempfile.mkdtemp(prefix="invalid-output-", dir=args.output))
    output = attempt / "result"
    parameters = args.parameters.read_text()
    parameters = re.sub(r"(?m)^  set Output directory = .*$",
                        f"  set Output directory = {output}", parameters)
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
    diagnostics = output / "invalid-result"
    metadata = json.loads((diagnostics / "diagnostic.json").read_text())
    assert metadata["diagnostic_only"] and not metadata["validation_valid"]
    assert not metadata["generation_complete"] and not metadata["archive_adoptable"]
    assert metadata["omitted_vessels"] == 0
    assert not (output / "network").exists()
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
