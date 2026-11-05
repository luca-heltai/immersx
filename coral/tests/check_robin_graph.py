"""Run a scalar/vector Robin graph and check an independent affine solution."""

import argparse
import base64
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
import zlib


def values(array, vtk):
    if array.get("format") == "ascii":
        return [float(value) for value in array.text.split()]
    encoded = "".join(array.text.split())
    endian = "<" if vtk.get("byte_order") == "LittleEndian" else ">"
    word = "Q" if vtk.get("header_type") == "UInt64" else "I"
    width = struct.calcsize(word)
    if vtk.get("compressor") == "vtkZLibDataCompressor":
        initial = base64.b64decode(encoded[: ((3 * width + 2) // 3) * 4])
        blocks, block_size, last_size = struct.unpack_from(endian + 3 * word, initial)
        header_length = ((3 + blocks) * width + 2) // 3 * 4
        header = base64.b64decode(encoded[:header_length])
        sizes = struct.unpack_from(endian + blocks * word, header, 3 * width)
        packed = base64.b64decode(encoded[header_length:])
        offset, chunks = 0, []
        for size in sizes:
            chunks.append(zlib.decompress(packed[offset : offset + size]))
            offset += size
        data = b"".join(chunks)
        assert len(data) == (blocks - 1) * block_size + last_size
    else:
        data = base64.b64decode(encoded)
        size = struct.unpack_from(endian + word, data)[0]
        if len(data) == width + size:
            data = data[width:]
        else:
            data = base64.b64decode(encoded[((width + 2) // 3) * 4 :])
        assert len(data) == size
    number = {"Float32": "f", "Float64": "d"}[array.get("type")]
    return list(struct.unpack(endian + str(len(data) // struct.calcsize(number)) + number, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coral", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path, required=True)
    parser.add_argument("--family", choices=["scalar", "elasticity"], required=True)
    parser.add_argument("--case", choices=["mms", "overlap", "negative", "nonfinite"], default="mms")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    work = args.working_directory.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copyfile(args.graph, work / args.graph.name)
    text = args.parameters.read_text()
    if args.case == "overlap":
        text, count = re.subn(r"set Dirichlet boundary ids =\s*\n", "set Dirichlet boundary ids = 0\n", text, count=1)
        assert count == 1
    elif args.case == "negative":
        text = text.replace("set Function expression = 2+x", "set Function expression = z-.25", 1)
    elif args.case == "nonfinite":
        expression = "exp(1000*z)" if args.family == "scalar" else "exp(1000*z);0;0"
        text, count = re.subn(r"(subsection Robin boundary conditions\s*\n\s*set Function expression = )[^\n]+", r"\g<1>" + expression, text, count=1)
        assert count == 1
    (work / args.parameters.name).write_text(text)
    # Bound every rank, including a rank left behind by a failed launcher.
    rank_command = ["timeout", "--verbose", "--kill-after=2s", "108s",
                    args.coral, "-p", args.plugin, "run",
                    str(work / args.graph.name), "--graph", str(work / "graph.dot")]
    command = rank_command
    if args.ranks > 1:
        command = ["timeout", "--verbose", "--kill-after=1s", "112s",
                   args.mpiexec, "-np", str(args.ranks)] + rank_command
    env = dict(os.environ, THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LC_ALL="C")
    if os.getuid() == 0:
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    with (work / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (work / "run.log").read_text()
    assert "timeout: sending signal" not in log, log
    if args.case != "mms":
        assert result.returncode not in (0, 124, 137), (result.returncode, log)
        expected = "Robin boundary ids must be disjoint" if args.case == "overlap" else "Robin coefficients must be finite and nonnegative"
        assert expected in log, log
        print(f"{args.family} {args.case}: expected collective Robin rejection")
        return
    assert result.returncode == 0, log
    assert (work / "output/solution.pvd").is_file(), log
    files = list((work / "output").glob("*.vtu"))
    assert files, log
    points, error = 0, 0.0
    for filename in files:
        vtk = ET.parse(filename).getroot()
        for piece in vtk.findall(".//Piece"):
            coordinates = values(piece.find("Points/DataArray"), vtk)
            array = piece.find(f"PointData/DataArray[@Name='{args.family}']")
            assert array is not None, ET.tostring(piece).decode()[:1500]
            field = values(array, vtk)
            components = 1 if args.family == "scalar" else 3
            assert int(array.get("NumberOfComponents", 1)) == components
            assert len(field) * 3 == len(coordinates) * components
            for point in range(len(coordinates) // 3):
                for component in range(components):
                    expected = component + 1 + coordinates[3 * point + component]
                    value = field[components * point + component]
                    assert math.isfinite(value)
                    error = max(error, abs(value - expected))
                points += 1
    # deal.II writes these point arrays as Float32 by default. The graph also
    # checks its full composed algebraic residual against 1e-10 before output.
    assert points > 0 and error < 2e-6, (points, error)
    print(f"{args.family} affine Robin MMS: {points} points, max error {error}")


if __name__ == "__main__":
    main()
