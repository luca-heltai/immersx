#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exercise surface meshing and imported arterial/venous growth through the CLI."""
import argparse
import csv
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def rows(directory, name, key="id"):
    result = {}
    for path in sorted(directory.glob(f"{name}-*.csv")):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                identity = int(row[key])
                assert identity not in result, (name, identity)
                result[identity] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--portal", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="liver-", dir=args.output) as temporary:
        scratch = Path(temporary)
        unrelated = scratch / "unrelated"
        unrelated.mkdir()
        output = scratch / "result"
        prm = scratch / "input.prm"
        text = re.sub(r"(?m)^  set Output directory = .*\n",
                      f"  set Output directory = {output}\n", args.parameters.read_text())
        prm.write_text(text)
        command = [str(args.executable.resolve()), str(prm)]
        if args.ranks > 1:
            command = [args.mpiexec, "-np", str(args.ranks)] + command
        process = subprocess.run(command, cwd=unrelated, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=110)
        assert process.returncode == 0, process.stdout
        nodes, vessels = rows(output, "nodes"), rows(output, "vessels")
        terminals, pairs = rows(output, "terminals"), rows(output, "pairs", "territory")
        expected = [(0, (.2, .5, .5), 1), (1, (.3, .5, .5), 0),
                    (2, (.8, .5, .5), 2), (3, (.7, .5, .5), 0)]
        for identity, point, port in expected:
            row = nodes[identity]
            actual = tuple(float(row[k]) for k in ("x", "y", "z"))
            assert math.dist(point, actual) < 1.e-14
            assert int(row["port"]) == port and int(row["provenance"]) == 0
        for identity, endpoints, radius in [(0, (0, 1), .003), (1, (2, 3), .004)]:
            row = vessels[identity]
            assert (int(row["node0"]), int(row["node1"])) == endpoints
            assert float(row["radius0"]) == radius == float(row["radius1"])
            assert int(row["provenance"]) == 0
        # Binary insertions add two vessels. Companion startup has both a
        # fixed-length stem and a continuation before its first insertion.
        generated = {0: 2 * 8 - 1, 1: 2 * 8}
        if args.portal:
            generated[2] = 2 * 8
            expected.extend([(4, (.2, .8, .5), 4), (5, (.3, .8, .5), 0)])
            for identity, point, port in expected[-2:]:
                row = nodes[identity]
                assert math.dist(point, tuple(float(row[k]) for k in ("x", "y", "z"))) < 1.e-14
                assert int(row["port"]) == port and int(row["provenance"]) == 0
            portal = vessels[2]
            assert (int(portal["node0"]), int(portal["node1"])) == (4, 5)
            assert float(portal["radius0"]) == .003 == float(portal["radius1"])
            assert int(portal["system"]) == 2 and int(portal["provenance"]) == 0
            assert len(vessels) == sum(generated.values()) + 3 and not pairs
            triplets = rows(output, "triplets", "territory")
            assert len(triplets) == 8
            used_triplets = set()
            for territory, t in triplets.items():
                endpoints = [int(t[k]) for k in ("arterial", "portal", "venous")]
                assert len(set(endpoints)) == 3 and not used_triplets.intersection(endpoints)
                used_triplets.update(endpoints)
                assert [int(terminals[i]["system"]) for i in endpoints] == [0, 2, 1]
                assert all(int(terminals[i]["territory"]) == territory for i in endpoints)
                assert len({float(terminals[i]["demand"]) for i in endpoints}) == 1
                points = [tuple(float(nodes[int(terminals[i]["node"])][k]) for k in ("x", "y", "z"))
                          for i in endpoints]
                assert math.isclose(math.dist(points[0], points[1]), .06, rel_tol=1.e-10)
                assert math.isclose(math.dist(points[0], points[2]), .04, rel_tol=1.e-10)
                qa, qp, qv = [float(t[k]) for k in ("arterial_flow", "portal_flow", "venous_flow")]
                assert min(qa, qp, qv) > 0 and math.isclose(qa + qp, qv, rel_tol=1.e-9)
                assert math.isclose(qp / qv, .7, rel_tol=1.e-8)
                assert all(float(t[k]) > 0 for k in ("arterial_resistance", "portal_resistance", "venous_resistance"))
        else:
            assert len(vessels) == sum(generated.values()) + 2 and len(pairs) == 8
        for system, expected_count in generated.items():
            assert sum(int(r["system"]) == system and int(r["provenance"]) == 1
                       for r in vessels.values()) == expected_count
        used = set()
        for territory, pair in pairs.items():
            a, v = int(pair["arterial"]), int(pair["venous"])
            assert a not in used and v not in used and a != v
            used.update((a, v))
            assert int(terminals[a]["system"]) == 0 and int(terminals[v]["system"]) == 1
            assert all(int(terminals[t]["territory"]) == territory for t in (a, v))
            points = [tuple(float(nodes[int(terminals[t]["node"])][k]) for k in ("x", "y", "z"))
                      for t in (a, v)]
            assert math.isclose(math.dist(*points), .04, rel_tol=1.e-10)
            assert float(pair["resistance"]) > 0.
        for name in ("liver", "vessels"):
            pvtu = output / f"{name}_0.pvtu"
            pieces = ET.parse(pvtu).findall(".//Piece")
            assert len(pieces) == args.ranks
            for piece in pieces:
                assert (output / piece.attrib["Source"]).is_file()
        tree = ET.parse(output / "vessels_0.pvtu")
        arrays = {a.attrib["Name"] for a in tree.findall(".//PPointData/PDataArray")}
        assert {"System", "Generated", "Radius"} <= arrays, arrays
        kind = "triplets" if args.portal else "pairs"
        print(f"Verified 8 tissue {kind} and preserved all input vessels on {args.ranks} rank(s)")


if __name__ == "__main__":
    main()
