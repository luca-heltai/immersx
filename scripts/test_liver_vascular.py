#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exercise surface meshing and imported arterial/venous growth through the CLI."""
import argparse
import csv
import math
from pathlib import Path
import re
import shutil
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


def check_robin_reference(nodes, vessels, terminals, pairs):
    """Check exported physical data, independently of the graph solver."""
    pressure = {i: float(row["pressure_Pa"]) for i, row in nodes.items()}
    inflow = {i: float(row["boundary_inflow_m3_s"]) for i, row in nodes.items()}
    assert all(math.isfinite(value) for value in [*pressure.values(), *inflow.values()])
    balance = dict.fromkeys(nodes, 0.)
    flows = []
    for row in vessels.values():
        i, j = int(row["node0"]), int(row["node1"])
        length = math.dist(tuple(float(nodes[i][k]) for k in ("x", "y", "z")),
                           tuple(float(nodes[j][k]) for k in ("x", "y", "z")))
        r0, r1 = float(row["radius0"]), float(row["radius1"])
        expected = (8 * float(row["viscosity"]) * length / math.pi *
                    (r0*r0 + r0*r1 + r1*r1) / (3 * r0**3 * r1**3))
        resistance, flow = float(row["resistance_Pa_s_per_m3"]), float(row["flow_m3_s"])
        assert math.isfinite(flow) and resistance > 0
        assert math.isclose(resistance, expected, rel_tol=1.e-12)
        # The CSV contains rounded total pressures, whereas the solver retains
        # high/low pressure parts. Bound subtraction of those serialized doubles
        # explicitly; retain the requested relative physical flow accuracy.
        rounding = 4 * (math.ulp(pressure[i]) + math.ulp(pressure[j]))
        assert abs(pressure[i] - pressure[j] - resistance*flow) <= (rounding +
                1.e-9 * resistance * abs(flow))
        balance[i] += flow
        balance[j] -= flow
        flows.append(flow)
    for pair in pairs.values():
        i, j = (int(terminals[int(pair[key])]["node"]) for key in ("arterial", "venous"))
        flow = (pressure[i] - pressure[j]) / float(pair["resistance"])
        assert flow > 0
        balance[i] += flow
        balance[j] -= flow
        flows.append(flow)
    scale = max(abs(value) for value in [*flows, *inflow.values()])
    assert scale > 0
    defect = max(abs(balance[i] - inflow[i]) for i in nodes)
    assert defect < 1.e-9 * scale, (defect, scale)
    assert abs(sum(inflow.values())) < 1.e-9 * scale
    # This fixture replaces the old p(0)=12000 boundary. g=9e-9 makes
    # ignoring the new condition distinguishable from that old Dirichlet case.
    assert abs(pressure[0] - 12000.) > 1.
    assert math.isclose(pressure[2], 4000., rel_tol=0., abs_tol=1.e-10)
    assert abs(inflow[0] - (9.e-9 - 5.e-13 * pressure[0])) < 1.e-9 * scale
    print(f"Robin p0={pressure[0]:.17g}, inflow={inflow[0]:.17g}, "
          f"maximum relative conservation defect={defect/scale:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--portal", action="store_true")
    parser.add_argument("--robin", action="store_true")
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
            watchdog = shutil.which("timeout") or shutil.which("gtimeout")
            assert watchdog, "MPI tests require GNU timeout or gtimeout"
            command = [args.mpiexec, "-np", str(args.ranks),
                       watchdog, "--kill-after=2s", "105s"] + command
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
        if args.robin:
            assert not args.portal
            check_robin_reference(nodes, vessels, terminals, pairs)
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
