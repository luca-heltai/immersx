#!/usr/bin/env python3
"""Check default fitted-mesh generation and its exported cube geometry."""
import argparse
import collections
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh, summary = output/'mesh.msh', output/'summary.json'
    subprocess.run([str(args.executable.resolve()), '--refinement', '0',
                    '--output', str(mesh), '--summary', str(summary)], check=True)
    data = json.loads(summary.read_text())
    assert math.isclose(data['radius'], math.sqrt(3.0605e-4/math.pi),
                        rel_tol=0., abs_tol=1.e-16)
    assert data['cells'] == 2016
    assert data['vessel_cells'] == 240
    assert data['wall_faces'] == 96 and data['open_wall_faces'] == 0
    assert data['min_cell_measure'] > 0
    assert abs(data['volume_relative_error']) < 1.e-12
    # The Q1 wall is a regular octagon, not an exact circular cylinder.
    expected_volume_error = 8*math.sin(2*math.pi/8)/(2*math.pi)-1
    assert math.isclose(data['vessel_volume_relative_error'], expected_volume_error,
                        rel_tol=0., abs_tol=1.e-12)
    assert data['max_wall_vertex_radius_error'] < 1.e-13*data['radius']

    # Read the legacy MSH format written by GridOut independently of the summary.
    lines = mesh.read_text().splitlines()
    start = lines.index('$NOD')
    count = int(lines[start+1])
    points = {int(p[0]): tuple(map(float, p[1:]))
              for line in lines[start+2:start+2+count] for p in [line.split()]}
    assert len(points) == count == 2509
    for d in range(3):
        assert math.isclose(min(p[d] for p in points.values()), -.120685,
                            rel_tol=0., abs_tol=1.e-14)
        assert math.isclose(max(p[d] for p in points.values()), .120685,
                            rel_tol=0., abs_tol=1.e-14)
    start = lines.index('$ELM')
    count = int(lines[start+1])
    materials = collections.Counter()
    for line in lines[start+2:start+2+count]:
        entry = list(map(int, line.split()))
        if entry[1] == 5:
            assert entry[4] == 8
            assert all(i in points for i in entry[5:])
            materials[entry[2]] += 1
    assert sorted(materials.values()) == [240, 1776]
    print('Verified default mesh counts, cube bounds and polygonal vessel volume')


if __name__ == '__main__':
    main()
