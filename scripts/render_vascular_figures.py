#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Render owned native C++ output tables; this script performs no growth."""
import argparse
from collections import defaultdict
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path

CASES = ('cube_uniform', 'cube_edge', 'sphere_uniform', 'sphere_center',
         'arteriovenous', 'rediscretized')
ASSETS = ('domains.svg', 'meshes.svg', 'length-distribution.svg',
          'terminal-density.svg', 'targets.svg', 'arteriovenous.svg',
          'rediscretization.svg', 'summary.csv', 'effective-parameters.txt', 'manifest.json')
COLORS = ('#b62737', '#2166ac', '#b05b16', '#277b55')
SCALE = 200
TITLES = ('Uniform cube', 'Cube: edge refinement', 'Uniform sphere', 'Sphere: central zoom')
EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
         (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7))


def tables(directory, kind):
    result = []
    files = sorted(directory.glob(kind + '-*.csv'))
    assert files, (directory, kind)
    for file in files:
        with file.open(newline='') as stream:
            for row in csv.DictReader(stream):
                assert None not in row and None not in row.values(), (file, row)
                result.append(row)
    return result


def keyed(rows):
    result = {int(row['id']): row for row in rows}
    assert len(result) == len(rows), 'Duplicate owned physical ids'
    return result


def xyz(row, suffix=''):
    return tuple(float(row[d + suffix]) for d in ('x', 'y', 'z'))


def close(a, b, tolerance=2e-9):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=1e-28), (a, b)


def validate(directory):
    with (directory / 'summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 1 and summary[0]['valid'] == '1', directory
    s = summary[0]
    data = {kind: tables(directory, kind) for kind in
            ('nodes', 'vessels', 'terminals', 'territories', 'pairs', 'intervals', 'background', 'quadrature')}
    for kind in ('nodes', 'vessels', 'terminals'):
        data[kind] = keyed(data[kind])
    assert len(data['vessels']) == int(s['vessels'])
    assert len(data['nodes']) == int(s['nodes'])
    assert len(data['intervals']) == int(s['numerical_cells'])
    assert len(data['background']) == int(s['background_cells'])
    assert int(s['arterial_terminals']) + int(s['venous_terminals']) == len(data['terminals'])
    assert int(s['pairs']) == len(data['pairs'])
    close(sum(float(q['weight']) for q in data['quadrature']), float(s['background_volume']))
    assert s['coverage_evaluated'] == '1' and s['reference_evaluated'] == '1'
    if s['mode'] == 'generate':
        assert s['calibration_status'] == 'converged'
        assert abs(float(s['volume_fraction']) / float(s['target_volume_fraction']) - 1) <= .05
        coverage = float(s['dual_coverage'] if data['pairs'] else s['arterial_coverage'])
        assert coverage >= float(s['minimum_coverage'])
    integrated = defaultdict(lambda: [0., 0.])
    intervals = defaultdict(list)
    for cell in data['intervals']:
        assert cell['inside'] == '1', 'These illustrated networks must be fully internal'
        vessel = data['vessels'][int(cell['vessel'])]
        a, b = float(cell['begin']), float(cell['end'])
        r0, r1 = float(vessel['radius0']), float(vessel['radius1'])
        ra, rb = r0 + a * (r1-r0), r0 + b * (r1-r0)
        length = (b-a) * float(vessel['length'])
        # Independent exact integrals of a linearly tapered physical radius.
        integrated[int(cell['vessel'])][0] += math.pi * length * (ra*ra + ra*rb + rb*rb) / 3
        integrated[int(cell['vessel'])][1] += 8 * float(vessel['viscosity']) * length * (ra*ra + ra*rb + rb*rb) / (3 * math.pi * ra**3 * rb**3)
        intervals[int(cell['vessel'])].append((a, b))
    children = defaultdict(list)
    for vessel in data['vessels'].values():
        children[int(vessel['node0'])].append(vessel)
    for id_, vessel in data['vessels'].items():
        close(integrated[id_][0], float(vessel['full_volume']))
        close(integrated[id_][1], float(vessel['resistance']))
        ordered = sorted(intervals[id_])
        close(ordered[0][0], 0.)
        close(ordered[-1][1], 1.)
        for left, right in zip(ordered, ordered[1:]):
            close(left[1], right[0])
        descendants = children[int(vessel['node1'])]
        if len(descendants) > 1:
            close(float(vessel['radius1'])**3, sum(float(v['radius0'])**3 for v in descendants))
    close(sum(float(v['full_volume']) for v in data['vessels'].values()), float(s['inside_volume']))
    seen = set()
    for bed in data['pairs']:
        assert int(bed['territory']) not in seen
        seen.add(int(bed['territory']))
        assert float(bed['resistance']) > 0 and float(bed['flow']) > 0
        a, v = (data['terminals'][int(bed[k])] for k in ('arterial_terminal', 'venous_terminal'))
        assert a['system'] == 'arterial' and v['system'] == 'venous' and xyz(a) != xyz(v)
        assert a['territory'] == v['territory'] == bed['territory']
        pa, pv = (float(data['nodes'][int(t['node'])]['pressure']) for t in (a, v))
        close(pa-pv, float(bed['resistance']) * float(bed['flow']))
    if data['pairs']:
        assert seen == {int(t['id']) for t in data['territories']}
    data['summary'] = s
    return data


def start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<title>{escape(title)}</title>',
            '<style>text{font-family:DejaVu Sans,sans-serif;fill:#24303b}line,path,polygon{stroke-linejoin:round}</style>']


def text(svg, x, y, value, size=14, anchor='start'):
    svg.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')


def line(svg, a, b, color='#87929b', width=.6, extra=''):
    svg.append(f'<line x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" y2="{b[1]:.3f}" stroke="{color}" stroke-width="{width:.4f}" {extra}/>')


def dot(svg, p, radius=2, color='#24303b'):
    svg.append(f'<circle cx="{p[0]:.3f}" cy="{p[1]:.3f}" r="{radius}" fill="{color}"/>')


def save(path, svg):
    path.write_text('\n'.join(svg + ['</svg>', '']))


def project(p, center, scale=SCALE):
    x, y, z = (value-.5 for value in p)
    return center[0] + scale * .82 * (x-y), center[1] + scale * (.4*(x+y)-.82*z)


def hull(points):
    points = sorted(set(points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def half(items):
        result = []
        for p in items:
            while len(result) > 1 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result[:-1]
    return half(points) + half(reversed(points))


def network(svg, data, center, numeric=False, pairs=False):
    polygon = hull([project(xyz(c, str(v)), center) for c in data['background'] for v in range(8)])
    svg.append('<polygon points="' + ' '.join(f'{x:.3f},{y:.3f}' for x, y in polygon) + '" fill="#fafbfd" stroke="#ccd2d8" stroke-width="1"/>')
    if pairs:
        for bed in data['pairs']:
            a, v = (data['terminals'][int(bed[k])] for k in ('arterial_terminal', 'venous_terminal'))
            line(svg, project(xyz(a), center), project(xyz(v), center), '#65727e', .8, 'stroke-dasharray="3 2"')
    if numeric:
        for cell in data['intervals']:
            vessel = data['vessels'][int(cell['vessel'])]
            a, b = project(xyz(cell, '0'), center), project(xyz(cell, '1'), center)
            line(svg, a, b, '#b62737', 8 * float(vessel['radius0']) * SCALE)
            dot(svg, a, .9, '#2166ac')
            dot(svg, b, .9, '#2166ac')
    else:
        for vessel in sorted(data['vessels'].values(), key=lambda v: float(v['y0']) + float(v['y1'])):
            color = COLORS[0] if vessel['system'] == 'arterial' else COLORS[1]
            width = 4 * (float(vessel['radius0']) + float(vessel['radius1'])) * SCALE
            line(svg, project(xyz(vessel, '0'), center), project(xyz(vessel, '1'), center), color, width)
    if pairs:
        for terminal in data['terminals'].values():
            dot(svg, project(xyz(terminal), center), 1.6,
                COLORS[0] if terminal['system'] == 'arterial' else COLORS[1])
    for node in data['nodes'].values():
        if node['port'] != '0':
            dot(svg, project(xyz(node), center), 3)


def mesh(svg, data, center, plane=.5):
    edges = set()
    for cell in data['background']:
        vertices = [xyz(cell, str(v)) for v in range(8)]
        if min(p[1] for p in vertices) > plane + 1e-12 or max(p[1] for p in vertices) < plane - 1e-12:
            continue
        for i, j in EDGES:
            edges.add(tuple(sorted((vertices[i], vertices[j]))))
    for a, b in sorted(edges):
        line(svg, project(a, center), project(b, center), '#677c8d', .45)


def axes(svg, origin, width, height, y_max, ticks):
    x, y = origin
    for value, label in ticks:
        yy = y - height * value / y_max
        line(svg, (x, yy), (x+width, yy), '#e1e5e9')
        text(svg, x-8, yy+4, label, 11, 'end')
    line(svg, (x, y), (x+width, y), '#67727c', 1)
    line(svg, (x, y), (x, y-height), '#67727c', 1)


def histogram(values, boundaries):
    counts = [0] * (len(boundaries)-1)
    for value in values:
        for i, end in enumerate(boundaries[1:]):
            if value <= end:
                counts[i] += 1
                break
    return counts


def render(root, output=None):
    data = {name: validate(root / name) for name in CASES}
    original, refined = data['cube_uniform'], data['rediscretized']
    for kind in ('nodes', 'vessels', 'terminals'):
        assert original[kind].keys() == refined[kind].keys()
        for id_, a in original[kind].items():
            b = refined[kind][id_]
            for key in a:
                if key not in ('owner', 'pressure', 'flow', 'shear0', 'shear1', 'target_flow', 'actual_outflow'):
                    assert a[key] == b[key], (kind, id_, key)
    close(float(original['summary']['volume_fraction']), float(refined['summary']['volume_fraction']))
    assert len(refined['intervals']) > len(original['intervals'])
    output = root / 'figures' if output is None else output
    output.mkdir(exist_ok=False)
    for name, title, draw in [('domains.svg', 'Physical vascular networks', network),
                              ('meshes.svg', 'Background refinement', mesh)]:
        svg = start(1120, 980, title)
        text(svg, 28, 30, title, 21)
        text(svg, 28, 54, 'Common oblique camera and spatial scale; coordinates and edges from native output.', 13)
        for i, case in enumerate(CASES[:4]):
            x, y = 280 + 560*(i % 2), 268 + 440*(i // 2)
            text(svg, x, y-170, TITLES[i], 18, 'middle')
            if name == 'meshes.svg':
                mesh(svg, data[case], (x, y), .0625 if i < 2 else .5)
            else:
                draw(svg, data[case], (x, y))
            s = data[case]['summary']
            text(svg, x, y+170, f"{s['arterial_terminals']} terminals; {s['vessels']} physical vessels; {s['numerical_cells']} line cells", 12, 'middle')
            text(svg, x, y+190, f"{s['background_cells']} background cells", 12, 'middle')
        footer = ('Red: arterial; black: inlet. Tube widths = 4 × physical diameter in every panel.'
                  if name == 'domains.svg' else 'Wireframes: cube cells intersect y = 0.0625; sphere cells intersect y = 0.5.')
        text(svg, 28, 958, footer, 13)
        save(output / name, svg)

    boundaries = [0., .25, .5, 1., 1.5, 2., math.inf]
    svg = start(1000, 460, 'Numerical interval length relative to local background diameter')
    text(svg, 25, 28, 'Numerical cells: length / h_min', 21)
    text(svg, 25, 52, 'h_min uses cells actually crossed by the interval. Short physical vessels remain physical vessels.', 13)
    axes(svg, (80, 355), 850, 260, 1., [(v, f'{v:.0%}') for v in (0, .25, .5, .75, 1)])
    for j, case in enumerate(CASES[:4]):
        values = [float(c['length_over_h']) for c in data[case]['intervals']]
        counts = histogram(values, boundaries)
        for i, count in enumerate(counts):
            x = 100 + i*140 + j*23
            height = 260 * count / len(values)
            svg.append(f'<rect x="{x}" y="{355-height:.3f}" width="20" height="{height:.3f}" fill="{COLORS[j]}"/>')
        text(svg, 80+j*230, 434, TITLES[j], 12)
        line(svg, (80+j*230, 412), (115+j*230, 412), COLORS[j], 4)
    for i, label in enumerate(('0–0.25', '0.25–0.5', '0.5–1', '1–1.5', '1.5–2', '>2')):
        text(svg, 145+i*140, 380, label, 12, 'middle')
    save(output / 'length-distribution.svg', svg)

    svg = start(1120, 480, 'Physical terminal density per tissue volume')
    text(svg, 25, 28, 'Physical arterial terminal density', 21)
    text(svg, 25, 51, 'Shell volumes estimated with native background quadrature; these are physical terminals, not line cells.', 13)
    for group, cases in enumerate((CASES[:2], CASES[2:4])):
        x0, y0 = 75 + 555*group, 365
        bins = [0., .125, .25, .5, .75, 1., math.sqrt(2)] if group == 0 else [0., .125, .25, .5, .75, 1.]
        title = 'Distance to chosen z edge / cube side' if group == 0 else 'Distance to sphere center / radius'
        axes(svg, (x0, y0), 450, 250, 5., [(i, str(i)) for i in range(6)])
        for j, case in enumerate(cases):
            def distance(row):
                p = xyz(row)
                return math.hypot(p[0], p[1]) if group == 0 else 2 * math.dist(p, (.5, .5, .5))
            counts = histogram([distance(t) for t in data[case]['terminals'].values() if t['system'] == 'arterial'], bins)
            volumes = [0.] * len(counts)
            for q in data[case]['quadrature']:
                for i, end in enumerate(bins[1:]):
                    if distance(q) <= end:
                        volumes[i] += float(q['weight'])
                        break
            previous = None
            for i, (count, volume) in enumerate(zip(counts, volumes)):
                if volume == 0:
                    previous = None
                    continue
                density = count / volume
                p = (x0 + 450*(i+.5)/len(counts), y0-250*math.log10(1+density)/5)
                if previous:
                    line(svg, previous, p, COLORS[2*group+j], 2)
                dot(svg, p, 3, COLORS[2*group+j])
                previous = p
            text(svg, x0+j*230, 441, TITLES[2*group+j], 12)
            line(svg, (x0+j*230, 418), (x0+35+j*230, 418), COLORS[2*group+j], 3)
        for i in range(len(bins)-1):
            text(svg, x0+450*(i+.5)/len(counts), 387, f'{bins[i]:g}–{bins[i+1]:.3g}', 11, 'middle')
        text(svg, x0+225, 463, title, 12, 'middle')
    text(svg, 25, 78, 'Vertical axis: log10(1 + terminals / m³). Gaps denote shells without quadrature volume.', 12)
    save(output / 'terminal-density.svg', svg)

    svg = start(1120, 460, 'Requested and achieved vascular metrics')
    text(svg, 25, 28, 'Volume fraction and geometric service coverage are separate targets', 21)
    for group, (title, maximum) in enumerate((('Volume fraction × 100000', 8.), ('Arterial geometric coverage', 1.))):
        x0, y0 = 75+555*group, 355
        axes(svg, (x0, y0), 450, 260, maximum, [(maximum*i/4, f'{maximum*i/4:g}') for i in range(5)])
        text(svg, x0, 62, title, 15)
        for i, case in enumerate(CASES[:4]):
            s = data[case]['summary']
            actual = float(s['volume_fraction'])*1e5 if group == 0 else float(s['arterial_coverage'])
            target = float(s['target_volume_fraction'])*1e5 if group == 0 else float(s['minimum_coverage'])
            x = x0+55+110*i
            height = 260*actual/maximum
            svg.append(f'<rect x="{x-22}" y="{y0-height:.3f}" width="44" height="{height:.3f}" fill="{COLORS[i]}"/>')
            yy = y0-260*target/maximum
            line(svg, (x-32, yy), (x+32, yy), '#172a3a', 2)
            if group == 0:
                line(svg, (x, y0-260*target*.95/maximum), (x, y0-260*target*1.05/maximum), '#172a3a', 2)
            text(svg, x, 378, str(i+1), 13, 'middle')
    text(svg, 25, 413, 'Bars: achieved. Black marks: requested value (±5% for volume) or minimum coverage.', 13)
    text(svg, 25, 438, '1 uniform cube; 2 edge cube; 3 uniform sphere; 4 central sphere. Coverage is not a physiological perfusion proof.', 12)
    save(output / 'targets.svg', svg)

    svg = start(700, 610, 'Arterial and venous common territories')
    text(svg, 25, 30, 'Arterial and venous common territories', 21)
    network(svg, data['arteriovenous'], (350, 300), pairs=True)
    text(svg, 25, 545, 'Red: arteries. Blue: veins. Dashed links: positive 0D beds, with no assigned capillary volume.', 12)
    text(svg, 25, 570, f"{len(data['arteriovenous']['pairs'])} distinct terminal pairs; actual paired reference flows are positive.", 13)
    text(svg, 25, 594, 'Common oblique camera; tube widths = 4 × physical diameter.', 12)
    save(output / 'arteriovenous.svg', svg)

    svg = start(1120, 520, 'Rediscretize the same physical graph')
    text(svg, 25, 30, 'The same physical graph on two numerical discretizations', 21)
    for i, case in enumerate(('cube_uniform', 'rediscretized')):
        d = data[case]
        network(svg, d, (280+560*i, 255), numeric=True)
        text(svg, 280+560*i, 445, f"{len(d['vessels'])} physical vessels; {len(d['intervals'])} numerical cells", 15, 'middle')
    text(svg, 25, 483, 'Blue dots: numerical vertices. Stable physical ids, connectivity, radii, viscosity, volume and resistance are checked.', 12)
    text(svg, 25, 507, 'Tube widths = 4 × physical diameter. No growth occurs in the right panel.', 12)
    save(output / 'rediscretization.svg', svg)

    columns = ('case', 'processes', 'background_cells', 'vessels', 'arterial_terminals', 'venous_terminals', 'pairs',
               'numerical_cells', 'volume_fraction', 'target_volume_fraction', 'arterial_coverage', 'dual_coverage',
               'minimum_coverage', 'calibration_status', 'calibration_attempts')
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, lineterminator='\n')
        writer.writeheader()
        for case in CASES:
            s = data[case]['summary']
            row = {key: s[key] for key in columns if key != 'case'}
            for key in ('volume_fraction', 'target_volume_fraction', 'arterial_coverage', 'dual_coverage', 'minimum_coverage'):
                row[key] = f'{float(row[key]):.9g}'
            writer.writerow(dict(row, case=case))
    (output / 'effective-parameters.txt').write_text('\n'.join(
        '# ' + case + '\n' + (root / case / 'parameters.prm').read_text().replace(str(root), '<run>')
        for case in CASES))
    manifest = {'schema': 1, 'source': 'Owned output of the native vascular_network application',
                'projection': 'x=.82*(x-y), y=.4*(x+y)-.82*z; fixed common scale',
                'diameter_exaggeration': 4, 'density_volume': 'Native QGauss(2) background weights binned by physical position',
                'assets_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                  for name in ASSETS if name != 'manifest.json'}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    render(parser.parse_args().run.resolve())
