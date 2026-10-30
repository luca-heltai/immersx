#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Revalidate completed native MPI experiments and publish measured SVG charts."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

from render_vascular_figures import dot, line, start, text
from run_vascular_benchmarks import digest, validate

COLORS = ('#1768ac', '#bd452d', '#398339', '#7954a1', '#9b691c')


def growth_work(directory, ranks):
    """Summarize recorded producer work, separately from vessel ownership."""
    rows, hashes = [], {}
    for rank in range(ranks):
        path = directory / f'growth-{rank}.csv'
        with path.open(newline='') as stream:
            rows.append(list(csv.DictReader(stream)))
        assert rows[-1], path
        hashes[path.name] = digest(path)
    result = {}
    for quantity in ('sampled_points', 'proposed_insertions', 'accepted_insertions'):
        counts = [sum(int(row[quantity]) for row in rank_rows) for rank_rows in rows]
        assert all(int(row[quantity]) >= 0 for rank_rows in rows for row in rank_rows)
        assert sum(counts) > 0, quantity
        result[quantity+'_by_rank'] = counts
        result[quantity+'_max_over_mean'] = max(counts)/(sum(counts)/ranks)
    result['startup_proposer_fraction'] = sum(int(rank_rows[0]['proposed_insertions']) > 0
                                               for rank_rows in rows)/ranks
    return result, hashes


def load(runs):
    records, provenance, names = [], [], set()
    for run in sorted(runs):
        manifest_path = run / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        identity = manifest['identity']
        for filename, key in (('vascular_network_profile', 'executable_sha256'),
                              ('libimmersx_lib.so', 'library_sha256')):
            assert digest(run / 'bin' / filename) == identity[key], filename
        planned = identity['cases']
        pending = [c['name'] for c in planned if not (run / c['name'] / 'case.json').is_file()
                   or json.loads((run / c['name'] / 'case.json').read_text())['status'] != 'complete']
        assert not pending, f'Experiment is incomplete: {run}: {pending}'
        provenance.append(dict(experiment=run.name, manifest_sha256=digest(manifest_path),
                               commit=manifest['commit'], executable_sha256=identity['executable_sha256'],
                               library_sha256=identity['library_sha256']))
        for case in planned:
            assert case['name'] not in names, f'Duplicate measurement: {case["name"]}'
            names.add(case['name'])
            path = run / case['name'] / 'case.json'
            record = json.loads(path.read_text())
            assert record['case'] == case and record['returncode'] == 0, path
            measured = validate(path.parent / 'network', case)
            assert measured == record['measurements'], f'Native tables changed: {path}'
            measured['work'], growth_hashes = growth_work(path.parent / 'network', case['ranks'])
            records.append((case, measured))
            provenance.append(dict(case=case['name'], case_sha256=digest(path),
                                   growth_sha256=growth_hashes))
    assert records, 'No completed measurements supplied'
    return sorted(records, key=lambda r: r[0]['name']), provenance


def group(case):
    base = case['terminals'] if case['kind'] == 'strong' else case['terminals'] // case['ranks']
    return case['kind'], base


def label(key):
    return f'{key[0]}, T={key[1]:,}' + (' per rank' if key[0] == 'weak' else '')


def chart(svg, origin, title, series, logarithmic=False):
    """Median curves with full observed seed ranges; no fitted performance model."""
    x0, y0 = origin
    width, height = 440, 205
    text(svg, x0, y0-28, title, 15)
    points = [p for values in series.values() for p in values]
    assert points
    xs = sorted({x for x, _ in points})
    ys = [y for _, y in points]
    assert all(math.isfinite(v) and v >= 0 for v in ys)
    transform = math.log10 if logarithmic else lambda y: y
    if logarithmic:
        assert min(ys) > 0
    lower = min(map(transform, ys)) if logarithmic else 0.
    upper = max(map(transform, ys))
    span = max(upper-lower, .1) if logarithmic else (upper or 1.)
    lower -= .03*span if logarithmic else 0.
    upper += .08*span
    px = lambda x: x0 + width*(math.log2(x)-math.log2(xs[0])) / max(math.log2(xs[-1]/xs[0]), 1.)
    py = lambda y: y0+height-height*(transform(y)-lower)/(upper-lower)
    for i in range(5):
        value = lower + i*(upper-lower)/4
        y = y0+height-i*height/4
        line(svg, (x0, y), (x0+width, y), '#dce1e5')
        text(svg, x0-8, y+4, f'{10**value if logarithmic else value:.3g}', 11, 'end')
    for x in xs:
        text(svg, px(x), y0+height+20, str(x), 12, 'middle')
    text(svg, x0+width/2, y0+height+40, 'MPI ranks', 12, 'middle')
    for index, (key, values) in enumerate(sorted(series.items())):
        color = COLORS[index % len(COLORS)]
        bins = defaultdict(list)
        for x, y in values:
            bins[x].append(y)
        previous = None
        for x, values in sorted(bins.items()):
            centre = (px(x), py(statistics.median(values)))
            line(svg, (px(x), py(min(values))), (px(x), py(max(values))), color, 1.5)
            for endpoint in (min(values), max(values)):
                line(svg, (px(x)-4, py(endpoint)), (px(x)+4, py(endpoint)), color, 1.5)
            if previous:
                line(svg, previous, centre, color, 2.)
            dot(svg, centre, 3, color)
            previous = centre
        text(svg, x0, y0+height+62+17*index, label(key), 11)
        line(svg, (x0+width-28, y0+height+58+17*index),
             (x0+width, y0+height+58+17*index), color, 3.)


def plot(records, metrics, title):
    svg = start(1180, 110+420*((len(metrics)+1)//2), title)
    text(svg, 40, 32, title, 21)
    text(svg, 40, 55, 'Medians and full observed seed ranges; each point uses a completed, independently rechecked native run.', 13)
    for index, (name, getter, logarithmic) in enumerate(metrics):
        series = defaultdict(list)
        for case, measured in records:
            series[group(case)].append((case['ranks'], getter(case, measured)))
        chart(svg, (95+575*(index % 2), 112+420*(index//2)), name, series, logarithmic)
    return '\n'.join(svg+['</svg>', '']).encode()


def render(records, provenance):
    baselines = {(group(c), c['seed']): float(m['summary']['generation_seconds'])
                 for c, m in records if c['ranks'] == 1}
    for case, _ in records:
        assert (group(case), case['seed']) in baselines, f'Missing one-rank baseline: {case["name"]}'
    ratio = lambda c, m: baselines[group(c), c['seed']] / float(m['summary']['generation_seconds'])
    maximum_phase = lambda m, phase: max(r[phase] for r in m['growth_rank_totals'])
    assets = {}
    assets['scaling.svg'] = plot(records, [
        ('Generation wall time [s], logarithmic', lambda c, m: float(m['summary']['generation_seconds']), True),
        ('Same-seed T(1) / T(P)', ratio, False),
    ], 'Measured vascular generation: strong and weak protocols')
    assets['resources.svg'] = plot(records, [
        ('Maximum per-rank peak RSS [MiB]', lambda c, m: max(float(r['peak_rss_kb']) for r in m['ranks'])/1024, False),
        ('Physical ownership: maximum / mean', lambda c, m: m['owned_vessel_max_over_mean'], False),
        ('Summed logical MPI input and posted bytes [MiB]', lambda c, m: (m['mpi']['posted_send_bytes']+m['mpi']['collective_input_bytes'])/2**20, True),
        ('Summed rank time inside selected MPI calls [s]', lambda c, m: m['mpi']['seconds'], True),
        ('Maximum rank physiological-update time [s]', lambda c, m: maximum_phase(m, 'physiology_seconds'), True),
        ('First leaf range: writer rank-batch fraction', lambda c, m: m['growth'][0]['active_writer_fraction'], False),
        ('Proposal counts: maximum / mean', lambda c, m: m['work']['proposed_insertions_max_over_mean'], False),
        ('First leaf range: fraction of ranks proposing', lambda c, m: m['work']['startup_proposer_fraction'], False),
    ], 'Measured memory, MPI activity, physiology, ownership and producer work')
    assets['quality.svg'] = plot(records, [
        ('Vascular volume fraction', lambda c, m: float(m['summary']['volume_fraction']), False),
        ('Geometric arterial coverage', lambda c, m: float(m['summary']['arterial_coverage']), False),
        ('Median physical vessel length / cube side', lambda c, m: m['length_quantiles']['0.5']/c['side'], False),
        ('Median junction-ray angle [degrees]', lambda c, m: m['junction_angle_quantiles']['0.5'], False),
    ], 'Observed partition and seed variability: synthetic tree quality')
    rows = []
    for case, measured in records:
        summary = measured['summary']
        row = dict(case=case['name'], kind=case['kind'], terminals=case['terminals'], ranks=case['ranks'], seed=case['seed'])
        row.update({k: summary[k] for k in ('generation_seconds', 'total_seconds', 'sampled_points', 'accepted_insertions',
                                          'vessels', 'numerical_cells', 'volume_fraction', 'arterial_coverage', 'maximum_flow_defect')})
        row.update(speedup_or_weak_time_ratio=ratio(case, measured),
                   maximum_rank_peak_rss_kib=max(float(r['peak_rss_kb']) for r in measured['ranks']),
                   summed_rank_peak_rss_kib=sum(float(r['peak_rss_kb']) for r in measured['ranks']),
                   owned_vessel_max_over_mean=measured['owned_vessel_max_over_mean'],
                   physiology_max_rank_seconds=maximum_phase(measured, 'physiology_seconds'),
                   startup_writer_fraction=measured['growth'][0]['active_writer_fraction'],
                   minimum_reference_area=measured['minimum_reference_area'])
        row.update({key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in measured['work'].items()})
        row.update({'mpi_'+key: value for key, value in measured['mpi'].items()})
        for quantity in ('length', 'radius', 'junction_angle', 'terminal_distance'):
            row.update({quantity+'_q'+str(round(float(q)*100)): v for q, v in measured[quantity+'_quantiles'].items()})
        rows.append(row)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    assets['measurements.csv'] = stream.getvalue().encode()
    assets['manifest.json'] = (json.dumps(dict(
        format_version=1, publisher_sha256=digest(Path(__file__)),
        conventions='Median curves; whiskers span all supplied seeds. Strong ratio is T(1)/T(P); weak ratio is the same elapsed-time ratio at increasing global size. MPI bytes are logical posted sends plus collective input, not wire traffic; MPI seconds sum rank time. RSS sums double-count shared pages and sum individual peaks rather than measuring simultaneous node memory. Writer ownership does not measure CPU-active proposal producers. Work imbalance uses cumulative rank-local sample, proposal and accepted-insertion counts. Startup proposer fraction counts ranks with at least one proposal in the first leaf range; it is not CPU utilization. No fitted model or physiological acceptance thresholds.',
        provenance=provenance,
        assets={name: hashlib.sha256(data).hexdigest() for name, data in assets.items()}), indent=2, sort_keys=True)+'\n').encode()
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Completed run_vascular_benchmarks.py experiment directories')
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).resolve().parents[1]/'doc'/'_static'/'vascular-benchmarks')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    runs = [p.resolve() for p in args.runs]
    target = args.destination.resolve()
    if any(target == run or run in target.parents or target in run.parents for run in runs):
        parser.error('Keep published assets separate from native experiment directories')
    records, provenance = load(runs)
    assets = render(records, provenance)
    for name, data in assets.items():
        if name.endswith('.svg'):
            ET.fromstring(data)
    if args.check:
        different = [name for name, data in assets.items() if not (args.destination/name).is_file()
                     or (args.destination/name).read_bytes() != data]
        if different:
            parser.error('Measured benchmark assets differ: '+', '.join(different))
        print(f'Checked {len(records)} native cases and {len(assets)} measured benchmark assets')
    else:
        args.destination.mkdir(parents=True, exist_ok=True)
        for name, data in assets.items():
            (args.destination/name).write_bytes(data)
        print(f'Published {len(assets)} measured benchmark assets from {len(records)} native cases in {args.destination}')


if __name__ == '__main__':
    main()
