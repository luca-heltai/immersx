#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Plot exact matched terminal radius/flow records; no generation or solve."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tables(directory, stem):
    records = []
    files = sorted(directory.glob(stem + '-*.csv'))
    assert len(files) == 8, (stem, len(files))
    for path in files:
        with path.open(newline='') as stream:
            records.extend(csv.DictReader(stream))
    return records


def extract(directory):
    nodes_raw = read_tables(directory, 'nodes')
    nodes = {int(row['id']): row for row in nodes_raw}
    terminals_raw = read_tables(directory, 'terminals')
    terminals = {int(row['id']): row for row in terminals_raw}
    assert len(nodes) == len(nodes_raw) and len(terminals) == len(terminals_raw)
    vessels = read_tables(directory, 'vessels')
    assert len({int(row['id']) for row in vessels}) == len(vessels)
    adjacent = defaultdict(list)
    for vessel in vessels:
        for endpoint in (0, 1):
            node = int(vessel['node' + str(endpoint)])
            assert node in nodes
            adjacent[node].append((vessel, endpoint))
    triplets = read_tables(directory, 'triplets')
    assert len(triplets) == 64 and len({int(row['territory']) for row in triplets}) == 64
    roles = [('arterial', 0), ('portal', 2), ('venous', 1)]
    rows = []
    for triplet in sorted(triplets, key=lambda row: int(row['territory'])):
        territory = int(triplet['territory'])
        for role, system in roles:
            terminal_id = int(triplet[role])
            terminal = terminals[terminal_id]
            node_id = int(terminal['node'])
            assert int(terminal['territory']) == territory and int(terminal['system']) == system
            assert len(adjacent[node_id]) == 1, (role, territory, node_id)
            vessel, endpoint = adjacent[node_id][0]
            assert int(vessel['system']) == system
            radius = float(vessel['radius' + str(endpoint)])
            signed_q = float(vessel['flow_m3_s'])
            q = abs(signed_q)
            q_triplet = float(triplet[role + '_flow'])
            assert all(math.isfinite(v) and v > 0 for v in (radius, q, q_triplet))
            relative_difference = abs(q - q_triplet) / max(q, q_triplet)
            assert relative_difference < 1e-9, (role, territory, relative_difference)
            rows.append(dict(territory_id=territory, role=role, system=system,
                             terminal_id=terminal_id, physical_node_id=node_id,
                             physical_vessel_id=int(vessel['id']), endpoint_index=endpoint,
                             radius_m=radius, absolute_flow_m3_s=q,
                             signed_leaf_vessel_flow_m3_s=signed_q,
                             triplet_role_flow_m3_s=q_triplet,
                             flow_relative_difference=relative_difference))
    assert len(rows) == 192 and len({r['terminal_id'] for r in rows}) == 192
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Rebuild scalar CSV from native owned CSV tables')
    parser.add_argument('--data-csv', type=Path, default=Path(__file__).resolve().parent / 'data.csv')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        rows = extract(args.source_dir.resolve())
        input_hashes = {p.name: sha(p) for p in sorted(args.source_dir.iterdir()) if p.is_file()}
    else:
        with args.data_csv.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        input_hashes = {args.data_csv.name: sha(args.data_csv)}
    assert len(rows) == 192
    for row in rows:
        assert math.isfinite(float(row['radius_m'])) and float(row['radius_m']) > 0
        assert math.isfinite(float(row['absolute_flow_m3_s'])) and float(row['absolute_flow_m3_s']) > 0
    assert len({(int(r['territory_id']), r['role']) for r in rows}) == 192
    with (out / 'data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, FixedLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'liver-matched-radius-flow-v1'})
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    fig.subplots_adjust(left=.12, right=.97, top=.80, bottom=.28)
    fig.suptitle('Matched terminal radius and reference flow', x=.12, y=.975,
                 ha='left', fontsize=16, fontweight='bold')
    fig.text(.12, .912, 'Native successful 64-triplet reference — 192 exact endpoint traces',
             fontsize=10.8, color='#444444')
    colours = {'arterial': '#c53e3e', 'portal': '#8b5caf', 'venous': '#2e73a5'}
    labels = {'arterial': 'Hepatic artery', 'portal': 'Portal vein', 'venous': 'Hepatic veins'}
    stats = {}
    for role in ('arterial', 'portal', 'venous'):
        group = [row for row in rows if row['role'] == role]
        assert len(group) == 64
        q = [float(row['absolute_flow_m3_s']) * 1e6 for row in group]
        r = [float(row['radius_m']) * 1e3 for row in group]
        ax.scatter(q, r, s=24, color=colours[role], alpha=.78, linewidths=.25,
                   edgecolors='white', label=labels[role] + ' (64)', zorder=3)
        stats[role] = {'terminals': len(group), 'radius_min_m': min(r) / 1e3,
                       'radius_max_m': max(r) / 1e3, 'absflow_min_m3_s': min(q) / 1e6,
                       'absflow_max_m3_s': max(q) / 1e6}
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Absolute terminal reference flow [mL/s]', labelpad=9)
    ax.set_ylabel('Terminal endpoint radius [mm]', labelpad=9)
    ax.set_ylim(.1, 2.4)
    ax.yaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.grid(which='major', color='#d3d9dd', linewidth=.8)
    ax.grid(which='minor', color='#e9ecee', linewidth=.45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=True, framealpha=.94, edgecolor='none', fontsize=10)
    fig.text(.12, .16,
             'Each radius is taken at the matched leaf-vessel endpoint; no junction averaging.\n'
             'Retained trunk and root radii are excluded. Flow is the exported rigid design reference.\n'
             'Portal reservoir pressure 8.5 mmHg is a modelling sensitivity, not patient calibration.\n'
             'These model relationships do not establish physiological validity or a coupled transient.',
             fontsize=9.2, color='#444444', va='top', linespacing=1.45)
    for suffix in ('png', 'svg'):
        metadata = {'Date': None} if suffix == 'svg' else {'Software': 'Matplotlib'}
        fig.savefig(out / ('terminal-radius-flow.' + suffix), dpi=200, facecolor='white', metadata=metadata)
    plt.close(fig)
    svg = out / 'terminal-radius-flow.svg'
    svg.write_text(''.join(line.rstrip() + '\n' for line in svg.read_text().splitlines()))
    provenance = {'kind': 'EXACT_MATCHED_MODEL_TERMINAL_RECORDS',
                  'native_case': 'probe-proximal-64-portal85-attempt-01',
                  'native_generation_success': True, 'patient_calibrated': False,
                  'coupled_PDE_steps': 0, 'portal_reservoir_assumption_mmHg': 8.5,
                  'input_model': 'proximal-anatomical-inputs-1p5-3-2mm-v1',
                  'matching': 'triplet TerminalId -> terminal.node -> unique incident vessel -> radius0 or radius1 at that exact endpoint',
                  'flow': 'abs(exported leaf-vessel signed flow), checked against the same triplet role flow',
                  'matching_flow_consistency_relative_tolerance': 1e-9,
                  'maximum_observed_relative_flow_difference': max(float(r['flow_relative_difference']) for r in rows),
                  'no_averaging_fitting_new_geometry_or_hydraulic_solve': True,
                  'source_hashes': input_hashes, 'script_sha256': sha(Path(__file__).resolve()),
                  'matplotlib_version': matplotlib.__version__, 'per_role_ranges': stats,
                  'text_normalization': {'csv': 'LF', 'svg': 'rstrip each line'},
                  'output_hashes': {name: sha(out / name) for name in ('data.csv', 'terminal-radius-flow.png', 'terminal-radius-flow.svg')}}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'rows': len(rows), 'stats': stats,
                      'maximum_observed_relative_flow_difference': provenance['maximum_observed_relative_flow_difference']}, indent=2))


if __name__ == '__main__':
    main()
