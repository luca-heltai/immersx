# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Plot archived setup counts and interval histograms; never launch a solver.

python3 plot.py --output-dir /tmp/liver-setup-figures
python3 plot.py --raw-runs SMALL_SETUP_DIR MEDIUM_SETUP_DIR --output-dir /tmp/rebuilt
Requires NumPy and Matplotlib. Raw runs need measured-setup-summary.json and
all native flow/mesh_scale-*.csv files. The published histogram retains counts,
not rounded probabilities. The inside flag means native overlap, not full
containment of the interval or its lifted wall.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LICENSE = '# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n'


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith('#')))


def write_csv(path, rows):
    with path.open('w') as stream:
        stream.write(LICENSE)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(runs, output):
    summaries, histograms, sources = [], [], {}
    edges = np.geomspace(1.e-5, 2., 81)
    for family, run in zip(('Small', 'Medium'), runs):
        path = run / 'measured-setup-summary.json'
        record = json.loads(path.read_text())
        assert record['setup_count_audit_pass']
        row = dict(family=family, background_tets=record['global_background_tets_independent_mesh_count'],
                   terminals_each_role=record['global_background_tets_independent_mesh_count'],
                   intervals=record['global_1d_cells'], short=record['discretization_short_cells'],
                   no_overlap=record['discretization_exterior_cells'], vertex_exterior=record['vertex_exterior_cells'])
        row.update(zip(('displacement', 'velocity', 'flow', 'multiplier'), record['coupled_block_dofs']))
        row['total_dofs'] = record['coupled_total_dofs']
        summaries.append(row)
        data = []
        pieces = sorted((run / 'flow').glob('mesh_scale-*.csv'))
        for piece in pieces:
            data.extend(read_csv(piece))
        assert len(data) == row['intervals']
        assert len({(v['vessel'], v['begin'], v['end']) for v in data}) == len(data)
        overlap = [v for v in data if int(v['inside'])]
        ratios = [float(v['length_m']) / float(v['h_max_m']) for v in overlap]
        assert sum(int(v['short_cell']) for v in data) == row['short']
        assert sum(r < .5 * (1. - 1.e-12) for r in ratios) == row['short']
        counts, _ = np.histogram(ratios, edges)
        assert int(sum(counts)) == len(overlap) == row['intervals'] - row['no_overlap']
        for low, high, count in zip(edges[:-1], edges[1:], counts):
            histograms.append(dict(family=family, lower_L_over_Hmax=format(low, '.17g'),
                                   upper_L_over_Hmax=format(high, '.17g'), count=int(count)))
        sources[family] = {'run': str(run), 'summary': record,
                           'sha256': {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [path, run / 'run.json', run / 'run.log'] + pieces},
                           'runtime': json.loads((run / 'run.json').read_text())['runtime_manifest']}
    write_csv(output / 'block-counts.csv', summaries)
    write_csv(output / 'interval-histograms.csv', histograms)
    provenance = {'SPDX-License-Identifier': 'Apache-2.0 WITH LLVM-exception', 'sources': sources,
                  'scope': 'Complete setup only, no initialization or IDA/time integration; no inf-sup certificate.',
                  'H_definition': 'Tissue cell diameter; Hmax is maximum diameter over native overlap cells, not volume^(1/3).',
                  'short_rule': 'L < 0.5 Hmax (1 - 1e-12); native short_cell flags crosschecked against ratios.',
                  'histogram': '80 logarithmic bins over [1e-5,2]; all overlap intervals retained, no-overlap intervals excluded.',
                  'generation_release': 'https://github.com/luca-heltai/immersx/releases/tag/liver-h-scaled-2026-11-10'}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def plot(data, output):
    rows = read_csv(data / 'block-counts.csv')
    hist = read_csv(data / 'interval-histograms.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'liver-setup-v1',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.6, 4.1), layout='constrained')
    colors = ('#4477AA', '#66CCEE', '#228833', '#CCBB44')
    left = np.zeros(len(rows))
    for key, label, color in zip(('displacement', 'velocity', 'flow', 'multiplier'),
                                  ('Displacement', 'Velocity', 'Native flow', 'Multiplier'), colors):
        values = np.array([int(row[key]) for row in rows])
        ax.barh(range(len(rows)), values / 1000., left=left / 1000., label=label, color=color, height=.45)
        left += values
    ax.set_yticks(range(len(rows)), [f"{r['family']}\n{int(r['terminals_each_role']):,} terminals/role" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Coupled degrees of freedom (thousands)')
    ax.set_xlim(0, 1110)
    for i, total in enumerate(left):
        ax.text(total / 1000. + 12, i, f'{int(total):,}', va='center')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.22), ncol=2, frameon=False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    save(fig, output / 'block-composition')
    fig, ax = plt.subplots(figsize=(7.6, 4.5), layout='constrained')
    ax.axvspan(1.e-5, .5, color='#999999', alpha=.12)
    for row, color in zip(rows, ('#4477AA', '#CC6677')):
        bins = [r for r in hist if r['family'] == row['family']]
        edges = [float(r['lower_L_over_Hmax']) for r in bins] + [float(bins[-1]['upper_L_over_Hmax'])]
        overlap = int(row['intervals']) - int(row['no_overlap'])
        counts = np.array([int(r['count']) for r in bins])
        assert sum(counts) == overlap
        ax.stairs(100. * counts / overlap, edges, color=color, linewidth=1.8,
                  label=f"{row['family']}: {overlap:,} overlap intervals")
    ax.axvline(.5, color='#333333', linestyle='--', linewidth=1.2, label='Short threshold: 0.5')
    ax.set_xscale('log')
    ax.set_xlim(1.e-5, 2.)
    ax.set_xlabel(r'Interval length $L/H_{\max}$ (overlap intervals only)')
    ax.set_ylabel('Overlap intervals per logarithmic bin (%)')
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    ax.grid(alpha=.15)
    save(fig, output / 'interval-distribution')


def save(fig, path):
    for suffix in ('svg', 'png'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=180,
                    metadata={'Date': None} if suffix == 'svg' else {})
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-runs', nargs=2, type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_runs:
        aggregate(args.raw_runs, args.output_dir)
    plot(args.output_dir if args.raw_runs else HERE, args.output_dir)


if __name__ == '__main__':
    main()
