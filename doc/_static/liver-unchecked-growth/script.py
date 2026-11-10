#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Plot previously computed rigid bounds; does not solve a hydraulic model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / 'input-r034-rigid-capacity-audit-v1-report.json'
    grown_path = data_dir / 'attempt02-rigid-capacity-audit-v1-report.json'
    sources = json.loads((data_dir / 'source-records.json').read_text())
    for source in sources:
        path = data_dir / (source['name'] + '-report.json')
        assert sha(path) == source['local_original_json_normalized_sha256']
    initial = json.loads(input_path.read_text())
    grown = json.loads(grown_path.read_text())
    roles = ('arterial', 'portal', 'venous')
    rows = []
    for role in roles:
        a = initial['roles'][role]
        b = grown['roles'][role]
        assert math.isclose(a['target_Q_m3_s'], b['Q_target_m3_s'], rel_tol=1e-14)
        assert math.isclose(a['available_head_Pa'], b['available_head_Pa'], rel_tol=1e-14)
        components = b['weighted_mean_drop_by_class_Pa']
        assert math.isclose(sum(components.values()), b['minimum_drop_Pa'], rel_tol=1e-12, abs_tol=1e-9)
        for scenario, total, retained, generated, source_name, solution in (
            ('input_only', a['ideal_manifold_drop_Pa'], a['ideal_manifold_drop_Pa'], 0.0,
             input_path.name, 'input_only_ideal_manifold_optimum'),
            ('grown_64', b['minimum_drop_Pa'], components['frozen'], components['generated'],
             grown_path.name, 'grown_network_ideal_manifold_optimum'),
        ):
            rows.append(dict(role=role, scenario=scenario, current_solution=solution,
                             Q_target_m3_s=b['Q_target_m3_s'], total_minimum_drop_Pa=total,
                             retained_contribution_Pa=retained, generated_contribution_Pa=generated,
                             available_head_Pa=b['available_head_Pa'], source_json=source_name))
    with (out / 'data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, StrMethodFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'liver-rigid-capacity-v1'})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.8))
    fig.subplots_adjust(left=.075, right=.985, top=.70, bottom=.29, wspace=.34)
    fig.suptitle('Optimistic rigid pressure-drop bounds', x=.075, ha='left', y=.975,
                 fontsize=17, fontweight='bold')
    fig.text(.075, .915, 'Anatomical 64-territory probe — native validation failed',
             fontsize=11, color='#4c4c4c')
    grey, blue, orange = '#aab0b5', '#416b8d', '#df922e'
    fig.legend(handles=[Patch(facecolor=grey, label='Input-only optimum'),
                        Patch(facecolor=blue, label='Retained input in grown optimum'),
                        Patch(facecolor=orange, label='CCO-generated in grown optimum')],
               loc='upper left', bbox_to_anchor=(.064, .873), frameon=False,
               ncol=3, fontsize=9.5, columnspacing=1.55)
    labels = {'arterial': 'Hepatic artery', 'portal': 'Portal vein', 'venous': 'Hepatic veins'}
    for ax, role in zip(axes, roles):
        a = initial['roles'][role]
        b = grown['roles'][role]
        part = b['weighted_mean_drop_by_class_Pa']
        ax.bar(0, a['ideal_manifold_drop_Pa'], width=.56, color=grey, zorder=3)
        ax.bar(1, part['frozen'], width=.56, color=blue, zorder=3)
        ax.bar(1, part['generated'], bottom=part['frozen'], width=.56,
               color=orange, zorder=3)
        budget = b['available_head_Pa']
        limit = 11000 if role == 'arterial' else 1000
        ax.axhline(budget, color='#262626', linestyle=(0, (4, 3)), linewidth=1.25, zorder=4)
        ax.text(.48, budget + .028 * limit, f'Budget {budget:,.0f} Pa',
                ha='center', fontsize=9, color='#262626',
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.2))
        for x, height in [(0, a['ideal_manifold_drop_Pa']), (1, b['minimum_drop_Pa'])]:
            ax.text(x, height + .025 * limit, f'{height:,.0f}', ha='center',
                    va='bottom', fontsize=10, fontweight='bold')
        ax.text(1, part['frozen'] + .70 * part['generated'],
                f"{part['generated']:,.0f} Pa", ha='center', va='center',
                fontsize=9, color='#222222')
        ax.set_title(f"{labels[role]}\nQ = {b['Q_target_m3_s'] * 1e6:.3f} mL/s", pad=13)
        ax.set_xticks([0, 1], ['Input only', '+ CCO'])
        ax.set_xlim(-.53, 1.53)
        ax.set_ylim(0, limit)
        ax.set_ylabel('Minimum pressure drop [Pa]')
        ax.yaxis.set_major_locator(MultipleLocator(2500 if role == 'arterial' else 250))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y', color='#d8dde1', linewidth=.65, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', length=0, pad=9)
    fig.text(.075, .20,
             'Stacked contributions use identical currents in the grown-tree optimum; panel scales differ.',
             fontsize=9.3, va='top')
    fig.text(.075, .145,
             'Necessary per-role tests only — not physiological or matched-network validation.\n'
             'Equipressure caps; 0D, other-role and root-boundary losses omitted.\n'
             'Portal and hepatic venous budgets assume forward portal/venous arms.',
             fontsize=9.1, color='#444444', linespacing=1.45, va='top')
    for suffix in ('png', 'svg'):
        metadata = {'Date': None} if suffix == 'svg' else {'Software': 'Matplotlib'}
        fig.savefig(out / ('rigid-capacity-bounds.' + suffix), dpi=200,
                    facecolor='white', metadata=metadata)
    svg_path = out / 'rigid-capacity-bounds.svg'
    svg_path.write_text(''.join(line.rstrip() + '\n' for line in svg_path.read_text().splitlines()))
    plt.close(fig)
    provenance = {
        'kind': 'PLOT_OF_EXISTING_NECESSARY_RIGID_BOUNDS',
        'not_physiological_validation': True,
        'text_output_normalization': {'csv': 'LF line terminator', 'svg': 'Trailing whitespace stripped line by line'},
        'anatomical_witness_status': 'INVALID',
        'new_CCO_PDE_or_hydraulic_solve': False,
        'generated_contribution_definition': 'weighted_mean_drop_by_class_Pa.generated in the grown-network optimum; never a difference of separately optimized networks',
        'source_records': sources,
        'input_file_hashes': {path.name: sha(path) for path in (input_path, grown_path, data_dir / 'source-records.json')},
        'script_sha256': sha(Path(__file__).resolve()),
        'matplotlib_version': matplotlib.__version__,
        'output_hashes': {name: sha(out / name) for name in ('data.csv', 'rigid-capacity-bounds.png', 'rigid-capacity-bounds.svg')},
        'units': {'pressure': 'Pa', 'flow_csv': 'm3/s', 'flow_figure': 'mL/s'},
        'bounds_include_root_boundary_resistance': False,
        'portal_venous_budget_condition': 'all portal and venous arms forward',
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'rows': len(rows), 'output': str(out),
                      'generated_contributions_Pa': {r: grown['roles'][r]['weighted_mean_drop_by_class_Pa']['generated'] for r in roles}}, indent=2))


if __name__ == '__main__':
    main()
