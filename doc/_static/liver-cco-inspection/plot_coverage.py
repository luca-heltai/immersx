"""Plot only the independent saved geometric diagnostics."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(sys.argv[1])
source = folder / 'coverage-report.json'
report = json.loads(source.read_text())
records = report['checkpoints']
assert records and all(not row['unlocated_terminal_indices'] for row in records)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
columns = ('mutable_leaves', 'requested', 'mean_terminals_per_cell',
           'maximum_terminals_per_cell', 'occupied_cells',
           'cell_centre_covered_volume_proxy',
           'sufficient_whole_cell_covered_volume_fraction',
           'possible_covered_volume_fraction_upper_bound',
           'maximum_cell_centre_distance_over_h')
table = folder / 'geometric-coverage.csv'
with table.open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', lineterminator=chr(10))
    writer.writeheader()
    writer.writerows(records)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
figure.subplots_adjust(left=.08, right=.98, top=.79, bottom=.25, wspace=.28)
count = [row['mutable_leaves'] for row in records]
lower = [100 * row['sufficient_whole_cell_covered_volume_fraction'] for row in records]
upper = [100 * row['possible_covered_volume_fraction_upper_bound'] for row in records]
proxy = [100 * row['cell_centre_covered_volume_proxy'] for row in records]
axis = axes[0]
axis.fill_between(count, lower, upper, color='#168649', alpha=.18)
axis.plot(count, lower, 'o-', color='#168649', label='Sufficient whole-cell coverage')
axis.plot(count, proxy, 's--', color='#596571', label='Cell-centre proxy')
axis.axhline(95, color='#8C969F', linestyle=':', label='Coverage target (95%)')
axis.set_ylim(94, 100.6)
axis.set_xlabel('Generated arterial terminals')
axis.set_ylabel('Generation-mesh volume (%)')
axis.set_title('Distance threshold: 2h')
axis.legend(loc='lower right', frameon=False, fontsize=9)
last = records[-1]
values = sorted((int(k), v) for k, v in last['histogram'].items())
axis = axes[1]
axis.bar([x for x, _ in values], [y for _, y in values], color='#C73339', width=.65)
axis.set_xticks([x for x, _ in values])
axis.set_yscale('log')
axis.set_ylim(.7, max(y for _, y in values) * 3)
for x, y in values:
    axis.text(x, y * 1.13, f'{y:,}', ha='center', va='bottom', fontsize=10)
axis.set_xlabel('Arterial terminals in one generation cell')
axis.set_ylabel('Number of cells (log scale)')
axis.set_title(f"Checkpoint: {last['mutable_leaves']:,} / {last['requested']:,} terminals")
for axis in axes:
    axis.spines[['top', 'right']].set_visible(False)
figure.suptitle('Arterial CCO growth: coverage and terminal density', fontsize=17, y=.965)
figure.text(.5, .89, f"{report['mesh_cells']:,} generation tetrahedra; h is the longest edge of each tissue cell",
            ha='center', color='#53606A')
figure.text(.5, .12, 'Coverage uses generated arterial terminals; shaded interval bounds geometric coverage in floating-point arithmetic.',
            ha='center', fontsize=9, color='#53606A')
figure.text(.5, .065, 'Partial growth snapshots. Cell-centre proxy is separate from native quadrature; no physiological or coupled-solution claim.',
            ha='center', fontsize=9, color='#53606A')
images = [folder / ('arterial-coverage-and-cell-counts.' + suffix) for suffix in ('png', 'svg')]
for path in images:
    figure.savefig(path, dpi=220, facecolor='white')
    if path.suffix == '.svg':
        path.write_text(chr(10).join(line.rstrip() for line in path.read_text().splitlines()) + chr(10))
plt.close(figure)
metadata = dict(partial_geometry_only=True, coupled_solution=False,
                report_sha256=sha(source), script_sha256=sha(Path(__file__)),
                output_sha256={p.name: sha(p) for p in images + [table]},
                visual_review='pending')
(folder / 'coverage-figure-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
