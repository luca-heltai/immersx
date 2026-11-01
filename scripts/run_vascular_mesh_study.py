#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the native generator at three uniform and three edge resolutions.

Every physical/growth control stays fixed. Only background refinement, random
seed and output location vary. Tables are native owned rows; no growth is
implemented here. Failed experiments remain available for inspection.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
from importlib.util import find_spec
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
import shutil
import signal
import subprocess
import time

sys.dont_write_bytecode = True

from render_vascular_figures import close, keyed, tables, xyz


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def parameters(case, output):
    # Coordinates, root radius, viscosity and total flow are declared SI data.
    return f'''subsection Vascularization
 set Length scale = local mesh size
 subsection Input and ports
  set Mode = generate
  set Seed position = .05,.25,.25
  set Seed direction = 1,0,0
  set Seed radius = .003
 end
 subsection Domain
  set Shape = cube
  set Cube side length = 1
  set Uniform refinements = {case['uniform']}
  set Refinement focus = {case['focus']}
  set Edge axis = 0
  set Edge sides = 1,1
  set Refinement fractions = {','.join(map(str, case['fractions']))}
 end
 subsection Sampling
  set Terminal count = 1
  set Terminal count multiplier factor = background cell count
  set Density mesh exponent = 3
  set Random seed = {case['seed']}
 end
 subsection Physiology
  set Dynamic viscosity = .004
  set Arterial Murray exponent = 3
  set Tissue demand = 1
  set Total new-tissue reference flow = 2e-9
  set Reference pressures = 0:12000
 end
 subsection Output
  set Directory = {output}
  set Write tables = true
  set Write meshes = true
  set Save physical archive = true
 end
end
'''


def validate(directory):
    summary, = read(directory / 'summary.csv')
    assert summary['valid'] == summary['reference_evaluated'] == '1'
    assert summary['length_scale'] == 'local mesh size'
    assert summary['calibration_status'] == 'not_requested'
    bg = tables(directory, 'background')
    nodes, vessels, terminals, territories = (keyed(tables(directory, k))
        for k in ('nodes', 'vessels', 'terminals', 'territories'))
    count = len(bg)
    assert len(terminals) == len(territories) == count
    assert int(summary['initial_terminal_count']) == count
    assert int(summary['effective_terminal_count']) == count
    assert len(vessels) == 2*count-1 and len(nodes) == 2*count
    children, parent = defaultdict(list), {}
    for vessel in vessels.values():
        a, b = int(vessel['node0']), int(vessel['node1'])
        assert b not in parent
        parent[b] = a
        children[a].append(b)
        assert float(vessel['normalized_length']) >= .1-1e-12
        assert vessel['normalization_complete'] == '1'
        close(float(vessel['viscosity']), .004)
    order, visited = [0], {0}
    for a in order:
        for b in children[a]:
            assert b not in visited
            visited.add(b)
            order.append(b)
    assert visited == set(nodes)
    mass, supplied = {}, []
    for terminal in terminals.values():
        value = float(terminal['demand'])
        assert math.isfinite(value) and value > 0
        mass[int(terminal['node'])] = value
        close(value, float(territories[int(terminal['territory'])]['demand']))
        flow = float(terminal['actual_outflow'])
        close(flow, 2e-9*value, tolerance=1e-8)
        supplied.append(flow)
    for node in reversed(order):
        if children[node]:
            mass[node] = math.fsum(mass[c] for c in children[node])
    volume = math.fsum(float(cell['volume']) for cell in bg)
    close(volume, 1.)
    close(mass[0], volume)
    close(math.fsum(supplied), 2e-9)
    for vessel in vessels.values():
        fraction = mass[int(vessel['node1'])]/mass[0]
        close(float(vessel['radius0']), .003*fraction**(1/3))
        close(float(vessel['radius1']), float(vessel['radius0']))
        close(float(vessel['flow']), 2e-9*fraction, tolerance=1e-8)
    # This mesh is axis-aligned. Choose the smallest containing cell on ties,
    # matching the public H convention without relying on its C++ search.
    by_h = defaultdict(lambda: dict(cells=0, volume=0., terminals=0))
    for cell in bg:
        h = float(cell['h'])
        by_h[h]['cells'] += 1
        by_h[h]['volume'] += float(cell['volume'])
    for terminal in terminals.values():
        point = xyz(terminal)
        containing = [c for c in bg if all(
            min(float(c[f'{d}{i}']) for i in range(8))-1e-13 <= point[k] <=
            max(float(c[f'{d}{i}']) for i in range(8))+1e-13
            for k, d in enumerate('xyz'))]
        assert containing
        by_h[min(float(c['h']) for c in containing)]['terminals'] += 1
    growth = tables(directory, 'growth')
    operation_names = [name[len('initialization_'):] for name in growth[0]
        if name.startswith('initialization_') and not name.endswith('_seconds')]
    work = {name: sum(int(g[name]) for g in growth) for name in operation_names}
    ordinary = {name: sum(int(g[name])-int(g['initialization_'+name])-int(g['finalization_'+name])
        for g in growth) for name in operation_names}
    assert all(value >= 0 for value in ordinary.values())
    return dict(terminals=count, cells=count, volume=volume,
        demand_per_tissue_volume=mass[0]/volume,
        flow_per_tissue_volume=math.fsum(supplied)/volume,
        normalized_lengths=[float(v['normalized_length']) for v in vessels.values()],
        normalized_radius_minima=[float(v['minimum_normalized_radius']) for v in vessels.values()],
        normalized_radius_maxima=[float(v['maximum_normalized_radius']) for v in vessels.values()],
        terminals_by_h=[dict(h=h, **row) for h, row in sorted(by_h.items())],
        summary=summary, work=work, ordinary_work=ordinary,
        global_quadrature_points=max(int(g['demand_quadrature_points']) for g in growth),
        global_quadrature_peak_points=max(int(g['demand_quadrature_peak_points']) for g in growth),
        ranks=read(directory/'ranks.csv'))


def figures(output, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for family in ('uniform', 'edge'):
        selected = [r for r in records if r['case']['family'] == family]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
        for level in sorted({r['case']['level'] for r in selected}):
            runs = [r for r in selected if r['case']['level'] == level]
            for index, field in enumerate(('normalized_lengths', 'normalized_radius_minima')):
                values = sorted(x for r in runs for x in r['measurements'][field])
                axes[index].plot(values, [(i+1)/len(values) for i in range(len(values))], label=f'level {level}')
            points = [(x['h'], x['terminals']/x['cells']) for r in runs
                for x in r['measurements']['terminals_by_h']]
            axes[2].scatter([x[0] for x in points], [x[1] for x in points], label=f'level {level}', alpha=.65)
        axes[0].set(xlabel='Whole-branch integral ds/H', ylabel='Empirical cumulative fraction')
        axes[1].set(xlabel='Minimum r/H along each branch', ylabel='Empirical cumulative fraction')
        axes[2].set(xlabel='Active cell diameter H [m]', ylabel='Accepted terminals / cell')
        for axis in axes:
            axis.legend()
            axis.grid(alpha=.2)
        fig.savefig(output/f'{family}-distributions.svg', bbox_inches='tight', pad_inches=.08)
        fig.savefig(output/f'{family}-distributions.png', dpi=180, bbox_inches='tight', pad_inches=.08)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', default='23,24,25')
    parser.add_argument('--levels', default='1,2,3')
    parser.add_argument('--families', default='uniform,edge')
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.ranks < 1 or args.timeout <= 0:
        parser.error('Ranks and timeout must be positive')
    if find_spec('matplotlib') is None:
        parser.error('Install matplotlib in the Python environment used for this study')
    if shutil.which(args.mpiexec) is None:
        parser.error('MPI launcher not found')
    seeds = [int(s) for s in args.seeds.split(',')]
    levels = [int(s) for s in args.levels.split(',')]
    families = args.families.split(',')
    if not levels or len(set(levels)) != len(levels) or not all(1 <= l <= 3 for l in levels):
        parser.error('Use distinct resolution levels from 1,2,3')
    if not families or len(set(families)) != len(families) or not set(families) <= {'uniform','edge'}:
        parser.error('Use distinct families from uniform,edge')
    if not seeds or len(set(seeds)) != len(seeds) or not all(s >= 0 for s in seeds):
        parser.error('Use distinct nonnegative seeds')
    output = args.output.resolve()
    executable = output/'bin'/args.executable.name
    library = output/'bin'/args.library.name
    template = parameters(dict(uniform=1, focus='none', fractions=[], seed=0), Path('OUTPUT'))
    common = re.sub(r' subsection (Domain|Output)\n.*?\n end\n', '', template, flags=re.S)
    identity = dict(executable_sha256=digest(args.executable),
        library_sha256=digest(args.library), script_sha256=digest(Path(__file__)),
        table_helpers_sha256=digest(Path(__file__).resolve().parent/'render_vascular_figures.py'),
        mpi_version=subprocess.check_output([args.mpiexec, '--version'], text=True),
        common_parameters_sha256=hashlib.sha256(common.encode()).hexdigest(),
        seeds=seeds, levels=levels, families=families, ranks=args.ranks, mpiexec=str(Path(shutil.which(args.mpiexec)).resolve()))
    if args.resume:
        assert json.loads((output/'manifest.json').read_text())['identity'] == identity
        assert digest(executable) == identity['executable_sha256']
        assert digest(library) == identity['library_sha256']
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'bin').mkdir()
        shutil.copy2(args.executable, executable)
        shutil.copy2(args.library, library)
    env = dict(os.environ, LD_LIBRARY_PATH=str(output/'bin')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    linked = subprocess.check_output(['ldd', str(executable)], env=env, text=True)
    assert str(library) in linked, 'Snapshot library was not selected by the dynamic loader'
    if not args.resume:
        source = Path(__file__).resolve().parents[1]
        save(output/'manifest.json', dict(identity=identity, command=sys.argv,
            linked_libraries=linked,
            host=platform.uname()._asdict(),
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else [],
            environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH')},
            commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=source, text=True).strip(),
            source_status=subprocess.check_output(['git','status','--porcelain'], cwd=source, text=True),
            source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=source)).hexdigest()))
        (output/'common-parameters.prm').write_text(common)
    records = []
    for family in families:
        for level in levels:
            for seed in seeds:
                case = dict(family=family, level=level, seed=seed,
                    uniform=level if family == 'uniform' else 1,
                    focus='none' if family == 'uniform' else 'edge',
                    fractions=[] if family == 'uniform' else [.45,.225,.1125][:level])
                run = output/f'{family}-l{level}-s{seed}-p{args.ranks}'
                prm = run/'input.prm'
                content = parameters(case, run/'network')
                canonical = re.sub(r' subsection (Domain|Output)\n.*?\n end\n', '', content, flags=re.S)
                canonical = canonical.replace(f"set Random seed = {seed}", 'set Random seed = 0')
                assert canonical == common
                if run.exists():
                    record = json.loads((run/'case.json').read_text())
                    assert record['status'] == 'complete', f'Audit failed case before restarting: {run}'
                    assert prm.read_text() == content
                    assert record['measurements'] == validate(run/'network')
                    records.append(record)
                    continue
                run.mkdir()
                prm.write_text(content)
                command = [args.mpiexec, '-np', str(args.ranks), str(executable), str(prm)]
                record = dict(case=case, command=command, status='running')
                save(run/'case.json', record)
                print(run.name, flush=True)
                start = time.monotonic()
                with (run/'application.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=run, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        record.update(status='timed_out', elapsed=time.monotonic()-start)
                        save(run/'case.json', record)
                        raise
                record.update(returncode=process.returncode, elapsed=time.monotonic()-start, status='failed')
                save(run/'case.json', record)
                assert process.returncode == 0, run
                record.update(measurements=validate(run/'network'), status='complete')
                save(run/'case.json', record)
                records.append(record)
                save(output/'results.json', records)
    rows = []
    for record in records:
        m = record['measurements']
        row = dict(record['case'])
        row.pop('fractions')
        row.update({k: m[k] for k in ('terminals', 'cells', 'volume', 'demand_per_tissue_volume', 'flow_per_tissue_volume')})
        row['generation_seconds'] = m['summary']['generation_seconds']
        row['peak_rss_kb'] = max(int(r['peak_rss_kb']) for r in m['ranks'])
        for field in ('normalized_lengths', 'normalized_radius_minima', 'normalized_radius_maxima'):
            values = sorted(m[field])
            for q in (0, .1, .5, .9, 1):
                row[field+'_'+str(q)] = values[round(q*(len(values)-1))]
        rows.append(row)
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(output, records)


if __name__ == '__main__':
    main()
