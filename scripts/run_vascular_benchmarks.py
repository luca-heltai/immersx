#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Measure the native vascular application; no growth is implemented in Python."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from render_vascular_figures import close, keyed, tables, xyz


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def quantiles(values):
    values = sorted(values)
    return {str(q): values[round(q * (len(values)-1))] for q in (0, .1, .5, .9, 1)}


def read_csv(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert all(None not in r and None not in r.values() for r in rows), path
    return rows


def validate(directory, case):
    """Independently check the synthetic single-inlet tree and reported work."""
    summaries = read_csv(directory / 'summary.csv')
    assert len(summaries) == 1
    summary = summaries[0]
    count, ranks = case['terminals'], case['ranks']
    side, radius, flow = case['side'], case['radius'], case['flow']
    assert summary['valid'] == summary['reference_evaluated'] == '1'
    assert summary['calibration_status'] == 'not_requested'
    assert int(summary['processes']) == ranks
    assert int(summary['arterial_terminals']) == count
    assert int(summary['venous_terminals']) == int(summary['pairs']) == 0
    assert int(summary['accepted_insertions']) == count-1
    assert int(summary['background_cells']) == 8**case['refinements']
    close(float(summary['background_volume']), side**3)
    nodes, vessels, terminals, territories = (keyed(tables(directory, kind)) for kind in
                                             ('nodes', 'vessels', 'terminals', 'territories'))
    assert len(nodes) == 2*count and len(vessels) == 2*count-1
    assert len(terminals) == len(territories) == count
    roots = {i for i, n in nodes.items() if n['port'] != '0'}
    assert roots == {0}, roots
    children, parent = defaultdict(list), {}
    for i, vessel in vessels.items():
        a, b = int(vessel['node0']), int(vessel['node1'])
        assert a in nodes and b in nodes and b not in parent
        assert vessel['system'] == 'arterial'
        parent[b] = i
        children[a].append((b, i))
        for d in 'xyz':
            close(float(vessel[d+'0']), float(nodes[a][d]))
            close(float(vessel[d+'1']), float(nodes[b][d]))
            assert min(float(vessel[d+'0'])-float(vessel['radius0']),
                       float(vessel[d+'1'])-float(vessel['radius1'])) >= 0
            assert max(float(vessel[d+'0'])+float(vessel['radius0']),
                       float(vessel[d+'1'])+float(vessel['radius1'])) <= side
    assert set(parent) == set(nodes)-roots
    order, seen = [0], {0}
    for node in order:
        for child, _ in children[node]:
            assert child not in seen
            seen.add(child)
            order.append(child)
    assert seen == set(nodes)
    branch_nodes = {node for node in children if children[node]}
    assert {int(t['node']) for t in terminals.values()} == set(nodes)-branch_nodes
    descendants = {}
    for node in reversed(order):
        descendants[node] = sum(descendants[c] for c, _ in children[node]) if children[node] else 1
    assert descendants[0] == count
    angles = []
    for node in order:
        rays = [tuple(x-y for x, y in zip(xyz(nodes[c]), xyz(nodes[node])))
                for c, _ in children[node]]
        if node in parent:
            a = int(vessels[parent[node]]['node0'])
            rays.append(tuple(x-y for x, y in zip(xyz(nodes[a]), xyz(nodes[node]))))
        for i, a in enumerate(rays):
            for b in rays[i+1:]:
                cosine = sum(x*y for x, y in zip(a, b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
                angles.append(math.degrees(math.acos(max(-1., min(1., cosine)))))
    assert not angles or min(angles) >= 15-1.e-9
    for vessel in vessels.values():
        a, b = (nodes[int(vessel[k])] for k in ('node0', 'node1'))
        n = descendants[int(vessel['node1'])]
        close(float(vessel['radius0']), radius*(n/count)**(1/3))
        close(float(vessel['radius1']), float(vessel['radius0']))
        assert math.isclose(float(vessel['flow']), flow*n/count, rel_tol=1.e-8)
        close(float(vessel['viscosity']), .004)
        close(float(vessel['length']), math.dist(xyz(a), xyz(b)))
        pressure_drop = float(a['pressure'])-float(b['pressure'])
        assert math.isclose(pressure_drop, float(vessel['flow'])*float(vessel['resistance']),
                            rel_tol=5.e-8, abs_tol=64*sys.float_info.epsilon*max(abs(float(a['pressure'])), abs(float(b['pressure']))))
    assert {int(t['territory']) for t in terminals.values()} == set(territories)
    for terminal in terminals.values():
        assert math.isclose(float(terminal['actual_outflow']), flow/count, rel_tol=1.e-8)
        close(float(terminal['demand']), 1.)
    integrals, intervals = defaultdict(lambda: [0., 0.]), defaultdict(list)
    for cell in tables(directory, 'intervals'):
        assert cell['inside'] == '1'
        i = int(cell['vessel'])
        vessel = vessels[i]
        a, b = float(cell['begin']), float(cell['end'])
        assert 0 <= a < b <= 1
        intervals[i].append((a, b))
        ra, rb = (float(vessel[k]) for k in ('radius0', 'radius1'))
        length = (b-a)*float(vessel['length'])
        integrals[i][0] += math.pi*length*(ra*ra+ra*rb+rb*rb)/3
        integrals[i][1] += 8*.004*length*(ra*ra+ra*rb+rb*rb)/(3*math.pi*ra**3*rb**3)
    for i, vessel in vessels.items():
        ordered = sorted(intervals[i])
        close(ordered[0][0], 0.)
        close(ordered[-1][1], 1.)
        for a, b in zip(ordered, ordered[1:]):
            close(a[1], b[0])
        close(integrals[i][0], float(vessel['full_volume']))
        close(integrals[i][1], float(vessel['resistance']))
    close(sum(float(v['full_volume']) for v in vessels.values()), float(summary['inside_volume']))
    assert sum(len(p) for p in intervals.values()) == int(summary['numerical_cells'])
    rank_rows = read_csv(directory / 'ranks.csv')
    assert {int(r['rank']) for r in rank_rows} == set(range(ranks))
    assert sum(int(r['vessels']) for r in rank_rows) == len(vessels)
    growth, mpi = [], []
    for rank in range(ranks):
        growth.append(read_csv(directory / f'growth-{rank}.csv'))
        mpi.append(read_csv(directory.parent / 'mpi' / f'mpi-{rank}.csv'))
        assert growth[-1] and mpi[-1]
        functions = [r['function'] for r in mpi[-1]]
        assert len(set(functions)) == len(functions)
        assert {'MPI_Isend', 'MPI_Allreduce', 'MPI_Waitall', 'MPI_Comm_split'} <= set(functions)
        assert all(g['system'] == 'arterial' and int(g['requested_terminals']) == count for g in growth[-1])
    assert sum(int(g['accepted_insertions']) for rows in growth for g in rows) == count-1
    assert sum(int(g['sampled_points']) for rows in growth for g in rows) == int(summary['sampled_points'])
    assert sum(int(rows[0]['owned_vessels_first']) for rows in growth) == 1
    assert sum(int(rows[-1]['owned_vessels_last']) for rows in growth) == len(vessels)
    assert all(float(r['seconds']) >= 0 and int(r['calls']) >= 0 for rows in mpi for r in rows)
    assert all(sum(int(r['calls']) for r in rows) > 0 for rows in mpi)
    assert all(len(g) == len(growth[0]) for g in growth)
    stages = []
    for i in range(len(growth[0])):
        rows = [g[i] for g in growth]
        batches = {int(r['batches']) for r in rows}
        assert len(batches) == 1 and next(iter(batches)) > 0
        stages.append(dict(first_terminals=int(rows[0]['first_terminals']),
                           last_terminals=int(rows[0]['last_terminals']),
                           batches=next(iter(batches)),
                           active_writer_fraction=sum(int(r['active_writer_batches']) for r in rows)/(ranks*next(iter(batches))),
                           maximum_owned_vessels=max(int(r['owned_vessels_maximum']) for r in rows),
                           **{k: max(float(r[k]) for r in rows) for k in rows[0] if k.endswith('_seconds')}))
    owned = [int(r['vessels']) for r in rank_rows]
    return dict(summary=summary, ranks=rank_rows, growth=stages,
                growth_rank_totals=[{k: sum(float(g[k]) for g in rows) for k in rows[0] if k.endswith('_seconds')}
                                    for rows in growth],
                mpi={k: sum(float(r[k]) if k == 'seconds' else int(r[k]) for rows in mpi for r in rows)
                     for k in ('calls', 'posted_send_bytes', 'collective_input_bytes', 'seconds')},
                owned_vessel_max_over_mean=max(owned)/(sum(owned)/ranks),
                minimum_reference_area=math.pi*min(float(v['radius0']) for v in vessels.values())**2,
                length_quantiles=quantiles([float(v['length']) for v in vessels.values()]),
                radius_quantiles=quantiles([float(v['radius0']) for v in vessels.values()]),
                junction_angle_quantiles=quantiles(angles) if angles else {},
                terminal_distance_quantiles=quantiles([math.dist(xyz(t), (side/2,)*3) for t in terminals.values()]))


def parameters(case, output):
    return f'''# Declared synthetic SI benchmark; not an organ calibration.
subsection Vascularization
 subsection Input and ports
  set Mode = generate
  set Seed position = {.2*case['side']},{.5*case['side']},{.5*case['side']}
  set Seed direction = 1,0,0
  set Seed radius = {case['radius']:.17g}
 end
 subsection Domain
  set Shape = cube
  set Cube side length = {case['side']:.17g}
  set Uniform refinements = {case['refinements']}
  set Refinement focus = none
 end
 subsection Growth
  set Seed length = .15
 end
 subsection Sampling
  set Terminal count = {case['terminals']}
  set Terminal count multiplier factor = none
  set Random seed = {case['seed']}
  set Density mesh exponent = 0
 end
 subsection Objectives
  set Target volume fraction = 0
  set Minimum tissue coverage = 0
  set Maximum service distance = .2
 end
 subsection Physiology
  set Dynamic viscosity = .004
  set Total new-tissue reference flow = {case['flow']:.17g}
  set Reference pressures = 0:12000
 end
 subsection Output
  set Directory = {output}
  set Write tables = true
  set Write meshes = false
  set Save physical archive = false
 end
end
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpiexec', default='mpirun')
    parser.add_argument('--mpi-argument', action='append', help='Repeat to replace OpenMPI binding arguments')
    parser.add_argument('--kind', choices=('strong', 'weak'), default='strong')
    parser.add_argument('--ranks', default='1,2,4,8')
    parser.add_argument('--terminals', default='1000,10000')
    parser.add_argument('--seeds', default='23,24,25')
    parser.add_argument('--timeout', type=float, default=14400)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ranks, counts, seeds = ([int(x) for x in values.split(',')] for values in
                            (args.ranks, args.terminals, args.seeds))
    if min(ranks) < 1 or min(counts) < 2 or min(seeds) < 0 or args.timeout <= 0:
        parser.error('Use positive ranks/timeouts, at least two terminals and nonnegative seeds')
    cases = []
    for count in counts:
        for seed in seeds:
            for n in ranks:
                side, refinements = 1., 3
                if args.kind == 'weak':
                    power = round(math.log2(n)/3)
                    if n != 8**power:
                        parser.error('Weak cube protocol requires ranks 1,8,64,...')
                    side, refinements = float(2**power), 3+power
                cases.append(dict(name=f'{args.kind}-t{count}-s{seed}-p{n}',
                                  kind=args.kind, ranks=n, seed=seed, side=side,
                                  terminals=count*(n if args.kind == 'weak' else 1),
                                  refinements=refinements, radius=.003*side,
                                  flow=2.e-9*(n if args.kind == 'weak' else 1)))
    output = args.output.resolve()
    source = Path(__file__).resolve().parents[1]
    mpiexec = shutil.which(args.mpiexec)
    if not mpiexec:
        parser.error('MPI launcher not found')
    binding = args.mpi_argument if args.mpi_argument is not None else ['--bind-to', 'core', '--report-bindings']
    identity = dict(cases=cases, executable_sha256=digest(args.executable),
                    library_sha256=digest(args.library), script_sha256=digest(Path(__file__)),
                    table_helpers_sha256=digest(source / 'scripts' / 'render_vascular_figures.py'),
                    mpiexec=str(Path(mpiexec).resolve()), mpi_arguments=binding)
    if args.resume:
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['identity'] == identity, 'Changed experiment: use a new output directory'
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'bin').mkdir()
        shutil.copy2(args.executable, output / 'bin' / 'vascular_network_profile')
        shutil.copy2(args.library, output / 'bin' / 'libimmersx_lib.so')
        manifest = dict(identity=identity, created=now(), host=platform.uname()._asdict(),
                        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else [],
                        environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_LIBRARY_PATH')},
                        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                        source_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True),
                        source_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=source)).hexdigest(),
                        mpi_version=subprocess.check_output([mpiexec, '--version'], text=True),
                        linked_libraries=subprocess.check_output(['ldd', str(output / 'bin' / 'vascular_network_profile')], text=True,
                            env=dict(os.environ, LD_LIBRARY_PATH=str(output / 'bin') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))))
        if Path('/proc/cpuinfo').exists():
            manifest['cpuinfo'] = Path('/proc/cpuinfo').read_text()
        save(output / 'manifest.json', manifest)
    assert digest(output / 'bin' / 'vascular_network_profile') == identity['executable_sha256']
    assert digest(output / 'bin' / 'libimmersx_lib.so') == identity['library_sha256']
    env = dict(os.environ, LD_LIBRARY_PATH=str(output / 'bin') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    for case in cases:
        run = output / case['name']
        if run.exists():
            record = json.loads((run / 'case.json').read_text())
            assert record['status'] == 'complete', f'Audit unfinished case before any restart: {run}'
            assert validate(run / 'network', case) == record['measurements']
            print('Verified completed case:', case['name'], flush=True)
            continue
        run.mkdir()
        (run / 'mpi').mkdir()
        filename = run / 'input.prm'
        filename.write_text(parameters(case, run / 'network'))
        command = [mpiexec] + binding + ['-np', str(case['ranks']), str(output / 'bin' / 'vascular_network_profile'), str(filename)]
        record = dict(case=case, command=command, started=now(), status='running')
        save(run / 'case.json', record)
        print('Running:', case['name'], flush=True)
        begin = time.monotonic()
        with (run / 'application.log').open('w') as log:
            process = subprocess.Popen(command, cwd=run, env=dict(env, IMMERSX_MPI_PROFILE_DIRECTORY=str(run / 'mpi')),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['launcher_pid'] = process.pid
            save(run / 'case.json', record)
            try:
                status = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(status='timed_out', finished=now())
                save(run / 'case.json', record)
                raise
        record.update(returncode=status, launcher_seconds=time.monotonic()-begin, finished=now(), status='failed')
        save(run / 'case.json', record)
        if status:
            raise RuntimeError(f'Native application failed; inspect {run}')
        diagnostics = (run / 'application.log').read_text()
        assert 'Cannot write MPI profile:' not in diagnostics and 'Cannot close MPI profile:' not in diagnostics
        record['measurements'] = validate(run / 'network', case)
        record['status'] = 'complete'
        save(run / 'case.json', record)
        print('Validated:', case['name'], 'generation seconds:', record['measurements']['summary']['generation_seconds'], flush=True)


if __name__ == '__main__':
    main()
