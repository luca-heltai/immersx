#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exercise the actual collective CLI and check its physical CSV output."""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile


def rows(directory, kind):
    result = []
    for file in sorted(directory.glob(kind + '-*.csv')):
        with file.open() as stream:
            for row in csv.DictReader(stream):
                assert None not in row and None not in row.values(), (file, row)
                result.append(row)
    return result


def section(groups):
    text = 'subsection Vascularization\n'
    for group, values in groups.items():
        text += '  subsection ' + group + '\n'
        for key, value in values.items():
            text += f'    set {key} = {value}\n'
        text += '  end\n'
    return text + 'end\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--mpiexec', default='mpirun')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cli-', dir=args.output) as scratch:
        root = Path(scratch)
        config = root / 'parameters'
        config.mkdir()
        cwd = root / 'unrelated'
        cwd.mkdir()
        command = [str(args.executable.resolve())]
        if args.ranks > 1:
            command = [args.mpiexec, '-np', str(args.ranks)] + command

        def invoke(arguments, success=True):
            result = subprocess.run(command + arguments, cwd=cwd,
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=120)
            assert (result.returncode == 0) == success, result.stdout
            return result.stdout

        template = args.parameters.read_text()

        def run(name, groups, success=True):
            groups = dict(groups)
            groups['Output'] = {'Directory': '../' + name}
            prm = config / (name + '.prm')
            prm.write_text(template + section(groups))
            invoke([str(prm)], success)
            output = root / name
            with (output / 'summary.csv').open() as stream:
                summary = list(csv.DictReader(stream))
            assert len(summary) == 1 and None not in summary[0]
            assert None not in summary[0].values()
            assert summary[0]['valid'] == str(int(success))
            assert int(summary[0]['processes']) == args.ranks
            with (output / 'ranks.csv').open() as stream:
                ranks = list(csv.DictReader(stream))
            assert len(ranks) == args.ranks
            assert all(None not in row and None not in row.values() for row in ranks)
            return output, summary[0]

        invoke(['--help'])
        defaults = config / 'defaults.prm'
        invoke(['--write-parameters', str(defaults)])
        assert 'subsection Vascularization' in defaults.read_text()
        invoke(['--write-parameters', str(defaults)], False)
        missing = config / 'missing.prm'
        invoke([str(missing)], False)
        assert not missing.exists()

        generation = {
            'Input and ports': {'Mode': 'generate', 'Seed position': '.2,.5,.5',
                                'Seed direction': '1,0,0', 'Seed radius': '.005'},
            'Sampling': {'Terminal count': 1},
            'Growth': {'Seed length': '.4'},
            'Physiology': {'Dynamic viscosity': '.004',
                           'Total new-tissue reference flow': '2e-9'},
        }
        original, summary = run('generated', generation)
        invoke([str(config / 'generated.prm')], False)
        vessels = rows(original, 'vessels')
        assert len(vessels) == 1
        vessel = vessels[0]
        assert math.isclose(float(vessel['length']), .4, rel_tol=1e-12)
        expected_volume = math.pi * .005**2 * .4
        expected_resistance = 8 * .004 * .4 / (math.pi * .005**4)
        assert math.isclose(float(vessel['full_volume']), expected_volume, rel_tol=1e-12)
        assert math.isclose(float(vessel['resistance']), expected_resistance, rel_tol=1e-12)
        assert math.isclose(float(vessel['flow']), 2e-9, rel_tol=1e-8)
        assert math.isclose(float(summary['volume_fraction']), expected_volume, rel_tol=1e-12)
        pressures = {int(row['id']): float(row['pressure']) for row in rows(original, 'nodes')}
        assert math.isclose(pressures[int(vessel['node0'])] - pressures[int(vessel['node1'])],
                            expected_resistance * 2e-9, rel_tol=1e-6)
        assert float(summary['arterial_coverage']) == 1.
        assert (original / 'network' / 'manifest').exists()
        assert list(original.glob('background*.vtu')) and list(original.glob('line*.vtu'))

        loaded, loaded_summary = run('loaded', {
            'Input and ports': {'Input archive': '../generated/network'},
            'Domain': {'Uniform refinements': 3},
            'Discretization': {'Target length factor': '.5', 'Lower length factor': '.25'},
        })
        loaded_vessels = rows(loaded, 'vessels')
        assert len(loaded_vessels) == 1
        for key in ('id', 'node0', 'node1', 'radius0', 'radius1', 'viscosity',
                    'length', 'full_volume', 'resistance'):
            assert loaded_vessels[0][key] == vessel[key], key
        assert int(loaded_summary['numerical_cells']) > int(summary['numerical_cells'])
        assert math.isclose(float(loaded_vessels[0]['flow']), 2e-9, rel_tol=1e-8)
        assert float(loaded_summary['volume_fraction']) == float(summary['volume_fraction'])
        assert all(row['target_flow'] == '' for row in rows(loaded, 'terminals'))

        completed, completed_summary = run('completed', {
            'Input and ports': {'Mode': 'complete', 'Input archive': '../generated/network'},
            'Sampling': {'Terminal count': 1},
            'Growth': {'Seed length': '.1'},
            'Physiology': {'Total new-tissue reference flow': '2e-9'},
        })
        completed_vessels = {row['id']: row for row in rows(completed, 'vessels')}
        assert len(completed_vessels) == 2
        for key in ('node0', 'node1', 'radius0', 'radius1', 'viscosity', 'length',
                    'full_volume', 'resistance'):
            assert completed_vessels[vessel['id']][key] == vessel[key], key
        assert int(completed_summary['arterial_terminals']) == 1
        assert len(rows(completed, 'territories')) == 1
        assert len(rows(original, 'vessels')) == 1

        for name, shape, focus in [('edge', 'cube', 'edge'),
                                   ('sphere', 'sphere', 'none'),
                                   ('center', 'sphere', 'center')]:
            case = dict(generation)
            case['Domain'] = {'Shape': shape, 'Refinement focus': focus}
            output, result = run(name, case)
            assert int(result['arterial_terminals']) == 1
            assert float(result['inside_volume']) > 0.
            assert rows(output, 'background') and rows(output, 'intervals')
            if focus != 'none':
                diameters = [float(row['h']) for row in rows(output, 'background')]
                assert min(diameters) < max(diameters)

        av = dict(generation)
        av['Growth'] = {'Seed length': '.2'}
        av['Input and ports'] = dict(generation['Input and ports'], **{'Seed radius': '.003'})
        av['Physiology'] = dict(generation['Physiology'], **{'Reference pressures': '0:12000,2:4000'})
        av['Venous network'] = {'Enabled': 'true', 'Seed position': '.8,.5,.5',
                                'Seed radius': '.004', 'Terminal separation': '.04'}
        paired, paired_summary = run('paired', av)
        assert int(paired_summary['pairs']) == 1
        assert int(paired_summary['venous_terminals']) == 1
        beds = rows(paired, 'pairs')
        assert len(beds) == 1 and float(beds[0]['resistance']) > 0.
        assert math.isclose(float(beds[0]['flow']), 2e-9, rel_tol=1e-8)
        terminals = {row['id']: row for row in rows(paired, 'terminals')}
        arterial = terminals[beds[0]['arterial_terminal']]
        venous = terminals[beds[0]['venous_terminal']]
        assert arterial['system'] == 'arterial' and venous['system'] == 'venous'
        assert any(arterial[d] != venous[d] for d in ('x', 'y', 'z'))

        invalid = dict(generation)
        invalid['Input and ports'] = dict(generation['Input and ports'], **{'Seed position': '2,2,2'})
        rejected, _ = run('rejected', invalid, False)
        assert not (rejected / 'network').exists()
        assert any(file.read_text() for file in rejected.glob('errors-*.txt'))
        print(f'Collective standalone checks passed on {args.ranks} rank(s).')


if __name__ == '__main__':
    main()
