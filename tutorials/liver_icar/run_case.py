#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prepare or run one coupled IRCAD12 profile using an extracted asset bundle."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


PROFILES = {
    'tiny': ('small', 1),
    'small': ('small', 4),
    'medium': ('medium', 4),
    'large': ('large', 4),
}
TOLERANCE_KEYS = {
    'Area absolute tolerance fraction', 'Velocity absolute tolerance',
    'Capacitor pressure absolute tolerance', 'Absolute error tolerance',
    'Relative error tolerance', 'Displacement absolute tolerance',
    'Wall pressure absolute tolerance',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_inflow(path, initial, final):
    previous = None
    times = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            time, flow = float(row['time_s']), float(row['flow_m3_s'])
            if not math.isfinite(time) or not math.isfinite(flow):
                raise ValueError('The inflow CSV contains nonfinite data')
            if previous is not None and time <= previous:
                raise ValueError('The inflow CSV times must increase strictly')
            previous = time
            times.append(time)
    if len(times) < 2 or times[0] > initial or times[-1] < final:
        raise ValueError('The inflow CSV does not cover the requested interval')


def render(text, tokens, assets, profile, output, build):
    for token, value in tokens.items():
        text = text.replace(token, str(value))
    text = text.replace(str(build / 'data/liver_icar/assets'), str(assets))
    text = text.replace(str(build / f'test_output/liver_icar/{profile}'), str(output))
    if re.search(r'@[A-Z_]+@', text):
        raise ValueError('An unresolved CMake token remains in a parameter file')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='New run directory; existing directories are refused')
    parser.add_argument('--profile', choices=PROFILES, default='tiny')
    parser.add_argument('--duration', type=float, default=.8,
                        help='Seconds after 7.2 s, up to the recorded endpoint 8 s')
    parser.add_argument('--ranks', type=int, help='Default: one for tiny, four otherwise')
    parser.add_argument('--solid-viscosity', type=float, default=4.2,
                        help='Native eta coefficient in Pa s; zero is the undamped control')
    parser.add_argument('--tolerance-scale', type=float, default=1.,
                        help='Scale every absolute tolerance and the relative tolerance')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Write concrete parameters and run metadata without execution')
    args = parser.parse_args()
    try:
        if not math.isfinite(args.duration) or not 0 < args.duration <= .8:
            raise ValueError('--duration must be finite and in (0, 0.8]')
        if not math.isfinite(args.solid_viscosity) or args.solid_viscosity < 0:
            raise ValueError('--solid-viscosity must be finite and nonnegative')
        if not math.isfinite(args.tolerance_scale) or args.tolerance_scale <= 0:
            raise ValueError('--tolerance-scale must be finite and positive')
        source = Path(__file__).resolve().parents[2]
        build, output = args.build_dir.resolve(), args.output.resolve()
        assets = (args.assets_dir or build / 'data/liver_icar/assets').resolve()
        for path in (source, build, output, assets):
            if '\n' in str(path) or '\r' in str(path):
                raise ValueError('Parameter paths cannot contain line breaks')
        mesh, default_ranks = PROFILES[args.profile]
        ranks = args.ranks if args.ranks is not None else default_ranks
        if ranks < 1 or (args.profile == 'tiny' and ranks != 1):
            raise ValueError('Ranks must be positive; tiny requires one rank for UMFPACK')
        inputs = [assets / f'meshes/{mesh}.msh', assets / 'inputs/inlet_ha.csv',
                  assets / 'network/manifest']
        for path in inputs:
            if not path.is_file():
                raise ValueError(f'Missing asset: {path}')
        inputs.extend(sorted(path for path in (assets / 'network').iterdir()
                             if path.is_file() and path.name != 'manifest'))
        initial, final = 7.2, min(8., 7.2 + args.duration)
        check_inflow(assets / 'inputs/inlet_ha.csv', initial, final)
        executable = build / 'metric_flow_x_elastodynamics'
        if not executable.is_file() and (build / (executable.name + '_debug')).is_file():
            executable = build / (executable.name + '_debug')
        if not args.prepare_only and not executable.is_file():
            raise ValueError(f'Missing coupled executable: {executable}')
        command = [str(executable), str(output / 'parameters.prm')]
        if ranks > 1:
            launcher = shutil.which('mpirun')
            if launcher is None and not args.prepare_only:
                raise ValueError('mpirun is not available')
            command = [launcher or 'mpirun', '-np', str(ranks)] + command
        templates = source / 'data/liver_icar'
        tokens = {'@IMMERSX_SOURCE_DIR@': source, '@IMMERSX_BINARY_DIR@': build,
                  '@TEST_DATA_DIR@': build / 'data',
                  '@TEST_OUTPUT_DIR@': build / 'test_output'}
        common = render((templates / 'common.prm.in').read_text(), tokens,
                        assets, args.profile, output, build)
        lines = []
        for line in common.splitlines():
            match = re.fullmatch(r'(\s*set ([^=]+?)\s*=\s*)([^#]+)', line)
            if match:
                key, value = match[2].strip(), match[3].strip()
                if key in TOLERANCE_KEYS:
                    line = match[1] + format(float(value) * args.tolerance_scale, '.17g')
                elif key == 'Final time':
                    line = match[1] + format(final, '.17g')
                elif key == 'Viscosity eta':
                    line = match[1] + format(args.solid_viscosity, '.17g')
            lines.append(line)
        common = '\n'.join(lines) + '\n'
        parameters = render((templates / f'{args.profile}.prm.in').read_text(),
                            tokens, assets, args.profile, output, build)
        parameters = re.sub(r'^include .+$', f'include {output / "common.prm"}',
                            parameters, count=1, flags=re.MULTILINE)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'common.prm').write_text(common)
        (output / 'parameters.prm').write_text(parameters)
        metadata = {'profile': args.profile, 'ranks': ranks,
                    'initial_time_s': initial, 'final_time_s': final,
                    'solid_viscosity_Pa_s': args.solid_viscosity,
                    'tolerance_scale': args.tolerance_scale, 'command': command,
                    'status': 'prepared',
                    'unvalidated_scale_profile': args.profile in ('medium', 'large'),
                    'sha256': {str(path.relative_to(assets)): sha256(path) for path in inputs}}
        metadata['sha256']['common.prm'] = sha256(output / 'common.prm')
        metadata['sha256']['parameters.prm'] = sha256(output / 'parameters.prm')
        manifest = output / 'run.json'
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        print(shlex.join(command), flush=True)
        if args.profile in ('medium', 'large'):
            print('This scale profile has no validated coupled runtime.', file=sys.stderr)
        if args.prepare_only:
            return 0
        with (output / 'run.log').open('w') as log:
            result = subprocess.run(command, cwd=output, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        metadata.update(status='finished', exit_code=result.returncode)
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Exit {result.returncode}; log: {output / "run.log"}')
        return result.returncode
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
