#!/usr/bin/env python3
"""Check fresh native steady correction, the legacy default, and restart rates."""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from run_metric_flow_x_elastostatic_restart import (
    RATE_NAMES, compare_states, read_state, run, set_entry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--parameter', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--mpiexec', type=Path)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='initialization-', dir=args.output_directory))
    command = [str(args.executable.resolve())]
    if args.mpiexec:
        command = [str(args.mpiexec), '-np', str(args.ranks)] + command
    text = args.parameter.read_text()
    text = set_entry(text, 'Final time', '0.0002')
    text = set_entry(text, 'Time interval between each output', '0.0001')
    text = set_entry(text, 'Absolute error tolerance', '1e-12')
    text = set_entry(text, 'Relative error tolerance', '1e-10')
    text += '\nsubsection MetricFlowX coupling\n  set Pressure feedback = true\nend\n'

    def execute(name, parameters, extra=()):
        target = directory / name
        target.mkdir()
        parameter = target / 'parameters.prm'
        parameter.write_text(set_entry(parameters, 'Output directory', target))
        state = target / 'state.txt'
        run(command + [str(parameter)] + list(extra), state, target / 'run.log',
            args.timeout, target)
        return state

    default = re.sub(r'^\s*set Correction type at initial time\s*=.*$', '', text,
                     flags=re.MULTILINE)
    default_state = execute('default', default)
    explicit_state = execute('use-y-diff', text)
    for suffix in ('', '.initial', '.rate', '.initial.rate'):
        compare_states(Path(str(default_state) + suffix),
                       Path(str(explicit_state) + suffix),
                       'default equals explicit use_y_diff', absolute=0, relative=0,
                       names=RATE_NAMES if suffix.endswith('.rate') else
                       ('displacement', 'flow', 'multiplier'))

    # Nonzero, time-independent forcing makes a resting flow guess non-steady.
    steady = set_entry(text, 'Inflow function', '1e-8')
    steady = set_entry(steady, 'Absolute error tolerance', '1e-9')
    steady = set_entry(steady, 'Relative error tolerance', '1e-7')
    if steady.count('0;(1+1000*t)*y;(1+1000*t)*z') != 1:
        raise AssertionError('The initialization fixture must contain the expected body load')
    steady = steady.replace('0;(1+1000*t)*y;(1+1000*t)*z', '0;y;z')
    steady = set_entry(steady, 'Correction type at initial time', 'use_y_dot')
    checkpoint = directory / 'checkpoint'
    steady_state = execute('use-y-dot', steady,
                           ('--final-time', '0.0001', '--checkpoint', str(checkpoint)))
    initial = Path(str(steady_state) + '.initial')
    rates = read_state(Path(str(initial) + '.rate'), RATE_NAMES)
    if any(value != 0 for block in rates for value in block):
        raise AssertionError('IDA_Y_INIT changed the prescribed zero initial rates')
    def check_residual(snapshot, label):
        residual = read_state(Path(str(snapshot) + '.residual'))
        for name, block in zip(('solid', 'flow', 'wall'), residual):
            norm = math.sqrt(math.fsum(value * value for value in block))
            print(f'{label} initial residual {name}: L2={norm:.17g}')
            if norm > 1e-9:
                raise AssertionError(f'Native coupled initial residual is not small: {name}')

    check_residual(initial, 'use_y_dot')
    transient_state = execute('same-forcing-use-y-diff',
                              set_entry(steady, 'Correction type at initial time', 'use_y_diff'))
    transient_initial = Path(str(transient_state) + '.initial')
    check_residual(transient_initial, 'use_y_diff')
    transient_rates = read_state(Path(str(transient_initial) + '.rate'), RATE_NAMES)
    if max(abs(value) for block in transient_rates for value in block) <= 1e-10:
        raise AssertionError('use_y_diff did not expose nonzero rates under the same forcing')
    flow = read_state(initial)[1]
    transient_flow = read_state(transient_initial)[1]
    if max(abs(a - b) for a, b in zip(flow, transient_flow)) <= 1e-10:
        raise AssertionError('The two native IC policies did not produce distinct flow states')

    restored = directory / 'restart'
    restored.mkdir()
    restored_state = restored / 'state.txt'
    run(command + ['--restart', str(checkpoint), '--final-time', '0.0002'],
        restored_state, restored / 'run.log', args.timeout, restored)
    for suffix in ('', '.rate'):
        compare_states(Path(str(steady_state) + suffix),
                       Path(str(restored_state) + '.initial' + suffix),
                       'use_y_dot checkpoint restored without fresh correction',
                       absolute=0, relative=0,
                       names=RATE_NAMES if suffix else
                       ('displacement', 'flow', 'multiplier'),
                       skip=('displacement rate',) if suffix else ())

    rejected = directory / 'none'
    rejected.mkdir()
    parameter = rejected / 'parameters.prm'
    parameter.write_text(set_entry(text, 'Correction type at initial time', 'none'))
    env = os.environ.copy()
    env.pop('IMMERSX_ELASTOSTATIC_STATE_OUTPUT', None)
    result = subprocess.run(command + [str(parameter)], cwd=rejected, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=min(args.timeout, 30), text=True, check=False)
    (rejected / 'run.log').write_text(result.stdout)
    if result.returncode == 0 or 'requires coupled IDA use_y_diff or use_y_dot' not in ' '.join(result.stdout.split()):
        raise AssertionError('Fresh none correction was not rejected by the application guard')
    print('Fresh steady initialization, default, and restart checks passed')


if __name__ == '__main__':
    main()
