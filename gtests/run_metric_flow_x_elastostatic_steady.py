#!/usr/bin/env python3
"""Check native steady-only KINSOL acceptance and failure before output."""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from run_metric_flow_x_elastostatic_restart import RATE_NAMES, compare_states, read_state, set_entry


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
    directory = Path(tempfile.mkdtemp(prefix='steady-', dir=args.output_directory))
    command = [str(args.executable.resolve())]
    if args.mpiexec:
        command = [str(args.mpiexec), '-np', str(args.ranks)] + command
    text = args.parameter.read_text()
    text = set_entry(text, 'Inflow function', '1e-8')
    body = '0;(1+1000*t)*y;(1+1000*t)*z'
    if text.count(body) != 1:
        raise AssertionError('Unexpected initialization fixture body load')
    text = text.replace(body, '0;y;z')
    text += '''
subsection MetricFlowX coupling
 set Pressure feedback = true
 subsection Steady solver
  # Tighten this tiny-case solve to retain its independent raw-F oracle.
  set Function norm tolerance = 1e-12
 end
end
subsection MetricFlowSystem<1, 3>
 subsection IDA parameters
  set Print nonlinear diagnostics = true
  subsection Linear solver
   set Type = mumps
  end
 end
end
'''

    def execute(name, parameters, extra, success=True):
        target = directory / name
        target.mkdir()
        parameter = target / 'parameters.prm'
        parameter.write_text(set_entry(parameters, 'Output directory', target))
        state = target / 'state.txt'
        env = os.environ.copy()
        env['IMMERSX_ELASTOSTATIC_STATE_OUTPUT'] = str(state)
        with (target / 'run.log').open('w') as log:
            result = subprocess.run(command + [str(parameter)] + extra,
                                    env=env, cwd=target, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout,
                                    check=False)
        output = (target / 'run.log').read_text()
        if (result.returncode == 0) != success:
            raise AssertionError(f'Unexpected return code {result.returncode}; inspect {target}/run.log')
        return state, output

    checkpoint = directory / 'checkpoint'
    state, output = execute('accepted', text,
                            ['--steady-only', '--checkpoint', str(checkpoint)])
    if not (checkpoint / 'COMPLETE').is_file():
        raise AssertionError('No complete native equilibrium checkpoint')
    marker = re.search(r'Steady KINSOL post-gate scaled F max: ([^;]+); target: ([^;]+); finite/positive/zero-rate: 1', output)
    if marker is None or not math.isfinite(float(marker[1])) or float(marker[1]) > float(marker[2]):
        raise AssertionError('Missing successful native post-gate')
    iterations = re.search(r'steady KINSOL iterations: (\d+)', output)
    setups = re.findall(r'IDA nonlinear Jacobian setup=(\d+) t=0 alpha=0', output)
    if iterations is None or int(iterations[1]) < 2 or len(setups) < int(iterations[1]):
        raise AssertionError('Nonlinear fixture did not exercise repeated fresh Jacobian setups')
    rates = read_state(Path(str(state) + '.rate'), RATE_NAMES)
    if any(value != 0 for block in rates for value in block):
        raise AssertionError('Steady-only changed a prescribed zero rate')
    for block in read_state(Path(str(state) + '.initial.residual')):
        if math.sqrt(math.fsum(value * value for value in block)) > 1e-9:
            raise AssertionError('Native steady residual exceeds the independent tiny-case threshold')
    flow_pvd = state.parent / 'flow.pvd'
    times = [float(item.attrib['timestep']) for item in ET.parse(flow_pvd).iter('DataSet')]
    if times != [0.0]:
        raise AssertionError(f'Steady-only emitted transient output times {times}')

    restored = directory / 'restored'
    restored.mkdir()
    restored_state = restored / 'state.txt'
    env = os.environ.copy()
    env['IMMERSX_ELASTOSTATIC_STATE_OUTPUT'] = str(restored_state)
    with (restored / 'run.log').open('w') as log:
        result = subprocess.run(command + ['--restart', str(checkpoint), '--final-time', '0.0001'],
                                env=env, cwd=restored, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.timeout, check=False)
    if result.returncode:
        raise AssertionError(f'Restart of equilibrium failed; inspect {restored}/run.log')
    for suffix in ('', '.rate'):
        compare_states(Path(str(state) + suffix),
                       Path(str(restored_state) + '.initial' + suffix),
                       'steady checkpoint at t0 restored exactly', absolute=0, relative=0,
                       names=RATE_NAMES if suffix else ('displacement', 'flow', 'multiplier'),
                       skip=('displacement rate',) if suffix else ())

    rejected_checkpoint = directory / 'rejected-checkpoint'
    rejected = text + '''
subsection MetricFlowX coupling
 subsection Steady solver
  set Function norm tolerance = 1e-30
  set Scaled step tolerance = 1e6
 end
end
'''
    failed_state, failed_log = execute('small-step', rejected,
                                       ['--steady-only', '--checkpoint', str(rejected_checkpoint)],
                                       success=False)
    if 'Steady KINSOL post-gate failed' not in ' '.join(failed_log.split()):
        raise AssertionError('The controlled small-step result did not reach the final residual gate')
    if failed_state.exists() or rejected_checkpoint.exists() or list(failed_state.parent.glob('*.pvtu')):
        raise AssertionError('Rejected small-step result emitted an accepted state or checkpoint')
    print(f'Steady-only native equilibrium and rejection checks passed; logs in {directory}')


if __name__ == '__main__':
    main()
