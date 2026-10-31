#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the native vascular examples, validate their tables and render SVGs."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

from render_vascular_figures import CASES, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--parameters', type=Path, required=True,
                        help='Directory containing generated build-tree .prm files')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path, help='New build-tree run directory')
    destination.add_argument('--output-parent', type=Path,
                             help='Create a unique run below this directory (for CTest)')
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--mpiexec', default='mpirun')
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('--ranks must be positive')
    if args.output_parent:
        args.output_parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='examples-', dir=args.output_parent)).resolve()
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    parameters = output / 'parameters'
    parameters.mkdir()
    working = output / 'unrelated-working-directory'
    working.mkdir()
    command = [str(args.executable.resolve())]
    if args.ranks > 1:
        command = [args.mpiexec, '-np', str(args.ranks)] + command
    for name in CASES:
        template = (args.parameters / (name + '.prm')).read_text()
        if '@TEST_' in template or '@IMMERSX_' in template:
            parser.error('Pass CMake-generated .prm files, not source templates')
        overrides = ('\nsubsection Vascularization\n  subsection Output\n'
                     f'    set Directory = {output / name}\n  end\n')
        if name == 'rediscretized':
            overrides += ('  subsection Input and ports\n'
                          f'    set Input archive = {output / "cube_uniform" / "network"}\n  end\n')
        filename = parameters / (name + '.prm')
        filename.write_text(template + overrides + 'end\n')
        log = output / (name + '.log')
        print(f'Running {name} on {args.ranks} rank(s)', flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command + [str(filename)], cwd=working,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=900)
        if result.returncode:
            raise RuntimeError(f'{name} failed; retained run: {output}\n{log.read_text()[-6000:]}')
    render(output)
    print(f'Validated native examples and rendered figures in {output / "figures"}')


if __name__ == '__main__':
    main()
