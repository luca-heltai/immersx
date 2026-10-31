#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Publish or check figures rendered from fresh native vascular examples."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

from render_vascular_figures import ASSETS, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Build-tree native examples run directory')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / 'doc' / '_static' / 'vascular-generation'
    if args.run.resolve() == target.resolve():
        parser.error('Use fresh build-tree outputs, not the published assets')
    # Revalidate native tables and rerender instead of trusting saved plot files.
    with tempfile.TemporaryDirectory(prefix='vascular-figures-') as scratch:
        source = Path(scratch) / 'figures'
        render(args.run.resolve(), source)
        for name in ASSETS:
            if name.endswith('.svg'):
                ET.parse(source / name)
        if args.check:
            different = [name for name in ASSETS
                         if not (target / name).is_file()
                         or (source / name).read_bytes() != (target / name).read_bytes()]
            if different:
                print('Published vascular assets differ: ' + ', '.join(different), file=sys.stderr)
                return 1
            print('Published vascular assets match the fresh run')
        else:
            target.mkdir(parents=True, exist_ok=True)
            for name in ASSETS:
                shutil.copyfile(source / name, target / name)
            print(f'Published {len(ASSETS)} native-run assets in {target}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
