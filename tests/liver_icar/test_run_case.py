# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prepare-only launcher tests use fixture files, never a physical liver run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[2] / 'tutorials/liver_icar/run_case.py'


class LiverRunner(unittest.TestCase):
    def test_tiny_rank_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / 'assets'
            for name, content in {
                    'meshes/small.msh': 'prepare-only mesh placeholder',
                    'network/manifest': 'prepare-only manifest placeholder',
                    'inputs/inlet_ha.csv': 'time_s,flow_m3_s\n7.2,0\n8.0,1\n'}.items():
                path = assets / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            for ranks, solver in ((1, 'umfpack'), (4, 'iterative')):
                output = root / f'run-{ranks}'
                subprocess.run([sys.executable, str(RUNNER), '--profile', 'tiny',
                                '--ranks', str(ranks), '--prepare-only',
                                '--build-dir', str(root / 'build'),
                                '--assets-dir', str(assets), '--output', str(output)],
                               check=True, capture_output=True, text=True)
                parameters = (output / 'parameters.prm').read_text()
                self.assertIn('set Type = ' + solver, parameters)
                if ranks > 1:
                    self.assertIn('set Maximum replicated flow inverse DoFs = 2048', parameters)
                    self.assertIn('set Maximum materialized Schur DoFs = 512', parameters)
                metadata = json.loads((output / 'run.json').read_text())
                self.assertEqual(metadata['ranks'], ranks)
                common = (output / 'common.prm').read_text().split('subsection MetricFlowSystem')[0]
                self.assertNotIn('Number of refinement cycles', common)


if __name__ == '__main__':
    unittest.main()
