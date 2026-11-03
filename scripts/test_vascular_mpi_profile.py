#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check standard MPI profiling counters against known posted payloads."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import struct
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--mpiexec', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='profile-', dir=args.output) as scratch:
        env = dict(os.environ, IMMERSX_MPI_PROFILE_DIRECTORY=str(Path(scratch).resolve()))
        subprocess.run([args.mpiexec, '-np', str(args.ranks), str(args.executable.resolve())],
                       env=env, check=True, cwd=scratch, timeout=60)
        for rank in range(args.ranks):
            with (Path(scratch) / f'mpi-{rank}.csv').open() as stream:
                rows = {row['function']: row for row in csv.DictReader(stream)}
            double_bytes = struct.calcsize('d')
            int_bytes = struct.calcsize('i')
            ull_bytes = struct.calcsize('Q')
            expected = {'MPI_Send': (0, 0), 'MPI_Iallreduce': (0, double_bytes), 'MPI_Isend': (3*double_bytes, 0), 'MPI_Irecv': (0, 0),
                        'MPI_Sendrecv': (3*ull_bytes, 0),
                        'MPI_Allreduce': (0, double_bytes),
                        'MPI_Bcast': (0, 2*double_bytes if rank == 0 else 0),
                        'MPI_Allgatherv': (0, 2*int_bytes),
                        'MPI_Alltoallv': (0, int_bytes*args.ranks),
                        'MPI_Waitall': (0, 0), 'MPI_Wait': (0, 0), 'MPI_Ibarrier': (0, 0)}
            for name, (sent, collective) in expected.items():
                row = rows[name]
                assert int(row['calls']) == (2 if name in ['MPI_Sendrecv', 'MPI_Wait'] else 1), (rank, name, row)
                assert int(row['posted_send_bytes']) == sent, (rank, name, row)
                assert int(row['collective_input_bytes']) == collective, (rank, name, row)
                assert float(row['seconds']) >= 0
            for name, row in rows.items():
                if name not in expected:
                    assert int(row['calls']) == 0, (rank, name, row)
            with (Path(scratch) / f'growth-mpi-{rank}.csv').open() as stream:
                phases = {row['phase']: row for row in csv.DictReader(stream)}
            assert set(phases) == {'initialization', 'batch', 'finalization'}, phases
            expected_phases = {
                'initialization': (1, 1, 3, 1, 3*double_bytes, 0, 0),
                'batch': (4, 2, 6, 2, 3*ull_bytes, 2*double_bytes, 2),
                'finalization': (8, 1, 2, 0, 0, 0, 0),
            }
            for phase, values in expected_phases.items():
                row = phases[phase]
                assert int(row['run']) == 1 and row['system'] == 'arterial', row
                assert int(row['requested_terminals']) == 8, row
                actual = tuple(int(row[name]) for name in [
                    'lower_terminals', 'entries', 'calls', 'posted_sends',
                    'posted_send_bytes', 'collective_input_bytes', 'reductions'])
                assert actual == values, (rank, phase, actual, values)
                assert float(row['seconds']) >= 0, row
        snapshots = {p.name: p.read_bytes() for p in Path(scratch).glob('*.csv')}
        # Interposition also forwards in-place calls safely when profiling is off.
        disabled = dict(os.environ)
        disabled.pop('IMMERSX_MPI_PROFILE_DIRECTORY', None)
        subprocess.run([args.mpiexec, '-np', str(args.ranks), str(args.executable.resolve())],
                       env=disabled, check=True, cwd=scratch, timeout=60)
        # An empty value also disables profiling and must not target /mpi-RANK.csv.
        disabled['IMMERSX_MPI_PROFILE_DIRECTORY'] = ''
        subprocess.run([args.mpiexec, '-np', str(args.ranks), str(args.executable.resolve())],
                       env=disabled, check=True, cwd=scratch, timeout=60)
        assert snapshots == {p.name: p.read_bytes() for p in Path(scratch).glob('*.csv')}
    print('MPI counters, phase scopes and in-place forwarding match the independent payload oracle.')


if __name__ == '__main__':
    main()
