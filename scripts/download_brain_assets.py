#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Download the brain and vessel grids from the ImmersX v0.3 release."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


RELEASE_URL = "https://github.com/luca-heltai/immersx/releases/download/v0.3"
ASSETS = {
    "BG001_skull_back_actuator.inp":
        "bf74412b4e3f67073d41d4a0743c1f61d31578297fad431ea8cc2cddf7c65105",
    "BG0001_alex_scale.vtk":
        "c908c5592e645168a9d6fdd395bbf48df9d162238422f97a3d8f59209c5dd0fc",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(directory, name, expected):
    target = directory / name
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"Refusing to overwrite {target}: SHA-256 mismatch")
        print(f"Verified existing {target}")
        return

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{name}.",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            print(f"Downloading {name}", flush=True)
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--retry", "3", "--connect-timeout", "30", "--max-time", "300",
             "--output", str(temporary), f"{RELEASE_URL}/{name}"],
            check=True)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded {name}: SHA-256 mismatch")
        temporary.replace(target)
        print(f"Verified {target}")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "brain",
        help="Destination directory (default: <repository>/data/brain)")
    arguments = parser.parse_args()
    try:
        directory = arguments.directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for name, expected in ASSETS.items():
            download(directory, name, expected)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
