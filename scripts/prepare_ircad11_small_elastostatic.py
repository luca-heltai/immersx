#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Download the existing IRCAD11 small mesh and three grown native networks.

Python standard library only. No vascular growth or meshing is performed.
The asset retains the source attribution, license, provenance and CSV audits.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request

ASSET_NAME = "ircad11-small-grown-networks-4804-20261112-v1.tar.gz"
ASSET_URL = ("https://github.com/luca-heltai/immersx/releases/download/"
             "liver-h-scaled-2026-11-10/" + ASSET_NAME)
ASSET_SHA256 = "20a8056214b565af988828025259334ea26fa6ec8504aea7d6719112f2456992"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_archive(data):
    if sha256(data) != ASSET_SHA256:
        raise ValueError("Downloaded asset SHA-256 mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or
                    ".." in name.parts or member.name in files):
                raise ValueError("Unsafe or repeated asset member: " + member.name)
            files[member.name] = archive.extractfile(member).read()
    for line in files["SHA256SUMS"].decode().splitlines():
        expected, name = line.split("  ", 1)
        if sha256(files[name]) != expected:
            raise ValueError("Asset member SHA-256 mismatch: " + name)
    manifest = json.loads(files["manifest.json"])
    for name, expected in manifest["sha256"].items():
        if sha256(files[name]) != expected:
            raise ValueError("Manifest SHA-256 mismatch: " + name)
    terminals = list(csv.DictReader(io.StringIO(files["audit/terminals.csv"].decode())))
    paired = [r for r in terminals if r["territory"] != str(2**64 - 1)]
    counts = Counter(r["system"] for r in paired)
    triplets = list(csv.DictReader(io.StringIO(files["audit/triplets.csv"].decode())))
    vessels = list(csv.DictReader(io.StringIO(files["audit/vessels.csv"].decode())))
    if (counts != {"0": 4804, "1": 4804, "2": 4804} or
            len(triplets) != 4804 or len(vessels) != 30011 or
            manifest["background_tetrahedra"] != 4804):
        raise ValueError("Unexpected IRCAD11 small network/cell counts")
    if len([name for name in files if name.startswith("network/")]) != 31:
        raise ValueError("Incomplete native physical archive")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory, normally <build>/data/ircad11-small")
    parser.add_argument("--archive", type=Path,
                        help="Verify and reuse a previously downloaded asset")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Output exists; choose a new directory")
    if args.archive:
        data = args.archive.read_bytes()
    else:
        request = urllib.request.Request(
            ASSET_URL, headers={"User-Agent": "ImmersX-IRCAD11-data-preparation"})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
    files = verify_archive(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name+".", dir=output.parent) as tmp:
        prepared = Path(tmp)
        for name, payload in files.items():
            target = prepared / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        record = dict(url=ASSET_URL, sha256=ASSET_SHA256, bytes=len(data),
                      growth_performed=False, meshing_performed=False)
        (prepared/"downloaded-asset.json").write_text(json.dumps(record, indent=2)+"\n")
        prepared.rename(output)
    print(f"Prepared {output}: 4804 cells; 4804 paired terminals per role; "
          "4804 triplets; 30011 physical vessels; no growth performed")


if __name__ == "__main__":
    main()
