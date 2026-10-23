# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Prepare the exact measured source workspace without importing Torch or CUDA."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument(
        "--receipt",
        help="Restore sources recorded by this bundled receipt instead of the final revision",
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    archive = ROOT / "measured_sources.tar.gz"
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != manifest["files"][archive.name]:
        raise SystemExit("Measured-source archive hash mismatch")
    destination = args.prepare.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("Choose a new or empty preparation directory")
    destination.mkdir(parents=True, exist_ok=True)
    expected = manifest["measured_source_sha256"]
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive.read_bytes()))) as tar:
        members = tar.getmembers()
        if {member.name for member in members} != set(expected):
            raise SystemExit("Unexpected measured-source archive members")
        sources = {}
        for member in members:
            if not member.isfile():
                raise SystemExit("Only regular source files are accepted")
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise SystemExit("Measured source hash mismatch: " + member.name)
            sources[hashlib.sha256(data).hexdigest()] = data
    if args.receipt:
        if args.receipt not in manifest["receipt_source_sha256"]:
            raise SystemExit("Choose a receipt listed in MANIFEST.json")
        history = ROOT / "source_history.tar.gz"
        if (
            hashlib.sha256(history.read_bytes()).hexdigest()
            != manifest["files"][history.name]
        ):
            raise SystemExit("Source-history archive hash mismatch")
        with tarfile.open(
            fileobj=io.BytesIO(gzip.decompress(history.read_bytes()))
        ) as tar:
            for member in tar:
                if (
                    not member.isfile()
                    or member.name not in manifest["historical_source_sha256"]
                ):
                    raise SystemExit("Unexpected source-history member")
                data = tar.extractfile(member).read()
                digest = hashlib.sha256(data).hexdigest()
                if digest != manifest["historical_source_sha256"][member.name]:
                    raise SystemExit("Source-history hash mismatch")
                sources[digest] = data
        expected = manifest["receipt_source_sha256"][args.receipt]
    for name, digest in expected.items():
        target = (destination / name).resolve()
        if destination not in target.parents:
            raise SystemExit("Archive path escapes the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sources[digest])
    (destination / "receipts").mkdir(exist_ok=True)
    print(
        json.dumps(
            {
                "status": "PREPARED_CPU_ONLY",
                "workspace": str(destination),
                "receipt": args.receipt,
            }
        )
    )


if __name__ == "__main__":
    main()
