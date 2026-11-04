#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install repository dependencies after FlagGems backend setup (Python 3.11+)."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", help="default: reuse FlagGems vendor detection")
    parser.add_argument("--dry-run", action="store_true")
    args, pip_args = parser.parse_known_args()
    vendor = args.vendor
    if vendor is None:
        import flag_gems

        vendor = flag_gems.vendor_name
    root = Path(__file__).resolve().parent.parent
    with (root / "pyproject.toml").open("rb") as source:
        extras = tomllib.load(source)["project"]["optional-dependencies"]
    if vendor not in extras or vendor in ("test", "vllm-build"):
        parser.error(f"no repository dependency group declared for vendor {vendor!r}")
    requirement = f"{root}[{vendor}]"
    if not args.dry_run and sys.prefix == sys.base_prefix:
        parser.error("activate the backend virtual environment before installing")
    # Like setup.sh, install the selected compiler after framework dependencies:
    # FlagTree owns the triton namespace but does not provide its package name.
    compiler = None
    try:
        distribution = metadata.distribution("flagtree")
        compiler = f"flagtree=={distribution.version}"
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        url = urlparse(origin.get("url", ""))
        if url.scheme == "file" and Path(unquote(url.path)).is_file():
            compiler = origin["url"]
    except metadata.PackageNotFoundError:
        pass
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-e",
        requirement,
        *pip_args,
    ]
    print(f"Vendor: {vendor}", flush=True)
    environment = os.environ.copy()
    if vendor in ("thead", "metax"):
        environment.update(
            VLLM_TARGET_DEVICE="empty",
            VLLM_VERSION_OVERRIDE="0.24.0+empty",
            VLLM_REQUIRE_RUST_FRONTEND="0",
            CARGO_NET_OFFLINE="true",
        )
        build = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-e",
            f"{root}[vllm-build]",
            *pip_args,
        ]
        print(shlex.join(build), flush=True)
        if not args.dry_run:
            subprocess.run(build, check=True, env=environment)
        print(
            "Building pinned vLLM 0.24 in empty mode with the existing torch.",
            flush=True,
        )
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, env=environment)
        if compiler is not None:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "triton"],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    compiler,
                    *pip_args,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
