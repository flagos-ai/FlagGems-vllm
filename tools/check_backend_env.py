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

"""Run a backend-neutral import, device, and float32 tensor smoke check."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

REQUIRED_MODULES = (
    "einops",
    "numpy",
    "packaging",
    "pytest",
    "sqlalchemy",
    "torch",
    "triton",
    "yaml",
)


def module_version(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-vendor", required=True)
    parser.add_argument("--require-flaggems", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("python:", sys.version.replace("\n", " "))
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        print(f"{name}:", module_version(module))

    if args.require_flaggems:
        flag_gems = importlib.import_module("flag_gems")
        print("flag_gems:", module_version(flag_gems))

    import torch

    import flaggems_vllm
    from flaggems_vllm import runtime

    print("flaggems_vllm:", flaggems_vllm.__version__)
    print("vendor:", flaggems_vllm.vendor_name)
    print("device:", flaggems_vllm.device)
    print("device count:", runtime.device.device_count)

    expected_vendor = args.expected_vendor.lower()
    configured_vendor = os.environ.get("FLAGGEMS_VENDOR", "").lower()
    if configured_vendor != expected_vendor:
        raise RuntimeError(
            "FLAGGEMS_VENDOR mismatch: "
            f"expected {expected_vendor!r}, got {configured_vendor!r}"
        )
    if flaggems_vllm.vendor_name != expected_vendor:
        raise RuntimeError(
            "detected vendor mismatch: "
            f"expected {expected_vendor!r}, got {flaggems_vllm.vendor_name!r}"
        )
    if runtime.device.device_count < 1:
        raise RuntimeError(f"no {expected_vendor} torch device is available")

    left = torch.ones(4, dtype=torch.float32, device=flaggems_vllm.device)
    right = torch.ones(4, dtype=torch.float32, device=flaggems_vllm.device)
    result = left + right
    values = result.detach().to("cpu").tolist()
    if values != [2.0, 2.0, 2.0, 2.0]:
        raise RuntimeError(f"device add returned unexpected values: {values!r}")

    print("portable smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
