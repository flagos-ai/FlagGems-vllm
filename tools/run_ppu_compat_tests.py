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

"""Run the bounded PPU compatibility suite and reject missing/skipped core tests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TARGETS = (
    "tests/test_optional_runtime_imports.py",
    "tests/test_thead_optional_imports.py",
    "tests/test_concat_and_cache_mla_padding.py",
    "tests/test_mhc_post_graph.py",
    "tests/test_qsa_sparse_paged_attention_unpacked.py",
    "tests/test_mhc_ops.py::test_mhc_post_optional_compiler_knob_restored",
)
REQUIRED_FAMILIES = (
    "test_nvidia_mnnvl_optional_dependency_contract",
    "test_mhc_post_changed_input_graph",
    "test_unpacked_qsa_public_changed_input_graph",
    "test_padded_rows_graph_replay_changed_inputs",
    "test_non_tle_moe_graph_replays_changed_routes",
    "test_mhc_post_optional_compiler_knob_restored",
)


def validate_report(path: Path) -> None:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases or any(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    ):
        raise RuntimeError("PPU compatibility suite is empty or contains failures")
    for family in REQUIRED_FAMILIES:
        selected = [
            case for case in cases if case.get("name", "").split("[", 1)[0] == family
        ]
        if not selected or any(case.find("skipped") is not None for case in selected):
            raise RuntimeError(f"Required compatibility family did not run: {family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch

    import flaggems_vllm

    if (
        flaggems_vllm.runtime.device.vendor_name != "thead"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("PPU compatibility tests require an available T-Head device")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "pythonpath=",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(output / "pytest-tmp"),
        "--record",
        "json",
        "--output",
        str(output / "results.json"),
        "--junitxml",
        str(output / "junit.xml"),
        *TARGETS,
    ]
    print(json.dumps({"installed_package": flaggems_vllm.__file__, "command": command}))
    subprocess.run(command, cwd=root, env=env, check=True)
    validate_report(output / "junit.xml")


if __name__ == "__main__":
    main()
