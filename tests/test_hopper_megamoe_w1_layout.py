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

"""CPU-only parity checks for the SM90 MegaMoE W1 layout contract."""

import importlib.util
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGAMOE_ROOT = (
    PROJECT_ROOT
    / "src"
    / "flaggems_vllm"
    / "runtime"
    / "backend"
    / "_nvidia"
    / "hopper"
    / "mega"
    / "megamoe"
)
QWEN_DATA_PATH = MEGAMOE_ROOT / "qwen3_fp8_shared_data.py"
QWEN_DATA_SPEC = importlib.util.spec_from_file_location(
    "flaggems_vllm_hopper_megamoe_qwen3_data", QWEN_DATA_PATH
)
assert QWEN_DATA_SPEC is not None and QWEN_DATA_SPEC.loader is not None
QWEN_DATA_MODULE = importlib.util.module_from_spec(QWEN_DATA_SPEC)
QWEN_DATA_SPEC.loader.exec_module(QWEN_DATA_MODULE)
interleave_l1_gate_up_rows = QWEN_DATA_MODULE.interleave_l1_gate_up_rows


def test_gran8_gate_up_interleave():
    source = torch.arange(32, dtype=torch.uint8).reshape(1, 32, 1)
    actual = interleave_l1_gate_up_rows(source, torch).flatten().tolist()
    expected = (
        list(range(0, 8))
        + list(range(16, 24))
        + list(range(8, 16))
        + list(range(24, 32))
    )
    assert actual == expected


def test_w1_scale_rows_stay_in_nl1n():
    nl1n = 24
    gate_rows = [n_block // 2 for n_block in range(nl1n)]
    up_rows = [nl1n // 2 + n_block // 2 for n_block in range(nl1n)]

    assert gate_rows == [value for value in range(12) for _ in range(2)]
    assert up_rows == [value for value in range(12, 24) for _ in range(2)]
    assert max(gate_rows + up_rows) < nl1n


def test_kernel_embeds_corrected_w1_contract():
    kernel_path = MEGAMOE_ROOT / "kernel.py"
    source = kernel_path.read_text(encoding="utf-8")

    for forbidden in ("cur_e * 2 * NL1N", "EPR, 2 * NL1N, NK1"):
        assert forbidden not in source
    for required in (
        "cur_e * NL1N",
        "NL1N // 2",
        "interleave_l1_gate_up_rows",
        "EPR, NL1N, NK1",
    ):
        assert required in source
