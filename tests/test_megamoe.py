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

"""Tests for the MegaMoE data-layout contract."""

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe.launcher import (
    MegaMoEConfig,
    _parse_benchmark_output,
    _worker_cli_args,
    parse_megamoe_config,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe.qwen3_fp8_shared_data import (
    interleave_l1_gate_up_rows,
)

MEGAMOE_LAYOUT_CONFIGS = [
    # (num_experts, rows, hidden_size, tile_rows)
    (1, 32, 1, 8),
    (2, 24, 3, 4),
]

MEGAMOE_INVALID_LAYOUT_CONFIGS = [
    # (shape, tile_rows, expected error)
    ((4, 8), 8, "rank 3"),
    ((1, 15, 2), 1, "row count must be even"),
    ((1, 20, 2), 8, "must be divisible"),
]


def _reference_interleave(weight, granularity):
    half = weight.shape[1] // 2
    chunks = []
    for offset in range(0, half, granularity):
        chunks.extend(
            (
                weight[:, offset : offset + granularity],
                weight[:, half + offset : half + offset + granularity],
            )
        )
    return torch.cat(chunks, dim=1)


@pytest.mark.megamoe
@pytest.mark.parametrize("config", MEGAMOE_LAYOUT_CONFIGS)
def test_megamoe_w1_layout(config):
    experts, rows, hidden, granularity = config
    source = torch.arange(experts * rows * hidden, dtype=torch.uint8).reshape(
        experts, rows, hidden
    )

    actual = interleave_l1_gate_up_rows(source, torch, granularity)
    expected = _reference_interleave(source, granularity)

    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.megamoe
@pytest.mark.parametrize("config", MEGAMOE_INVALID_LAYOUT_CONFIGS)
def test_megamoe_w1_layout_rejects_invalid_shape(config):
    shape, granularity, error = config
    weight = torch.empty(shape, dtype=torch.uint8)

    with pytest.raises(ValueError, match=error):
        interleave_l1_gate_up_rows(weight, torch, granularity)


@pytest.mark.megamoe
def test_megamoe_launcher_parses_every_rank_latency():
    config = MegaMoEConfig(benchmark=True)
    stdout = "\n".join(
        (
            "[rank 1/2] BENCH h=256 ih=128 E=16 k=4 tokens=128 "
            "recv=64 experts=8 | 1500.0 us",
            "[rank 0/2] BENCH h=256 ih=128 E=16 k=4 tokens=128 "
            "recv=64 experts=8 | 1250.0 us",
        )
    )

    assert _parse_benchmark_output(stdout, config) == (1.25, 1.5)


@pytest.mark.megamoe
def test_megamoe_launcher_rejects_missing_rank_latency():
    config = MegaMoEConfig(benchmark=True)
    stdout = (
        "[rank 0/2] BENCH h=256 ih=128 E=16 k=4 tokens=128 "
        "recv=64 experts=8 | 1250.0 us"
    )

    with pytest.raises(RuntimeError, match="every rank"):
        _parse_benchmark_output(stdout, config)


@pytest.mark.megamoe
def test_megamoe_worker_cli_preserves_kernel_config(tmp_path):
    config = MegaMoEConfig(
        num_ranks=4,
        tokens=256,
        hidden_size=512,
        intermediate_size=256,
        num_experts=32,
        topk=8,
        stages=3,
        num_sms=64,
        max_recv=1024,
        drop_rate=0,
        benchmark=True,
        warmup=10,
        iterations=30,
        reduce="mean",
        gpu_start_barrier=True,
        data_dir=tmp_path,
        verify_data_sha256=True,
        inject_fault="queue",
        cuda_home=tmp_path / "cuda",
        nvshmem_home=tmp_path / "nvshmem",
    )

    worker = parse_megamoe_config(_worker_cli_args(config))

    assert worker.worker
    for field in (
        "num_ranks",
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "topk",
        "stages",
        "num_sms",
        "max_recv",
        "drop_rate",
        "benchmark",
        "warmup",
        "iterations",
        "reduce",
        "gpu_start_barrier",
        "data_dir",
        "verify_data_sha256",
        "inject_fault",
        "cuda_home",
        "nvshmem_home",
    ):
        assert getattr(worker, field) == getattr(config, field)
