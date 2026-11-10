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

"""Tests for the public Hopper MegaMoE entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_IS_ACCURACY_WORKER = "--accuracy-worker" in sys.argv
if _IS_ACCURACY_WORKER:
    _local_rank = os.environ.get("LOCAL_RANK", "0")
    _cache_root = Path(
        os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")
    )
    os.environ["TRITON_CACHE_DIR"] = str(
        _cache_root.parent / f"{_cache_root.name}-megamoe-test-rank-{_local_rank}"
    )

import pytest  # noqa: E402
import torch  # noqa: E402

if _IS_ACCURACY_WORKER:
    torch.cuda.set_device(int(_local_rank))

# The worker cache and CUDA device must be selected before these imports.
# isort: off
import flaggems_vllm  # noqa: E402
from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (  # noqa: E402
    megamoe as hopper_megamoe,
)
from flaggems_vllm.testing.megamoe import StandaloneNvshmem  # noqa: E402
from flaggems_vllm.testing.megamoe import (  # noqa: E402
    configure_descriptor_allocator,
    interleave_l1_gate_up_rows,
    make_context,
    pool_tokens,
)
from flaggems_vllm.utils.triton_version_utils import has_triton_tle  # noqa: E402

# isort: on

pytestmark = pytest.mark.megamoe


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamoe_fp8_is_exported_on_hopper():
    assert callable(hopper_megamoe.megamoe_fp8)
    assert flaggems_vllm.megamoe_fp8 is hopper_megamoe.megamoe_fp8


@dataclass(frozen=True)
class _AccuracyConfig:
    num_ranks: int = 2
    tokens: int = 2
    hidden: int = 128
    intermediate: int = 128
    num_experts: int = 2
    topk: int = 1
    stages: int = 2
    max_recv: int = 128


def _make_accuracy_inputs(config: _AccuracyConfig, rank: int, device):
    generator = torch.Generator().manual_seed(20261105)
    fp8 = torch.float8_e4m3fn

    inputs = torch.randn(
        config.num_ranks,
        config.tokens,
        config.hidden,
        generator=generator,
    ).mul_(0.25)
    input_scales = (
        torch.rand(
            config.num_ranks,
            config.tokens,
            config.hidden // 128,
            generator=generator,
        )
        .mul_(0.25)
        .add_(0.75)
    )
    w1_logical = torch.randn(
        config.num_experts,
        2 * config.intermediate,
        config.hidden,
        generator=generator,
    ).mul_(0.125)
    w1_scales = (
        torch.rand(
            config.num_experts,
            config.intermediate // 64,
            config.hidden // 128,
            generator=generator,
        )
        .mul_(0.25)
        .add_(0.75)
    )
    w2 = torch.randn(
        config.num_experts,
        config.hidden,
        config.intermediate,
        generator=generator,
    ).mul_(0.125)
    w2_scales = (
        torch.rand(
            config.num_experts,
            config.hidden // 128,
            config.intermediate // 128,
            generator=generator,
        )
        .mul_(0.25)
        .add_(0.75)
    )

    inputs = inputs.to(device=device, dtype=fp8)
    input_scales = input_scales.to(device)
    w1_logical = w1_logical.to(device=device, dtype=fp8)
    w1_scales = w1_scales.to(device)
    w2 = w2.to(device=device, dtype=fp8)
    w2_scales = w2_scales.to(device)
    w1_packed = interleave_l1_gate_up_rows(w1_logical)

    topk_all = torch.empty(
        config.num_ranks,
        config.tokens,
        config.topk,
        dtype=torch.int64,
    )
    for source_rank in range(config.num_ranks):
        topk_all[source_rank].fill_(1 - source_rank)
    topk_weights = torch.tensor([[0.75], [1.25]], dtype=torch.float32, device=device)

    local_slice = slice(rank, rank + 1)
    operands = (
        inputs[rank].contiguous(),
        input_scales[rank].contiguous(),
        topk_all[rank].to(device),
        topk_weights,
        w1_packed[local_slice].contiguous(),
        w1_scales[local_slice].contiguous(),
        w2[local_slice].contiguous(),
        w2_scales[local_slice].contiguous(),
    )
    reference_inputs = (
        inputs[rank],
        input_scales[rank],
        topk_all[rank].to(device),
        topk_weights,
        w1_logical,
        w1_scales,
        w2,
        w2_scales,
    )
    return operands, reference_inputs, topk_all


def _reference_megamoe(
    input_fp8,
    input_scales,
    topk_ids,
    topk_weights,
    w1,
    w1_scales,
    w2,
    w2_scales,
):
    tokens, hidden = input_fp8.shape
    intermediate = w2.shape[-1]
    output = torch.zeros(tokens, hidden, dtype=torch.float32, device=input_fp8.device)

    for token in range(tokens):
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            l1 = torch.zeros(
                2 * intermediate, dtype=torch.float32, device=input_fp8.device
            )
            for n_block in range((2 * intermediate) // 128):
                n_start = n_block * 128
                for k_block in range(hidden // 128):
                    k_start = k_block * 128
                    product = (
                        input_fp8[token, k_start : k_start + 128].float()
                        @ w1[
                            expert,
                            n_start : n_start + 128,
                            k_start : k_start + 128,
                        ]
                        .float()
                        .T
                    )
                    l1[n_start : n_start + 128] += product * (
                        input_scales[token, k_block]
                        * w1_scales[expert, n_block, k_block]
                    )

            gate, up = l1.chunk(2)
            swiglu = gate / (1.0 + torch.exp(-gate)) * up
            route_weight = topk_weights[token, route]
            segments = swiglu.reshape(intermediate // 64, 64)
            amax = segments.abs().amax(dim=1) * route_weight.abs()
            scaled = amax / 448.0
            segment_scales = torch.where(
                scaled > 0,
                torch.exp2(torch.ceil(torch.log2(scaled))),
                torch.ones_like(scaled),
            )
            quantized = (segments * (route_weight / segment_scales)[:, None]).to(
                torch.float8_e4m3fn
            )

            partial = torch.zeros(hidden, dtype=torch.float32, device=input_fp8.device)
            for n_block in range(hidden // 128):
                n_start = n_block * 128
                for k_block in range(intermediate // 128):
                    k_start = k_block * 128
                    weight_scale = w2_scales[expert, n_block, k_block]
                    low = (
                        quantized[2 * k_block].float()
                        @ w2[
                            expert,
                            n_start : n_start + 128,
                            k_start : k_start + 64,
                        ]
                        .float()
                        .T
                    )
                    high = (
                        quantized[2 * k_block + 1].float()
                        @ w2[
                            expert,
                            n_start : n_start + 128,
                            k_start + 64 : k_start + 128,
                        ]
                        .float()
                        .T
                    )
                    partial[n_start : n_start + 128] += weight_scale * (
                        low * segment_scales[2 * k_block]
                        + high * segment_scales[2 * k_block + 1]
                    )
            output[token] += partial.to(torch.bfloat16).float()

    return output.to(torch.bfloat16)


def _run_accuracy_worker() -> None:
    config = _AccuracyConfig()
    with StandaloneNvshmem() as runtime:
        assert runtime.world_size == config.num_ranks
        configure_descriptor_allocator(runtime.device)
        operands, reference_inputs, topk_all = _make_accuracy_inputs(
            config, runtime.rank, runtime.device
        )
        pool_size = pool_tokens(topk_all, config)
        context = make_context(config, runtime, pool_size)
        output = hopper_megamoe.megamoe_fp8(
            *operands,
            context,
            stages=config.stages,
        )
        torch.cuda.synchronize(runtime.device)
        reference = _reference_megamoe(*reference_inputs)

        relative_l2 = (
            output.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
        assert float(relative_l2) < 0.02, relative_l2
        assert float(cosine) > 0.999, cosine


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamoe_fp8_accuracy():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible Hopper GPUs")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "127.0.0.1:0",
        "--nproc-per-node",
        "2",
        str(Path(__file__).resolve()),
        "--accuracy-worker",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + completed.stderr)


if __name__ == "__main__" and _IS_ACCURACY_WORKER:
    _run_accuracy_worker()
