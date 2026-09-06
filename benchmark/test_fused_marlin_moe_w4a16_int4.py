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

"""Benchmark W4A16 INT4 fused Marlin MoE against the vLLM implementation."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

from . import base

try:
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False


GROUP_SIZE = 128
MODEL_GEOMETRY = (256, 4096, 256, 6)
PR5140_TRACE = (
    (1, 172),
    (2, 172),
    (4, 172),
    (8, 172),
    (16, 172),
    (24, 172),
    (32, 172),
    (40, 172),
    (48, 172),
    (56, 172),
    (64, 172),
    (72, 172),
    (80, 172),
    (88, 172),
    (96, 172),
    (104, 172),
    (112, 172),
    (120, 172),
    (128, 172),
    (136, 172),
    (144, 172),
    (152, 172),
    (160, 172),
    (168, 172),
    (176, 172),
    (184, 172),
    (192, 172),
    (200, 172),
    (208, 172),
    (216, 172),
    (224, 172),
    (232, 172),
    (240, 172),
    (248, 172),
    (256, 172),
    (272, 172),
    (288, 172),
    (304, 172),
    (320, 172),
    (336, 172),
    (352, 172),
    (368, 172),
    (384, 172),
    (400, 172),
    (416, 172),
    (432, 172),
    (448, 172),
    (464, 172),
    (480, 172),
    (496, 344),
    (512, 344),
    (2048, 43),
    (16384, 946),
)


def _pack_gptq_int32(weight):
    """Convert output-major uint8 pairs to GPTQ INT32 packing."""
    experts, output_size, packed_k = weight.shape
    packed = torch.empty(
        (experts, packed_k // 4, output_size),
        device=weight.device,
        dtype=torch.int32,
    )
    for expert in range(experts):
        bytes4 = weight[expert].to(torch.int32).reshape(output_size, packed_k // 4, 4)
        words = (
            bytes4[..., 0]
            | (bytes4[..., 1] << 8)
            | (bytes4[..., 2] << 16)
            | (bytes4[..., 3] << 24)
        )
        packed[expert].copy_(words.transpose(0, 1))
    return packed


def _to_vllm_marlin(weight, scales, size_k, size_n):
    qweight = _pack_gptq_int32(weight)
    empty_perm = torch.empty(
        (weight.size(0), 0), device=weight.device, dtype=torch.int32
    )
    qweight = vllm_ops.gptq_marlin_moe_repack(
        qweight,
        empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
    )
    scales = marlin_moe_permute_scales(
        scales.transpose(1, 2).contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=GROUP_SIZE,
    )
    return qweight, scales


def _make_weights(num_experts, hidden_size, intermediate_size, dtype):
    torch.manual_seed(7)
    device = flaggems_vllm.device
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1_scale = (
        torch.rand(
            (num_experts, 2 * intermediate_size, hidden_size // GROUP_SIZE),
            device=device,
            dtype=dtype,
        )
        * 0.03
    )
    w2_scale = (
        torch.rand(
            (num_experts, hidden_size, intermediate_size // GROUP_SIZE),
            device=device,
            dtype=dtype,
        )
        * 0.03
    )
    vllm_w1, vllm_w1_scale = _to_vllm_marlin(
        w1, w1_scale, hidden_size, 2 * intermediate_size
    )
    vllm_w2, vllm_w2_scale = _to_vllm_marlin(
        w2, w2_scale, intermediate_size, hidden_size
    )
    return (
        w1,
        w2,
        w1_scale,
        w2_scale,
        vllm_w1,
        vllm_w2,
        vllm_w1_scale,
        vllm_w2_scale,
    )


class FusedMarlinMoEW4A16INT4Benchmark(base.Benchmark):
    """Use the production trace from FlagGems PR 5140."""

    def set_shapes(self, shape_file_path=None):
        num_experts, hidden_size, intermediate_size, top_k = MODEL_GEOMETRY
        self.shapes = [
            (m, num_experts, hidden_size, intermediate_size, top_k, count)
            for m, count in PR5140_TRACE
        ]
        self.shape_desc = "M, E, K, N, top_k, call_count"

    def get_input_iter(self, dtype):
        num_experts, hidden_size, intermediate_size, top_k = MODEL_GEOMETRY
        weights = _make_weights(num_experts, hidden_size, intermediate_size, dtype)
        for num_tokens, call_count in PR5140_TRACE:
            torch.manual_seed(7 + num_tokens)
            hidden_states = (
                torch.randn(
                    (num_tokens, hidden_size),
                    device=flaggems_vllm.device,
                    dtype=dtype,
                )
                * 0.1
            )
            topk_ids = (
                torch.rand((num_tokens, num_experts), device=flaggems_vllm.device)
                .topk(top_k, dim=-1)
                .indices
            )
            topk_weights = torch.softmax(
                torch.randn((num_tokens, top_k), device=flaggems_vllm.device),
                dim=-1,
            ).to(torch.float32)
            yield (hidden_states, *weights, topk_weights, topk_ids, call_count)


def _vllm_baseline(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    vllm_w1,
    vllm_w2,
    vllm_w1_scale,
    vllm_w2_scale,
    topk_weights,
    topk_ids,
    call_count,
):
    del w1, w2, w1_scale, w2_scale, call_count
    return vllm_fused_marlin_moe(
        hidden_states,
        vllm_w1,
        vllm_w2,
        None,
        None,
        vllm_w1_scale,
        vllm_w2_scale,
        None,
        topk_weights,
        topk_ids,
        scalar_types.uint4b8.id,
    )


def _gems_call(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    vllm_w1,
    vllm_w2,
    vllm_w1_scale,
    vllm_w2_scale,
    topk_weights,
    topk_ids,
    call_count,
):
    del vllm_w1, vllm_w2, vllm_w1_scale, vllm_w2_scale, call_count
    return flaggems_vllm.fused_marlin_moe_w4a16_int4(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        QUANT_TYPE_UINT4B8,
    )


@pytest.mark.fused_marlin_moe_w4a16_int4
@pytest.mark.skipif(
    not HAS_VLLM_FUSED_MARLIN_MOE,
    reason="vLLM fused_marlin_moe is not installed",
)
def test_fused_marlin_moe_w4a16_int4():
    bench = FusedMarlinMoEW4A16INT4Benchmark(
        op_name="fused_marlin_moe_w4a16_int4",
        torch_op=_vllm_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call)
    bench.run()
