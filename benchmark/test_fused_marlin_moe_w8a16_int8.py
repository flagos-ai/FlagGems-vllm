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

import pytest
import torch

# vLLM imports (baseline). Optional: when vllm is not installed (e.g. in CI),
# the entire benchmark is skipped via the skipif marker below.
try:
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_INT8 = scalar_types.uint8b128
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flaggems_vllm

# FlagGems wrapper under test
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT8B128
from flaggems_vllm.ops.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe

from . import base

try:
    from vllm import _custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
    )
    from vllm.scalar_type import scalar_types as fp8_scalar_types

    VLLM_QUANT_TYPE_FP8 = fp8_scalar_types.float8_e4m3fn
    HAS_VLLM_FP8_MARLIN_MOE = HAS_VLLM_FUSED_MARLIN_MOE
except ImportError:
    HAS_VLLM_FP8_MARLIN_MOE = False


def is_supported_device():
    if flaggems_vllm.vendor_name == "thead":
        return True
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


SUPPORTED_DEVICE = is_supported_device()
HAS_REQUIRED_VLLM = (
    HAS_VLLM_FP8_MARLIN_MOE
    if flaggems_vllm.vendor_name == "thead"
    else HAS_VLLM_FUSED_MARLIN_MOE
)

GROUP_SIZE = 128


def _wna16_quantize_per_expert_int8(w_fp):
    """
    Per-expert GPTQ-style INT8 quantization for FlagGems wna16 kernel layout.
    INT8 is one byte per element — no nibble packing — so K-dim stays in_dim.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim), uint8
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        _, q_e, sc_e, _ = quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE_INT8, GROUP_SIZE, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        w_q[e] = q_e
        scales[e] = sc_e
    return w_q, scales


def _marlin_quantize_per_expert_int8(w_fp):
    """Per-expert Marlin-layout INT8 quantization for vLLM's fused_marlin_moe."""
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        _, qw, sc, _, _, _ = marlin_quantize(
            w_fp[e].T.contiguous(), VLLM_QUANT_TYPE_INT8, GROUP_SIZE, act_order=False
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


def _pack_gptq_int32_fp8(weight):
    """Pack four E4M3 bytes per GPTQ INT32 word."""
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


def _to_vllm_marlin_fp8(weight, scales, size_k, size_n):
    qweight = _pack_gptq_int32_fp8(weight.view(torch.uint8))
    empty_perm = torch.empty(
        (weight.size(0), 0), device=weight.device, dtype=torch.int32
    )
    qweight = vllm_ops.gptq_marlin_moe_repack(
        qweight,
        empty_perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
    )
    scales = marlin_moe_permute_scales(
        scales.transpose(1, 2).contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=GROUP_SIZE,
    )
    return qweight, fp8_fused_exponent_bias_into_scales(scales)


def _make_ppu_fp8_weights(num_experts, hidden_size, intermediate_size, dtype):
    torch.manual_seed(7)
    weights = []
    scales = []
    native_weights = []
    native_scales = []
    for output_size, input_size in (
        (2 * intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    ):
        raw = torch.randint(
            0,
            256,
            (num_experts, output_size, input_size),
            device=flaggems_vllm.device,
            dtype=torch.uint8,
        )
        raw = torch.where((raw & 127) == 127, raw - 1, raw)
        weight = raw.view(torch.float8_e4m3fn)
        scale = (
            torch.rand(
                (num_experts, output_size, input_size // GROUP_SIZE),
                device=flaggems_vllm.device,
            )
            * 0.001
            + 0.0005
        ).to(dtype)
        native_weight, native_scale = _to_vllm_marlin_fp8(
            weight, scale, input_size, output_size
        )
        weights.append(weight)
        scales.append(scale)
        native_weights.append(native_weight)
        native_scales.append(native_scale)
    return (*weights, *scales, *native_weights, *native_scales)


class FusedMarlinMoEW8A16INT8Benchmark(base.Benchmark):
    """
    Benchmark for fused_marlin_moe W8A16 INT8/FP8 (fused-dequant MoE GEMM).
    Sister of FusedMarlinMoEW4A16INT4Benchmark (W4A16 INT4).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-8x7B-like
            (1, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            # DeepSeek-V3-like (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        if flaggems_vllm.vendor_name == "thead":
            yield from self._get_ppu_input_iter(cur_dtype)
            return
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _get_ppu_input_iter(self, dtype):
        geometry = None
        weights = None
        for config in self.shapes:
            num_tokens, num_experts, hidden_size, intermediate_size, top_k = config
            next_geometry = (num_experts, hidden_size, intermediate_size)
            if geometry != next_geometry:
                weights = _make_ppu_fp8_weights(
                    num_experts, hidden_size, intermediate_size, dtype
                )
                geometry = next_geometry
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
            yield (hidden_states, *weights, topk_weights, topk_ids)

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        w1_fp = (
            torch.randn(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )
        w2_fp = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )

        # FlagGems wna16 INT8 layout (unpacked)
        w1_q_wna16, w1_scale_wna16 = _wna16_quantize_per_expert_int8(w1_fp)
        w2_q_wna16, w2_scale_wna16 = _wna16_quantize_per_expert_int8(w2_fp)

        # vLLM Marlin INT8 layout
        w1_q_marlin, w1_scale_marlin = _marlin_quantize_per_expert_int8(w1_fp)
        w2_q_marlin, w2_scale_marlin = _marlin_quantize_per_expert_int8(w2_fp)

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        yield (
            hidden_states,
            w1_q_wna16,
            w2_q_wna16,
            w1_scale_wna16,
            w2_scale_wna16,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            topk_weights,
            topk_ids,
        )


def _vllm_baseline_int8(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe (INT8)."""
    quant_type = (
        VLLM_QUANT_TYPE_FP8
        if flaggems_vllm.vendor_name == "thead"
        else VLLM_QUANT_TYPE_INT8
    )
    return vllm_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_marlin,
        w2=w2_q_marlin,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_marlin,
        w2_scale=w2_scale_marlin,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=quant_type.id,
    )


def _gems_call_int8(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """FlagGems' Triton wna16 fused_marlin_moe W8A16."""
    is_thead = flaggems_vllm.vendor_name == "thead"
    gems_op = flaggems_vllm.fused_marlin_moe if is_thead else gems_fused_marlin_moe
    quant_type_id = QUANT_TYPE_FP8_E4M3 if is_thead else QUANT_TYPE_UINT8B128
    return gems_op(
        hidden_states=hidden_states,
        w1=w1_q_wna16,
        w2=w2_q_wna16,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_wna16,
        w2_scale=w2_scale_wna16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=quant_type_id,
        group_size=GROUP_SIZE,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not HAS_REQUIRED_VLLM, reason="required vLLM Marlin baseline is unavailable"
)
@pytest.mark.skipif(not SUPPORTED_DEVICE, reason="requires NVIDIA Hopper or T-Head PPU")
def test_fused_marlin_moe_w8a16_int8():
    """
    Benchmark FlagGems fused_marlin_moe W8A16 (Triton wna16) vs vLLM
    fused_marlin_moe W8A16 (CUDA Marlin). Both run GPTQ uint8b128 + per-group-128.
    """
    bench = FusedMarlinMoEW8A16INT8Benchmark(
        op_name=(
            "fused_marlin_moe_w8a16_fp8"
            if flaggems_vllm.vendor_name == "thead"
            else "fused_marlin_moe_w8a16_int8"
        ),
        torch_op=_vllm_baseline_int8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call_int8)
    bench.run()
