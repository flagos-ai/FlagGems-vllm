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

"""dsv41_o_proj (wo_a part) vs vLLM ``deep_gemm_fp8_o_proj`` on Hopper.

Both sides run inverse RoPE + FP8 quant + grouped FP8 GEMM; wo_b is an identity
callback on the vLLM side, so only the wo_a stage is timed.

Small batches take one kernel launch and large ones two; see ``dsv41_o_proj``. Also
run ``--mode operator``: it counts the CPU launch time the default mode partly hides.
"""

import math
import types

import pytest
import torch

import flaggems_vllm

from . import base

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
QUANT_BLOCK = 128
NOPE_DIM, ROPE_DIM = 448, 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
MAX_POSITION = 8192


def is_hopper():
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


HOPPER = is_hopper()

try:
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    HAS_VLLM = is_deep_gemm_supported()
except ImportError:
    HAS_VLLM = False


def _make_cos_sin_cache(device):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(MAX_POSITION, dtype=torch.float32), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device)


def _per_block_cast_to_fp8(w):
    n, k = w.shape
    wb = w.float().view(n // QUANT_BLOCK, QUANT_BLOCK, k // QUANT_BLOCK, QUANT_BLOCK)
    scale = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4) / FP8_MAX
    q = (wb / scale).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.view(n // QUANT_BLOCK, k // QUANT_BLOCK)


def _make_inputs(num_tokens, n_groups, hpg, rank, device, seed=0):
    torch.manual_seed(seed)
    k = hpg * HEAD_DIM
    attn_out = torch.randn(
        num_tokens, n_groups * hpg, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    pos = torch.randint(0, MAX_POSITION, (num_tokens,), device=device)
    csc = _make_cos_sin_cache(device)
    w, ws = _per_block_cast_to_fp8(
        torch.randn(n_groups * rank, k, device=device) / math.sqrt(k)
    )
    return (
        attn_out,
        pos,
        csc,
        w.view(n_groups, rank, k),
        ws.view(n_groups, rank // QUANT_BLOCK, k // QUANT_BLOCK),
    )


class Dsv41OProjBenchmark(base.Benchmark):
    """Shape = (num_tokens, n_groups, heads_per_group, o_lora_rank)."""

    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        tokens = (1, 4, 8, 16, 32, 64, 96, 128, 512, 2048, 8192, 16384)
        groups = ((8, 8, 1024), (16, 8, 1024), (2, 8, 1024))
        return [(t, g, h, r) for (g, h, r) in groups for t in tokens]

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        device = flaggems_vllm.device
        for t, g, h, r in self.shapes:
            inputs = _make_inputs(t, g, h, r, device)
            # Compile / tune outside the timed region.
            _vllm_wrapper(*inputs)
            _gems_wrapper(*inputs)
            torch.cuda.synchronize()
            yield inputs

    def get_tflops(self, op, *args, **kwargs):
        attn_out, _, _, w3, _ = args
        g, n, k = w3.shape
        return 2.0 * attn_out.shape[0] * g * n * k


def _vllm_wrapper(attn_out, pos, csc, w3, ws3):
    g, rank, k = w3.shape
    wo_a = types.SimpleNamespace(weight=w3, weight_scale_inv=ws3)
    # Identity wo_b: both sides allocate and return z, nothing else is timed.
    return deep_gemm_fp8_o_proj(
        attn_out,
        pos,
        csc,
        wo_a,
        lambda flat: flat,
        n_groups=g,
        heads_per_group=k // HEAD_DIM,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=rank,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )


def _gems_wrapper(attn_out, pos, csc, w3, ws3):
    g, rank, k = w3.shape
    return flaggems_vllm.dsv41_o_proj(
        attn_out,
        pos,
        csc,
        w3,
        ws3,
        n_groups=g,
        heads_per_group=k // HEAD_DIM,
        o_lora_rank=rank,
    )


@pytest.mark.dsv41_o_proj
@pytest.mark.skipif(
    not (HAS_VLLM and HOPPER),
    reason="requires NVIDIA Hopper and vLLM with DeepGEMM support",
)
def test_perf_dsv41_o_proj_gems_vs_vllm():
    bench = Dsv41OProjBenchmark(
        op_name="dsv41_o_proj",
        torch_op=_vllm_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_wrapper)
    bench.run()
