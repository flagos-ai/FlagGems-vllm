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

"""CUDA baseline collection contract for dsv41_o_proj.

Native: vLLM 0.24.0 ``deep_gemm_fp8_o_proj``, the Hopper general path that
DeepSeek-V4.1's ``dsv41_o_proj`` calls (0.24.0 has no V4.1 model). It runs
inverse RoPE + per-128 FP8 quant + DeepGEMM grouped FP8 GEMM. ``wo_b`` is an
identity callback, as in benchmark/test_dsv41_o_proj.py, so only wo_a is timed.

Inputs, shapes and the shape key follow benchmark/test_dsv41_o_proj.py and
benchmark/core_shapes.yaml, so the benchmark's first tensor (``attn_out``)
matches ``key_shape``.
"""

import importlib
import math
import types

import torch

OP_NAME = "dsv41_o_proj"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

NOPE_DIM, ROPE_DIM = 448, 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
QUANT_BLOCK = 128
MAX_POSITION = 8192
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
EINSUM_RECIPE = (1, 128, 128)  # Hopper: per-128 activation scales, 128x128 weight

# (num_tokens, n_groups, heads_per_group, o_lora_rank), as in core_shapes.yaml.
# n_groups=8 is DeepSeek-V4-Flash's wo_a (64 heads); 16 and 1 cover larger
# and smaller head counts.
_SHAPES = [
    (1, 8, 8, 1024),
    (32, 8, 8, 1024),
    (64, 8, 8, 1024),
    (128, 8, 8, 1024),
    (257, 8, 8, 1024),
    (512, 8, 8, 1024),
    (4096, 8, 8, 1024),
    (1, 16, 8, 1024),
    (32, 16, 8, 1024),
    (128, 16, 8, 1024),
    (4096, 16, 8, 1024),
    (1, 1, 8, 1024),
    (32, 1, 8, 1024),
    (128, 1, 8, 1024),
    (1024, 1, 8, 1024),
]


def native():
    """Resolve vLLM's ``deep_gemm_fp8_o_proj``; None if unavailable."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        o_proj = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.o_proj")
        deep_gemm = importlib.import_module("vllm.utils.deep_gemm")
        supported = deep_gemm.is_deep_gemm_supported()
    except Exception as e:  # noqa: BLE001 - any import failure means no baseline
        NATIVE_UNAVAILABLE_REASON = f"{type(e).__name__}: {e}"
        return None
    if not supported:
        NATIVE_UNAVAILABLE_REASON = "DeepGEMM is not supported on this device"
        return None
    op = getattr(o_proj, "deep_gemm_fp8_o_proj", None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "vLLM has no deep_gemm_fp8_o_proj"
        return None
    return op


def grid():
    return [
        {
            "num_tokens": t,
            "n_groups": g,
            "heads_per_group": h,
            "o_lora_rank": r,
        }
        for t, g, h, r in _SHAPES
    ]


def _identity(z):
    return z


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


def build_inputs(binding, dtype, device):
    """Same tensors as the benchmark's ``_make_inputs`` (seed 0)."""
    t = binding["num_tokens"]
    g = binding["n_groups"]
    h = binding["heads_per_group"]
    r = binding["o_lora_rank"]
    k = h * HEAD_DIM
    torch.manual_seed(0)
    attn_out = torch.randn(t, g * h, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.randint(0, MAX_POSITION, (t,), device=device)
    cos_sin_cache = _make_cos_sin_cache(device)
    w, ws = _per_block_cast_to_fp8(torch.randn(g * r, k, device=device) / math.sqrt(k))
    # vLLM reads wo_a.weight / wo_a.weight_scale_inv after DeepGEMM post-processing.
    wo_a = types.SimpleNamespace(
        weight=w.view(g, r, k),
        weight_scale_inv=ws.view(g, r // QUANT_BLOCK, k // QUANT_BLOCK),
    )
    args = (attn_out, positions, cos_sin_cache, wo_a, _identity)
    kwargs = {
        "n_groups": g,
        "heads_per_group": h,
        "nope_dim": NOPE_DIM,
        "rope_dim": ROPE_DIM,
        "o_lora_rank": r,
        "einsum_recipe": EINSUM_RECIPE,
        "tma_aligned_scales": False,
    }
    return args, kwargs


def key_shape(binding):
    """``attn_out`` shape, the first tensor in the benchmark's shape detail."""
    return [
        binding["num_tokens"],
        binding["n_groups"] * binding["heads_per_group"],
        HEAD_DIM,
    ]


def config(binding, dtype):
    t = binding["num_tokens"]
    g = binding["n_groups"]
    h = binding["heads_per_group"]
    r = binding["o_lora_rank"]
    k = h * HEAD_DIM
    return {
        "inputs": {
            "attn_out": {"shape": [t, g * h, HEAD_DIM], "dtype": str(dtype)},
            "positions": {"shape": [t], "dtype": "torch.int64"},
            "cos_sin_cache": {
                "shape": [MAX_POSITION, ROPE_DIM],
                "dtype": "torch.float32",
            },
            "wo_a.weight": {"shape": [g, r, k], "dtype": "torch.float8_e4m3fn"},
            "wo_a.weight_scale_inv": {
                "shape": [g, r // QUANT_BLOCK, k // QUANT_BLOCK],
                "dtype": "torch.float32",
            },
            "n_groups": {"scalar": g},
            "heads_per_group": {"scalar": h},
            "nope_dim": {"scalar": NOPE_DIM},
            "rope_dim": {"scalar": ROPE_DIM},
            "o_lora_rank": {"scalar": r},
        },
        "outputs": {
            "z": {"shape": [t, g * r], "dtype": str(dtype), "note": "wo_b 为恒等回调"},
        },
        "dims": {
            "num_tokens": t,
            "n_groups": g,
            "heads_per_group": h,
            "o_lora_rank": r,
            "einsum_recipe": list(EINSUM_RECIPE),
        },
    }
