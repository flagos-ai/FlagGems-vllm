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

"""deep_gemm_fp8_o_proj baseline, aligned with the FlagGems-vllm benchmark.

native: ``vllm.models.deepseek_v4.nvidia.ops.o_proj.deep_gemm_fp8_o_proj``
        (inverse RoPE + per-128 FP8 quant + block-wise FP8 einsum). ``wo_b``
        is the identity callable, exactly as in
        ``FlagGems-vllm/benchmark/test_deep_gemm_fp8_o_proj.py``, so the timed
        interval is quant + einsum only -- the same scope the FlagGems-vllm
        operator implements and the benchmark compares against.

Shapes mirror ``benchmark/core_shapes.yaml`` and the benchmark's
``set_more_shapes()``: ``(tokens, n_groups, group_dim, o_lora_rank)`` with
``group_dim = 4096``, ``o_lora_rank = 1024``, ``head_dim = 512``,
``nope_dim = 448``, ``rope_dim = 64``.

``key_shape`` returns the first benchmark tensor's shape,
``[tokens, n_groups * heads_per_group, head_dim]``, so the collected JSON key
matches ``str(first tensor shape)`` in ``lookup_base_record_ex`` directly.

Works with both vLLM 0.24 and 0.32: 0.32 keeps the tensor-input path (the
``QuantizedActivation`` fast path is only taken when the attention kernel has
already rotated and quantized), and its ``wo_a`` may expose ``weight_scale``
instead of ``weight_scale_inv`` -- the module below keeps the latter, which
0.32 still accepts as a fallback.
"""

import importlib

import torch
import torch.nn as nn

OP_NAME = "deep_gemm_fp8_o_proj"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 返回新的输出张量

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
GROUP_DIM = 4096
O_LORA_RANK = 1024
BLOCK = 128
MAX_POS = 65536
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

# FlagGems-vllm benchmark shapes: core level (tokens 1/16/128/4096) plus
# set_more_shapes() extras (4/64/512/2048/8192).
_TOKENS = [1, 4, 16, 64, 128, 512, 2048, 4096, 8192]
_GROUPS = [8, 16]

_CANDIDATES = [
    ("vllm.models.deepseek_v4.nvidia.ops.o_proj", "deep_gemm_fp8_o_proj"),
    ("vllm.models.deepseek_v4.nvidia.ops", "deep_gemm_fp8_o_proj"),
]


def _per_block_cast_to_fp8(x, block_n=BLOCK, block_k=BLOCK):
    """128x128 block-wise FP8 cast, identical to the benchmark helper."""
    m, n = x.shape
    x_view = x.view(m // block_n, block_n, n // block_k, block_k)
    amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    scale = amax / FP8_MAX
    return (x_view / scale).to(torch.float8_e4m3fn).view(m, n), scale.view(
        m // block_n, n // block_k
    )


def _make_cos_sin_cache(max_pos, device):
    """``[max_pos, rope_dim]`` fp32, laid out as ``cos || sin``."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


class _QuantWoA(nn.Module):
    """FP8 block-wise ``wo_a`` holder, mirroring vLLM's module layout."""

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.weight_scale_inv = scale


def _identity(flat):
    """The benchmark compares quant+einsum only; ``wo_b`` is not measured."""
    return flat


def _recipe():
    """Hopper: (1, 128, 128) with FP32 block scales; SM100+: (1, 1, 128)."""
    try:
        mod = importlib.import_module("vllm.models.deepseek_v4.nvidia.ops.o_proj")
        return mod.compute_fp8_einsum_recipe()
    except Exception:
        cap = torch.cuda.get_device_capability()
        return ((1, 128, 128), False) if cap[0] <= 9 else ((1, 1, 128), True)


def native():
    for module, symbol in _CANDIDATES:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        op = getattr(mod, symbol, None)
        if callable(op):
            return op
    return None


def grid():
    return [
        {"tokens": tokens, "n_groups": n_groups}
        for n_groups in _GROUPS
        for tokens in _TOKENS
    ]


def build_inputs(binding, dtype, device):
    tokens = binding["tokens"]
    n_groups = binding["n_groups"]
    heads_per_group = GROUP_DIM // HEAD_DIM

    o = torch.randn(
        (tokens, n_groups * heads_per_group, HEAD_DIM), dtype=dtype, device=device
    )
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int64, device=device)
    cos_sin_cache = _make_cos_sin_cache(MAX_POS, device)

    weight = torch.randn(
        (n_groups, O_LORA_RANK, GROUP_DIM), dtype=torch.bfloat16, device=device
    )
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(
        (n_groups, O_LORA_RANK // BLOCK, GROUP_DIM // BLOCK),
        dtype=torch.float32,
        device=device,
    )
    # One group at a time keeps the fp32 dequant scratch small.
    for group in range(n_groups):
        weight_fp8[group], weight_scale[group] = _per_block_cast_to_fp8(weight[group])

    wo_a = _QuantWoA(weight_fp8, weight_scale)
    recipe, tma_aligned = _recipe()
    args = (o, positions, cos_sin_cache, wo_a, _identity)
    kwargs = {
        "n_groups": n_groups,
        "heads_per_group": heads_per_group,
        "nope_dim": NOPE_DIM,
        "rope_dim": ROPE_DIM,
        "o_lora_rank": O_LORA_RANK,
        "einsum_recipe": recipe,
        "tma_aligned_scales": tma_aligned,
    }
    return args, kwargs


def key_shape(binding):
    """First benchmark tensor (``o``): ``[tokens, num_heads, head_dim]``."""
    heads_per_group = GROUP_DIM // HEAD_DIM
    return [binding["tokens"], binding["n_groups"] * heads_per_group, HEAD_DIM]


def config(binding, dtype):
    tokens = binding["tokens"]
    n_groups = binding["n_groups"]
    heads_per_group = GROUP_DIM // HEAD_DIM
    reduction = heads_per_group * HEAD_DIM
    dt = str(dtype)
    return {
        "inputs": {
            "o": {
                "shape": [tokens, n_groups * heads_per_group, HEAD_DIM],
                "dtype": dt,
                "fill": "randn",
            },
            "positions": {"shape": [tokens], "dtype": "torch.int64"},
            "cos_sin_cache": {
                "shape": [MAX_POS, ROPE_DIM],
                "dtype": "torch.float32",
            },
            "wo_a.weight": {
                "shape": [n_groups, O_LORA_RANK, reduction],
                "dtype": "torch.float8_e4m3fn",
            },
            "wo_a.weight_scale_inv": {
                "shape": [
                    n_groups,
                    O_LORA_RANK // BLOCK,
                    reduction // BLOCK,
                ],
                "dtype": "torch.float32",
            },
            "n_groups": {"scalar": n_groups},
            "heads_per_group": {"scalar": heads_per_group},
            "nope_dim": {"scalar": NOPE_DIM},
            "rope_dim": {"scalar": ROPE_DIM},
            "o_lora_rank": {"scalar": O_LORA_RANK},
        },
        "outputs": {
            "out": {"shape": [tokens, n_groups * O_LORA_RANK], "dtype": dt},
        },
        "dims": {
            "tokens": tokens,
            "n_groups": n_groups,
            "group_dim": GROUP_DIM,
            "o_lora_rank": O_LORA_RANK,
        },
    }
