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

"""fused_marlin_moe W8A16 FP8 baseline（方案 B）。

native：vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe(...)
    内部两段 ops.moe_wna16_marlin_gemm（CUDA Marlin）。W8A16 FP8：权重 float8_e4m3fn
    per-group scale，激活 16-bit（input_dtype=None）。quant_type_id=float8_e4m3fn.id
    （marlin_moe.py#L297-304 的 assert 列表含 scalar_types.float8_e4m3fn）。
    src: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L235

shape 为 vllm 源码推断 + 复用 int8 benchmark 的 MoE 架构档，非 FlagGems-vllm 专属基准：
    CSV 明确「未找到对应 benchmark，仅有 w8a16_int8 版本」。这里**完全沿用**
    test_fused_marlin_moe_w8a16_int8.py::set_shapes 的 28 组
    (num_tokens, num_experts, hidden, intermediate, topk) 生产档，量化换成 FP8。

FP8 Marlin 权重打包用 vllm marlin_utils_fp8.marlin_quant_fp8_torch（#L612）：逐专家
对 (out_dim, in_dim) 权重量化 → pack_fp8_to_int32 + gptq_marlin_repack + 缩放。
group_size=128（marlin_quant_fp8_torch 的 per-group scale）。

离线（无 vllm 的 stub）无法做 FP8 打包，build_inputs 走占位回退；native() 此时返回 None，
采集器会跳过。
"""

import importlib

import torch

OP_NAME = "fused_marlin_moe_w8a16_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_MODULE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_SYMBOL = "fused_marlin_moe"
_GROUP_SIZE = 128

# shape 源码推断：复用 w8a16_int8 benchmark 的生产 MoE 架构档（无 fp8 专属 benchmark）。
# 与 test_fused_marlin_moe_w8a16_int8.py::set_shapes 完全对齐：
# 4 个 MoE 架构 (experts,hidden,intermediate,topk) × 7 个 token 档，共 28 组。
# (num_tokens, num_experts, hidden, intermediate, topk)
_SHAPES = [
    (tokens, experts, hidden, intermediate, topk)
    for (experts, hidden, intermediate, topk) in (
        (8, 4096, 14336, 2),    # Mixtral-8x7B
        (256, 7168, 2048, 8),   # DeepSeek-V3 (TP=8)
        (512, 4096, 1024, 10),  # Qwen3.5-397B-A17B
        (256, 4096, 2048, 6),   # DeepSeek-V4-Flash
    )
    for tokens in (1, 16, 64, 256, 1024, 4096, 16384)
]


def native():
    """解析 experts/marlin_moe.py 的 fused_marlin_moe；解析不到返回 None。"""
    try:
        mod = importlib.import_module(_MODULE)
    except ImportError:
        return None
    op = getattr(mod, _SYMBOL, None)
    return op if callable(op) else None


def _resolve_quant():
    """取 FP8 Marlin 打包工具 + float8_e4m3fn quant type；离线拿不到返回 None。"""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            marlin_quant_fp8_torch,
        )
        from vllm.scalar_type import scalar_types
    except ImportError:
        return None
    return marlin_quant_fp8_torch, scalar_types.float8_e4m3fn


def _make_routing(t, e, k, device):
    """构造 (topk_weights fp32, topk_ids)。真 torch 走 softmax+topk；stub 走占位。"""
    gating = torch.randn(t, e, dtype=torch.float32, device=device)
    if hasattr(torch, "topk") and hasattr(torch, "softmax"):
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating, dim=-1), k, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids
    topk_weights = torch.rand(t, k, dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, e, (t, k), device=device)
    return topk_weights, topk_ids


def _marlin_fp8_quantize_per_expert(w_fp, marlin_quant_fp8_torch):
    """逐专家 FP8 Marlin 打包。marlin_quant_fp8_torch 期望 (out_dim, in_dim)，
    返回 (weight_ref.T, marlin_qweight, marlin_scales)。"""
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        _, qw, sc = marlin_quant_fp8_torch(w_fp[e], _GROUP_SIZE)
        qweight_l.append(qw)
        scales_l.append(sc)
    return (torch.stack(qweight_l, dim=0).contiguous(),
            torch.stack(scales_l, dim=0).contiguous())


def grid():
    return [
        {"num_tokens": t, "num_experts": e, "hidden": h,
         "intermediate": i, "topk": k}
        for (t, e, h, i, k) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t = binding["num_tokens"]
    e = binding["num_experts"]
    h = binding["hidden"]
    i = binding["intermediate"]
    k = binding["topk"]

    hidden_states = torch.randn(t, h, dtype=dtype, device=device)
    topk_weights, topk_ids = _make_routing(t, e, k, device)

    resolved = _resolve_quant()
    if resolved is None:
        # 离线占位：未量化 FP 权重，仅供契约冒烟。
        w1_q = torch.randn(e, i * 2, h, dtype=dtype, device=device)
        w2_q = torch.randn(e, h, i, dtype=dtype, device=device)
        w1_s = torch.randn(e, i * 2, h // _GROUP_SIZE, dtype=dtype, device=device)
        w2_s = torch.randn(e, h, i // _GROUP_SIZE, dtype=dtype, device=device)
        kwargs = dict(
            hidden_states=hidden_states, w1=w1_q, w2=w2_q, bias1=None, bias2=None,
            w1_scale=w1_s, w2_scale=w2_s, topk_weights=topk_weights,
            topk_ids=topk_ids, quant_type_id=0,
        )
        return (), kwargs

    marlin_quant_fp8_torch, quant_type = resolved
    w1_fp = torch.randn(e, i * 2, h, dtype=dtype, device=device) / 10.0
    w2_fp = torch.randn(e, h, i, dtype=dtype, device=device) / 10.0
    w1_q, w1_s = _marlin_fp8_quantize_per_expert(w1_fp, marlin_quant_fp8_torch)
    w2_q, w2_s = _marlin_fp8_quantize_per_expert(w2_fp, marlin_quant_fp8_torch)

    kwargs = dict(
        hidden_states=hidden_states, w1=w1_q, w2=w2_q, bias1=None, bias2=None,
        w1_scale=w1_s, w2_scale=w2_s, topk_weights=topk_weights,
        topk_ids=topk_ids, quant_type_id=quant_type.id,
    )
    return (), kwargs


def key_shape(binding):
    return (f"t{binding['num_tokens']}_e{binding['num_experts']}"
            f"_h{binding['hidden']}_i{binding['intermediate']}"
            f"_topk{binding['topk']}")


def config(binding, dtype):
    t = binding["num_tokens"]
    e = binding["num_experts"]
    h = binding["hidden"]
    i = binding["intermediate"]
    k = binding["topk"]
    return {
        "quant": {"scheme": "W8A16", "weight_type": "float8_e4m3fn",
                  "group_size": _GROUP_SIZE},
        "shape_source": "vllm 源码推断（复用 int8 benchmark 生产档），非 FlagGems-vllm 基准",
        "inputs": {
            "hidden_states": {"shape": [t, h], "dtype": str(dtype)},
            "w1": {"shape": [e, i * 2, h], "note": "Marlin packed fp8_e4m3fn"},
            "w2": {"shape": [e, h, i], "note": "Marlin packed fp8_e4m3fn"},
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "int64"},
        },
        "outputs": {"output": {"shape": [t, h], "dtype": str(dtype)}},
        "dims": {"num_tokens": t, "num_experts": e, "hidden": h,
                 "intermediate": i, "topk": k},
    }
