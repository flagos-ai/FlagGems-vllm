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

"""fused_marlin_moe W8A16 INT8 baseline（方案 B）。

native：vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe(...)
    内部两段 ops.moe_wna16_marlin_gemm（CUDA Marlin）。W8A16：权重 GPTQ uint8b128
    per-group-128，激活 16-bit（不量化）。quant_type_id=uint8b128.id。
    src: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L235

输入构造复刻 FlagGems-vllm/benchmark/test_fused_marlin_moe_w8a16_int8.py：
    与 int4 版同构，仅量化类型换成 uint8b128（marlin_quantize 逐专家打包）。
    shape 来源：benchmark set_shapes（4 个 MoE 架构 Mixtral / DeepSeek-V3 /
    Qwen3.5-397B-A17B / DeepSeek-V4-Flash × token 档 1/16/64/256/1024/4096/16384，
    共 28 组，与 benchmark 完全一致）。

离线（无 vllm 的 stub）无法做 Marlin 打包，build_inputs 走占位回退；native() 此时
返回 None，采集器会跳过。
"""

import importlib

import torch

OP_NAME = "fused_marlin_moe_w8a16_int8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_MODULE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_SYMBOL = "fused_marlin_moe"
_GROUP_SIZE = 128

# benchmark set_shapes：4 个 MoE 架构 (experts,hidden,intermediate,topk) × 7 个 token
# 档 (1/16/64/256/1024/4096/16384) 的笛卡尔积，共 28 组。
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
    """取 marlin_quantize + uint8b128 quant type；离线拿不到返回 None。"""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            marlin_quantize,
        )
        from vllm.scalar_type import scalar_types
    except ImportError:
        return None
    return marlin_quantize, scalar_types.uint8b128


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


def _marlin_quantize_per_expert(w_fp, marlin_quantize, quant_type):
    """逐专家 Marlin 打包（复刻 benchmark _marlin_quantize_per_expert_int8）。"""
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        _, qw, sc, _, _, _ = marlin_quantize(
            w_fp[e].T.contiguous(), quant_type, _GROUP_SIZE, act_order=False
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


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

    marlin_quantize, quant_type = resolved
    w1_fp = torch.randn(e, i * 2, h, dtype=dtype, device=device) / 10.0
    w2_fp = torch.randn(e, h, i, dtype=dtype, device=device) / 10.0
    w1_q, w1_s = _marlin_quantize_per_expert(w1_fp, marlin_quantize, quant_type)
    w2_q, w2_s = _marlin_quantize_per_expert(w2_fp, marlin_quantize, quant_type)

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
        "quant": {"scheme": "W8A16", "weight_type": "uint8b128",
                  "group_size": _GROUP_SIZE},
        "inputs": {
            "hidden_states": {"shape": [t, h], "dtype": str(dtype)},
            "w1": {"shape": [e, i * 2, h], "note": "Marlin packed uint8b128"},
            "w2": {"shape": [e, h, i], "note": "Marlin packed uint8b128"},
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "int64"},
        },
        "outputs": {"output": {"shape": [t, h], "dtype": str(dtype)}},
        "dims": {"num_tokens": t, "num_experts": e, "hidden": h,
                 "intermediate": i, "topk": k},
    }
