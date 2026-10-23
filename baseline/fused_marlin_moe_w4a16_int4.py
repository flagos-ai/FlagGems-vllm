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

"""fused_marlin_moe W4A16 INT4 baseline（方案 B）。

native：vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe(
            hidden_states, w1, w2, bias1, bias2, w1_scale, w2_scale,
            topk_weights, topk_ids, quant_type_id, ...)
    内部两段 ops.moe_wna16_marlin_gemm（CUDA Marlin）。W4A16：权重 GPTQ uint4b8
    per-group-128，激活 16-bit（input_dtype=None，不量化激活）。quant_type_id 选择
    量化类型：uint4b8。返回 output 张量。
    src: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L235
    注意：CSV 旧引用的 vllm.model_executor.layers.fused_moe.fused_marlin_moe 路径已迁移，
    实际定义在 experts/marlin_moe.py。这里从新路径解析。

输入构造复刻 FlagGems-vllm/benchmark/test_fused_marlin_moe_w4a16_int4.py：
    hidden_states = randn(num_tokens, hidden_size)  bf16
    w1_fp = randn(E, intermediate*2, hidden)/10 → 逐专家 marlin_quantize(w1_fp[e].T, uint4b8, 128)
    w2_fp = randn(E, hidden, intermediate)/10   → 逐专家 marlin_quantize(w2_fp[e].T, uint4b8, 128)
    topk_weights/topk_ids 由 softmax+topk 得，topk_weights 必须 fp32。
    shape 来源：benchmark set_shapes（4 个生产 MoE 架构 × decode token 档
    1/4/8/16/32/64/128/256，共 32 组，与 benchmark 完全一致）。

离线（无 vllm 的 stub）无法做 Marlin 打包，build_inputs 走占位回退：返回未量化的
FP 权重占位张量，仅用于契约冒烟；native() 此时返回 None，采集器会跳过，不会真正喂给 kernel。
"""

import importlib

import torch

OP_NAME = "fused_marlin_moe_w4a16_int4"
DTYPES = [torch.bfloat16]  # benchmark 只跑 bf16；marlin_moe assert dtype∈{fp16,bf16}
IS_INPLACE = False

_MODULE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_SYMBOL = "fused_marlin_moe"
_GROUP_SIZE = 128

# benchmark set_shapes：(num_tokens, num_experts, hidden, intermediate, topk)
# 与 benchmark 完全一致：4 个生产 MoE 架构 × token 档 {1,4,8,16,32,64,128,256}，共 32 组。
# 注意：本算子逐专家用纯 Python marlin_quantize（marlin_utils_test）打包，耗时随
# num_experts 线性增长且与 num_tokens 无关（同一 (e,h,i) 的多个 token 档会把相同权重
# 重复量化多次）。大 E（256/512）全档在单算子子进程里可能逼近 --op-timeout（缺省
# 1800s）；实跑若超时，用 --op-timeout 调大或 --blacklist 跳过最大档单独处理。
_SHAPES = [
    # Mixtral-8x7B
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (8, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (32, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    # DeepSeek-V3 (TP=8 shard)
    (1, 256, 7168, 2048, 8),
    (4, 256, 7168, 2048, 8),
    (8, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (32, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
    (128, 256, 7168, 2048, 8),
    (256, 256, 7168, 2048, 8),
    # Qwen3-5-397B-A17B
    (1, 512, 4096, 1024, 10),
    (4, 512, 4096, 1024, 10),
    (8, 512, 4096, 1024, 10),
    (16, 512, 4096, 1024, 10),
    (32, 512, 4096, 1024, 10),
    (64, 512, 4096, 1024, 10),
    (128, 512, 4096, 1024, 10),
    (256, 512, 4096, 1024, 10),
    # DeepSeek-V4-Flash
    (1, 256, 4096, 2048, 6),
    (4, 256, 4096, 2048, 6),
    (8, 256, 4096, 2048, 6),
    (16, 256, 4096, 2048, 6),
    (32, 256, 4096, 2048, 6),
    (64, 256, 4096, 2048, 6),
    (128, 256, 4096, 2048, 6),
    (256, 256, 4096, 2048, 6),
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
    """取 marlin_quantize + uint4b8 quant type；离线拿不到返回 None。"""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            marlin_quantize,
        )
        from vllm.scalar_type import scalar_types
    except ImportError:
        return None
    return marlin_quantize, scalar_types.uint4b8


def _make_routing(t, e, k, device):
    """构造 (topk_weights fp32, topk_ids)。真 torch 走 softmax+topk（复刻
    benchmark 路由分布）；stub 无 topk/softmax 时直接造占位路由张量。"""
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
    """逐专家 Marlin 打包（复刻 benchmark _marlin_quantize_per_expert）。

    marlin_quantize 期望 (in_dim, out_dim)，故传 w_fp[e].T。"""
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

    # routing：topk_weights 必须 fp32（vllm assert）。
    topk_weights, topk_ids = _make_routing(t, e, k, device)

    resolved = _resolve_quant()
    if resolved is None:
        # 离线占位：未量化 FP 权重，仅供契约冒烟（native() 为 None，不会真跑）。
        w1_q = torch.randn(e, i * 2, h, dtype=dtype, device=device)
        w2_q = torch.randn(e, h, i, dtype=dtype, device=device)
        w1_s = torch.randn(e, i * 2, h // _GROUP_SIZE, dtype=dtype, device=device)
        w2_s = torch.randn(e, h, i // _GROUP_SIZE, dtype=dtype, device=device)
        quant_type_id = 0
        kwargs = dict(
            hidden_states=hidden_states, w1=w1_q, w2=w2_q, bias1=None, bias2=None,
            w1_scale=w1_s, w2_scale=w2_s, topk_weights=topk_weights,
            topk_ids=topk_ids, quant_type_id=quant_type_id,
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
        "quant": {"scheme": "W4A16", "weight_type": "uint4b8",
                  "group_size": _GROUP_SIZE},
        "inputs": {
            "hidden_states": {"shape": [t, h], "dtype": str(dtype)},
            "w1": {"shape": [e, i * 2, h], "note": "Marlin packed uint4b8"},
            "w2": {"shape": [e, h, i], "note": "Marlin packed uint4b8"},
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "int64"},
        },
        "outputs": {"output": {"shape": [t, h], "dtype": str(dtype)}},
        "dims": {"num_tokens": t, "num_experts": e, "hidden": h,
                 "intermediate": i, "topk": k},
    }
