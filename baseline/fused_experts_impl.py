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

"""fused_experts_impl baseline（方案 B）。

native：vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids, activation="silu", ...)
    未量化（W16A16）MoE 专家计算：per-expert GEMM + SiLU + 加权求和，内部调用一批
    NV 原生 Triton/CUDA moe kernel。返回 out_hidden_states（torch.empty_like，非原地）。
    src: vllm/model_executor/layers/fused_moe/fused_moe.py#L1656

输入构造复刻 FlagGems-vllm/benchmark/test_inplace_fused_experts.py 的 _fused_moe_input_fn：
    hidden_states = randn(num_tokens, hidden)
    w1 = randn(E, intermediate*2, hidden)
    w2 = randn(E, hidden, intermediate)
    gating = randn(num_tokens, E) fp32 → softmax+topk → topk_weights(.to(dtype)) / topk_ids
    shape 来源：benchmark set_shapes（Mixtral / DeepSeek-V3 生产档，token 1..512）。

注意：benchmark 的 vLLM wrapper 传了 inplace=True，但当前源码 fused_experts_impl 签名
无 inplace 形参（#L1656-1681），故这里按源码只传 activation="silu"，不传 inplace。
"""

import importlib

import torch

OP_NAME = "fused_experts_impl"
DTYPES = [torch.bfloat16, torch.float16]  # benchmark 两种；源码 assert 含 fp32/16/bf16
IS_INPLACE = False  # 源码返回 torch.empty_like(hidden_states)

_MODULE = "vllm.model_executor.layers.fused_moe.fused_moe"
_SYMBOL = "fused_experts_impl"

# benchmark set_shapes：(num_tokens, num_experts, hidden, intermediate, topk)
_SHAPES = [
    # Mixtral-like
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    (512, 8, 4096, 14336, 2),
    # DeepSeek-V3-like (TP=8 shard)
    (1, 256, 7168, 2048, 8),
    (4, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
    (128, 256, 7168, 2048, 8),
    (256, 256, 7168, 2048, 8),
]


def native():
    """解析 fused_moe.fused_experts_impl；解析不到返回 None。"""
    try:
        mod = importlib.import_module(_MODULE)
    except ImportError:
        return None
    op = getattr(mod, _SYMBOL, None)
    return op if callable(op) else None


def _make_routing(t, e, k, dtype, device):
    """(topk_weights, topk_ids)。真 torch 走 softmax+topk 并 .to(dtype)；stub 走占位。"""
    gating = torch.randn(t, e, dtype=torch.float32, device=device)
    if hasattr(torch, "topk") and hasattr(torch, "softmax"):
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating, dim=-1), k, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)
        return topk_weights, topk_ids
    topk_weights = torch.rand(t, k, dtype=dtype, device=device)
    topk_ids = torch.randint(0, e, (t, k), device=device)
    return topk_weights, topk_ids


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
    w1 = torch.randn(e, i * 2, h, dtype=dtype, device=device)
    w2 = torch.randn(e, h, i, dtype=dtype, device=device)
    topk_weights, topk_ids = _make_routing(t, e, k, dtype, device)

    args = (hidden_states, w1, w2, topk_weights, topk_ids)
    kwargs = {"activation": "silu"}
    return args, kwargs


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
        "inputs": {
            "hidden_states": {"shape": [t, h], "dtype": str(dtype)},
            "w1": {"shape": [e, i * 2, h], "dtype": str(dtype)},
            "w2": {"shape": [e, h, i], "dtype": str(dtype)},
            "topk_weights": {"shape": [t, k], "dtype": str(dtype)},
            "topk_ids": {"shape": [t, k], "dtype": "int64"},
            "activation": {"scalar": "silu"},
        },
        "outputs": {"output": {"shape": [t, h], "dtype": str(dtype)}},
        "dims": {"num_tokens": t, "num_experts": e, "hidden": h,
                 "intermediate": i, "topk": k},
    }
