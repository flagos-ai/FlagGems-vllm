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

"""fused_marlin_moe W4A16 MXFP4 baseline（方案 B）。

native：vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe(...)
    内部两段 ops.moe_wna16_marlin_gemm（CUDA Marlin）。MXFP4：权重 FP4 E2M1 +
    per-32 E8M0 scale，激活 16-bit。quant_type_id=float4_e2m1f.id。
    src: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L235

输入构造复刻 FlagGems-vllm/benchmark/test_fused_marlin_moe_w4a16_mxfp4.py：
    RTN MXFP4 量化 → nibble 打包 → vllm_ops.gptq_marlin_repack + marlin_permute_scales
    + mxfp4_marlin_process_scales → float8_e8m0fnu scale。group_size=32。
    shape 来源：benchmark set_shapes（四个生产 MoE 架构 × token 档 1/16/64/256）。

离线（无 vllm 的 stub）无法做 FP4 打包/repack，build_inputs 走占位回退；native()
此时返回 None，采集器会跳过。
"""

import importlib

import torch

OP_NAME = "fused_marlin_moe_w4a16_mxfp4"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_MODULE = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_SYMBOL = "fused_marlin_moe"
_MXFP4_GROUP_SIZE = 32
_E2M1_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_MAX = 6.0

# benchmark set_shapes：(num_tokens, num_experts, hidden, intermediate, topk)
_SHAPES = [
    # Mixtral-8x7B
    (1, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    # DeepSeek-V3 (TP=8 shard)
    (1, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
    (256, 256, 7168, 2048, 8),
    # Qwen3-5-397B-A17B
    (1, 512, 4096, 1024, 10),
    (16, 512, 4096, 1024, 10),
    (64, 512, 4096, 1024, 10),
    (256, 512, 4096, 1024, 10),
    # DeepSeek-V4-Flash
    (1, 256, 4096, 2048, 6),
    (16, 256, 4096, 2048, 6),
    (64, 256, 4096, 2048, 6),
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
    """取 MXFP4 打包需要的 vllm 工具；离线拿不到返回 None。"""
    try:
        import vllm._custom_ops as vllm_ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            mxfp4_marlin_process_scales,
        )
        from vllm.scalar_type import scalar_types
    except ImportError:
        return None
    return (vllm_ops, marlin_permute_scales, mxfp4_marlin_process_scales,
            scalar_types.float4_e2m1f)


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


def _quantize_mxfp4_2d(w_2d, group_size):
    """RTN MXFP4：返回 nibble (uint8 [0,15]) + E8M0 scale（复刻 benchmark）。"""
    out_dim, in_dim = w_2d.shape
    ng = in_dim // group_size
    device = w_2d.device
    wg = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    exp = torch.ceil(
        torch.log2((amax / _E2M1_MAX).clamp(min=1e-30))
    ).clamp(-127, 127)
    scale = torch.exp2(exp)
    e8m0_byte = (exp + 127.0).to(torch.uint8)
    wn = wg / scale
    sign = wn < 0
    a = wn.abs().clamp(max=_E2M1_MAX)
    mag = torch.bucketize(a, torch.tensor(_E2M1_MID, device=device))
    nibbles = (sign.to(torch.uint8) * 8 + mag.to(torch.uint8)).reshape(
        out_dim, in_dim
    )
    scale_e8m0 = e8m0_byte.squeeze(-1).view(torch.float8_e8m0fnu)
    return nibbles, scale_e8m0


def _marlin_mxfp4_quantize_per_expert(w_fp, vllm_ops, permute, process):
    """逐专家 vLLM Marlin MXFP4 打包（复刻 benchmark _marlin_mxfp4_quantize_per_expert）。"""
    E, out_dim, in_dim = w_fp.shape
    dtype = w_fp.dtype
    qweight_l, scales_l = [], []
    for e in range(E):
        nib, sc = _quantize_mxfp4_2d(w_fp[e], _MXFP4_GROUP_SIZE)
        packed = (nib[:, 1::2] * 16 + nib[:, ::2]).to(torch.uint8)
        perm = torch.empty(0, dtype=torch.int, device=w_fp.device)
        qw = vllm_ops.gptq_marlin_repack(
            packed.view(torch.int32).T.contiguous(), perm, in_dim, out_dim, 4, False
        )
        ms = permute(sc.T.to(dtype), in_dim, out_dim, _MXFP4_GROUP_SIZE, False)
        ms = process(ms, input_dtype=None).to(torch.float8_e8m0fnu)
        qweight_l.append(qw)
        scales_l.append(ms)
    return (torch.stack(qweight_l, 0).contiguous(),
            torch.stack(scales_l, 0).contiguous())


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
        # 离线占位：packed nibble 形状 (E, out, in//2) uint8 + E8M0 scale 占位。
        w1_q = torch.empty(e, i * 2, h // 2, dtype=torch.uint8, device=device)
        w2_q = torch.empty(e, h, i // 2, dtype=torch.uint8, device=device)
        w1_s = torch.empty(
            e, i * 2, h // _MXFP4_GROUP_SIZE, dtype=torch.uint8, device=device
        )
        w2_s = torch.empty(
            e, h, i // _MXFP4_GROUP_SIZE, dtype=torch.uint8, device=device
        )
        kwargs = dict(
            hidden_states=hidden_states, w1=w1_q, w2=w2_q, bias1=None, bias2=None,
            w1_scale=w1_s, w2_scale=w2_s, topk_weights=topk_weights,
            topk_ids=topk_ids, quant_type_id=0,
        )
        return (), kwargs

    vllm_ops, permute, process, quant_type = resolved
    w1_fp = torch.randn(e, i * 2, h, dtype=dtype, device=device) / 10.0
    w2_fp = torch.randn(e, h, i, dtype=dtype, device=device) / 10.0
    w1_q, w1_s = _marlin_mxfp4_quantize_per_expert(w1_fp, vllm_ops, permute, process)
    w2_q, w2_s = _marlin_mxfp4_quantize_per_expert(w2_fp, vllm_ops, permute, process)

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
        "quant": {"scheme": "W4A16", "weight_type": "float4_e2m1f",
                  "scale_type": "float8_e8m0fnu",
                  "group_size": _MXFP4_GROUP_SIZE},
        "inputs": {
            "hidden_states": {"shape": [t, h], "dtype": str(dtype)},
            "w1": {"shape": [e, i * 2, h], "note": "Marlin repacked FP4"},
            "w2": {"shape": [e, h, i], "note": "Marlin repacked FP4"},
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "int64"},
        },
        "outputs": {"output": {"shape": [t, h], "dtype": str(dtype)}},
        "dims": {"num_tokens": t, "num_experts": e, "hidden": h,
                 "intermediate": i, "topk": k},
    }
