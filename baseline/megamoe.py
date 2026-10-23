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

"""megamoe baseline（方案 B）。

native：vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe.prepare_megamoe_inputs(
            hidden_states, topk_weights, topk_ids, x_fp8, x_sf,
            topk_idx_out, topk_weights_out,
            is_padding=None, shared_x_sf=None, shared_block_m=None) -> None
    DeepSeek-V4 MegaMoE 输入预处理：把 hidden_states 逐 token 做 UE8M0 block 量化
    写入 x_fp8/x_sf（BLOCK_K=128, GROUP_K=32，每 int32 打包 4 个 group 的指数），
    并把 topk_ids/topk_weights 拷入 topk_idx_out(int64)/topk_weights_out。
    内部通过 _prepare_megamoe_inputs_kernel[grid](...) 启动 triton kernel，
    但 prepare_megamoe_inputs 本身是普通 Python 可调用入口，可直接 native 解析。
    grid = (num_tokens, cdiv(hidden_size, 128))；原地写各输出缓冲，返回 None。
    src: vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py#L148

CSV 备注 megamoe 走注册表（library=flag_gems, status=planned），vLLM-nvidia 侧对应
Triton 预处理即 prepare_megamoe_inputs（MoE 主计算另走 flashinfer MoEEpMegaLayer，
不在本基准范围）。这里基准化可直接调用的 prepare_megamoe_inputs。

无直接对应 benchmark（benchmark 目录仅有相关但不同的 stage_deepseek_v4_mega_moe_inputs）。
**shape 为 vllm 源码推断，非 FlagGems-vllm 基准**。
推断依据（见源码约束与调用点 vllm/models/deepseek_v4/nvidia/model.py#L705）：
    hidden_states     : [num_tokens, hidden_size] bf16    (hidden_size % 128 == 0)
    topk_weights      : [num_tokens, top_k]  fp32
    topk_ids          : [num_tokens, top_k]  int32
    x_fp8   (out)     : [num_tokens, hidden_size] float8_e4m3fn
    x_sf    (out)     : [num_tokens, hidden_size//128] int32  (packed UE8M0)
    topk_idx_out (out): [num_tokens, top_k] int64
    topk_weights_out  : [num_tokens, top_k] fp32
    is_padding/shared_x_sf/shared_block_m 取默认 None（不启用 shared-expert 路径）。
DeepSeek-V4 典型档位：hidden_size=7168, top_k=8。
"""

import importlib

import torch

OP_NAME = "megamoe"
DTYPES = [torch.bfloat16]  # hidden_states dtype
IS_INPLACE = True  # 原地写 x_fp8 / x_sf / topk_idx_out / topk_weights_out，返回 None

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_BLOCK_K = 128  # 源码 block_k

# (num_tokens, hidden_size, top_k)；hidden_size 须为 128 的倍数。
_SHAPES = [
    (1, 7168, 8),
    (16, 7168, 8),
    (64, 7168, 8),
    (128, 7168, 8),
    (256, 7168, 8),
]


def native():
    """解析 prepare_megamoe_inputs；解析不到返回 None。

    该符号是普通 Python 函数（内部再启动 triton kernel），可直接返回。
    """
    try:
        mod = importlib.import_module(
            "vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe"
        )
    except ImportError:
        return None
    op = getattr(mod, "prepare_megamoe_inputs", None)
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "hidden_size": h, "top_k": k}
        for (t, h, k) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t = binding["num_tokens"]
    h = binding["hidden_size"]
    k = binding["top_k"]
    sf_k = h // _BLOCK_K

    hidden_states = torch.randn(t, h, dtype=dtype, device=device)
    topk_weights = torch.rand(t, k, dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, 256, (t, k), dtype=torch.int32, device=device)

    x_fp8 = torch.empty(t, h, dtype=_FP8_DTYPE, device=device)
    x_sf = torch.empty(t, sf_k, dtype=torch.int32, device=device)
    topk_idx_out = torch.empty(t, k, dtype=torch.int64, device=device)
    topk_weights_out = torch.empty(t, k, dtype=torch.float32, device=device)

    args = (
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )
    # is_padding / shared_x_sf / shared_block_m 取默认 None。
    return args, {}


def key_shape(binding):
    return [binding["num_tokens"], binding["hidden_size"], binding["top_k"]]


def config(binding, dtype):
    t = binding["num_tokens"]
    h = binding["hidden_size"]
    k = binding["top_k"]
    sf_k = h // _BLOCK_K
    dt = str(dtype)
    fp8 = str(_FP8_DTYPE)
    return {
        "inputs": {
            "hidden_states": {"shape": [t, h], "dtype": dt},
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "torch.int32"},
            "x_fp8": {"shape": [t, h], "dtype": fp8, "note": "原地写（量化输出）"},
            "x_sf": {"shape": [t, sf_k], "dtype": "torch.int32",
                     "note": "原地写（packed UE8M0 scale）"},
            "topk_idx_out": {"shape": [t, k], "dtype": "torch.int64",
                             "note": "原地写"},
            "topk_weights_out": {"shape": [t, k], "dtype": "torch.float32",
                                 "note": "原地写"},
        },
        "outputs": {
            "x_fp8": {"shape": [t, h], "dtype": fp8},
            "x_sf": {"shape": [t, sf_k], "dtype": "torch.int32"},
            "topk_idx_out": {"shape": [t, k], "dtype": "torch.int64"},
            "topk_weights_out": {"shape": [t, k], "dtype": "torch.float32"},
        },
        "dims": {"num_tokens": t, "hidden_size": h, "top_k": k,
                 "block_k": _BLOCK_K},
        "note": "shape 为 vllm 源码推断，非 FlagGems-vllm 基准",
    }
