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

"""fused_deepseek_v4_qnorm_rope_kv_rope_insert baseline（DSV4 bf16 全精度 cache 融合插入）。

融合算子：Q 侧逐 head RMSNorm(无权重) + GPT-J RoPE（bf16 原地重写 q）；
KV 侧 GPT-J RoPE + 写入 [num_blocks, block_size, 512] 的连续 bf16 paged cache。

存疑点（命名映射，务必核对）：
  vllm 源码中 **没有** 名为 fused_deepseek_v4_qnorm_rope_kv_rope_insert 的符号，
  也 **没有** 对应 benchmark（FlagGems-vllm/benchmark 下仅有 quant 版）。
  非 quant / bf16 全 cache 变体在源码里叫
  fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert，本模块 native 即解析
  该符号。若上游后续新增精确同名符号，应改 _NATIVE_SYM。

native：torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
        q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, eps,
        cache_block_size) -> ()。q 原地写、k_cache 原地写。
    C++ 声明见 csrc/libtorch_stable/ops.h#L272；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L440。模型调用点见
    vllm/models/deepseek_v4/attention.py#L747、nvidia/dspark.py#L273。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

shape 为 **vllm 源码推断，非 FlagGems-vllm 基准**：无对应 benchmark，故沿用
quant 版 benchmark 的维度约定（HEAD_DIM=512, block_size=64, eps=1e-6），并按 bf16
全 cache 变体的调用点张量维度推断：
    q            = randn(num_tokens, num_heads, HEAD_DIM) bf16          （原地重写）
    kv           = randn(num_tokens, HEAD_DIM) bf16
    positions    = arange(num_tokens) int64
    cos_sin_cache= randn(max(max_pos,num_tokens), ROPE_DIM) fp32        （占位）
    slot_mapping = arange(num_tokens) int64
    k_cache      = zeros(num_blocks, block_size, HEAD_DIM) bf16
                   （attention.py 断言 swa_kv_cache.shape[1:]==(block_size, head_dim)）
    num_blocks   = (num_tokens + block_size - 1)//block_size + 1
    num_tokens ∈ {1,4,17,64,1024,2048,8192}，num_heads ∈ {64,128}（沿用 quant 网格中小档）。
"""

import importlib

import torch

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # q 原地重写、k_cache 原地写

# 源码无同名符号，映射到 bf16 全 cache 变体：
_NATIVE_SYM = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert"

_HEAD_DIM = 512
_ROPE_DIM = 64
_BLOCK_SIZE = 64
_MAX_POS = 4096
_EPS = 1e-6

# 源码推断网格（非 benchmark）：(num_tokens, num_heads)
_SHAPES = [
    (1, 64), (1, 128),
    (4, 64), (4, 128),
    (17, 64), (17, 128),
    (64, 64), (64, 128),
    (1024, 64), (1024, 128),
    (2048, 64), (2048, 128),
    (8192, 64), (8192, 128),
]


def native():
    """解析 torch.ops._C.<bf16 全 cache 变体>；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), _NATIVE_SYM, None)
    return op if callable(op) else None


def grid():
    return [{"num_tokens": t, "num_heads": h} for (t, h) in _SHAPES]


def _num_blocks(num_tokens):
    return (num_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE + 1


def build_inputs(binding, dtype, device):
    t, h = binding["num_tokens"], binding["num_heads"]
    max_pos = max(_MAX_POS, t)
    q = torch.randn(t, h, _HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(t, _HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(t, dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(max_pos, _ROPE_DIM, dtype=torch.float32, device=device)
    nb = _num_blocks(t)
    slot_mapping = torch.arange(t, dtype=torch.int64, device=device)
    # bf16 plain-row cache：[num_blocks, block_size, head_dim]
    k_cache = torch.zeros(nb, _BLOCK_SIZE, _HEAD_DIM, dtype=dtype, device=device)
    return (q, kv, k_cache, slot_mapping, positions, cos_sin_cache,
            _EPS, _BLOCK_SIZE), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_heads"], _HEAD_DIM]


def config(binding, dtype):
    t, h = binding["num_tokens"], binding["num_heads"]
    max_pos = max(_MAX_POS, t)
    nb = _num_blocks(t)
    return {
        "inputs": {
            "q": {"shape": [t, h, _HEAD_DIM], "dtype": repr(dtype),
                  "note": "bf16 原地重写"},
            "kv": {"shape": [t, _HEAD_DIM], "dtype": repr(dtype)},
            "k_cache": {"shape": [nb, _BLOCK_SIZE, _HEAD_DIM], "dtype": repr(dtype),
                        "note": "连续 bf16 paged cache，原地写"},
            "slot_mapping": {"shape": [t], "dtype": "torch.int64"},
            "position_ids": {"shape": [t], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [max_pos, _ROPE_DIM], "dtype": "torch.float32",
                              "note": "randn 占位（真实为 cos/sin 拼接）"},
            "eps": {"scalar": _EPS},
            "cache_block_size": {"scalar": _BLOCK_SIZE},
        },
        "outputs": {
            "q": {"shape": [t, h, _HEAD_DIM], "dtype": repr(dtype),
                  "note": "原地重写"},
            "k_cache": {"shape": [nb, _BLOCK_SIZE, _HEAD_DIM], "dtype": repr(dtype)},
        },
        "dims": {"num_tokens": t, "num_heads": h, "head_dim": _HEAD_DIM,
                 "rope_dim": _ROPE_DIM, "block_size": _BLOCK_SIZE, "num_blocks": nb},
    }
