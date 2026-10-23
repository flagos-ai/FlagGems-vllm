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

"""fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert baseline（DSV4 fp8_ds_mla 融合插入）。

融合算子：Q 侧逐 head RMSNorm(无权重) + GPT-J RoPE 并 zero-fill padding head；
KV 侧 GPT-J RoPE + UE8M0 FP8 量化 + 写 paged cache。返回 padded 后的 q 张量。

native：torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
        q_head_padded, eps, cache_block_size) -> Tensor。
    C++ 声明见 csrc/libtorch_stable/ops.h#L265；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L432。模型调用点见
    vllm/models/deepseek_v4/attention.py#L727、nvidia/dspark.py#L260。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/
test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py 的
FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input：
    HEAD_DIM=512, ROPE_DIM=64, HEAD_BYTES=584, block_size=64, max_pos=4096, eps=1e-6
    q            = randn(num_tokens, num_heads, HEAD_DIM) bf16
    kv           = randn(num_tokens, HEAD_DIM) bf16
    positions    = arange(num_tokens) int64
    cos_sin_cache= [max(max_pos,num_tokens), ROPE_DIM] fp32
    num_blocks   = (num_tokens + block_size - 1)//block_size + 1
    slot_mapping = arange(num_tokens_insert=num_tokens) int64
    k_cache      = zeros(num_blocks, block_size*HEAD_BYTES) uint8

存疑点（务必核对）：
  1. benchmark 的 get_input_iter 只 yield 8 个位置参数
     (q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size)，
     而当前 vllm 源码注册的 schema 是 9 参（eps 前多一个 int q_head_padded）。
     此处按 **当前源码 9 参签名** 构造，q_head_padded 取 num_heads（benchmark 的 q
     未做 head padding，模型侧则传 self.padded_heads）。benchmark 疑似基于旧签名。
  2. cos_sin_cache 用 randn 占位（真实值应为 make_cos_sin_cache 的 cos/sin 拼接），
     仅供 NCU 计时基准，不做数值正确性校验；离线冒烟 stub 也不支持 einsum/cos。

shape 来自 benchmark get_performance_test_params（与 benchmark 完全一致，全量 22 档）：
    num_tokens ∈ {1,4,17,64,1024,2048,8192,32768,65536,98304,131072}，num_heads ∈ {64,128}。
    注意：大档（num_tokens>=32768）单个 q 张量达数 GB（131072*128*512*2B≈17GB），
    真卡采集时可能 OOM；此处按「与 benchmark 一致」要求保留全部档位，实跑若遇显存
    不足可用 --ops 单独跑或临时裁剪。
"""

import importlib

import torch

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # k_cache 原地写；同时返回 padded q

_NATIVE_SYM = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"

_HEAD_DIM = 512
_ROPE_DIM = 64
_HEAD_BYTES = 584
_BLOCK_SIZE = 64
_MAX_POS = 4096
_EPS = 1e-6

# benchmark get_performance_test_params 全量网格：num_tokens × num_heads
# num_tokens ∈ {1,4,17,64,1024,2048,8192,32768,65536,98304,131072}, num_heads ∈ {64,128}
_NUM_TOKENS = [1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072]
_NUM_HEADS = [64, 128]
_SHAPES = [(t, h) for t in _NUM_TOKENS for h in _NUM_HEADS]


def native():
    """解析 torch.ops._C.<sym>；解析不到返回 None。

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
    # cos_sin_cache 真实值应为 make_cos_sin_cache 的 cos/sin 拼接，这里 randn 占位
    cos_sin_cache = torch.randn(max_pos, _ROPE_DIM, dtype=torch.float32, device=device)
    nb = _num_blocks(t)
    slot_mapping = torch.arange(t, dtype=torch.int64, device=device)
    k_cache = torch.zeros(nb, _BLOCK_SIZE * _HEAD_BYTES, dtype=torch.uint8, device=device)
    # 当前源码 9 参签名：q_head_padded 置 num_heads（benchmark q 未做 head padding）
    q_head_padded = h
    return (q, kv, k_cache, slot_mapping, positions, cos_sin_cache,
            q_head_padded, _EPS, _BLOCK_SIZE), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_heads"], _HEAD_DIM]


def config(binding, dtype):
    t, h = binding["num_tokens"], binding["num_heads"]
    max_pos = max(_MAX_POS, t)
    nb = _num_blocks(t)
    return {
        "inputs": {
            "q_in": {"shape": [t, h, _HEAD_DIM], "dtype": repr(dtype)},
            "kv": {"shape": [t, _HEAD_DIM], "dtype": repr(dtype)},
            "k_cache": {"shape": [nb, _BLOCK_SIZE * _HEAD_BYTES], "dtype": "torch.uint8",
                        "note": "fp8_ds_mla UE8M0 paged 布局，原地写"},
            "slot_mapping": {"shape": [t], "dtype": "torch.int64"},
            "position_ids": {"shape": [t], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [max_pos, _ROPE_DIM], "dtype": "torch.float32",
                              "note": "randn 占位（真实为 cos/sin 拼接）"},
            "q_head_padded": {"scalar": h,
                              "note": "当前源码 9 参签名新增；benchmark 旧签名无此参"},
            "eps": {"scalar": _EPS},
            "cache_block_size": {"scalar": _BLOCK_SIZE},
        },
        "outputs": {
            "q_out": {"shape": [t, h, _HEAD_DIM], "dtype": repr(dtype),
                      "note": "返回 padded 后的 q"},
            "k_cache": {"shape": [nb, _BLOCK_SIZE * _HEAD_BYTES], "dtype": "torch.uint8"},
        },
        "dims": {"num_tokens": t, "num_heads": h, "head_dim": _HEAD_DIM,
                 "rope_dim": _ROPE_DIM, "head_bytes": _HEAD_BYTES,
                 "block_size": _BLOCK_SIZE, "num_blocks": nb},
    }
