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

"""moe_align_block_size baseline（方案 B）。

native：vllm._custom_ops.moe_align_block_size(
            topk_ids, num_experts, block_size,
            sorted_token_ids, experts_ids, num_tokens_post_pad, expert_map=None)
    内部调用 torch.ops._moe_C.moe_align_block_size；原地写三个输出缓冲，返回 None。
    src: vllm/_custom_ops.py#L2318

输入构造复刻 FlagGems-vllm/benchmark/test_moe_align_block_size_triton.py 的
_input_fn（shape=(num_experts, block_size, num_tokens, topk)）：
    topk_ids = randint(0, num_experts, (num_tokens, topk))  int32
    max_num_tokens_padded = ceil(num_experts/32)*32
    sorted_ids = empty(max_num_tokens_padded)         int32
    expert_ids = empty(max_num_tokens_padded//block_size)  int32
    num_tokens_post_pad = empty(1)                    int32
shape 取 benchmark set_shapes 里的 (512,64,*,10) 系列。
"""

import importlib

import torch

OP_NAME = "moe_align_block_size"
DTYPES = [torch.int32]
IS_INPLACE = True  # 原地写 sorted_ids/expert_ids/num_tokens_post_pad

_WARP_SIZE = 32

# benchmark set_shapes：(num_experts, block_size, num_tokens, topk)。
_SHAPES = [
    (512, 64, 16384, 10),
    (512, 64, 6152, 10),
    (512, 64, 4727, 10),
    (512, 64, 1905, 10),
    (512, 64, 11575, 10),
    (512, 64, 1032, 10),
    (512, 64, 4201, 10),
    (512, 64, 2056, 10),
    (512, 64, 7561, 10),
    (512, 64, 4104, 10),
    (512, 64, 14281, 10),
]


def native():
    """解析 vllm._custom_ops.moe_align_block_size；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "moe_align_block_size", None)
    return op if callable(op) else None


def grid():
    return [{"num_experts": e, "block_size": b, "num_tokens": t, "topk": k}
            for (e, b, t, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    e = binding["num_experts"]
    b = binding["block_size"]
    t = binding["num_tokens"]
    k = binding["topk"]
    topk_ids = torch.randint(0, e, (t, k), dtype=torch.int32, device=device)
    max_num_tokens_padded = ((e + _WARP_SIZE - 1) // _WARP_SIZE) * _WARP_SIZE
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32,
                             device=device)
    max_num_m_blocks = max_num_tokens_padded // b
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32,
                             device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    args = (topk_ids, e, b, sorted_ids, expert_ids, num_tokens_post_pad)
    return args, {}


def key_shape(binding):
    return [binding["num_experts"], binding["block_size"],
            binding["num_tokens"], binding["topk"]]


def config(binding, dtype):
    e = binding["num_experts"]
    b = binding["block_size"]
    t = binding["num_tokens"]
    k = binding["topk"]
    max_num_tokens_padded = ((e + _WARP_SIZE - 1) // _WARP_SIZE) * _WARP_SIZE
    return {
        "inputs": {
            "topk_ids": {"shape": [t, k], "dtype": "torch.int32"},
            "num_experts": {"scalar": e},
            "block_size": {"scalar": b},
            "sorted_token_ids": {"shape": [max_num_tokens_padded],
                                 "dtype": "torch.int32", "note": "原地写回"},
            "experts_ids": {"shape": [max_num_tokens_padded // b],
                            "dtype": "torch.int32", "note": "原地写回"},
            "num_tokens_post_pad": {"shape": [1], "dtype": "torch.int32",
                                    "note": "原地写回"},
        },
        "outputs": {
            "sorted_token_ids": {"shape": [max_num_tokens_padded],
                                 "dtype": "torch.int32"},
            "experts_ids": {"shape": [max_num_tokens_padded // b],
                            "dtype": "torch.int32"},
            "num_tokens_post_pad": {"shape": [1], "dtype": "torch.int32"},
        },
        "dims": {"num_experts": e, "block_size": b,
                 "num_tokens": t, "topk": k},
    }
