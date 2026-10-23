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

"""reshape_and_cache_flash baseline（方案 B）。

native：vllm._custom_ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale) -> None（底层
        torch.ops._C_cache_ops.reshape_and_cache_flash，原地写 key_cache/value_cache）。
        签名见 vllm/_custom_ops.py#L2680（已回源核对）。

输入构造复刻 FlagGems-vllm/benchmark/test_reshape_and_cache_flash.py 的 input_kwargs，
shape=(num_tokens, num_heads, head_size, block_size, num_blocks)：
    num_slots = block_size * num_blocks
    slot_mapping = sample(range(num_slots), num_tokens)        long
    qkv = randn(num_tokens, 3, num_heads, head_size); _, key, value = unbind(dim=1)
    key_value_cache = empty(num_blocks, 2, block_size, num_heads, head_size).uniform_
    key_cache   = key_value_cache[:, 0].contiguous()
    value_cache = key_value_cache[:, 1].contiguous()
    k_scale = (key.amax()/64).float; v_scale = (value.amax()/64).float
    kwargs: kv_cache_dtype="auto", k_scale, v_scale
shape 取自 core_shapes.yaml「ReshapeAndCacheFlashBenchmark」。
"""

import importlib
import random

import torch

OP_NAME = "reshape_and_cache_flash"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # key_cache/value_cache 原地写回

# core_shapes.yaml「ReshapeAndCacheFlashBenchmark」：
# (num_tokens, num_heads, head_size, block_size, num_blocks)
_SHAPES = [
    (21, 4, 64, 8, 512),
    (21, 4, 64, 8, 1024),
    (21, 4, 80, 16, 10000),
    (42, 8, 64, 8, 1024),
    (42, 8, 80, 16, 10000),
]

_KV_CACHE_DTYPE = "auto"


def native():
    """解析 vllm._custom_ops.reshape_and_cache_flash；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "reshape_and_cache_flash", None)
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "num_heads": h, "head_size": d,
         "block_size": b, "num_blocks": n}
        for (t, h, d, b, n) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t = binding["num_tokens"]
    h = binding["num_heads"]
    d = binding["head_size"]
    b = binding["block_size"]
    n = binding["num_blocks"]

    num_slots = b * n
    slot_mapping_lst = random.sample(range(num_slots), t)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.int64, device=device)

    # benchmark 由 qkv=randn(t,3,h,d) unbind 出 key/value；两者等价于直接构造
    # (t,h,d)（采集只关心形状/dtype），这里直接构造以兼容离线冒烟 stub。
    key = torch.randn(t, h, d, dtype=dtype, device=device)
    value = torch.randn(t, h, d, dtype=dtype, device=device)

    scale = d ** -0.5
    key_value_cache = torch.empty(n, 2, b, h, d, dtype=dtype, device=device)
    key_value_cache.uniform_(-scale, scale)
    key_cache = key_value_cache[:, 0].contiguous()
    value_cache = key_value_cache[:, 1].contiguous()

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    args = (key, value, key_cache, value_cache, slot_mapping,
            _KV_CACHE_DTYPE, k_scale, v_scale)
    return args, {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_heads"], binding["head_size"],
            binding["block_size"], binding["num_blocks"]]


def config(binding, dtype):
    t = binding["num_tokens"]
    h = binding["num_heads"]
    d = binding["head_size"]
    b = binding["block_size"]
    n = binding["num_blocks"]
    dt = str(dtype)
    return {
        "inputs": {
            "key": {"shape": [t, h, d], "dtype": dt},
            "value": {"shape": [t, h, d], "dtype": dt},
            "key_cache": {"shape": [n, b, h, d], "dtype": dt, "note": "原地写回"},
            "value_cache": {"shape": [n, b, h, d], "dtype": dt, "note": "原地写回"},
            "slot_mapping": {"shape": [t], "dtype": "torch.int64"},
            "kv_cache_dtype": {"scalar": _KV_CACHE_DTYPE},
            "k_scale": {"shape": [], "dtype": "torch.float32"},
            "v_scale": {"shape": [], "dtype": "torch.float32"},
        },
        "outputs": {
            "key_cache": {"shape": [n, b, h, d], "dtype": dt},
            "value_cache": {"shape": [n, b, h, d], "dtype": dt},
        },
        "dims": {"num_tokens": t, "num_heads": h, "head_size": d,
                 "block_size": b, "num_blocks": n},
    }
