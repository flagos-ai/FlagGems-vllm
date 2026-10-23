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

"""moe_sum baseline（方案 B）。

native：vllm._custom_ops.moe_sum(input, output, topk_ids=None, expert_map=None)
    内部调用 torch.ops._moe_C.moe_sum；把 [num_tokens, topk, hidden] 沿 topk 维
    求和写入 output[num_tokens, hidden]。原地写 output，返回 None。
    src: vllm/_custom_ops.py#L2309

输入构造复刻 FlagGems-vllm/benchmark/test_moe_sum.py 的 _input_fn：
    input  = randn(num_tokens, topk, hidden)
    output = empty(num_tokens, hidden)
benchmark 用 GenericBenchmarkExcluse1D，op_name "moe_sum" 不在 core_shapes.yaml，
按 MRO 命中类名键「GenericBenchmarkExcluse1D」（core_shapes.yaml#L234）：
    [64,64] [1024,1024] [4096,4096] [64,512,512] [1024,1024,1024]
_input_fn 把 2D (M,N) 补成 (M,1,N)，3D (num_tokens,topk,hidden) 原样用。
这里按该规则逐一映射为 (num_tokens, topk, hidden)。
"""

import importlib

import torch

OP_NAME = "moe_sum"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # 原地写 output

# core_shapes.yaml「GenericBenchmarkExcluse1D」键，经 _input_fn 规则映射：
#   2D (M,N) -> (M, 1, N)；3D 原样 (num_tokens, topk, hidden)。
_SHAPES = [
    (64, 1, 64),          # [64, 64]
    (1024, 1, 1024),      # [1024, 1024]
    (4096, 1, 4096),      # [4096, 4096]
    (64, 512, 512),       # [64, 512, 512]
    (1024, 1024, 1024),   # [1024, 1024, 1024]
]


def native():
    """解析 vllm._custom_ops.moe_sum；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "moe_sum", None)
    return op if callable(op) else None


def grid():
    return [{"num_tokens": t, "topk": k, "hidden": h} for (t, k, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, k, h = binding["num_tokens"], binding["topk"], binding["hidden"]
    inp = torch.randn(t, k, h, dtype=dtype, device=device)
    out = torch.empty(t, h, dtype=dtype, device=device)
    # topk_ids/expert_map 缺省 None，走纯求和路径。
    return (inp, out), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["topk"], binding["hidden"]]


def config(binding, dtype):
    t, k, h = binding["num_tokens"], binding["topk"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [t, k, h], "dtype": dt},
            "output": {"shape": [t, h], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "output": {"shape": [t, h], "dtype": dt},
        },
        "dims": {"num_tokens": t, "topk": k, "hidden": h},
    }
