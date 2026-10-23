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

"""topk_softplus_sqrt baseline（方案 B）。

native：vllm._custom_ops.topk_hash_softplus_sqrt(
            topk_weights, topk_indices, token_expert_indices, gating_output,
            renormalize=False, routed_scaling_factor=1.0,
            e_score_correction_bias=None, input_tokens=None,
            hash_indices_table=None, is_padding=None) -> None
    内部调用 torch.ops._moe_C.topk_softplus_sqrt；原地写 topk_weights/topk_indices/
    token_expert_indices。src: vllm/_custom_ops.py#L2493

输入构造复刻 FlagGems-vllm/benchmark/test_topk_softplus_sqrt.py 的
TopkSoftplusSqrtBenchmark.get_input_iter（yield 7 个位置参数，对应到 native 的前 7 个）：
    gating_output = randn(num_tokens, num_experts)      遍历 dtype
    correction_bias = randn(num_experts,)               fp32
    topk_weights = empty(num_tokens, topk)              fp32（输出）
    topk_indices = empty(num_tokens, topk)              int32（输出）
    token_expert_indices = empty(num_tokens, topk)      int32（输出）
    renormalize=True, routed_scaling_factor=1.0
"""

import importlib

import torch

OP_NAME = "topk_softplus_sqrt"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 三个输出缓冲原地写

# benchmark TopkSoftplusSqrtBenchmark.set_shapes：(num_tokens, num_experts, topk)。
_SHAPES = [
    (1, 256, 6),
    (10, 256, 6),
    (16, 256, 6),
    (128, 256, 6),
    (512, 256, 6),
    (1024, 256, 6),
    (2048, 256, 6),
    (4096, 256, 6),
]

_RENORMALIZE = True
_ROUTED_SCALING_FACTOR = 1.0


def native():
    """解析 vllm._custom_ops.topk_hash_softplus_sqrt；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "topk_hash_softplus_sqrt", None)
    return op if callable(op) else None


def grid():
    return [{"num_tokens": t, "num_experts": e, "topk": k}
            for (t, e, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, e, k = binding["num_tokens"], binding["num_experts"], binding["topk"]
    gating_output = torch.randn(t, e, dtype=dtype, device=device)
    correction_bias = torch.randn(e, dtype=torch.float32, device=device)
    topk_weights = torch.empty(t, k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(t, k, dtype=torch.int32, device=device)
    token_expert_indices = torch.empty(t, k, dtype=torch.int32, device=device)
    args = (
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        _RENORMALIZE,
        _ROUTED_SCALING_FACTOR,
        correction_bias,
    )
    return args, {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_experts"], binding["topk"]]


def config(binding, dtype):
    t, e, k = binding["num_tokens"], binding["num_experts"], binding["topk"]
    dt = str(dtype)
    return {
        "inputs": {
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32",
                             "note": "原地写回（输出）"},
            "topk_indices": {"shape": [t, k], "dtype": "torch.int32",
                             "note": "原地写回（输出）"},
            "token_expert_indices": {"shape": [t, k], "dtype": "torch.int32",
                                     "note": "原地写回（输出）"},
            "gating_output": {"shape": [t, e], "dtype": dt},
            "renormalize": {"scalar": _RENORMALIZE},
            "routed_scaling_factor": {"scalar": _ROUTED_SCALING_FACTOR},
            "e_score_correction_bias": {"shape": [e], "dtype": "torch.float32"},
        },
        "outputs": {
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_indices": {"shape": [t, k], "dtype": "torch.int32"},
            "token_expert_indices": {"shape": [t, k], "dtype": "torch.int32"},
        },
        "dims": {"num_tokens": t, "num_experts": e, "topk": k},
    }
