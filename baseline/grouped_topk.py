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

import importlib

import torch

OP_NAME = "grouped_topk"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False


def native():
    try:
        importlib.import_module("vllm._custom_ops")
        if hasattr(torch.ops._moe_C, "grouped_topk"):
            op = torch.ops._moe_C.grouped_topk
        else:
            op = None
    except (ImportError, AttributeError):
        return None
    return op if callable(op) else None


def grid():
    # num_tokens, num_experts, n_group, topk_group, topk, renormalize, scoring_func
    return [
        # Deepseek-3.2
        (num_tokens, 256, 8, 4, 8, renormalize, scoring_func)
        for num_tokens in [1, 8, 32, 64, 128, 256, 496, 512, 16384]
        for renormalize, scoring_func in [
            (False, 0),
            (True, 0),
            (True, 1),
        ]
    ] + [
        # GLM-5.2
        (num_tokens, 256, 1, 1, 8, renormalize, scoring_func)
        for num_tokens in [1, 8, 64, 193, 449, 16384]
        for renormalize, scoring_func in [
            (False, 0),
            (True, 0),
            (True, 1),
        ]
    ]


def build_inputs(binding, dtype, device):
    num_tokens, num_experts, n_group, topk_group, topk, renormalize, scoring_func = (
        binding
    )
    routed_scaling_factor = 1.0
    scores = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    args = (
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    return args, {}


def key_shape(binding):
    num_tokens, num_experts, *_ = binding
    return [num_tokens, num_experts]


def config(binding, dtype):
    num_tokens, num_experts, n_group, topk_group, topk, renormalize, scoring_func = (
        binding
    )
    dtype_name = str(dtype)
    return {
        "inputs": {
            "scores": {"shape": [num_tokens, num_experts], "dtype": dtype_name},
            "n_group": {"scalar": n_group},
            "topk_group": {"scalar": topk_group},
            "topk": {"scalar": topk},
            "renormalize": {"scalar": renormalize},
            "routed_scaling_factor": {"scalar": 1.0},
            "bias": {"shape": [num_experts], "dtype": "torch.float32"},
            "scoring_func": {"scalar": scoring_func},
        },
        "outputs": {
            "topk_weights": {"shape": [num_experts, topk], "dtype": "torch.float32"},
            "topk_ids": {"shape": [num_experts, topk], "dtype": "torch.int32"},
        },
        "dims": {
            "T": num_tokens,
            "E": num_experts,
            "n_group": n_group,
            "topk_group": topk_group,
            "topk": topk,
        },
    }
