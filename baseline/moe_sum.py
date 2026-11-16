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

OP_NAME = "moe_sum"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False

# Active (topk, hidden_size) pairs from mainstream MoE models. Hunyuan
# Hy4-preview and GLM-5.x share the same pair, so it is listed once.
MODEL_TOP_K_HIDDEN = [
    (6, 4096),
    (8, 6144),
    (4, 6144),
    (16, 7168),
]

NUM_TOKENS = [512, 2048, 8192, 16384]

# Benchmark shapes: (num_tokens, topk, hidden_size).
MOE_SUM_SHAPES = [
    (num_tokens, topk, hidden_size)
    for topk, hidden_size in MODEL_TOP_K_HIDDEN
    for num_tokens in NUM_TOKENS
]


def native():
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "moe_sum", None)
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in MOE_SUM_SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, topk, hidden_size = binding["shape"]
    input_tensor = torch.randn(
        num_tokens, topk, hidden_size, dtype=dtype, device=device
    )
    output_tensor = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
    args = (input_tensor, output_tensor)
    return args, {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    num_tokens, topk, hidden_size = binding["shape"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [num_tokens, topk, hidden_size], "dtype": dt},
        },
        "outputs": {
            "output": {"shape": [num_tokens, hidden_size], "dtype": dt},
        },
        "dims": {
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden_size": hidden_size,
        },
    }
