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

OP_NAME = "topk_softplus_sqrt"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# (num_tokens, num_experts, topk) — mirrors the shape set used by the
# existing pytest benchmark (benchmark/test_topk_softplus_sqrt.py).
TOPK_SOFTPLUS_SQRT_SHAPES = [
    (1, 256, 6),
    (10, 256, 6),
    (16, 256, 6),
    (128, 256, 6),
    (512, 256, 6),
    (1024, 256, 6),
    (2048, 256, 6),
    (4096, 256, 6),
]


def native():
    """Resolve vLLM's native topk_hash_softplus_sqrt kernel; None if unavailable.
    Mirrors the HAS_VLLM probe in the pytest benchmark: this baseline is only
    meaningful when the real vLLM CUDA kernel is importable, not the pure
    PyTorch fallback reference used there when vLLM is absent.
    """
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "topk_hash_softplus_sqrt", None)
    return op if callable(op) else None


def grid():
    return [
        {"shape": list(shape)}
        for shape in TOPK_SOFTPLUS_SQRT_SHAPES
    ]


def _unpack(binding):
    num_tokens, num_experts, topk = binding["shape"]
    return num_tokens, num_experts, topk


def build_inputs(binding, dtype, device):
    num_tokens, num_experts, topk = _unpack(binding)
    torch.manual_seed(0)

    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device=device
    )
    correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device=device
    )
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=device
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )

    # Matches the vLLM native call signature used by the pytest wrapper:
    # renormalize=True, routed_scaling_factor=1.0, input_ids/tid2eid unused
    # (identity top-k path, not the hashed id2eid lookup path).
    args = (
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        True,
        1.0,
        correction_bias,
    )
    kwargs = {
        "input_ids": None,
        "tid2eid": None,
    }
    return args, kwargs


def key_shape(binding):
    num_tokens, num_experts, topk = _unpack(binding)
    return [num_tokens, num_experts, topk]


def config(binding, dtype):
    num_tokens, num_experts, topk = _unpack(binding)
    dt = str(dtype)

    return {
        "inputs": {
            "gating_output": {"shape": [num_tokens, num_experts], "dtype": dt},
            "correction_bias": {
                "shape": [num_experts],
                "dtype": "torch.float32",
            },
            "renormalize": {"value": True},
            "routed_scaling_factor": {"value": 1.0},
        },
        "outputs": {
            "topk_weights": {
                "shape": [num_tokens, topk],
                "dtype": "torch.float32",
            },
            "topk_indices": {
                "shape": [num_tokens, topk],
                "dtype": "torch.int32",
            },
            "token_expert_indices": {
                "shape": [num_tokens, topk],
                "dtype": "torch.int32",
            },
        },
        "dims": {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
        },
    }

