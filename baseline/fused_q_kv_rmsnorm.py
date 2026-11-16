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

OP_NAME = "fused_q_kv_rmsnorm"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

EPS = 1e-6

# DeepSeek-V4 workload: (num_tokens, q_size, kv_size); identical to the
# benchmark shape grid.
FUSED_Q_KV_RMSNORM_SHAPES = [
    (1, 1536, 512),
    (32, 1536, 512),
    (128, 1536, 512),
    (512, 1536, 512),
    (2048, 1536, 512),
    (32, 64 * 576, 576),
    (128, 64 * 576, 576),
]


def native():
    """Resolve the vLLM fused_q_kv_rmsnorm CUDA baseline; no torch fallback."""
    global NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    try:
        mod = importlib.import_module("vllm.models.deepseek_v4.common.ops")
    except (ImportError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    op = getattr(mod, "fused_q_kv_rmsnorm", None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = (
            "vllm.models.deepseek_v4.common.ops has no fused_q_kv_rmsnorm"
        )
        return None
    return op


def grid():
    return [{"shape": list(shape)} for shape in FUSED_Q_KV_RMSNORM_SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, q_size, kv_size = binding["shape"]
    qr = torch.randn((num_tokens, q_size), dtype=dtype, device=device)
    kv = torch.randn((num_tokens, kv_size), dtype=dtype, device=device)
    q_weight = torch.randn((q_size,), dtype=dtype, device=device)
    kv_weight = torch.randn((kv_size,), dtype=dtype, device=device)
    return (qr, kv, q_weight, kv_weight, EPS), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    num_tokens, q_size, kv_size = binding["shape"]
    dt = str(dtype)
    return {
        "inputs": {
            "qr": {"shape": [num_tokens, q_size], "dtype": dt},
            "kv": {"shape": [num_tokens, kv_size], "dtype": dt},
            "q_weight": {"shape": [q_size], "dtype": dt},
            "kv_weight": {"shape": [kv_size], "dtype": dt},
            "eps": {"scalar": EPS},
        },
        "outputs": {
            "q_out": {
                "shape": [num_tokens, q_size],
                "dtype": dt,
                "alias": "fresh; no input alias",
            },
            "kv_out": {
                "shape": [num_tokens, kv_size],
                "dtype": dt,
                "alias": "fresh; no input alias",
            },
        },
        "dims": {"num_tokens": num_tokens, "q_size": q_size, "kv_size": kv_size},
        "compute_dtype": "torch.float32",
    }
