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

import torch

OP_NAME = "fused_add_rms_norm"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)  # input and residual (positional args)
NATIVE_UNAVAILABLE_REASON = None

EPS = 1e-5
INPUT_SEED = 20261123

# Include cases exercised by tests/test_fused_add_rms_norm.py (which uses
# REDUCTION_SHAPES[:2]), plus common LLM hidden sizes and large-width cases.
# Each binding is a native two-dimensional input shape [M, N].
SHAPES = [
    (1, 4096),  # stable and representative first case for --check
    (64, 64),
    (4096, 4096),
    (1, 2),
    (2, 32),
    (4096, 256),
    (200, 40999),
    (16, 4096),
    (128, 4096),
    (512, 4096),
    (1, 8192),
    (128, 8192),
    (1, 16384),
    (32, 16384),
    (1, 32768),
    (1, 65536),
]


def native():
    """Resolve the real vLLM CUDA custom operator, or return None.

    vLLM's signature is (input, residual, weight, epsilon), *without*
    ``normalized_shape``; its return value is None because it mutates buffers.
    """
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm import _custom_ops

        op = getattr(_custom_ops, "fused_add_rms_norm", None)
    except (ImportError, AttributeError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None

    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "vllm._custom_ops.fused_add_rms_norm unavailable"
        return None

    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [{"M": m, "N": n} for m, n in SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["M"], binding["N"]
    generator = torch.Generator(device=device).manual_seed(INPUT_SEED)
    inp = torch.randn((m, n), dtype=dtype, device=device, generator=generator)
    residual = torch.randn((m, n), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((n,), dtype=dtype, device=device, generator=generator)
    # vLLM native API: (input, residual, weight, epsilon)
    return (inp, residual, weight, EPS), {}


def key_shape(binding):
    return [binding["M"], binding["N"]]


def config(binding, dtype):
    m, n = binding["M"], binding["N"]
    dtype_name = str(dtype)
    matrix = {"shape": [m, n], "dtype": dtype_name, "stride": [n, 1]}
    return {
        "inputs": {
            "input": dict(matrix),
            "residual": dict(matrix),
            "weight": {"shape": [n], "dtype": dtype_name, "stride": [1]},
            "epsilon": {"scalar": EPS},
        },
        "outputs": {
            "input": {**matrix, "aliases": "input"},
            "residual": {**matrix, "aliases": "residual"},
        },
        "dims": {"M": m, "N": n, "normalized_shape": [n]},
        "accumulate_dtype": "torch.float32",
        "input_seed": INPUT_SEED,
        "mutation": ["input", "residual"],
        "graph": False,
    }
