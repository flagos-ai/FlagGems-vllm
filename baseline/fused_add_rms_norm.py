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
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)  # input and residual (positional args)
NATIVE_UNAVAILABLE_REASON = None

EPS = 1e-5
INPUT_SEED = 20261123

# Keep the exact workload order of the benchmark:
# - Core: benchmark/core_shapes.yaml -> GenericBenchmarkExcluse1D
# - Comprehensive: GenericBenchmark.set_more_shapes(), excluding the 1D case
# Input ranks are intentionally preserved (both 2D and 3D).
CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]
COMPREHENSIVE_EXTRA_SHAPES = [
    (10000, 1),
    (10000, 256),
    (10000, 65536),
    (100, 1, 100),
    (100, 256, 100),
    (100, 65536, 100),
]
SHAPES = CORE_SHAPES + COMPREHENSIVE_EXTRA_SHAPES


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
    return [{"shape": shape} for shape in SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    generator = torch.Generator(device=device).manual_seed(INPUT_SEED)
    inp = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    residual = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    weight = torch.randn((shape[-1],), dtype=dtype, device=device, generator=generator)
    # vLLM native signature excludes the benchmark-only ``layer_shape`` arg.
    return (inp, residual, weight, EPS), {}


def key_shape(binding):
    return list(binding["shape"])


def _contiguous_strides(shape):
    strides = []
    stride = 1
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= dimension
    return list(reversed(strides))


def config(binding, dtype):
    shape = tuple(binding["shape"])
    dtype_name = str(dtype)
    tensor = {
        "shape": list(shape),
        "dtype": dtype_name,
        "stride": _contiguous_strides(shape),
    }
    rows = 1
    for size in shape[:-1]:
        rows *= size
    hidden_size = shape[-1]
    return {
        "inputs": {
            "input": dict(tensor),
            "residual": dict(tensor),
            "weight": {
                "shape": [hidden_size],
                "dtype": dtype_name,
                "stride": [1],
            },
            "epsilon": {"scalar": EPS},
        },
        "outputs": {
            "input": {**tensor, "aliases": "input"},
            "residual": {**tensor, "aliases": "residual"},
        },
        "dims": {
            "shape": list(shape),
            "rows": rows,
            "hidden_size": hidden_size,
            "normalized_shape": [hidden_size],
        },
        "accumulate_dtype": "torch.float32",
        "input_seed": INPUT_SEED,
        "mutation": ["input", "residual"],
        "graph": False,
    }
