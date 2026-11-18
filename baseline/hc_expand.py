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

"""Shared baseline contract for the mHC ``hc_expand`` operator.

Functional tests use a local PyTorch reference matching vLLM's implementation.
vLLM's NVIDIA Triton implementation is the performance baseline used by the
benchmark suite.
"""

import torch

try:
    from vllm.models.deepseek_v41.nvidia.ops.mhc import hc_expand as native
except (ImportError, OSError, RuntimeError):
    native = None

OP_NAME = "hc_expand"
DTYPES = [torch.bfloat16, torch.float32]
IS_INPLACE = False
INPUT_LAYOUTS = (
    "contiguous",
    "strided",
)

BENCHMARK_SHAPES = [
    (7, 17, 8),
    (33, 5137, 4),
    (8192, 5120, 4),
    (1, 5120, 4),
    (16, 5120, 4),
    (256, 5120, 4),
    (1024, 5120, 4),
    (4096, 5120, 4),
    (16384, 5120, 4),
    (32768, 5120, 4),
    (1025, 5120, 4),
    (8193, 5120, 4),
]

FUNCTIONAL_SHAPES = [
    *BENCHMARK_SHAPES,
    (0, 17, 4),
    (1, 513, 1),
    (2, 5120, 8),
    (4, 5120, 2),
    (8, 5120, 3),
    (32, 5120, 5),
    (64, 5120, 6),
    (128, 5120, 7),
    (512, 5120, 9),
    (2048, 5120, 16),
    (1, 5120, 2),
    (7, 5120, 3),
    (33, 5137, 5),
    (1025, 5120, 6),
    (8193, 5120, 7),
]

def torch_reference(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """PyTorch reference copied from vllm/model_executor/layers/mhc.py."""
    return x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "hc_mult": hc_mult,
        }
        for num_tokens, hidden_size, hc_mult in BENCHMARK_SHAPES
    ]


def build_inputs(
    binding,
    dtype,
    device,
    *,
    strided=False,
    layout=None,
    special_values=True,
):
    num_tokens = binding["num_tokens"]
    hidden_size = binding["hidden_size"]
    hc_mult = binding["hc_mult"]

    if layout is None:
        layout = "strided" if strided else "contiguous"

    if layout == "contiguous":
        x = torch.randn(
            num_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
        )
    elif layout == "strided":
        # Both axes are non-unit stride: (2 * hidden + 1, 2).
        storage = torch.randn(
            num_tokens,
            2 * hidden_size + 1,
            dtype=dtype,
            device=device,
        )
        x = storage[:, 1::2]
    else:
        raise ValueError(f"unsupported input layout: {layout!r}")

    # Bitwise validation must also cover non-finite values and signed zero.
    if special_values and num_tokens and hidden_size:
        x[0, 0] = float("nan")
        x[0, -1] = -0.0
        if hidden_size > 1:
            x[0, 1] = float("inf")
        if hidden_size > 2:
            x[0, 2] = float("-inf")

    return (x, hc_mult), {}


def key_shape(binding):
    return (
        f"tokens{binding['num_tokens']}_hidden{binding['hidden_size']}_"
        f"hc{binding['hc_mult']}"
    )


def config(binding, dtype, *, layout="contiguous"):
    num_tokens = binding["num_tokens"]
    hidden_size = binding["hidden_size"]
    hc_mult = binding["hc_mult"]
    stride = {
        "contiguous": [hidden_size, 1],
        "strided": [2 * hidden_size + 1, 2],
    }.get(layout)
    if stride is None:
        raise ValueError(f"unsupported input layout: {layout!r}")
    return {
        "inputs": {
            "x": {
                "shape": [num_tokens, hidden_size],
                "dtype": str(dtype),
                "stride": stride,
            },
            "hc_mult": {"scalar": hc_mult},
        },
        "outputs": {
            "out": {
                "shape": [num_tokens, hc_mult, hidden_size],
                "dtype": str(dtype),
                "layout": "contiguous",
            }
        },
        "dims": dict(binding),
    }
