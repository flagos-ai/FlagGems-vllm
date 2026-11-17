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

"""Collect native vLLM cooperative_topk on the FP32 full-row core workloads.

The collection contract follows the repository's pinned vLLM 0.24.0:
https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/cooperative_topk.cu
Its 32-row limit leaves the 64-row workload explicitly unsupported.

Run on a supported NVIDIA GPU with a native vLLM CUDA build:
    python tools/collect_baseline_nvidia.py --check --ops cooperative_topk
    python tools/collect_baseline_nvidia.py --ops cooperative_topk \
        --output cooperative_topk_baseline.json --reference-chip H800

Set --reference-chip to the actual collection GPU. Consumers must use
op_name="cooperative_topk" and the same full-row lengths and padded row strides.
"""

import importlib

import torch

OP_NAME = "cooperative_topk"
DTYPES = [torch.float32]
# Output and workspace are overwritten scratch: native writes each workspace
# element before reading it. No input state needs restoration between calls.
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

# (rows, width, k): short rows, the 16384 boundary, padded and wide rows.
CORE_SHAPES = [
    (1, 127, 512),
    (4, 4097, 512),
    (16, 16383, 1024),
    (16, 16384, 1024),
    (16, 16385, 1024),
    (4, 65539, 2048),
    (1, 262144, 512),
    (64, 262144, 2048),
    (1, 1048576, 1024),
]
WORKSPACE_BYTES_PER_ROW = 16384
INPUT_SEED = 0


def native():
    """Resolve the CUDA kernel, with no Torch or FlagGems fallback."""
    global NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    if not torch.cuda.is_available():
        NATIVE_UNAVAILABLE_REASON = "CUDA device required"
        return None
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        NATIVE_UNAVAILABLE_REASON = "vLLM cooperative_topk requires SM90+"
        return None
    try:
        importlib.import_module("vllm._custom_ops")
        op = getattr(torch.ops._C, OP_NAME, None)
        if not callable(op) or not torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::cooperative_topk", "CUDA"
        ):
            NATIVE_UNAVAILABLE_REASON = "vLLM build has no CUDA cooperative_topk kernel"
            return None
    except (ImportError, OSError, RuntimeError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    return op


def grid():
    return [{"shape": list(shape)} for shape in CORE_SHAPES]


def unsupported_reason(binding, dtype):
    if binding["shape"][0] > 32:
        return "Collection is limited to <=32 rows by the vLLM 0.24.0 native contract"
    return None


def build_inputs(binding, dtype, device):
    rows, width, k = binding["shape"]
    # Native vectorized loads require a four-element-aligned row stride.
    # Keep the logical width intact, including non-multiples of four.
    stride = (width + 3) // 4 * 4
    generator = torch.Generator(device=device).manual_seed(INPUT_SEED)
    logits = torch.randn(rows, stride, dtype=dtype, device=device, generator=generator)[
        :, :width
    ]
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    output = torch.empty((rows, k), dtype=torch.int32, device=device)
    workspace = torch.empty(
        rows * WORKSPACE_BYTES_PER_ROW, dtype=torch.uint8, device=device
    )
    return (logits, lengths, output, workspace, k, width), {}


def key_shape(binding):
    # Include k: distinct selections over the same logits must not overwrite
    # each other. config.inputs enables the benchmark's signature lookup.
    return list(binding["shape"])


def config(binding, dtype):
    rows, width, k = binding["shape"]
    stride = (width + 3) // 4 * 4
    return {
        "inputs": {
            "logits": {
                "shape": [rows, width],
                "stride": [stride, 1],
                "dtype": str(dtype),
                "distribution": "standard normal",
            },
            "lengths": {
                "shape": [rows],
                "dtype": "torch.int32",
                "value": width,
            },
            "output": {"shape": [rows, k], "dtype": "torch.int32"},
            "workspace": {
                "shape": [rows * WORKSPACE_BYTES_PER_ROW],
                "dtype": "torch.uint8",
            },
            "k": {"scalar": k},
            "max_seq_len": {"scalar": width},
        },
        "outputs": {
            "output": {
                "shape": [rows, k],
                "dtype": "torch.int32",
                "aliases": "output",
                "note": "unsorted top-k indices; short rows are padded with -1",
            },
        },
        "dims": {
            "rows": rows,
            "width": width,
            "k": k,
            "row_stride": stride,
            "lengths_mode": "full-row",
        },
        "input_seed": INPUT_SEED,
        "mutation": ["output", "workspace (scratch, no persistent state)"],
        "native_unsupported_reason": unsupported_reason(binding, dtype),
    }
