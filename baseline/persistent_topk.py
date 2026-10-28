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

"""CUDA baseline collection contract for persistent_topk.

Consumed by tools/collect_baseline_nvidia.py: grid() enumerates the
collection points, build_inputs() builds the real call arguments, config()
describes them for shape matching, and key_shape() names each JSON entry.

The shape grid must stay identical to benchmark/test_persistent_topk.py so a
collected baseline can be matched by every platform.
"""

import importlib

import torch

OP_NAME = "persistent_topk"
DTYPES = [torch.float32]
IS_INPLACE = False

K = 512
# The vLLM baseline layout; the same size is allocated by the benchmark.
WORKSPACE_BYTES = 1 << 20

# (num_rows, seq_len, max_seq_len, stride) -- mirrors benchmark set_shapes().
# (num_rows, seq_len, max_seq_len, stride) -- mirrors benchmark set_shapes().
#
# Batches with num_rows <= 64 go to cooperative_topk under vLLM's auto policy;
# persistent_topk covers the larger ones. cooperative_topk is newer, has no
# FlagGems-vllm counterpart, and before it existed these shapes all ran on
# persistent_topk, so the small-batch rows are kept deliberately.
_HOMO_SHAPES = [
    # (num_rows, seq_len, max_seq_len, stride) -- mirrors benchmark set_shapes().
    #
    # Batches with num_rows <= 64 go to cooperative_topk under vLLM's auto policy;
    # persistent_topk covers the larger ones. cooperative_topk is newer, has no
    # FlagGems-vllm counterpart, and before it existed these shapes all ran on
    # persistent_topk, so the small-batch rows are kept deliberately.
    # DeepSeek-V4.1-Flash, 1k context: p1024d1024 run filtered by the
    # serving-start sentinel (11,136 real calls, num_rows 67-508); seq_len is the
    # p50 of each ratio class (1044 / 522), stride == max_seq_len.
    (67, 1044, 65536, 65536),
    (128, 1044, 65536, 65536),
    (256, 1044, 65536, 65536),
    (508, 1044, 65536, 65536),
    (67, 522, 32768, 32768),
    (128, 522, 32768, 32768),
    (256, 522, 32768, 32768),
    (508, 522, 32768, 32768),
    # DeepSeek-V4.1-Flash, 4k context (ratio 1 / ratio 2).
    (16, 4096, 65536, 65536),
    (125, 4096, 65536, 65536),
    (48, 2048, 32768, 32768),
    (120, 2048, 32768, 32768),
    # DeepSeek-V4.1-Flash, 32k context.
    (3, 32769, 65536, 65536),
    (5, 16384, 32768, 32768),
    # DeepSeek-V4-Flash: fixed 262144-wide buffer, seq_len == max_seq_len.
    (12, 4105, 4105, 262144),
    (24, 4108, 4108, 262144),
    (10, 1055, 1055, 262144),
    (1, 32773, 32773, 262144),
]


def native():
    """Resolve vLLM's CUDA persistent_topk, or None when it is unavailable."""
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "persistent_topk", None)
    return op if callable(op) else None


def grid():
    """Every collection point: one entry per (num_rows, seq_len, msl, stride)."""
    return [
        {
            "kind": "homo",
            "num_rows": num_rows,
            "seq_len": seq_len,
            "max_seq_len": max_seq_len,
            "stride": stride,
        }
        for num_rows, seq_len, max_seq_len, stride in _HOMO_SHAPES
    ]


def build_inputs(binding, dtype, device):
    """Real call arguments, identical to what the benchmark feeds the op."""
    num_rows = binding["num_rows"]
    seq_len = binding["seq_len"]
    max_seq_len = binding["max_seq_len"]
    stride = binding["stride"]

    logits = torch.full((num_rows, stride), float("-inf"), dtype=dtype, device=device)
    logits[:, :seq_len] = torch.randn(num_rows, seq_len, dtype=dtype, device=device)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)
    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return (logits, lengths, indices, workspace, K, max_seq_len), {}


def key_shape(binding):
    """JSON entry name: unique across grid() and readable in the baseline dump."""
    return [
        binding["num_rows"],
        binding["seq_len"],
        binding["max_seq_len"],
        binding["stride"],
    ]


def config(binding, dtype):
    """Shape description matching what the benchmark records."""
    num_rows = binding["num_rows"]
    seq_len = binding["seq_len"]
    max_seq_len = binding["max_seq_len"]
    stride = binding["stride"]
    dtype_name = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [num_rows, stride], "dtype": dtype_name},
            "lengths": {"shape": [num_rows], "dtype": "torch.int32"},
            "output": {
                "shape": [num_rows, K],
                "dtype": "torch.int32",
                "note": "原地写回（top-k 索引）",
            },
            "workspace": {"shape": [WORKSPACE_BYTES], "dtype": "torch.uint8"},
            "k": {"scalar": K},
            "max_seq_len": {"scalar": max_seq_len},
        },
        "outputs": {
            "output": {"shape": [num_rows, K], "dtype": "torch.int32"},
        },
        "dims": {
            "kind": binding["kind"],
            "num_rows": num_rows,
            "seq_len": seq_len,
            "max_seq_len": max_seq_len,
            "stride": stride,
            "K": K,
        },
    }
