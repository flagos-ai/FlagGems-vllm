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

"""CUDA baseline collection contract for persistent_topk."""

import importlib

import torch

OP_NAME = "persistent_topk"
DTYPES = [torch.float32]
IS_INPLACE = False

K = 512
WORKSPACE_BYTES = 1 << 20

_HOMO_SHAPES = [
    # DeepSeek-V4.1-Flash, 1k context, ratio 1
    (67, 1044, 65536, 65536),
    (128, 1044, 65536, 65536),
    (256, 1044, 65536, 65536),
    (508, 1044, 65536, 65536),
    # DeepSeek-V4.1-Flash, 1k context, ratio 2
    (67, 522, 32768, 32768),
    (128, 522, 32768, 32768),
    (256, 522, 32768, 32768),
    (508, 522, 32768, 32768),
    # DeepSeek-V4.1-Flash, 4k context (ratio 1 / ratio 2)
    (16, 4096, 65536, 65536),
    (125, 4096, 65536, 65536),
    (48, 2048, 32768, 32768),
    (120, 2048, 32768, 32768),
    # DeepSeek-V4.1-Flash, 32k context (ratio 1 / ratio 2)
    (3, 32769, 65536, 65536),
    (5, 16384, 32768, 32768),
    # DeepSeek-V4-Flash
    (12, 4105, 4105, 262144),
    (24, 4108, 4108, 262144),
    (10, 1055, 1055, 262144),
    (1, 32773, 32773, 262144),
]


def native():
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "persistent_topk", None)
    return op if callable(op) else None


def grid():
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
    return [
        binding["num_rows"],
        binding["seq_len"],
        binding["max_seq_len"],
        binding["stride"],
    ]


def config(binding, dtype):
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
