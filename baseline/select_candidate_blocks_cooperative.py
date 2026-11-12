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

"""CUDA baseline collection contract for select_candidate_blocks_cooperative.

The native reference is vLLM's own
``vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk.
select_candidate_blocks_cooperative`` (two Triton kernels + the CUDA
``cooperative_topk``). It is only present on vLLM builds that ship the
specialized-decode indexer; ``native()`` returns ``None`` otherwise and the
collector records the workloads as unsupported without launching them.
"""

import importlib

import torch

OP_NAME = "select_candidate_blocks_cooperative"
DTYPES = [torch.float32]
IS_INPLACE = False

BLOCK_SIZE = 8
# DeepSeek-V4.1 decode: candidate_block_size=8, candidate_topk_blocks up to
# 2048, a 1M-token context -> 131072 blocks, at most 64 rows per rank.
_WIDTH = 131072 * BLOCK_SIZE

# (num_rows, seq_len, k); width/block_size are fixed above.
_SHAPES = [
    (1, 4096, 2048),
    (1, 1048576, 2048),
    (4, 32768, 2048),
    (4, 1048576, 2048),
    (14, 262144, 2048),
    (16, 65536, 2048),
    (32, 131072, 2048),
    (64, 16384, 2048),
    (64, 524288, 2048),
    (16, 131072, 512),
    (16, 131072, 1024),
]


def native():
    try:
        mod = importlib.import_module(
            "vllm.models.deepseek_v41.nvidia.specialized_decode."
            "attention.candidate_topk"
        )
    except Exception:
        return None
    op = getattr(mod, "select_candidate_blocks_cooperative", None)
    return op if callable(op) else None


def grid():
    return [
        {
            "kind": "homo",
            "num_rows": num_rows,
            "seq_len": seq_len,
            "width": _WIDTH,
            "block_size": BLOCK_SIZE,
            "k": k,
        }
        for num_rows, seq_len, k in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    num_rows = binding["num_rows"]
    seq_len = binding["seq_len"]
    width = binding["width"]
    block_size = binding["block_size"]
    k = binding["k"]

    logits = torch.randn(num_rows, width, dtype=dtype, device=device)
    row_ends = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)
    out = torch.empty((num_rows, k), dtype=torch.int32, device=device)
    return (logits, row_ends, block_size, out), {}


def key_shape(binding):
    return [
        binding["num_rows"],
        binding["seq_len"],
        binding["width"],
        binding["block_size"],
        binding["k"],
    ]


def config(binding, dtype):
    num_rows = binding["num_rows"]
    seq_len = binding["seq_len"]
    width = binding["width"]
    block_size = binding["block_size"]
    k = binding["k"]
    dtype_name = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [num_rows, width], "dtype": dtype_name},
            "row_ends": {"shape": [num_rows], "dtype": "torch.int32"},
            "block_size": {"scalar": block_size},
            "out": {
                "shape": [num_rows, k],
                "dtype": "torch.int32",
                "note": "原地写回（候选块 id，-1 补齐）",
            },
        },
        "outputs": {
            "out": {"shape": [num_rows, k], "dtype": "torch.int32"},
        },
        "dims": {
            "kind": binding["kind"],
            "num_rows": num_rows,
            "seq_len": seq_len,
            "width": width,
            "block_size": block_size,
            "nblocks": width // block_size,
            "k": k,
        },
    }
