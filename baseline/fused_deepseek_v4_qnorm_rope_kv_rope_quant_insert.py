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

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
HEAD_DIM = 512
ROPE_DIM = 64
HEAD_BYTES = 584


def native():
    try:
        importlib.import_module("vllm._custom_ops")
        if hasattr(torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"):
            op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
        else:
            op = None
    except (ImportError, AttributeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        (num_tokens, num_heads)
        for num_tokens in [
            1,
            4,
            17,
            64,
            1024,
            2048,
            8192,
            32768,
            65536,
            98304,
            131072,
        ]
        for num_heads in [64, 128]
    ]


def build_inputs(binding, dtype, device):
    num_tokens, num_heads = binding
    num_tokens_insert = num_tokens
    q_head_padded = num_heads
    block_size = 64
    max_pos = 4096
    eps = 1e-6

    def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
        if max_pos <= 8192:
            base = 10000.0
        elif max_pos <= 32768:
            base = 20000.0
        elif max_pos <= 65536:
            base = 40000.0
        elif max_pos <= 98304:
            base = 60000.0
        else:
            base = 100000.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
        return cache.to(dtype)

    q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.arange(num_tokens_insert, dtype=torch.int64, device=device)
    k_cache = torch.zeros(
        num_blocks,
        block_size * HEAD_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    args = (
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        q_head_padded,
        eps,
        block_size,
    )
    return args, {}


def key_shape(binding):
    num_tokens, num_heads = binding
    return [num_tokens, num_heads]


def config(binding, dtype):
    num_tokens, num_heads = binding
    num_tokens_insert = num_tokens
    q_head_padded = num_heads
    block_size = 64
    max_pos = 4096
    eps = 1e-6
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    dtype_name = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [num_tokens, num_heads, HEAD_DIM], "dtype": dtype_name},
            "kv": {"shape": [num_tokens, HEAD_DIM], "dtype": dtype_name},
            "k_cache": {
                "shape": [num_blocks, block_size * HEAD_BYTES],
                "dtype": "torch.uint8",
            },
            "slot_mapping": {"shape": [num_tokens_insert], "dtype": "torch.int64"},
            "position_ids": {"shape": [num_tokens], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [max_pos, ROPE_DIM], "dtype": "torch.float32"},
            "q_head_padded": {"scalar": q_head_padded},
            "eps": {"scalar": eps},
            "cache_block_size": {"scalar": block_size},
        },
        "outputs": {
            "q_out": {
                "shape": [num_tokens, q_head_padded, HEAD_DIM],
                "dtype": dtype_name,
            },
        },
        "dims": {
            "num_tokens": num_tokens,
            "num_heads": num_heads,
        },
    }
