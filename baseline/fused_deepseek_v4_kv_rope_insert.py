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

"""CUDA baseline collection contract for fused_deepseek_v4_kv_rope_insert."""

import importlib

import torch

OP_NAME = "fused_deepseek_v4_kv_rope_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

# Input KV is always BF16; cache format is an independent collection dimension.
CACHE_FORMATS = ("bf16", "fp8", "packed_v4", "mxfp8")

# FlagOSTune covers concurrency 1/2/4/8/64/100, up to 2048 sequences and
# 16384 batched tokens. vLLM uses 32-token pages for this SWA cache.
BENCHMARK_SHAPES = [
    (1, 1, 32),
    (2, 2, 32),
    (4, 4, 32),
    (8, 8, 32),
    (64, 64, 32),
    (100, 100, 32),
    (1027, 1024, 32),
    (2048, 2048, 32),
    (8192, 8192, 32),
    (16384, 16384, 32),
]


def native():
    try:
        importlib.import_module("vllm._custom_ops")
    except (ImportError, AttributeError, RuntimeError, OSError):
        return None
    op = getattr(torch.ops._C, OP_NAME, None)
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "num_tokens_insert": num_insert,
            "cache_block_size": block_size,
            "cache_format": cache_format,
        }
        for num_tokens, num_insert, block_size in BENCHMARK_SHAPES
        for cache_format in CACHE_FORMATS
    ]


def _cache_spec(cache_format):
    return {
        "bf16": (torch.bfloat16, 512),
        "fp8": (torch.float8_e4m3fn, 512),
        "packed_v4": (torch.uint8, 584),
        "mxfp8": (torch.uint8, 528),
    }[cache_format]


def build_inputs(
    binding,
    dtype,
    device,
    *,
    padded_pages=False,
    slot_mapping=None,
    position_offset=0,
    scale_value=0.5,
):
    num_tokens = binding["num_tokens"]
    num_insert = binding["num_tokens_insert"]
    block_size = binding["cache_block_size"]
    cache_format = binding["cache_format"]
    cache_dtype, width = _cache_spec(cache_format)
    kv = torch.randn(num_tokens, 512, dtype=dtype, device=device)
    num_blocks = (num_insert + block_size - 1) // block_size + 1
    if padded_pages:
        page_stride = ((block_size * width + 575) // 576) * 576 + 576
        backing = torch.full((num_blocks, page_stride), 3.0, device=device).to(
            cache_dtype
        )
        cache = backing.as_strided(
            (num_blocks, block_size, width), (page_stride, width, 1)
        )
    else:
        cache = torch.zeros(
            num_blocks, block_size, width, dtype=cache_dtype, device=device
        )
    # Unique shuffled slots stress paged writes; larger cases mask one row.
    if slot_mapping is None:
        slots = torch.randperm(num_insert, dtype=torch.int64, device=device)
        if num_insert > 1:
            slots[-1] = -1
    else:
        slots = slot_mapping
    positions = torch.arange(
        position_offset,
        position_offset + num_tokens,
        dtype=torch.int64,
        device=device,
    )
    angle_positions = torch.arange(
        position_offset + num_tokens, dtype=torch.int64, device=device
    )
    angles = (
        angle_positions.float()[:, None]
        * (10000.0 ** (-torch.arange(0, 64, 2, device=device).float() / 64))[None, :]
    )
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    scale = (
        torch.tensor([scale_value], dtype=torch.float32, device=device)
        if cache_format == "fp8"
        else None
    )
    return (
        kv,
        cache,
        slots,
        positions,
        cos_sin,
        block_size,
        scale,
        cache_format == "mxfp8",
    ), {}


def key_shape(binding):
    return (
        f"tokens{binding['num_tokens']}_insert{binding['num_tokens_insert']}_"
        f"bs{binding['cache_block_size']}_{binding['cache_format']}"
    )


def config(binding, dtype):
    num_tokens = binding["num_tokens"]
    num_insert = binding["num_tokens_insert"]
    block_size = binding["cache_block_size"]
    cache_format = binding["cache_format"]
    cache_dtype, width = _cache_spec(cache_format)
    num_blocks = (num_insert + block_size - 1) // block_size + 1
    cache = {
        "shape": [num_blocks, block_size, width],
        "dtype": str(cache_dtype),
    }
    return {
        "inputs": {
            "kv": {"shape": [num_tokens, 512], "dtype": str(dtype)},
            "k_cache": dict(cache),
            "slot_mapping": {"shape": [num_insert], "dtype": "torch.int64"},
            "position_ids": {"shape": [num_tokens], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [num_tokens, 64], "dtype": "torch.float32"},
            "cache_block_size": {"scalar": block_size},
            "fp8_scale": (
                {"shape": [1], "dtype": "torch.float32", "value": 0.5}
                if cache_format == "fp8"
                else {"scalar": None}
            ),
            "kv_mxfp8": {"scalar": cache_format == "mxfp8"},
        },
        "outputs": {"k_cache": dict(cache)},
        "dims": dict(binding),
    }
