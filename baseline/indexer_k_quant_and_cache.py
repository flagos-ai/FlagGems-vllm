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

OP_NAME = "indexer_k_quant_and_cache"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (1,)
NATIVE_UNAVAILABLE_REASON = None

_SCALE_FMT = "ue8m0"
SHAPE_FIELDS = (
    "num_tokens",
    "num_blocks",
    "block_size",
    "head_dim",
    "quant_block_size",
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm._custom_ops import indexer_k_quant_and_cache

        getattr(torch.ops._C_cache_ops, OP_NAME)
    except (ImportError, AttributeError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return indexer_k_quant_and_cache


def grid():
    # Match the token sweeps and the two 285B profiling cases in the benchmark.
    shapes = [
        (tokens, (2 * tokens + block - 1) // block, block, 512, 128)
        for block, token_counts in (
            (
                16,
                (
                    1,
                    2,
                    4,
                    8,
                    16,
                    17,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                ),
            ),
            (64, (8192, 32768, 65536)),
        )
        for tokens in token_counts
    ]
    shapes.extend([(8192, 1302, 16, 128, 128), (128, 20, 16, 128, 128)])
    return [dict(zip(SHAPE_FIELDS, shape)) for shape in shapes]


def key_shape(binding):
    return [binding[field] for field in SHAPE_FIELDS]


def _cache_shape(binding):
    head_dim = binding["head_dim"]
    scale_bytes = 4 * (head_dim // binding["quant_block_size"])
    return [binding["num_blocks"], binding["block_size"], head_dim + scale_bytes]


def build_inputs(binding, dtype, device):
    tokens = binding["num_tokens"]
    generator = torch.Generator(device=device).manual_seed(1)
    k = torch.randn(
        tokens, binding["head_dim"], dtype=dtype, device=device, generator=generator
    )
    num_slots = binding["num_blocks"] * binding["block_size"]
    slots = torch.randperm(
        num_slots, dtype=torch.int64, device=device, generator=generator
    )[:tokens]
    cache = torch.empty(_cache_shape(binding), dtype=torch.uint8, device=device)
    # The vLLM Python wrapper names its fifth argument kv_cache_dtype.
    return (k, cache, slots, binding["quant_block_size"], _SCALE_FMT), {}


def config(binding, dtype):
    cache_shape = _cache_shape(binding)
    cache = {"shape": cache_shape, "dtype": "torch.uint8"}
    tokens = binding["num_tokens"]
    return {
        "inputs": {
            "k": {"shape": [tokens, binding["head_dim"]], "dtype": str(dtype)},
            "kv_cache": {**cache, "note": "原地写回量化 K + scale"},
            "slot_mapping": {"shape": [tokens], "dtype": "torch.int64"},
            "quant_block_size": {"scalar": binding["quant_block_size"]},
            "kv_cache_dtype": {"scalar": _SCALE_FMT},
        },
        "outputs": {"kv_cache": cache},
        "dims": {**binding, "cache_stride": cache_shape[-1]},
    }
