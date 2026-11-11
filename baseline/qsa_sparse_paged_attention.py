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

"""Native vLLM QSA baseline (v0.32-compatible packed-index/FP8 interface).

The collector imports this module without FlagGems-vllm or pytest. The benchmark
uses the same input builder and shape set, which is defined locally so collection
does not depend on repository package imports or data-file loading.
"""

import importlib

import torch

OP_NAME = "qsa_sparse_paged_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_HEAD_DIM = 256
_SELECTION_WIDTH = 2051


def native():
    """Resolve the vLLM NVIDIA implementation, or return None if unavailable."""
    try:
        module = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
    except ImportError:
        return None
    op = getattr(module, OP_NAME, None)
    return op if callable(op) else None


def grid():
    shapes = [
        [1, 6, 1, 784, False, 1, True, 88, 2712],
        [16, 6, 1, 784, False, 16, True, 88, 512],
        [32, 6, 1, 784, False, 32, True, 88, 2712],
        [40, 6, 1, 784, False, 40, True, 88, 512],
        [64, 6, 1, 784, False, 64, True, 88, 512],
        [72, 6, 1, 784, False, 72, True, 88, 2712],
        [128, 6, 1, 784, False, 128, True, 88, 512],
        [136, 6, 1, 784, False, 136, True, 88, 512],
        [272, 6, 1, 784, False, 272, True, 88, 512],
        [336, 6, 1, 784, False, 336, True, 88, 512],
        [512, 6, 1, 784, False, 512, True, 88, 2712],
        [902, 6, 1, 784, True, 451, True, 88, 2712],
        [2369, 6, 1, 784, True, 54, True, 88, 2712],
        [6177, 6, 1, 784, True, 57, True, 88, 2712],
        [16384, 6, 1, 784, True, 1, True, 88, 2712],
        [16384, 6, 1, 784, True, 16, True, 88, 2712],
        [16384, 6, 1, 784, True, 30, True, 88, 2712],
        [16384, 6, 1, 784, True, 31, True, 88, 2712],
        [16384, 6, 1, 784, True, 57, True, 88, 2712],
        [16384, 6, 1, 784, True, 2048, True, 88, 2712],
    ]
    return [{"shape": shape} for shape in shapes]


def _shape(binding):
    shape = list(binding["shape"])
    pages = shape[7] if len(shape) > 7 else 65
    blocks = shape[8] if len(shape) > 8 else shape[5] * pages
    return (*shape[:7], pages, blocks)


def _make_packed_indices(num_rows, selection_width, context_capacity, device):
    """Packed [rows, selection_width + 1] indices with a trailing count column.

    Every row selects distinct random indices over the context capacity;
    the trailing count is the full selection width.
    """
    packed = torch.empty(
        (num_rows, selection_width + 1), device=device, dtype=torch.int32
    )
    for r in range(num_rows):
        perm = torch.randperm(context_capacity, device=device)[:selection_width]
        packed[r, :selection_width] = perm.to(torch.int32)
    packed[:, -1] = selection_width
    return packed


def build_inputs(binding, dtype, device):
    (
        num_rows,
        num_query_heads,
        num_kv_heads,
        page_size,
        use_prefill_config,
        num_requests,
        fp8,
        num_pages_per_request,
        num_cache_blocks,
    ) = _shape(binding)
    torch.manual_seed(2)
    head_dim = _HEAD_DIM
    selection_width = _SELECTION_WIDTH
    q = torch.randn(
        num_rows,
        num_query_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    output_gate = torch.randn_like(q)
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device=device,
        dtype=dtype,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    total_pages = num_requests * num_pages_per_request
    if num_cache_blocks >= total_pages:
        block_table = (
            torch.randperm(num_cache_blocks, device=device)[:total_pages]
            .reshape(num_requests, num_pages_per_request)
            .to(torch.int32)
        )
    else:
        # Synthetic page mappings allow repeats when the pool is smaller
        # than the logical table.
        block_table = torch.randint(
            0,
            num_cache_blocks,
            (num_requests, num_pages_per_request),
            device=device,
            dtype=torch.int32,
        )
    rows = torch.arange(num_rows, device=device, dtype=torch.int32)
    rows_per_request = (num_rows + num_requests - 1) // num_requests
    token_to_req = rows // rows_per_request

    context_capacity = num_pages_per_request * page_size
    # Worst case: every row selects the full width, so no row hits the
    # short-row direct-write or empty-split paths.
    logical_indices = _make_packed_indices(
        num_rows,
        selection_width,
        context_capacity,
        device,
    )

    if fp8:
        # A fixed non-unit pair (k != v) exercises the host-side scale folding
        # and catches a k/v swap; scales are host floats, as the layer exposes
        # them. Stored values are the scaled ones, as reshape_and_cache does.
        k_scale, v_scale = 0.5, 2.0
        # Quantize into the fused KV buffer and then split, so K/V keep
        # the production interleaved layout (head stride 2*head_dim,
        # token stride num_kv_heads*2*head_dim).
        kv_fp8 = torch.empty_like(kv_cache, dtype=torch.float8_e4m3fn)
        k_cache, v_cache = kv_fp8.split(head_dim, dim=-1)
        k_cache.copy_((kv_cache[..., :head_dim] / k_scale).to(torch.float8_e4m3fn))
        v_cache.copy_((kv_cache[..., head_dim:] / v_scale).to(torch.float8_e4m3fn))
    else:
        k_scale, v_scale = 1.0, 1.0

    return (q, k_cache, v_cache, logical_indices, block_table, token_to_req), {
        "use_prefill_config": use_prefill_config,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "output_gate": output_gate,
    }


def key_shape(binding):
    # Q alone is insufficient: requests, physical cache size and KV dtype vary.
    return list(_shape(binding))


def config(binding, dtype):
    (
        rows,
        query_heads,
        kv_heads,
        page_size,
        prefill,
        requests,
        fp8,
        pages,
        blocks,
    ) = _shape(binding)
    q_shape = [rows, query_heads, _HEAD_DIM]
    kv_shape = [blocks, page_size, kv_heads, _HEAD_DIM]
    kv_stride = [
        page_size * kv_heads * 2 * _HEAD_DIM,
        kv_heads * 2 * _HEAD_DIM,
        2 * _HEAD_DIM,
        1,
    ]
    kv_dtype = str(torch.float8_e4m3fn if fp8 else dtype)
    return {
        "inputs": {
            "q": {"shape": q_shape, "dtype": str(dtype)},
            "k_cache": {
                "shape": kv_shape,
                "dtype": kv_dtype,
                "stride": kv_stride,
                "storage_offset": 0,
            },
            "v_cache": {
                "shape": kv_shape,
                "dtype": kv_dtype,
                "stride": kv_stride,
                "storage_offset": _HEAD_DIM,
            },
            "logical_indices": {
                "shape": [rows, _SELECTION_WIDTH + 1],
                "dtype": "torch.int32",
            },
            "block_table": {"shape": [requests, pages], "dtype": "torch.int32"},
            "token_to_req": {"shape": [rows], "dtype": "torch.int32"},
            "output_gate": {"shape": q_shape, "dtype": str(dtype)},
            "use_prefill_config": {"scalar": prefill},
            "k_scale": {"scalar": 0.5 if fp8 else 1.0},
            "v_scale": {"scalar": 2.0 if fp8 else 1.0},
        },
        "outputs": {"out": {"shape": q_shape, "dtype": str(dtype)}},
        "dims": {
            "shape": key_shape(binding),
            "head_dim": _HEAD_DIM,
            "selection_width": _SELECTION_WIDTH,
            "selection": "full_width_random",
        },
    }
