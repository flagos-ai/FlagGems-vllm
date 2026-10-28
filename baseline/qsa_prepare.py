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

"""CUDA baseline collection contract for qsa_prepare.

Consumed by tools/collect_baseline_nvidia.py: grid() enumerates the
collection points, build_inputs() builds the real call arguments, config()
describes them for shape matching, and key_shape() names each JSON entry.

The shape grid must stay identical to benchmark/test_qsa_prepare.py so a
collected baseline can be matched by every platform.

Unlike most operators this one takes about two dozen operands whose shapes
are derived from a batch layout (seq_lens / query_lens / history_lens) rather
than being independent.  Both this contract and the benchmark therefore build
their inputs through ``tests.test_qsa_prepare.QsaPrepareCase``; run the
collector with that repository on PYTHONPATH.
"""

import torch

OP_NAME = "qsa_prepare"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# Model constants, mirrored from tests/test_qsa_prepare.py.
HEAD_DIM = 128
QUERY_HEADS = 4
COMPRESS_RATIO = 4
COMPRESSED_PAGE_SIZE = 4
COS_SIN_ROWS = 32768
MAIN_QUERY_HEADS = 6
MAIN_KV_HEADS = 2
MAIN_HEAD_DIM = 256
MAIN_PAGE_SIZE = 16
NORM_EPS = 1e-6
MAIN_EPS = 1e-6

FP8 = torch.float8_e4m3fn


def _single(tokens, **overrides):
    """One single-request prefill of ``tokens`` queries with no history."""

    return {
        "seq_lens": [tokens],
        "query_lens": [tokens],
        "history_lens": [0],
        "overrides": overrides,
    }


def _decode(n_requests, seq_len=512, history=8, **overrides):
    """One pure decode step: ``n_requests`` single-token requests behind histories."""

    return {
        "seq_lens": [seq_len] * n_requests,
        "query_lens": [1] * n_requests,
        "history_lens": [history] * n_requests,
        "overrides": overrides,
    }


def _steady_state(n_decode, n_prefill, chunk=512, decode_seq=512):
    """A steady-state step: decode requests and prefill chunks in one batch."""

    return {
        "seq_lens": [decode_seq] * n_decode + [chunk] * n_prefill,
        "query_lens": [1] * n_decode + [chunk] * n_prefill,
        "history_lens": [8] * n_decode + [0] * n_prefill,
        "overrides": {},
    }


# Mirrors benchmark/test_qsa_prepare.py::SHAPES order and values.
_LAYOUTS = [
    _steady_state(53, 1, chunk=16331, decode_seq=32768),
    _steady_state(54, 1, chunk=16330, decode_seq=32768),
    _steady_state(56, 1, chunk=16328, decode_seq=32768),
    _steady_state(32, 1, chunk=16352, decode_seq=32768),
    _steady_state(16, 1, chunk=16368, decode_seq=32768),
    _steady_state(8, 1, chunk=16376, decode_seq=32768),
    _single(16384),
    _single(6173),
    _single(2372),
    _decode(53, seq_len=32768),
    _decode(54, seq_len=32768),
    {
        "seq_lens": [260, 259, 138],
        "query_lens": [1, 1, 37],
        "history_lens": [8, 8, 8],
        "overrides": {},
    },
    _decode(16, state_size=8),
    _single(4097, indexer_dtype=FP8),
]


def _geometry(layout):
    """Derive every tensor dimension from one batch layout, no allocation."""

    seq_lens = layout["seq_lens"]
    query_lens = layout["query_lens"]
    overrides = layout["overrides"]

    num_tokens = sum(query_lens)
    num_requests = len(seq_lens)
    state_size = overrides.get("state_size", 4)
    is_2d_positions = overrides.get("is_2d_positions", True)
    cache_rope_positions = overrides.get("cache_rope_positions", True)
    indexer_dtype = overrides.get("indexer_dtype", torch.bfloat16)

    compressed_counts = [
        (seq_len // COMPRESS_RATIO + COMPRESSED_PAGE_SIZE - 1) // COMPRESSED_PAGE_SIZE
        for seq_len in seq_lens
    ]
    raw_width = HEAD_DIM + 12 if cache_rope_positions else HEAD_DIM

    return {
        "num_tokens": num_tokens,
        "num_requests": num_requests,
        "state_size": state_size,
        "raw_width": raw_width,
        "num_compressed_blocks": sum(compressed_counts),
        "num_blocks": num_tokens // MAIN_PAGE_SIZE + 2,
        "max_k_work": (num_tokens + (COMPRESS_RATIO - 1) * num_requests)
        // COMPRESS_RATIO,
        "positions_shape": [3, num_tokens] if is_2d_positions else [num_tokens],
        "indexer_dtype": indexer_dtype,
    }


def _with_geometry(layout):
    binding = dict(layout)
    binding["geometry"] = _geometry(layout)
    return binding


def grid():
    return [_with_geometry(layout) for layout in _LAYOUTS]


def native():
    """Return vLLM's qwen4_exp implementation, or None when unavailable."""

    try:
        from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import qsa_prepare
    except ImportError:
        return None
    return qsa_prepare


def _make_case(binding, dtype, device):
    from tests.test_qsa_prepare import QsaPrepareCase

    # A binding may carry its own indexer_dtype (the fp8 layout); the collector's
    # dtype loop only varies the element type, not the cache format.
    overrides = dict(binding["overrides"])
    if dtype == FP8:
        overrides["indexer_dtype"] = FP8
    return QsaPrepareCase(
        seq_lens=binding["seq_lens"],
        query_lens=binding["query_lens"],
        history_lens=binding["history_lens"],
        device=device,
        **overrides,
    )


def build_inputs(binding, dtype, device):
    """Same operands the benchmark hands to both implementations."""

    case = _make_case(binding, dtype, device)
    args = case._launch_args(case.outputs())
    kwargs = {
        "compress_ratio": COMPRESS_RATIO,
        "mrope_section": (11, 11, 10),
        "rope_pos_offset": HEAD_DIM,
        **case.main,
    }
    return args, kwargs


def key_shape(binding):
    """Semantic key; the first tensor shape repeats across T=16384 layouts."""

    geometry = binding["geometry"]
    suffix = "_fp8" if geometry["indexer_dtype"] == FP8 else ""
    return "req%d_tok%d_ss%d%s" % (
        geometry["num_requests"],
        geometry["num_tokens"],
        geometry["state_size"],
        suffix,
    )


def config(binding, dtype):
    """Describe the operands that the tensor-signature fallback matches on.

    ``inputs`` must hold exactly the tensors that
    ``benchmark/test_qsa_prepare.py::record_shapes`` reports — the lookup builds
    its index from this set and compares it against the benchmark's multiset.
    The remaining operands are listed under ``dims`` for readability; they are
    derived from the same batch layout and cannot distinguish two entries.
    """

    g = binding["geometry"]
    num_tokens = g["num_tokens"]
    num_requests = g["num_requests"]
    state_size = g["state_size"]
    raw_width = g["raw_width"]
    num_compressed_blocks = g["num_compressed_blocks"]
    num_blocks = g["num_blocks"]
    max_k_work = g["max_k_work"]
    indexer = "torch.float8_e4m3fn" if g["indexer_dtype"] == FP8 else "torch.bfloat16"

    return {
        "inputs": {
            # Exactly the four tensors record_shapes() reports.
            "query": {
                "shape": [num_tokens, QUERY_HEADS * HEAD_DIM],
                "dtype": "torch.bfloat16",
            },
            "state_cache": {
                "shape": [num_requests, state_size, 1, raw_width],
                "dtype": "torch.bfloat16",
            },
            "main_qkv": {
                "shape": [
                    num_tokens,
                    2 * (MAIN_QUERY_HEADS + MAIN_KV_HEADS) * MAIN_HEAD_DIM,
                ],
                "dtype": "torch.bfloat16",
            },
            "main_kv_cache": {
                "shape": [num_blocks, MAIN_PAGE_SIZE, MAIN_KV_HEADS, 2 * MAIN_HEAD_DIM],
                "dtype": indexer,
            },
        },
        "outputs": {
            "query_out": {
                "shape": [num_tokens, QUERY_HEADS, HEAD_DIM],
                "dtype": indexer,
            },
            "raw_cache": {
                "shape": [num_requests, state_size, 1, raw_width],
                "dtype": "torch.bfloat16",
            },
            "main_q_out": {
                "shape": [num_tokens, MAIN_QUERY_HEADS, MAIN_HEAD_DIM],
                "dtype": "torch.bfloat16",
            },
            "main_gate_out": {
                "shape": [num_tokens, MAIN_QUERY_HEADS, MAIN_HEAD_DIM],
                "dtype": "torch.bfloat16",
            },
            "main_kv_cache": {
                "shape": [num_blocks, MAIN_PAGE_SIZE, MAIN_KV_HEADS, 2 * MAIN_HEAD_DIM],
                "dtype": indexer,
            },
        },
        "dims": {
            "num_tokens": num_tokens,
            "num_requests": num_requests,
            "state_size": state_size,
            "num_compressed_blocks": num_compressed_blocks,
            "max_k_work": max_k_work,
            # The remaining call operands, all derived from the batch layout.
            "key": [num_tokens, HEAD_DIM],
            "positions": g["positions_shape"],
            "cos_sin_cache": [COS_SIN_ROWS, HEAD_DIM // 2],
            "query_norm_weight": [HEAD_DIM],
            "key_norm_weight": [HEAD_DIM],
            "norm_eps": NORM_EPS,
            "state_slots": [num_tokens],
            "raw_block_table": [num_requests, 1],
            "query_start_loc": [num_requests + 1],
            "logical_positions": [num_tokens],
            "compressed_cache": [
                num_compressed_blocks,
                COMPRESSED_PAGE_SIZE,
                1,
                HEAD_DIM,
            ],
            "compressed_slots": [num_tokens],
            "key_work_metadata": [max_k_work, 2],
            "compress_ratio": COMPRESS_RATIO,
            "mrope_section": (11, 11, 10),
            "rope_pos_offset": HEAD_DIM,
            "main_q_norm_weight": [MAIN_HEAD_DIM],
            "main_k_norm_weight": [MAIN_HEAD_DIM],
            "main_eps": MAIN_EPS,
            "main_slot_mapping": [num_tokens],
            "main_k_scale": 0.5,
            "main_v_scale": 2.0,
        },
    }
