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

import inspect

import pytest
import torch
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.persistent_topk import persistent_topk

from . import conftest as cfg

device = flaggems_vllm.device


def _has_histogram_mask():
    if not hasattr(tl, "histogram"):
        return False
    try:
        return "mask" in inspect.signature(tl.histogram).parameters
    except (ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_histogram_mask(),
    reason="tl.histogram with mask parameter not available",
)

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = True
except (ImportError, AttributeError):
    pass

STRIDE = 262144
K = 512

if cfg.QUICK_MODE:
    SHAPES = [(1, 32773, 32773), (16, 262144, 1048576), (512, 262144, 1048576)]
    DATA_TYPES = ["random"]
else:
    SHAPES = [
        (1, 1, 1),
        (1, 4102, 4102),
        (10, 1055, 1055),
        (12, 4105, 4105),
        (20, 4107, 4107),
        (28, 4109, 4109),
        (1, 10000, 10000),
        (1, 16384, 16384),
        (1, 20000, 20000),
        (1, 24576, 24576),
        (1, 32768, 32768),
        (4, 16384, 16384),
        (8, 16384, 16384),
        (12, 20000, 20000),
        (16, 32768, 32768),
        (24, 32768, 32768),
        (32, 32768, 32768),
        (1, 32773, 32773),
        (1, 32774, 32774),
        (1, 262144, 1048576),
        (2, 262144, 1048576),
        (4, 262144, 1048576),
        (8, 262144, 1048576),
        (16, 262144, 1048576),
        (24, 262144, 1048576),
        (32, 262144, 1048576),
        (496, 262144, 1048576),
        (512, 262144, 1048576),
    ]
    DATA_TYPES = ["random", "many_ties"]


def _padded_logits(num_rows, seq_len, data_type):
    logits = torch.full(
        (num_rows, STRIDE), float("-inf"), dtype=torch.float32, device=device
    )
    if data_type == "random":
        logits[:, :seq_len] = torch.randn(num_rows, seq_len, device=device)
    elif data_type == "many_ties":
        logits[:, :seq_len] = (
            torch.randint(0, 10, (num_rows, seq_len), device=device).float() / 10.0
        )
    return logits


def _selected_values(logits, indices):
    num_rows = indices.shape[0]
    vals = []
    for i in range(num_rows):
        valid = indices[i][indices[i] >= 0].long()
        if valid.numel() > 0:
            vals.append(logits[i].gather(0, valid).sort(descending=True)[0])
    if vals:
        return torch.cat(vals)
    return torch.empty(0, device=device)


def _torch_ref(logits, seq_lens, top_k):
    num_rows = logits.shape[0]
    ref = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    for i in range(num_rows):
        k = min(top_k, seq_lens[i])
        ref[i, :k] = logits[i, : seq_lens[i]].topk(k, dim=-1)[1]
        ref[i, k:] = -1
    return ref


def _vllm_persistent_topk(logits, seq_lens, max_seq_len, top_k):
    num_rows = logits.shape[0]
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    torch.ops._C.persistent_topk(
        logits, lengths, indices, workspace, top_k, max_seq_len
    )
    return indices


def _gems_decode(logits, seq_lens, top_k, max_seq_len=None):
    num_rows = logits.shape[0]
    n_blocks = (logits.shape[1] + 4096 - 1) // 4096
    ws_bytes = num_rows * (n_blocks * 256 + 4 + 2) * 4
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    persistent_topk(
        logits, lengths, indices, workspace, k=top_k, max_seq_len=max_seq_len
    )
    return indices


@pytest.mark.persistent_topk
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.parametrize("num_rows, seq_len, max_seq_len", SHAPES)
@pytest.mark.parametrize("data_type", DATA_TYPES)
@torch.inference_mode()
def test_persistent_topk_cross_agreement(num_rows, seq_len, max_seq_len, data_type):
    seq_lens = [seq_len] * num_rows
    logits = _padded_logits(num_rows, seq_len, data_type)

    p = _vllm_persistent_topk(logits.clone(), seq_lens, max_seq_len, K)
    g = _gems_decode(logits.clone(), seq_lens, K, max_seq_len=max_seq_len)

    vals_p = _selected_values(logits, p)
    vals_g = _selected_values(logits, g)
    torch.testing.assert_close(vals_p, vals_g, rtol=1e-4, atol=1e-4)


@pytest.mark.persistent_topk
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.parametrize("num_rows, seq_len, max_seq_len", SHAPES)
@pytest.mark.parametrize("data_type", DATA_TYPES)
@torch.inference_mode()
def test_persistent_topk_vs_torch(num_rows, seq_len, max_seq_len, data_type):
    seq_lens = [seq_len] * num_rows
    logits = _padded_logits(num_rows, seq_len, data_type)

    g = _gems_decode(logits.clone(), seq_lens, K, max_seq_len=max_seq_len)
    t = _torch_ref(logits, seq_lens, K)

    vals_g = _selected_values(logits, g)
    vals_t = _selected_values(logits, t)
    torch.testing.assert_close(vals_g, vals_t, rtol=1e-4, atol=1e-4)


HETERO_CASES = [
    (1, [1, 1, 1, 1], "all_trivial"),
    (1055, [1, 256, 515, 1055], "p1024_range"),
    (4120, [4120], "p4096_single"),
    (16390, [16383, 16384, 16385, 16390], "all_medium"),
    (20000, [8000, 16384, 20000], "decode_medium_mix"),
    (32772, [32767, 32768, 32769], "medium_large_boundary"),
    (32774, [32773, 32774], "p32768_multi"),
    (1048576, [262144], "full_single"),
    (1048576, [1, 32773, 32773, 262144], "full_mixed"),
    (1048576, [1, 1, 1, 262144], "full_one_large"),
    (32, [1, 32], "rare_32"),
]


def _make_hetero_logits(lengths_list):
    num_rows = len(lengths_list)
    logits = torch.full(
        (num_rows, STRIDE), float("-inf"), dtype=torch.float32, device=device
    )
    for i, sl in enumerate(lengths_list):
        logits[i, :sl] = torch.randn(sl, device=device)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
    return logits, lengths


@pytest.mark.persistent_topk
@pytest.mark.parametrize(
    "max_seq_len,lengths_list,case_id", HETERO_CASES, ids=lambda x: x
)
@torch.inference_mode()
def test_persistent_topk_heterogeneous(max_seq_len, lengths_list, case_id):
    logits, lengths = _make_hetero_logits(lengths_list)
    num_rows = len(lengths_list)

    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)

    persistent_topk(logits, lengths, indices, ws, k=K, max_seq_len=max_seq_len)

    ref = _torch_ref(logits, lengths_list, K)
    vals_g = _selected_values(logits, indices)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


@pytest.mark.persistent_topk
@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize(
    "lengths_list,max_seq_len",
    [
        ([262144], 1048576),
        ([32773, 32774], 32774),
    ],
)
@torch.inference_mode()
def test_persistent_topk_k_values(k, lengths_list, max_seq_len):
    logits, lengths = _make_hetero_logits(lengths_list)
    num_rows = len(lengths_list)

    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    indices = torch.empty((num_rows, k), dtype=torch.int32, device=device)

    persistent_topk(logits, lengths, indices, ws, k=k, max_seq_len=max_seq_len)

    ref = _torch_ref(logits, lengths_list, k)
    vals_g = _selected_values(logits, indices)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


@pytest.mark.persistent_topk
@torch.inference_mode()
def test_persistent_topk_mtp():
    next_n = 2
    lengths_2d = torch.tensor([[32773, 32774]], dtype=torch.int32, device=device)
    seq_lens = lengths_2d.reshape(-1)
    row_ends = [
        int(seq_lens[0] - next_n + 0 + 1),  # pid=0: 32772
        int(seq_lens[0] - next_n + 1 + 1),  # pid=1: 32773
    ]
    num_rows = len(row_ends)

    logits = torch.full(
        (num_rows, STRIDE), float("-inf"), dtype=torch.float32, device=device
    )
    for i, sl in enumerate(row_ends):
        logits[i, :sl] = torch.randn(sl, device=device)

    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    persistent_topk(logits, lengths_2d, indices, ws, k=K, max_seq_len=32774)

    ref = _torch_ref(logits, row_ends, K)
    vals_g = _selected_values(logits, indices)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


BOUNDARY_CASES = [
    ([262144] * 5, 262144, "5_rows_no_cluster"),
    ([65535], 65535, "max_65535_no_cluster"),
    ([65536], 65536, "max_65536_cluster"),
]


@pytest.mark.persistent_topk
@pytest.mark.parametrize("lengths_list,max_seq_len,case_id", BOUNDARY_CASES)
@torch.inference_mode()
def test_persistent_topk_boundary(lengths_list, max_seq_len, case_id):
    logits, lengths = _make_hetero_logits(lengths_list)
    num_rows = len(lengths_list)

    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)

    persistent_topk(logits, lengths, indices, ws, k=K, max_seq_len=max_seq_len)

    ref = _torch_ref(logits, lengths_list, K)
    vals_g = _selected_values(logits, indices)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


def _make_hetero_logits_dist(lengths_list, dist="random"):
    num_rows = len(lengths_list)
    logits = torch.full(
        (num_rows, STRIDE), float("-inf"), dtype=torch.float32, device=device
    )
    for i, sl in enumerate(lengths_list):
        if dist == "sorted_asc":
            logits[i, :sl] = torch.arange(sl, dtype=torch.float32, device=device)
        elif dist == "sorted_desc":
            logits[i, :sl] = torch.arange(sl, 0, -1, dtype=torch.float32, device=device)
        elif dist == "all_same":
            logits[i, :sl] = torch.ones(sl, dtype=torch.float32, device=device)
        elif dist == "small_differences":
            base = torch.randn(sl, dtype=torch.float32, device=device)
            noise = torch.randn(sl, dtype=torch.float32, device=device) * 1e-6
            logits[i, :sl] = base + noise
        else:
            logits[i, :sl] = torch.randn(sl, dtype=torch.float32, device=device)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
    return logits, lengths


@pytest.mark.persistent_topk
@torch.inference_mode()
def test_persistent_topk_random_stress():
    max_seq_len = 1048576
    for seed in range(3):
        torch.manual_seed(seed)
        B = torch.randint(1, 32, (1,)).item()
        seq_lens = torch.randint(100, 262144, (B,)).tolist()

        logits = torch.full(
            (B, STRIDE), float("-inf"), dtype=torch.float32, device=device
        )
        for i, sl in enumerate(seq_lens):
            logits[i, :sl] = torch.randn(sl, dtype=torch.float32, device=device)
        lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)

        ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
        indices = torch.empty((B, K), dtype=torch.int32, device=device)

        persistent_topk(logits, lengths, indices, ws, k=K, max_seq_len=max_seq_len)

        ref = _torch_ref(logits, seq_lens, K)
        vals_g = _selected_values(logits, indices)
        vals_t = _selected_values(logits, ref)
        torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


DATA_DIST_CASES = [
    ([5000, 10000], "sorted_asc"),
    ([5000, 10000], "sorted_desc"),
    ([5000, 10000], "all_same"),
    ([5000, 10000], "small_differences"),
]


@pytest.mark.persistent_topk
@pytest.mark.parametrize("lengths_list,dist", DATA_DIST_CASES)
@torch.inference_mode()
def test_persistent_topk_data_distributions(lengths_list, dist):
    logits, lengths = _make_hetero_logits_dist(lengths_list, dist)
    num_rows = len(lengths_list)

    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)

    persistent_topk(logits, lengths, indices, ws, k=K, max_seq_len=max(lengths_list))

    ref = _torch_ref(logits, lengths_list, K)
    vals_g = _selected_values(logits, indices)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)
