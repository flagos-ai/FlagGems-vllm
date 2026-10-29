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

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.qsa_select_paged_decode import qsa_select_paged_decode

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.qsa_select_paged_decode

try:
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        qsa_select_paged_decode as vllm_qsa_select_paged_decode,
    )

    _HAS_VLLM_QSA = True
except Exception:
    vllm_qsa_select_paged_decode = None
    _HAS_VLLM_QSA = False


device = flaggems_vllm.device
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()
if not _HAS_DEVICE:
    _HAS_DEVICE = bool(flaggems_vllm.runtime.torch_device_fn.device_count())

PAGE_SIZE = 16
HEAD_DIM = 128

# (num_requests, decode_query_len, num_heads, max_pages)
SHAPE_LIST = (
    [(4, 1, 4, 128)]
    if cfg.QUICK_MODE
    else [
        (1, 1, 4, 64),
        (4, 1, 4, 128),
        (16, 1, 4, 128),
        (16, 2, 4, 128),
        (8, 4, 4, 128),
        (8, 1, 8, 128),
        (80, 1, 4, 128),
        (64, 1, 4, 512),
        (8, 1, 4, 1024),
        (32, 1, 4, 2048),
    ]
)
# (token_topk, compress_ratio) — production-legal block_topk values are
# 512 and 2048 (vllm Qwen4Exp config validation)
TOPK_LIST = [(2048, 4)] if cfg.QUICK_MODE else [(2048, 4), (8192, 4)]


def make_inputs(
    num_requests,
    decode_query_len,
    num_heads,
    max_pages,
    token_topk,
    compress_ratio,
    dtype,
    seed,
    min_visible=None,
    max_visible=None,
    page_size=PAGE_SIZE,
    num_blocks=None,
):
    """Synthetic decode batch.

    ``num_blocks`` overrides the physical block-pool size directly (used by
    the fp8 production trace tests to model both the 2712-block shared pool
    and the 512-block small pool).  When omitted, ``page_size == 196``
    implies the 2712-block shared pool, and any other page_size gives every
    request private pages.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    rows = num_requests * decode_query_len
    block_topk = token_topk // compress_ratio
    columns = max_pages * page_size
    if num_blocks is None:
        if page_size == 196:
            num_blocks = 2712
        else:
            num_blocks = num_requests * max_pages

    q = torch.randn(
        rows,
        num_heads,
        HEAD_DIM,
        generator=g,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_blocks,
        page_size,
        1,
        HEAD_DIM,
        generator=g,
        device=device,
        dtype=torch.bfloat16,
    )
    if dtype != torch.bfloat16:
        q = q.to(dtype)
        k_cache = k_cache.to(dtype)

    perm = torch.randperm(num_blocks, generator=g, device=device)
    entry_ids = torch.arange(num_requests * max_pages, device=device)
    page_table = (
        perm[entry_ids % num_blocks].view(num_requests, max_pages).to(torch.int32)
    )

    lo = min_visible if min_visible is not None else block_topk
    vis_req = torch.randint(
        lo,
        columns + 1,
        (num_requests,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    visible_blocks = (
        vis_req.repeat_interleave(decode_query_len)
        + torch.arange(decode_query_len, device=device, dtype=torch.int32).repeat(
            num_requests
        )
    ).clamp_(max=columns)
    if max_visible is not None:
        visible_blocks = visible_blocks.clamp(max=max_visible)

    block_indices = torch.full((rows, block_topk), -1, dtype=torch.int32, device=device)
    return (
        q,
        k_cache,
        page_table,
        visible_blocks,
        token_topk,
        compress_ratio,
        decode_query_len,
        block_indices,
    )


@torch.no_grad()
def reference_scores(args):
    """fp32 golden scores [rows, columns]; -inf beyond each row's visibility."""
    (q, k_cache, page_table, visible_blocks, _, _, decode_query_len, _) = args
    rows, _, head_dim = q.shape
    page_size = k_cache.shape[1]
    columns = page_table.shape[1] * page_size

    k_flat = k_cache.view(k_cache.shape[0], page_size, head_dim)
    page_table = page_table.long()
    qf = q.float()
    scores = torch.empty(rows, columns, dtype=torch.float32, device=q.device)
    for r in range(rows):
        keys = k_flat[page_table[r // decode_query_len]]
        keys = keys.reshape(columns, head_dim).float()
        scores[r] = torch.relu(keys @ qf[r].T).sum(-1)
        scores[r, int(visible_blocks[r].item()) :] = float("-inf")
    return scores


def assert_structural(block_indices, visible_blocks):
    assert block_indices.dtype == torch.int32
    for r in range(block_indices.shape[0]):
        row = block_indices[r]
        vis = int(visible_blocks[r].item())
        assert int(row.min()) >= 0, f"row {r}: unwritten entry"
        assert int(row.max()) < vis, f"row {r}: index >= visible_blocks"
        assert row.unique().numel() == row.numel(), f"row {r}: duplicates"


def assert_score_capture(block_indices, scores, block_topk, min_ratio):
    """Semantic top-k check, robust to tie-breaking among equal scores.

    ``sum(scores[selected]) / sum(top-k scores)`` must be close to 1: any
    tie-broken selection among equal-scored columns captures the same total.
    """
    top_vals = torch.topk(scores, block_topk, dim=1).values.sum(-1)
    sel_vals = torch.gather(scores, 1, block_indices.long()).sum(-1)
    ratio = sel_vals / top_vals.clamp_min(1e-6)
    min_r = ratio.min().item()
    assert min_r >= min_ratio, (
        f"score capture ratio {min_r:.6f} < {min_ratio} "
        f"(worst row {ratio.argmin().item()})"
    )


# (shape, topk_cfg) pairs; combos whose column count is below block_topk
# are the underfilled case, covered by test_qsa_select_paged_decode_underfilled_rows
_ACCURACY_CASES = [
    (shape, topk_cfg)
    for shape in SHAPE_LIST
    for topk_cfg in TOPK_LIST
    if shape[3] * PAGE_SIZE >= topk_cfg[0] // topk_cfg[1]
]


@pytest.mark.parametrize("shape, topk_cfg", _ACCURACY_CASES)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_accuracy(shape, topk_cfg):
    num_requests, dql, num_heads, max_pages = shape
    token_topk, compress_ratio = topk_cfg
    block_topk = token_topk // compress_ratio
    args = make_inputs(
        num_requests,
        dql,
        num_heads,
        max_pages,
        token_topk,
        compress_ratio,
        torch.bfloat16,
        seed=hash((shape, topk_cfg)) % 10000,
    )
    qsa_select_paged_decode(*args)

    block_indices = args[-1]
    assert_structural(block_indices, args[3])
    scores = reference_scores(args)
    # bf16 tensor-core vs fp32 reference rounding may flip near-tied
    # boundary columns; the captured-score ratio absorbs those exchanges.
    assert_score_capture(block_indices, scores, block_topk, min_ratio=1 - 1e-4)


@pytest.mark.parametrize(
    "shape",
    (
        [(4, 1, 4, 128)]
        if cfg.QUICK_MODE
        else [
            (4, 1, 4, 128),
            (16, 2, 4, 128),
            (80, 1, 4, 128),
        ]
    ),
)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_strict_sets(shape):
    """Exact index-set equality, with min_visible=2K so the selection
    boundary stays far above the exact-zero score ties."""
    num_requests, dql, num_heads, max_pages = shape
    args = make_inputs(
        num_requests,
        dql,
        num_heads,
        max_pages,
        2048,
        4,
        torch.bfloat16,
        seed=hash(shape) % 10000,
        min_visible=1024,
    )
    qsa_select_paged_decode(*args)

    block_indices = args[-1]
    assert_structural(block_indices, args[3])
    scores = reference_scores(args)
    ref_idx = torch.topk(scores, 512, dim=1).indices
    res_sorted = torch.sort(block_indices.long(), dim=1).values
    ref_sorted = torch.sort(ref_idx, dim=1).values
    utils.gems_assert_equal(
        utils.to_reference(res_sorted), utils.to_reference(ref_sorted)
    )


@pytest.mark.parametrize("shape", SHAPE_LIST)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_accuracy_fp8(shape):
    num_requests, dql, num_heads, max_pages = shape
    token_topk, compress_ratio = 2048, 4
    block_topk = token_topk // compress_ratio
    args = make_inputs(
        num_requests,
        dql,
        num_heads,
        max_pages,
        token_topk,
        compress_ratio,
        torch.float8_e4m3fn,
        seed=hash(shape) % 10000,
    )
    qsa_select_paged_decode(*args)

    block_indices = args[-1]
    assert_structural(block_indices, args[3])
    # e4m3 quantization noise flips near-tied scores more often than bf16.
    assert_score_capture(
        block_indices, reference_scores(args), block_topk, min_ratio=0.99
    )


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_underfilled_rows():
    # Production short-context case (vllm reference semantics): rows whose
    # visible column count is below block_topk must select ALL visible
    # columns and pad the remaining slots with -1 (the real caller hands us
    # an uninitialized output buffer).
    args = list(
        make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=7, max_visible=200)
    )
    args[-1] = torch.empty_like(args[-1])  # uninitialized, like vllm
    qsa_select_paged_decode(*args)

    block_indices, visible_blocks = args[-1], args[3]
    for r in range(block_indices.shape[0]):
        v = int(visible_blocks[r].item())
        assert v < 512
        row = block_indices[r]
        sel = row[row >= 0]
        assert sel.numel() == v, f"row {r}: {sel.numel()} != visible {v}"
        expected = torch.arange(v, device=device, dtype=torch.int64)
        utils.gems_assert_equal(
            utils.to_reference(torch.sort(sel.long()).values),
            utils.to_reference(expected),
        )
        assert int(row[v:].min().item()) == -1, f"row {r}: padding != -1"


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_mixed_fill_rows():
    args = list(make_inputs(16, 1, 4, 128, 2048, 4, torch.bfloat16, seed=11))
    visible_blocks = args[3].clone()
    visible_blocks[::2] = torch.randint(1, 512, (8,), device=device, dtype=torch.int32)
    args[3] = visible_blocks
    args[-1] = torch.empty_like(args[-1])
    qsa_select_paged_decode(*args)

    block_indices = args[-1]
    scores = reference_scores(args)
    for r in range(16):
        v = int(visible_blocks[r].item())
        row = block_indices[r]
        sel = row[row >= 0]
        assert sel.numel() == min(v, 512)
        assert int(row[min(v, 512) :].min().item()) == -1 if v < 512 else True
        if v < 512:
            expected = torch.arange(v, device=device, dtype=torch.int64)
            utils.gems_assert_equal(
                utils.to_reference(torch.sort(sel.long()).values),
                utils.to_reference(expected),
            )
        else:
            # normal row: captured-score ratio against the fp32 reference
            top_sum = torch.topk(scores[r], 512).values.sum()
            sel_sum = scores[r][sel.long()].sum()
            assert (sel_sum / top_sum).item() >= 1 - 1e-4


# Production scenarios captured from Qwen3.8-Flash-Next serving runs
# (page_size=16 geometry, columns 512~8448; complementary to the fp8 table
# below).  (num_requests, max_pages, min_visible, page_size)
QSA_DECODE_SCENARIOS = [
    # columns=512; many rows under-filled (visible < 512)
    (1, 32, 256, 16),
    (19, 32, 256, 16),
    (30, 32, 256, 16),
    (48, 32, 256, 16),
    (63, 32, 256, 16),
    # columns=1280
    (1, 80, 1024, 16),
    (6, 80, 1024, 16),
    (10, 80, 1024, 16),
    (61, 80, 1024, 16),
    # columns=4352
    (1, 272, 4096, 16),
    (2, 272, 4096, 16),
    (5, 272, 4096, 16),
    (7, 272, 4096, 16),
    (10, 272, 4096, 16),
    (19, 272, 4096, 16),
    (30, 272, 4096, 16),
    (48, 272, 4096, 16),
    (61, 272, 4096, 16),
    (63, 272, 4096, 16),
    # columns=8448
    (1, 528, 8192, 16),
    (2, 528, 8192, 16),
    (3, 528, 8192, 16),
    (4, 528, 8192, 16),
    (5, 528, 8192, 16),
    (6, 528, 8192, 16),
    (7, 528, 8192, 16),
]
# Qwen3.8-Flash-Next-FP8-p32768d1024 production trace.  Every shape shares
# page_size=196, max_pages=88 (=> columns=17248), heads=4, head_dim=128,
# token_topk=2048, compress_ratio=4 (=> block_topk=512) and dql=1.  Only
# (num_requests, num_blocks) varies across two physical block-pool layouts:
# a shared pool of 2712 blocks and a small pool of 512 blocks.
#
# This table is a superset of the benchmark/test_qsa_select_paged_decode.py
# scenario table (it exercises every shape the benchmark covers, plus
# additional row counts from the trace).  It is NOT the full 149-shape
# trace: low-row shapes are sampled here, and the full trace remains in
# the source data file.
# (num_requests, num_blocks, max_pages, min_visible, page_size)
QSA_DECODE_SCENARIOS_FP8 = [
    # --- 2712-block shared pool: benchmark row sweep + the 104 point ---
    (1, 2712, 88, 8192, 196),
    (2, 2712, 88, 8192, 196),
    (4, 2712, 88, 8192, 196),
    (8, 2712, 88, 8192, 196),
    (16, 2712, 88, 8192, 196),
    (24, 2712, 88, 8192, 196),
    (32, 2712, 88, 8192, 196),
    (40, 2712, 88, 8192, 196),
    (48, 2712, 88, 8192, 196),
    (53, 2712, 88, 8192, 196),
    (56, 2712, 88, 8192, 196),
    (64, 2712, 88, 8192, 196),
    (96, 2712, 88, 8192, 196),
    (104, 2712, 88, 8192, 196),
    (128, 2712, 88, 8192, 196),
    (168, 2712, 88, 8192, 196),
    (256, 2712, 88, 8192, 196),
    (320, 2712, 88, 8192, 196),
    (451, 2712, 88, 8192, 196),
    (512, 2712, 88, 8192, 196),
    # --- 512-block small pool: matches the benchmark's small-pool layout ---
    (1, 512, 88, 8192, 196),
    (32, 512, 88, 8192, 196),
    (128, 512, 88, 8192, 196),
    (256, 512, 88, 8192, 196),
    (480, 512, 88, 8192, 196),
    (512, 512, 88, 8192, 196),
]


# (scen, dtype) pairs over both production tables: the page_size=16 table
# runs both dtypes, the fp8-trace table runs fp8 only.
_PRODUCTION_DTYPES = (
    [torch.bfloat16] if cfg.QUICK_MODE else [torch.bfloat16, torch.float8_e4m3fn]
)
_PRODUCTION_CASES = [
    (scen, dtype) for scen in QSA_DECODE_SCENARIOS for dtype in _PRODUCTION_DTYPES
] + [(scen, torch.float8_e4m3fn) for scen in QSA_DECODE_SCENARIOS_FP8]


@pytest.mark.parametrize(
    "scen, dtype",
    [((1, 32, 256, 16), torch.bfloat16)] if cfg.QUICK_MODE else _PRODUCTION_CASES,
)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_production_shapes(scen, dtype):
    """Every production scenario must satisfy the per-row contract."""
    if len(scen) == 5:
        num_requests, num_blocks, max_pages, min_visible, page_size = scen
    else:
        num_requests, max_pages, min_visible, page_size = scen
        num_blocks = None
    token_topk, compress_ratio = 2048, 4
    block_topk = token_topk // compress_ratio
    args = make_inputs(
        num_requests,
        1,
        4,
        max_pages,
        token_topk,
        compress_ratio,
        dtype,
        seed=hash(scen) % 10000,
        min_visible=min_visible,
        page_size=page_size,
        num_blocks=num_blocks,
    )
    qsa_select_paged_decode(*args)

    block_indices, visible_blocks = args[-1], args[3]
    scores = reference_scores(args)
    min_ratio = 0.99 if dtype != torch.bfloat16 else 1 - 1e-4
    _check_production_rows(block_indices, visible_blocks, scores, block_topk, min_ratio)


def _check_production_rows(
    block_indices, visible_blocks, scores, block_topk, min_ratio
):
    """Per-row contract shared by the production scenario tests."""
    for r in range(block_indices.shape[0]):
        v = int(visible_blocks[r].item())
        row = block_indices[r]
        sel = row[row >= 0]
        assert sel.numel() == min(v, block_topk), f"row {r}: count"
        if sel.numel():
            assert int(sel.min()) >= 0 and int(sel.max()) < v, f"row {r}: range"
            assert sel.unique().numel() == sel.numel(), f"row {r}: duplicates"
        if v < block_topk:
            assert int(row[v:].min().item()) == -1, f"row {r}: padding"
            expected = torch.arange(v, device=device, dtype=torch.int64)
            utils.gems_assert_equal(
                utils.to_reference(torch.sort(sel.long()).values),
                utils.to_reference(expected),
            )
        else:
            top_sum = torch.topk(scores[r], block_topk).values.sum()
            sel_sum = scores[r][sel.long()].sum()
            ratio = (sel_sum / top_sum.clamp_min(1e-6)).item()
            assert ratio >= min_ratio, f"row {r}: capture {ratio:.6f}"


@pytest.mark.parametrize(
    "shape",
    (
        [(4, 1, 4, 128)]
        if cfg.QUICK_MODE
        else [
            (4, 1, 4, 128),
            (16, 2, 4, 128),
            (80, 1, 4, 128),
        ]
    ),
)
@pytest.mark.skipif(
    (not _HAS_DEVICE) or (not _HAS_VLLM_QSA),
    reason="requires an accelerator and the vllm baseline",
)
def test_qsa_select_paged_decode_vs_vllm(shape):
    num_requests, dql, num_heads, max_pages = shape
    args = make_inputs(
        num_requests,
        dql,
        num_heads,
        max_pages,
        2048,
        4,
        torch.bfloat16,
        seed=hash(shape) % 10000,
    )
    qsa_select_paged_decode(*args)

    vllm_args = list(args)
    vllm_args[-1] = torch.full_like(args[-1], -1)
    vllm_qsa_select_paged_decode(*vllm_args)

    # Two independent top-k implementations may break equal-score ties
    # differently near the selection boundary, so raw index-set equality is
    # too strict.  Validate instead:
    #   1. Each side captures (almost) the full fp32 top-K score sum, i.e.
    #      it really is a top-K selection;
    #   2. The two captured score sums agree, i.e. any index-set difference
    #      is confined to a genuine equal-score tie.
    scores = reference_scores(args)
    top_sum = torch.topk(scores, 512, dim=1).values.sum(-1)

    res_sel = args[-1].long()
    ref_sel = vllm_args[-1].long()
    res_capture = torch.gather(scores, 1, res_sel).sum(-1)
    ref_capture = torch.gather(scores, 1, ref_sel).sum(-1)

    for name, capture in (("flaggems", res_capture), ("vllm", ref_capture)):
        ratio = (capture / top_sum.clamp_min(1e-6)).min().item()
        assert ratio >= 0.999, (
            f"{name} captured-score ratio {ratio:.6f} < 0.999 "
            f"for shape {shape}: not a real top-K selection"
        )

    torch.testing.assert_close(
        res_capture,
        ref_capture,
        rtol=1e-3,
        atol=1e-2,
        msg=f"captured-score mismatch between flaggems and vllm for {shape}",
    )


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_unsupported_dtype():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[0] = args[0].to(torch.float16)
    args[1] = args[1].to(torch.float16)
    with pytest.raises(NotImplementedError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_unsupported_block_topk():
    # token_topk // compress_ratio = 8192 exceeds the supported maximum 4096
    args = make_inputs(4, 1, 4, 512, 32768, 4, torch.bfloat16, seed=0)
    with pytest.raises(NotImplementedError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_bad_output_shape():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[-1] = torch.full((4, 256), -1, dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)
