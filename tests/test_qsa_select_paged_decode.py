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

from unittest.mock import patch

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

qsa_select_paged_decode = flaggems_vllm.qsa_select_paged_decode

pytestmark = pytest.mark.qsa_select_paged_decode


def _hygon_qsa_select_paged_decode(
    q,
    k_cache,
    page_table,
    visible_blocks,
    token_topk,
    compress_ratio,
    decode_query_len,
    block_indices,
):
    from vllm.models.qwen4_exp.amd.ops.qsa import qsa_mqa_paged

    rows = q.shape[0]
    requests = rows // decode_query_len
    token_to_req = torch.arange(requests, device=q.device).repeat_interleave(
        decode_query_len
    )
    sequence_lengths = (
        visible_blocks.view(requests, decode_query_len).max(dim=1).values
        * compress_ratio
    ).to(torch.int32)
    query_positions = visible_blocks * compress_ratio - 1
    logits, actual_visible = qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        actual_visible,
        block_indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        token_topk // compress_ratio,
    )


def _ppu_device_capability(cls, device_id=0):
    from vllm.platforms.interface import DeviceCapability

    major, minor = torch.cuda.get_device_capability(device_id)
    return DeviceCapability(major=major, minor=minor)


try:
    if flaggems_vllm.vendor_name == "metax":
        from vllm_metax.models.qwen4_exp.metax.ops.qsa_indexer_fallback import (
            qsa_select_paged_decode as vllm_qsa_select_paged_decode,
        )
    elif flaggems_vllm.vendor_name == "hygon":
        from vllm.platforms import current_platform

        current_platform.import_kernels()
        vllm_qsa_select_paged_decode = _hygon_qsa_select_paged_decode
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
            qsa_select_paged_decode as vllm_qsa_select_paged_decode,
        )

    _HAS_VLLM_QSA = True
    _VLLM_QSA_ERROR = ""
except Exception as exc:
    vllm_qsa_select_paged_decode = None
    _HAS_VLLM_QSA = False
    _VLLM_QSA_ERROR = f"{type(exc).__name__}: {exc}"


@pytest.fixture(scope="module", autouse=True)
def _scoped_ppu_device_capability():
    if flaggems_vllm.vendor_name != "thead":
        yield
        return

    from vllm.platforms import current_platform

    assert current_platform.is_ppu(), "run PPU tests with VLLM_PLUGINS=ppu"
    current_platform.import_kernels()
    with patch.object(
        type(current_platform),
        "get_device_capability",
        classmethod(_ppu_device_capability),
    ):
        yield


device = flaggems_vllm.device
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()
if not _HAS_DEVICE:
    _HAS_DEVICE = bool(flaggems_vllm.runtime.torch_device_fn.device_count())

PAGE_SIZE = 16
HEAD_DIM = 128
DTYPES = (torch.bfloat16, torch.float8_e4m3fn)

# vLLM QSA decode coverage: requests, decode query length, heads, max pages.
VLLM_DECODE_SHAPES = (
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
VLLM_BASELINE_SHAPES = {(4, 1, 4, 128), (16, 2, 4, 128), (80, 1, 4, 128)}

# The same active set as benchmark/test_qsa_select_paged_decode.py:
# (requests, physical blocks). Each request maps 88 logical pages of 196
# compressed tokens onto the physical block pool.
PRODUCTION_SCENARIOS = [
    (1, 2712),
    (8, 2712),
    (32, 2712),
    (53, 2712),
    (56, 2712),
    (64, 2712),
    (128, 2712),
    (256, 2712),
    (320, 2712),
    (512, 2712),
    (32, 512),
    (128, 512),
    (256, 512),
    (512, 512),
]
if cfg.QUICK_MODE:
    PRODUCTION_SCENARIOS = PRODUCTION_SCENARIOS[:1]

GENERAL_CASES = [
    (shape, 2048, dtype) for shape in VLLM_DECODE_SHAPES for dtype in DTYPES
]
if not cfg.QUICK_MODE:
    GENERAL_CASES += [
        ((4, 1, 4, 128), token_topk, dtype)
        for token_topk in (4096, 8192)
        for dtype in DTYPES
    ]
PRODUCTION_CASES = [
    (scenario, dtype) for scenario in PRODUCTION_SCENARIOS for dtype in DTYPES
]


def _to_dtype(value, dtype):
    if dtype == torch.float8_e4m3fn and flaggems_vllm.vendor_name == "metax":
        result = torch.empty(value.shape, dtype=dtype, device=value.device)
        result.copy_(value.cpu().to(dtype))
        return result
    return value.to(dtype)


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
    """
    Build synthetic inputs with per-row compressed visibility.

    With DQL>1, later query rows intentionally see one extra compressed column,
    exercising speculative/per-row visibility rather than shared-KV decode.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    rows = num_requests * decode_query_len
    block_topk = token_topk // compress_ratio
    columns = max_pages * page_size
    if num_blocks is None:
        num_blocks = 2712 if page_size == 196 else num_requests * max_pages

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
    q = _to_dtype(q, dtype)
    k_cache = _to_dtype(k_cache, dtype)

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

    block_indices = torch.empty((rows, block_topk), dtype=torch.int32, device=device)
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
    q, k_cache, page_table, visible_blocks, _, _, decode_query_len, _ = args
    rows, _, head_dim = q.shape
    page_size = k_cache.shape[1]
    columns = page_table.shape[1] * page_size
    k_flat = k_cache.view(k_cache.shape[0], page_size, head_dim)
    page_table = page_table.long()
    qf = q.float()
    scores = torch.empty(rows, columns, dtype=torch.float32, device=q.device)
    for row in range(rows):
        keys = k_flat[page_table[row // decode_query_len]]
        keys = keys.reshape(columns, head_dim).float()
        scores[row] = torch.relu(keys @ qf[row].T).sum(-1)
        scores[row, int(visible_blocks[row].item()) :] = float("-inf")
    return scores


def assert_structural_full(block_indices, visible_blocks):
    """Check rows with visible >= block_topk and no -1 padding."""
    assert block_indices.dtype == torch.int32
    for row in range(block_indices.shape[0]):
        indices = block_indices[row]
        visible = int(visible_blocks[row].item())
        assert int(indices.min()) >= 0, f"row {row}: unwritten entry"
        assert int(indices.max()) < visible, f"row {row}: index out of range"
        assert indices.unique().numel() == indices.numel(), f"row {row}: duplicates"


def assert_score_capture_full(block_indices, scores, block_topk, min_ratio):
    """Check score capture for full rows without -1 padding."""
    assert (block_indices >= 0).all(), "use assert_rows for underfilled rows"
    top_values = torch.topk(scores, block_topk, dim=1).values.sum(-1)
    selected = torch.gather(scores, 1, block_indices.long()).sum(-1)
    ratios = selected / top_values.clamp_min(1e-6)
    minimum = ratios.min().item()
    assert minimum >= min_ratio, (
        f"score capture ratio {minimum:.6f} < {min_ratio} "
        f"(worst row {ratios.argmin().item()})"
    )


def assert_rows(block_indices, visible_blocks, scores, block_topk, min_ratio):
    for row_index in range(block_indices.shape[0]):
        visible = int(visible_blocks[row_index].item())
        row = block_indices[row_index]
        selected = row[row >= 0]
        assert selected.numel() == min(visible, block_topk), f"row {row_index}: count"
        if selected.numel():
            assert int(selected.min()) >= 0 and int(selected.max()) < visible
            assert (
                selected.unique().numel() == selected.numel()
            ), f"row {row_index}: duplicates"
        if visible < block_topk:
            assert int(row[visible:].min().item()) == -1, f"row {row_index}: padding"
            # Every visible block is required; order is unspecified.
            expected = torch.arange(visible, device=device, dtype=torch.int64)
            utils.gems_assert_equal(
                utils.to_reference(torch.sort(selected.long()).values),
                utils.to_reference(expected),
            )
        else:
            top_sum = torch.topk(scores[row_index], block_topk).values.sum()
            selected_sum = scores[row_index][selected.long()].sum()
            ratio = (selected_sum / top_sum.clamp_min(1e-6)).item()
            assert ratio >= min_ratio, f"row {row_index}: capture {ratio:.6f}"


def assert_vendor_agreement(args, scores):
    if not _HAS_VLLM_QSA:
        pytest.skip(f"requires the vendor vLLM baseline ({_VLLM_QSA_ERROR})")
    decode_query_len = args[6]
    if flaggems_vllm.vendor_name == "thead" and decode_query_len != 1:
        pytest.skip(
            "vendor PPU score is incorrect for DQL>1; FlagGems is already "
            "validated against the FP32 reference in this test"
        )

    vendor_args = list(args)
    vendor_args[-1] = torch.empty_like(args[-1])
    vllm_qsa_select_paged_decode(*vendor_args)

    block_topk = args[4] // args[5]
    min_ratio = 1 - 1e-4 if args[0].dtype == torch.bfloat16 else 0.99
    assert_structural_full(vendor_args[-1], args[3])
    assert_score_capture_full(vendor_args[-1], scores, block_topk, min_ratio)

    selected = torch.gather(scores, 1, args[-1].long()).sum(-1)
    vendor_selected = torch.gather(scores, 1, vendor_args[-1].long()).sum(-1)
    tolerance = 1e-3 if args[0].dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        selected,
        vendor_selected,
        rtol=tolerance,
        atol=1e-2 if args[0].dtype == torch.bfloat16 else 1e-1,
        msg="captured-score mismatch between FlagGems and the vendor baseline",
    )


@pytest.mark.parametrize("shape, token_topk, dtype", GENERAL_CASES)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_accuracy(shape, token_topk, dtype):
    num_requests, decode_query_len, num_heads, max_pages = shape
    args = make_inputs(
        num_requests,
        decode_query_len,
        num_heads,
        max_pages,
        token_topk,
        4,
        dtype,
        seed=1000 + sum(shape) + token_topk,
    )
    qsa_select_paged_decode(*args)

    block_topk = token_topk // 4
    scores = reference_scores(args)
    min_ratio = 1 - 1e-4 if dtype == torch.bfloat16 else 0.99
    assert_structural_full(args[-1], args[3])
    assert_score_capture_full(args[-1], scores, block_topk, min_ratio)

    if dtype == torch.bfloat16 and shape in VLLM_BASELINE_SHAPES:
        assert_vendor_agreement(args, scores)


@pytest.mark.parametrize("scenario, dtype", PRODUCTION_CASES)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_production_vs_vendor(scenario, dtype):
    num_requests, num_blocks = scenario
    args = make_inputs(
        num_requests,
        1,
        4,
        88,
        2048,
        4,
        dtype,
        seed=100 + num_requests + num_blocks,
        min_visible=8192,
        page_size=196,
        num_blocks=num_blocks,
    )
    qsa_select_paged_decode(*args)

    scores = reference_scores(args)
    min_ratio = 1 - 1e-4 if dtype == torch.bfloat16 else 0.99
    assert_structural_full(args[-1], args[3])
    assert_score_capture_full(args[-1], scores, 512, min_ratio)
    assert_vendor_agreement(args, scores)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_strict_sets():
    args = make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=29, min_visible=1024)
    qsa_select_paged_decode(*args)
    assert_structural_full(args[-1], args[3])
    scores = reference_scores(args)
    boundary = torch.topk(scores, 513, dim=1).values
    gap = (boundary[:, -2] - boundary[:, -1]).min().item()
    if gap < 1e-3:
        pytest.skip(f"top-k boundary gap {gap:.2e} is too close for strict sets")
    reference = torch.topk(scores, 512, dim=1).indices
    utils.gems_assert_equal(
        utils.to_reference(torch.sort(args[-1].long(), dim=1).values),
        utils.to_reference(torch.sort(reference, dim=1).values),
    )


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_underfilled_rows():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=7))
    args[3] = torch.tensor([0, 32, 199, 200], dtype=torch.int32, device=device)
    args[-1] = torch.empty_like(args[-1])
    qsa_select_paged_decode(*args)
    assert_rows(args[-1], args[3], reference_scores(args), 512, 1 - 1e-4)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_boundary_visible():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=7))
    args[3] = torch.tensor([511, 512, 513, 1024], dtype=torch.int32, device=device)
    args[-1] = torch.empty_like(args[-1])
    qsa_select_paged_decode(*args)
    assert_rows(args[-1], args[3], reference_scores(args), 512, 1 - 1e-4)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_mixed_fill_rows():
    args = list(make_inputs(16, 1, 4, 128, 2048, 4, torch.bfloat16, seed=11))
    visible_blocks = args[3].clone()
    visible_blocks[::2] = torch.randint(1, 512, (8,), device=device, dtype=torch.int32)
    args[3] = visible_blocks
    args[-1] = torch.empty_like(args[-1])
    qsa_select_paged_decode(*args)
    assert_rows(args[-1], args[3], reference_scores(args), 512, 1 - 1e-4)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_unsupported_dtype():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[0] = args[0].to(torch.float16)
    args[1] = args[1].to(torch.float16)
    with pytest.raises(NotImplementedError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_unsupported_block_topk():
    args = make_inputs(4, 1, 4, 512, 32768, 4, torch.bfloat16, seed=0)
    with pytest.raises(NotImplementedError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_bad_output_shape():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[-1] = torch.empty((4, 256), dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_non_contiguous_output():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    output = torch.empty((4, 1024), dtype=torch.int32, device=device)
    args[-1] = output[:, ::2]
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(
    (not _HAS_DEVICE) or flaggems_vllm.vendor_name != "thead",
    reason="requires an accelerator and the PPU-specific dense-output guard",
)
def test_qsa_select_paged_decode_non_dense_output():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    output = torch.empty((4, 1024), dtype=torch.int32, device=device)
    args[-1] = output[:, :512]
    assert args[-1].stride(-1) == 1
    assert not args[-1].is_contiguous()
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(
    (not _HAS_DEVICE) or flaggems_vllm.vendor_name != "thead",
    reason="requires an accelerator and the PPU-specific empty-cache guard",
)
def test_qsa_select_paged_decode_empty_k_cache():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[1] = args[1][:0]
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_invalid_token_topk():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[4] = 2047
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_invalid_decode_query_len():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[6] = 3
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_bad_page_table_rows():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[2] = args[2][:-1]
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_select_paged_decode_bad_visible_shape():
    args = list(make_inputs(4, 1, 4, 128, 2048, 4, torch.bfloat16, seed=0))
    args[3] = args[3][:-1]
    with pytest.raises(ValueError):
        qsa_select_paged_decode(*args)
