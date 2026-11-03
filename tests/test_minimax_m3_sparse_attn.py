# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Functional tests for MiniMax M3 sparse attention against vLLM's native op.

The reference is vLLM's own ``minimax_m3_sparse_attn`` / ``..._decode``, per
this repository's convention of aligning with vLLM rather than PyTorch. Both
consume the packed page layout ``[num_blocks, num_kv_heads, 128, 2*head_dim]``.

Selected-block ids follow the indexer's contract: valid ids first, ``-1``
padding only as a trailing suffix (see ``index_topk``, which emits
``where(off_t < valid_blocks, off_t, -1)``). Interior holes are not generated
by the producer and are not part of the contract under test.
"""

import itertools

import pytest
import torch

import flaggems_vllm
from tests import conftest as cfg

PAGE = 128

native = pytest.importorskip(
    "vllm.models.minimax_m3.common.ops.sparse_attn",
    reason="requires the vLLM MiniMax M3 operator",
)


def _native_takes_packed_pages():
    """Whether the installed vLLM operator accepts the packed 4D KV layout.

    The operator exists in older vLLM builds too, but those index
    ``kv_cache.stride(4)`` against the 5D layout
    ``[num_blocks, 2, 128, num_kv_heads, head_dim]`` and raise on a packed 4D
    cache. Probing is cheaper than version-sniffing and does not assume how
    the layout change was versioned.
    """
    if not torch.cuda.is_available():
        return False
    d = 128
    q = torch.zeros((1, 8, d), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, PAGE, 2 * d), device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros((1, 1, 1), device="cuda", dtype=torch.int32)
    bt = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    lens = torch.ones((1,), device="cuda", dtype=torch.int32)
    try:
        native.minimax_m3_sparse_attn(
            q,
            kv,
            idx,
            bt,
            cu,
            lens,
            torch.zeros_like(lens),
            1,
            1,
            d**-0.5,
            torch.empty_like(q),
        )
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.minimax_m3_sparse_attn,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available(),
        reason="requires an NVIDIA GPU",
    ),
    pytest.mark.skipif(
        not _native_takes_packed_pages(),
        reason="installed vLLM operator predates the packed 4D KV layout",
    ),
]


def _topk_rows(total, kv_heads, topk, pages_per_request, mode, n_valid=None):
    """Ascending ids with trailing -1 padding, matching the indexer."""
    rows = []
    for _ in range(total * kv_heads):
        if mode == "compact":
            row = torch.arange(topk, device="cuda") + max(0, pages_per_request - topk)
        else:
            row = torch.randperm(pages_per_request, device="cuda")[:topk]
        rows.append(torch.sort(row).values)
    top = torch.stack(rows).view(kv_heads, total, topk).to(torch.int32)
    if n_valid is not None:
        top[:, :, n_valid:] = -1
    return top


def make_inputs(
    qlens=(17, 13),
    d=128,
    kv_heads=1,
    dtype=torch.bfloat16,
    mode="scattered",
    topk=16,
    context=4096,
    n_valid=None,
    seed=20261029,
):
    torch.manual_seed(seed)
    total = sum(qlens)
    batch = len(qlens)
    pages_per_request = (context + PAGE - 1) // PAGE
    pages = pages_per_request * batch
    q = torch.randn((total, kv_heads * 8, d), device="cuda", dtype=torch.bfloat16)
    kv = (torch.randn((pages, kv_heads, PAGE, 2 * d), device="cuda") * 0.3).to(dtype)
    bt = torch.randperm(pages, device="cuda").to(torch.int32).view(batch, -1)
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), device="cuda", dtype=torch.int32
    )
    seq = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    prefix = seq - torch.tensor(qlens, device="cuda", dtype=torch.int32)
    top = _topk_rows(total, kv_heads, topk, pages_per_request, mode, n_valid)
    return (q, kv, top, bt, cu, seq, prefix, max(qlens), kv_heads, d**-0.5)


def make_decode_inputs(
    num_reqs=4,
    decode_query_len=1,
    d=128,
    kv_heads=1,
    dtype=torch.bfloat16,
    mode="scattered",
    topk=16,
    context=4096,
    n_valid=None,
    seed=20261029,
):
    torch.manual_seed(seed)
    total = num_reqs * decode_query_len
    pages_per_request = (context + PAGE - 1) // PAGE
    pages = pages_per_request * num_reqs
    q = torch.randn((total, kv_heads * 8, d), device="cuda", dtype=torch.bfloat16)
    kv = (torch.randn((pages, kv_heads, PAGE, 2 * d), device="cuda") * 0.3).to(dtype)
    bt = torch.randperm(pages, device="cuda").to(torch.int32).view(num_reqs, -1)
    seq = torch.full((num_reqs,), context, device="cuda", dtype=torch.int32)
    top = _topk_rows(total, kv_heads, topk, pages_per_request, mode, n_valid)
    return (q, kv, top, bt, seq, kv_heads, d**-0.5, decode_query_len)


def assert_matches_native(actual, expected):
    """Compare only where both are finite.

    A row whose every selected block lies beyond its causal position has a
    zero softmax denominator; both implementations may represent that as NaN
    or zero, and that boundary is pinned separately below rather than folded
    into every comparison.
    """
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    assert bool(finite.any()), "fixture produced no comparable elements"
    torch.testing.assert_close(
        actual.float()[finite], expected.float()[finite], atol=0.02, rtol=0
    )


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("mode", ["scattered", "compact"])
@pytest.mark.minimax_m3_sparse_attn
def test_prefill_matches_native(d, kv_heads, mode):
    if cfg.QUICK_MODE and (kv_heads == 2 and mode == "compact"):
        pytest.skip("non-quick case")
    args = make_inputs(qlens=(17, 13), d=d, kv_heads=kv_heads, mode=mode, topk=17)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn(*args, output=reference)
    returned = flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    assert returned is out
    assert_matches_native(out, reference)


@pytest.mark.parametrize("n_valid", [1, 4, 12])
@pytest.mark.minimax_m3_sparse_attn
def test_prefill_trailing_padding(n_valid):
    """Short contexts make the indexer emit a -1 suffix; honor the count."""
    args = make_inputs(qlens=(17, 13), topk=16, n_valid=n_valid)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn(*args, output=reference)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    assert_matches_native(out, reference)


@pytest.mark.parametrize("qlens", [(1,), (384, 256), (2048,)])
@pytest.mark.minimax_m3_sparse_attn
def test_prefill_query_lengths(qlens):
    """Query lengths that select different internal prefill schedules."""
    if cfg.QUICK_MODE and qlens == (2048,):
        pytest.skip("non-quick case")
    args = make_inputs(qlens=qlens, context=32768)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn(*args, output=reference)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    assert_matches_native(out, reference)


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.minimax_m3_sparse_attn
def test_prefill_does_not_write_out_of_bounds(d):
    """A guard band catches a store tail that is not masked.

    head_dim 96 pads to a 128-wide tile, which is where an unmasked store
    walks into the next row of the output.
    """
    args = make_inputs(qlens=(17, 13), d=d, topk=17)
    q = args[0]
    sentinel = 1.5
    guard = torch.full((q.numel() + 128,), sentinel, device=q.device, dtype=q.dtype)
    out = guard[64:-64].view_as(q)
    reference = torch.empty_like(q)
    native.minimax_m3_sparse_attn(*args, output=reference)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    assert bool((guard[:64] == sentinel).all()), "wrote before the output"
    assert bool((guard[-64:] == sentinel).all()), "wrote past the output"
    assert_matches_native(out, reference)


@pytest.mark.minimax_m3_sparse_attn
def test_prefill_fp8_kv_matches_native():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("typed FP8 KV requires SM89+")
    args = make_inputs(qlens=(17, 13), dtype=torch.float8_e4m3fn)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn(*args, output=reference)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    assert_matches_native(out, reference)


@pytest.mark.minimax_m3_sparse_attn
def test_prefill_fully_masked_rows_agree_with_native():
    """Rows whose selected blocks are all in the future are a boundary case.

    Pin that the shim reports whatever the native operator reports, rather
        than substituting a different representation of "no contribution".
    """
    args = make_inputs(qlens=(2048,), mode="compact", context=32768, topk=16)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn(*args, output=reference)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    masked = ~torch.isfinite(reference)
    if bool(masked.any()):
        assert bool(
            (~torch.isfinite(out[masked])).all()
        ), "native reports no contribution; shim must agree"
    else:
        torch.testing.assert_close(
            out[~torch.isfinite(out)].float(),
            reference[~torch.isfinite(out)].float(),
            atol=0.02,
            rtol=0,
        )


@pytest.mark.minimax_m3_sparse_attn
def test_prefill_cuda_graph_replay():
    """Replay must recompute from device metadata, not captured values."""
    args = make_inputs(qlens=(384, 256), kv_heads=2)
    q, kv, top, bt, cu, seq, prefix = args[:7]
    out = torch.empty_like(q)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    for seed, mode in ((11, "scattered"), (12, "compact")):
        update = make_inputs(qlens=(384, 256), kv_heads=2, mode=mode, seed=seed)
        for dst, src in zip((q, kv, top, bt, cu, seq, prefix), update[:7]):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        reference = torch.empty_like(q)
        native.minimax_m3_sparse_attn(*args, output=reference)
        assert_matches_native(out, reference)


@pytest.mark.minimax_m3_sparse_attn
def test_prefill_empty_query_is_a_noop():
    args = list(make_inputs())
    args[0] = args[0][:0]
    args[2] = args[2][:, :0]
    args[7] = 0
    out = torch.empty_like(args[0])
    assert flaggems_vllm.minimax_m3_sparse_attn(*args, output=out) is out


@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.minimax_m3_sparse_attn_decode
def test_decode_matches_native(d, kv_heads):
    args = make_decode_inputs(num_reqs=5, d=d, kv_heads=kv_heads, topk=17)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn_decode(
        *args[:7], output=reference, decode_query_len=args[7]
    )
    returned = flaggems_vllm.minimax_m3_sparse_attn_decode(
        *args[:7], output=out, decode_query_len=args[7]
    )
    assert returned is out
    assert_matches_native(out, reference)


@pytest.mark.parametrize("num_reqs", [1, 16, 48, 128])
@pytest.mark.minimax_m3_sparse_attn_decode
def test_decode_split_k_chunk_counts(num_reqs):
    """Batch sizes that move the internal split-K chunk count."""
    if cfg.QUICK_MODE and num_reqs not in (1, 48):
        pytest.skip("non-quick case")
    args = make_decode_inputs(num_reqs=num_reqs, context=8192)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn_decode(
        *args[:7], output=reference, decode_query_len=args[7]
    )
    flaggems_vllm.minimax_m3_sparse_attn_decode(
        *args[:7], output=out, decode_query_len=args[7]
    )
    assert_matches_native(out, reference)


@pytest.mark.parametrize("decode_query_len", [1, 2, 4])
@pytest.mark.minimax_m3_sparse_attn_decode
def test_decode_multi_token(decode_query_len):
    args = make_decode_inputs(num_reqs=4, decode_query_len=decode_query_len)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn_decode(
        *args[:7], output=reference, decode_query_len=args[7]
    )
    flaggems_vllm.minimax_m3_sparse_attn_decode(
        *args[:7], output=out, decode_query_len=args[7]
    )
    assert_matches_native(out, reference)


@pytest.mark.parametrize("n_valid", [1, 8])
@pytest.mark.minimax_m3_sparse_attn_decode
def test_decode_trailing_padding(n_valid):
    args = make_decode_inputs(num_reqs=5, topk=16, n_valid=n_valid)
    out = torch.empty_like(args[0])
    reference = torch.empty_like(args[0])
    native.minimax_m3_sparse_attn_decode(
        *args[:7], output=reference, decode_query_len=args[7]
    )
    flaggems_vllm.minimax_m3_sparse_attn_decode(
        *args[:7], output=out, decode_query_len=args[7]
    )
    assert_matches_native(out, reference)


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda a: a.__setitem__(0, a[0].to(torch.float16)), NotImplementedError),
        (lambda a: a.__setitem__(8, 3), ValueError),
        (lambda a: a.__setitem__(9, float("nan")), ValueError),
        (lambda a: a.__setitem__(2, a[2].to(torch.int64)), ValueError),
        (lambda a: a.__setitem__(7, -1), ValueError),
    ],
)
@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.minimax_m3_sparse_attn_decode
def test_rejects_unsupported_contract(mutate, error):
    """Unsupported input must raise, never silently fall back to torch."""
    args = list(make_inputs())
    out = torch.empty_like(args[0])
    mutate(args)
    with pytest.raises(error):
        flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)


@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.minimax_m3_sparse_attn_decode
def test_scales_must_be_paired():
    args = make_inputs()
    out = torch.empty_like(args[0])
    scale = torch.ones((), device=out.device, dtype=torch.float32)
    with pytest.raises(ValueError, match="provided together"):
        flaggems_vllm.minimax_m3_sparse_attn(*args, output=out, k_scale=scale)


@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.minimax_m3_sparse_attn_decode
def test_exports_are_not_registered_as_aten_overrides():
    """Library-only APIs must stay out of the ATen registration config."""
    registered = {name for name, _ in flaggems_vllm._FULL_CONFIG}
    assert "minimax_m3_sparse_attn" not in registered
    assert "minimax_m3_sparse_attn_decode" not in registered


def topk_span_stats(top):
    """Per-row selected-id span statistics, reported by the benchmark.

    ``span`` is ``max_id - min_id + 1`` over a row's valid ids, so
    ``span / valid_count`` is the redundancy a span-scanning schedule pays.
    Sparse-attention latency depends strongly on how clustered the ids are, so
    a speedup is only interpretable next to the distribution behind it.
    """
    flat = top.reshape(-1, top.shape[-1]).long()
    valid = flat >= 0
    counts = valid.sum(-1)
    big = torch.iinfo(torch.int32).max
    lo = flat.masked_fill(~valid, big).min(-1).values
    hi = flat.masked_fill(~valid, -1).max(-1).values
    span = (hi - lo + 1).clamp_min(0)
    nonempty = counts > 0
    if not bool(nonempty.any()):
        return {"rows": int(flat.shape[0]), "nonempty_rows": 0}
    span_ne = span[nonempty].float()
    ratio = span_ne / counts[nonempty].float()
    return {
        "rows": int(flat.shape[0]),
        "nonempty_rows": int(nonempty.sum()),
        "span_median": round(float(span_ne.median()), 2),
        "redundancy_median": round(float(ratio.median()), 2),
    }
