# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
import importlib
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(
    ops.vendor_name != "metax", reason="MetaX specialization"
)


def check_replays(call, update, check, outputs=()):
    # No compilation, allocation growth or tuning is permitted during capture.
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        update(value)
        for output in outputs() if callable(outputs) else outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


def _cpu_sparse_attention(q, kv, ids, lengths, scale, sink):
    q, kv = q.cpu().double(), kv.cpu().double()
    output = torch.empty((*q.shape[:-1], 512), dtype=torch.float64)
    lse = torch.empty(q.shape[:-1], dtype=torch.float64)
    maximum = torch.empty_like(lse)
    for row in range(q.shape[0]):
        selected = ids[row, : int(lengths[row])].cpu().long()
        selected = selected[(selected >= 0) & (selected < kv.shape[0])]
        keys = kv[selected]
        score = (q[row] @ keys.t()) * scale
        if keys.shape[0] == 0:
            output[row] = 0
            lse[row] = float("inf")  # FlashMLA empty-row sentinel
            maximum[row] = -float("inf")
            continue
        logits = (
            torch.cat((score, sink.cpu().double()[:, None]), dim=1)
            if sink is not None
            else score
        )
        prob = logits.softmax(-1)[:, : keys.shape[0]]
        output[row] = prob @ keys[:, :512]
        lse[row] = score.logsumexp(-1)
        maximum[row] = score.amax(-1)
    return output.bfloat16(), lse.float(), maximum.float()


@pytest.mark.parametrize(
    "rows,topk,head_dim,index_dim",
    [(1, 16, 512, 2), (17, 64, 512, 2), (17, 64, 576, 2), (17, 64, 576, 3)],
)
@pytest.mark.flash_mla_sparse_fwd
def test_h16_prefill_reference_and_changed_input_graph(rows, topk, head_dim, index_dim):
    torch.manual_seed(104)
    q = (torch.randn(rows, 16, head_dim, device="cuda") * 0.1).bfloat16()
    kv = (torch.randn(128, head_dim, device="cuda") * 0.1).bfloat16()
    ids = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(rows, 1)
    lengths = torch.full((rows,), topk, dtype=torch.int32, device="cuda")
    sink = torch.randn(16, device="cuda") * 0.1
    result = torch.empty(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
    maximum = torch.empty(rows, 16, device="cuda")
    lse = torch.empty_like(maximum)
    scale = head_dim**-0.5

    def call():
        ops.flash_mla_sparse_fwd(
            q,
            kv[:, None],
            ids if index_dim == 2 else ids[:, None],
            scale,
            attn_sink=sink,
            topk_length=lengths,
            out=result,
            max_logits_out=maximum,
            lse_out=lse,
        )

    def check():
        expected, expected_lse, expected_max = _cpu_sparse_attention(
            q, kv, ids, lengths.cpu(), scale, sink
        )
        torch.testing.assert_close(result.cpu(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(maximum.cpu(), expected_max, atol=1e-4, rtol=1e-4)

    call()
    check()
    check_replays(
        call, lambda value: q.fill_(value), check, outputs=(result, maximum, lse)
    )


def _model1_cache(pages=2, block=256):
    # Test-only independent packing: 448 E4M3 values + 64 BF16 rope values,
    # followed by 7 power-of-two group scales and one padding byte per token.
    count = pages * block
    nope = (torch.randn(count, 448, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    rope = (torch.randn(count, 64, device="cuda") * 0.1).bfloat16()
    data = torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=1)
    cache = torch.zeros(pages, block, 1, 584, device="cuda", dtype=torch.uint8)
    flat = cache.view(pages, -1)
    flat[:, : block * 576] = data.view(pages, -1)
    encoded = torch.full((count, 8), 125, device="cuda", dtype=torch.uint8)
    flat[:, block * 576 :] = encoded.view(pages, -1)
    dequant = torch.cat((nope.float() * 0.25, rope.float()), dim=1)
    return cache, dequant


@contextmanager
def _record_launches(module, names):
    """Observe the public dispatch without replacing any GPU computation."""
    launches = []

    class Recorder:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def invoke(*args, **kwargs):
                launches.append(self.name)
                return launch(*args, **kwargs)

            return invoke

    with ExitStack() as stack:
        for name in names:
            stack.enter_context(
                patch.object(module, name, Recorder(getattr(module, name), name))
            )
        yield launches


def _decode_case(batch, heads):
    torch.manual_seed(104)
    q = (torch.randn(batch, 1, heads, 512, device="cuda") * 0.1).bfloat16()
    cache, decoded = _model1_cache()
    extra_cache, extra_decoded = _model1_cache(pages=1)
    indices = (
        torch.arange(512, device="cuda", dtype=torch.int32)
        .reshape(1, 1, 512)
        .repeat(batch, 1, 1)
    )
    extra_indices = (
        torch.arange(128, device="cuda", dtype=torch.int32)
        .reshape(1, 1, 128)
        .repeat(batch, 1, 1)
    )
    lengths = torch.full((batch,), 256, device="cuda", dtype=torch.int32)
    extra_lengths = torch.full((batch,), 64, device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda") * 0.1
    workspace = tuple(
        torch.empty(shape, device="cuda")
        for shape in (
            (64, 4, heads),
            (64, 4, heads),
            (64, 4, heads, 512),
        )
    )
    meta = ops.FlashMLASchedMeta()
    result = torch.empty_like(q)
    kwargs = dict(
        q=q,
        k_cache=cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=meta,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=lengths,
        softmax_scale=512**-0.5,
        attn_sink=sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lengths,
        out=result,
        split_workspace=workspace,
    )
    return kwargs, decoded, extra_decoded


def _check_split_decode(batch, heads):
    kwargs, decoded, extra_decoded = _decode_case(batch, heads)
    q, result, sink = (kwargs[name] for name in ("q", "out", "attn_sink"))
    # Non-tile-aligned lengths and masked IDs exercise each split's tail.
    kwargs["topk_length"] -= torch.arange(batch, device="cuda", dtype=torch.int32) % 7
    kwargs["extra_topk_length"] -= (
        torch.arange(batch, device="cuda", dtype=torch.int32) % 5
    )
    kwargs["indices"][..., 5] = -1
    kwargs["extra_indices_in_kvcache"][..., 7] = 9999
    observed = []

    def call():
        output, lse = ops.flash_mla_with_kvcache(**kwargs)
        assert output is result
        observed[:] = [lse]

    def check():
        main_ids = kwargs["indices"][:, 0].cpu()
        extra_ids = kwargs["extra_indices_in_kvcache"][:, 0].cpu()
        lengths = kwargs["topk_length"].cpu().tolist()
        extra_lengths = kwargs["extra_topk_length"].cpu().tolist()
        ids = torch.full((batch, 640), -1, dtype=torch.int32)
        for row, (length, extra_length) in enumerate(zip(lengths, extra_lengths)):
            ids[row, :length] = main_ids[row, :length]
            extra = extra_ids[row, :extra_length]
            ids[row, length : length + extra_length] = torch.where(
                (extra >= 0) & (extra < extra_decoded.shape[0]),
                extra + decoded.shape[0],
                -1,
            )
        expected, expected_lse, _ = _cpu_sparse_attention(
            q[:, 0],
            torch.cat((decoded, extra_decoded)),
            ids,
            [a + b for a, b in zip(lengths, extra_lengths)],
            512**-0.5,
            sink,
        )
        torch.testing.assert_close(result[:, 0].cpu(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(
            observed[0][:, :, 0].cpu(), expected_lse, atol=2e-4, rtol=2e-4
        )

    def update(value):
        q[:, :, :, :256].fill_(value)
        q[:, :, :, 256:].fill_(-value * 0.5)
        if heads == 16:
            q[:, :, 8:, :256].fill_(value * 1.5)

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.sparse_mla_decode"
    )
    names = (
        "_partial_model1_kernel_halfd",
        "_partial_model1_h8_kernel",
        "_reduce_split_kernel",
        "_sparse_decode_model1_kernel",
    )
    with _record_launches(module, names) as launches:
        call()
        if batch > 64 and heads == 16:
            assert launches == ["_sparse_decode_model1_kernel"]
        elif heads == 8:
            assert launches == [
                "_partial_model1_h8_kernel",
                "_reduce_split_kernel",
            ]
        else:
            assert launches == ["_partial_model1_kernel_halfd"] * 2 + [
                "_reduce_split_kernel"
            ]
        check()
        check_replays(call, update, check, outputs=lambda: (result, observed[0]))


@pytest.mark.parametrize("batch", [1, 3, 64, 65])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.flash_mla_with_kvcache
def test_decode_reference_workspace_and_changed_input_graph(batch, heads):
    _check_split_decode(batch, heads)


@pytest.mark.parametrize("bad_workspace", ["missing", "shape", "dtype", "layout"])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.flash_mla_with_kvcache
def test_split_decode_rejects_workspace_before_output_write(bad_workspace, heads):
    kwargs, _, _ = _decode_case(1, heads)
    kwargs["out"].fill_(17)
    workspace = list(kwargs["split_workspace"])
    if bad_workspace == "missing":
        kwargs["split_workspace"] = None
    else:
        if bad_workspace == "shape":
            workspace[0] = workspace[0][:1]
        elif bad_workspace == "dtype":
            workspace[0] = workspace[0].bfloat16()
        else:
            workspace[0] = torch.empty((64, 4, heads * 2), device="cuda")[..., ::2]
        kwargs["split_workspace"] = tuple(workspace)
    with pytest.raises(ValueError, match="workspace"):
        ops.flash_mla_with_kvcache(**kwargs)
    assert bool((kwargs["out"] == 17).all())


@pytest.mark.parametrize(
    "rows,head_dim,sink_on,all_masked",
    [
        (1, 512, True, False),
        (2, 512, True, False),
        (17, 512, True, False),
        (64, 512, False, False),
        (17, 576, True, False),
        (17, 512, True, True),
    ],
)
@pytest.mark.parametrize("index_dim", [2, 3])
@pytest.mark.flash_mla_sparse_fwd
def test_h8_prefill_reference_and_changed_input_graph(
    rows, head_dim, sink_on, all_masked, index_dim
):
    torch.manual_seed(104)
    q = (torch.randn(rows, 8, head_dim, device="cuda") * 0.1).bfloat16()
    kv = (torch.randn(128, 1, head_dim, device="cuda") * 0.1).bfloat16()
    ids = torch.arange(64, device="cuda", dtype=torch.int32).repeat(rows, 1)
    ids[:, 5], ids[:, 17] = -1, 9999
    lengths = 64 - torch.arange(rows, device="cuda", dtype=torch.int32) % 7
    if all_masked:
        ids.fill_(-1)
        lengths.zero_()
    sink = torch.linspace(-0.3, 0.9, 8, device="cuda") if sink_on else None
    output = torch.empty(rows, 8, 512, device="cuda", dtype=torch.bfloat16)
    maximum = torch.empty(rows, 8, device="cuda")
    lse = torch.empty_like(maximum)
    scale = head_dim**-0.5

    def call():
        result = ops.flash_mla_sparse_fwd(
            q,
            kv,
            ids[:, None] if index_dim == 3 else ids,
            scale,
            attn_sink=sink,
            topk_length=lengths,
            out=output,
            max_logits_out=maximum,
            lse_out=lse,
        )
        assert result[0] is output and result[1] is maximum and result[2] is lse

    def check():
        expected, expected_lse, expected_max = _cpu_sparse_attention(
            q,
            kv[:, 0],
            ids,
            lengths.cpu(),
            scale,
            sink,
        )
        torch.testing.assert_close(output.cpu(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(maximum.cpu(), expected_max, atol=1e-4, rtol=1e-4)

    def update(value):
        q[:, :, :256].fill_(value)
        q[:, :, 256:].fill_(-value * 0.5)

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.sparse_mla_prefill"
    )
    name = "triton_flash_mla_sparse_fwd_h8"
    with _record_launches(module, (name,)) as launches:
        call()
        assert launches == [name]
        check()
        check_replays(call, update, check, outputs=(output, maximum, lse))


@pytest.fixture(autouse=True)
def restore_sparse_rng_state():
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    yield
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(gpu)


@pytest.mark.flash_mla_sparse_fwd
def test_sparse_prefill_all_masked_rows_and_empty_batch_rejection():
    q = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(16, 1, 512, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((2, 16), -1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(2, device="cuda", dtype=torch.int32)
    sink = torch.zeros(16, device="cuda")
    output = torch.empty_like(q)
    maximum = torch.empty(2, 16, device="cuda")
    lse = torch.empty_like(maximum)

    def call():
        ops.flash_mla_sparse_fwd(
            q,
            kv,
            ids,
            512**-0.5,
            attn_sink=sink,
            topk_length=lengths,
            out=output,
            max_logits_out=maximum,
            lse_out=lse,
        )

    def check():
        assert bool((output == 0).all())
        # FlashMLA uses +inf as the no-valid-key LSE sentinel. The upstream
        # independent FlashmlaSparseTestKit.ref_fwd maps empty -inf to +inf.
        assert bool(torch.isposinf(lse).all())
        assert bool(torch.isneginf(maximum).all())

    check_replays(call, lambda value: q.fill_(value), check, outputs=(output, lse))
    output.fill_(17)
    with pytest.raises(ValueError, match="nonempty"):
        ops.flash_mla_sparse_fwd(
            q[:0],
            kv,
            ids[:0],
            512**-0.5,
            attn_sink=sink,
            topk_length=lengths[:0],
            out=output[:0],
            max_logits_out=maximum[:0],
            lse_out=lse[:0],
        )
    assert bool((output == 17).all())


@pytest.mark.parametrize(
    "case",
    [
        "multi_query",
        "no_lengths_or_sink",
        "empty_splits",
        "invalid_ids",
        "int8_cache",
        "fp8_cache",
    ],
)
@pytest.mark.flash_mla_with_kvcache
def test_native_h8_decode_general_routes_and_guard_buffers(case):
    kwargs, decoded, _ = _decode_case(3, 8)
    kwargs.update(
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
    )
    if case == "multi_query":
        kwargs["q"] = kwargs["q"].repeat(1, 2, 1, 1)
        kwargs["indices"] = kwargs["indices"].repeat(1, 2, 1)
    elif case == "no_lengths_or_sink":
        kwargs.update(topk_length=None, attn_sink=None)
    elif case == "empty_splits":
        # Includes splits with no keys, and a completely empty attention row.
        kwargs["topk_length"][:] = torch.tensor([0, 1, 3], device="cuda")
        extra, _ = _model1_cache(pages=1)
        kwargs.update(
            extra_k_cache=extra,
            extra_indices_in_kvcache=kwargs["indices"][..., :64].contiguous(),
            extra_topk_length=torch.zeros(3, device="cuda", dtype=torch.int32),
        )
    elif case == "invalid_ids":
        kwargs["indices"].fill_(-1)
    if case in ("int8_cache", "fp8_cache"):
        kwargs["k_cache"] = kwargs["k_cache"].view(
            torch.int8 if case == "int8_cache" else torch.float8_e4m3fn
        )
    q = kwargs["q"]
    # Prefix/suffix canaries catch writing virtual heads into physical H8 buffers.
    guards = []

    def guarded(shape, dtype):
        size = 1
        for n in shape:
            size *= n
        backing = torch.full((size + 64,), 17, device="cuda", dtype=dtype)
        guards.append(backing)
        return backing[32:-32].view(shape)

    kwargs["out"] = guarded(q.shape, q.dtype)
    kwargs["split_workspace"] = tuple(
        guarded(x.shape, x.dtype) for x in kwargs["split_workspace"]
    )
    observed = []

    def call():
        result, lse = ops.flash_mla_with_kvcache(**kwargs)
        observed[:] = [result, lse]

    def check():
        batch, sq = q.shape[:2]
        lens = kwargs["topk_length"]
        lens = [512] * batch if lens is None else lens.cpu().tolist()
        for i in range(sq):
            expected, expected_lse, _ = _cpu_sparse_attention(
                q[:, i],
                decoded,
                kwargs["indices"][:, i],
                lens,
                512**-0.5,
                kwargs["attn_sink"],
            )
            torch.testing.assert_close(
                observed[0][:, i].cpu(), expected, atol=2e-3, rtol=2e-2
            )
            torch.testing.assert_close(
                observed[1][:, :, i].cpu(), expected_lse, atol=2e-4, rtol=2e-4
            )
        for backing in guards:
            assert bool((backing[:32] == 17).all())
            assert bool((backing[-32:] == 17).all())

    call()
    check()
    check_replays(call, lambda value: q.fill_(value), check, outputs=lambda: observed)


@pytest.mark.parametrize("caller_outputs", [False, True])
@pytest.mark.flash_mla_sparse_fwd
def test_native_h8_prefill_optional_outputs_and_guard_buffers(caller_outputs):
    q = torch.randn(3, 8, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(128, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.arange(64, device="cuda", dtype=torch.int32).repeat(3, 1)
    # A fully masked chunk before valid chunks must not poison online softmax.
    ids[:, :16] = -1
    guards, kwargs = [], {}
    if caller_outputs:
        for name, shape, dtype in (
            ("out", (3, 8, 512), torch.bfloat16),
            ("max_logits_out", (3, 8), torch.float32),
            ("lse_out", (3, 8), torch.float32),
        ):
            size = 1
            for dim in shape:
                size *= dim
            backing = torch.full((size + 64,), 17, device="cuda", dtype=dtype)
            guards.append(backing)
            kwargs[name] = backing[32:-32].view(shape)
    observed = []

    def call():
        observed[:] = ops.flash_mla_sparse_fwd(q, kv, ids[:, None], 576**-0.5, **kwargs)

    def check():
        expected, expected_lse, expected_max = _cpu_sparse_attention(
            q, kv[:, 0], ids, [64] * 3, 576**-0.5, None
        )
        for actual, reference in zip(observed, (expected, expected_max, expected_lse)):
            torch.testing.assert_close(actual.cpu(), reference, atol=2e-3, rtol=2e-2)
        for backing in guards:
            assert bool((backing[:32] == 17).all())
            assert bool((backing[-32:] == 17).all())

    call()
    check()
    check_replays(call, lambda value: q.fill_(value), check, outputs=lambda: observed)


@pytest.mark.parametrize(
    "bad_input", ["q_stride", "sink_stride", "empty_batch", "empty_query"]
)
@pytest.mark.flash_mla_with_kvcache
def test_native_h8_decode_rejects_unsupported_inputs_before_state_write(bad_input):
    kwargs, _, _ = _decode_case(1, 8)
    kwargs["out"].fill_(17)
    if bad_input == "q_stride":
        kwargs["q"] = torch.empty(1, 1, 8, 1024, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif bad_input == "sink_stride":
        kwargs["attn_sink"] = torch.empty(16, device="cuda")[::2]
    elif bad_input == "empty_batch":
        kwargs["q"] = kwargs["q"][:0]
    else:
        kwargs["q"] = kwargs["q"][:, :0]
    with pytest.raises(ValueError, match="physical H8"):
        ops.flash_mla_with_kvcache(**kwargs)
    assert not kwargs["tile_scheduler_metadata"].have_initialized
    assert bool((kwargs["out"] == 17).all())


@pytest.mark.flash_mla_with_kvcache
def test_removed_decode_padded_head_override_rejected_before_state_write():
    kwargs, _, _ = _decode_case(1, 16)
    kwargs["out"].fill_(17)
    kwargs["real_num_heads"] = 8
    with pytest.raises(TypeError, match="real_num_heads"):
        ops.flash_mla_with_kvcache(**kwargs)
    assert not kwargs["tile_scheduler_metadata"].have_initialized
    assert bool((kwargs["out"] == 17).all())
