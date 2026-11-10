# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
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
        for output in outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


def _cpu_sparse_attention(q, kv, ids, lengths, scale, sink):
    q, kv = q.cpu().double(), kv.cpu().double()
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], dtype=torch.float64)
    for row in range(q.shape[0]):
        selected = ids[row, : int(lengths[row])].cpu().long()
        selected = selected[(selected >= 0) & (selected < kv.shape[0])]
        keys = kv[selected]
        score = (q[row] @ keys.t()) * scale
        logits = torch.cat((score, sink.cpu().double()[:, None]), dim=1)
        output[row] = logits.softmax(-1)[:, :-1] @ keys
        lse[row] = score.logsumexp(-1)
    return output.bfloat16(), lse.float()


@pytest.mark.parametrize("rows,topk", [(1, 16), (17, 64)])
@pytest.mark.flash_mla_sparse_fwd
def test_sparse_prefill_local_reference_and_changed_input_graph(rows, topk):
    torch.manual_seed(104)
    q = (torch.randn(rows, 16, 512, device="cuda") * 0.1).bfloat16()
    kv = (torch.randn(128, 512, device="cuda") * 0.1).bfloat16()
    ids = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(rows, 1)
    lengths = torch.full((rows,), topk, dtype=torch.int32, device="cuda")
    sink = torch.randn(16, device="cuda") * 0.1
    result = torch.empty_like(q)
    maximum = torch.empty(rows, 16, device="cuda")
    lse = torch.empty_like(maximum)
    scale = 512**-0.5

    def call():
        ops.flash_mla_sparse_fwd(
            q,
            kv[:, None],
            ids,
            scale,
            attn_sink=sink,
            topk_length=lengths,
            out=result,
            max_logits_out=maximum,
            lse_out=lse,
        )

    def check():
        expected, expected_lse = _cpu_sparse_attention(
            q, kv, ids, lengths.cpu(), scale, sink
        )
        torch.testing.assert_close(result.cpu(), expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=1e-4, rtol=1e-4)

    call()
    check()
    check_replays(call, lambda value: q.fill_(value), check, outputs=(result, lse))


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


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.flash_mla_with_kvcache
def test_split_decode_model1_reference_workspace_and_replay(batch):
    torch.manual_seed(104)
    q = (torch.randn(batch, 1, 16, 512, device="cuda") * 0.1).bfloat16()
    q[:, :, 8:].zero_()
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
    sink = torch.randn(16, device="cuda") * 0.1
    workspace = tuple(
        torch.empty(shape, device="cuda")
        for shape in ((64, 4, 16), (64, 4, 16), (64, 4, 16, 512))
    )
    meta = ops.FlashMLASchedMeta()
    result = torch.empty_like(q)
    observed = []

    def call():
        _, lse = ops.flash_mla_with_kvcache(
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
            real_num_heads=8,
            split_workspace=workspace,
        )
        observed[:] = [lse]

    def check():
        keys = torch.cat((decoded[:256], extra_decoded[:64]))
        ids = torch.arange(320, dtype=torch.int32).repeat(batch, 1)
        expected, expected_lse = _cpu_sparse_attention(
            q[:, 0], keys, ids, [320] * batch, 512**-0.5, sink
        )
        # Only the first eight heads are real; the remaining lanes are padding.
        torch.testing.assert_close(
            result[:, 0, :8].cpu(), expected[:, :8], atol=2e-3, rtol=2e-2
        )
        torch.testing.assert_close(
            observed[0][:, :8, 0].cpu(), expected_lse[:, :8], atol=2e-4, rtol=2e-4
        )

    call()
    check()
    check_replays(
        call, lambda value: q[:, :, :8].fill_(value), check, outputs=(result,)
    )


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
        assert bool(torch.isneginf(lse).all())

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
