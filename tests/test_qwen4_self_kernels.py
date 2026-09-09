"""Correctness coverage for eight self-developed and three vendor Qwen4 kernels.

Torch references live in this test module only. The production wrappers are
required to fail closed instead of dispatching a Torch compute fallback.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flaggems_vllm.ops.qwen4.hyperconnection import (  # noqa: E402
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
)
from flaggems_vllm.ops.qwen4.ple_state import (  # noqa: E402
    ple_state_gather,
    ple_state_scatter_,
)
from flaggems_vllm.ops.qwen4.qsa import (  # noqa: E402
    QWEN4_VENDOR_QSA_SOURCE,
    qwen4_compress_norm_mrope_store_groups,
    qwen4_qsa_mqa_paged_dot,
    qwen4_store_qsa_kv_rows,
    qwen4_vendor_compress_qsa_groups,
    qwen4_vendor_qsa_mqa_paged,
    qwen4_vendor_store_qsa_rows,
)

DEVICE = "cuda"
HC = 4
EPS = 1.0e-6


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen4 Triton correctness requires an accelerator")


def _hc_refs(x, weight, logits, normed, injection, block, residual):
    hidden = weight.numel() // HC
    x3 = x.reshape(-1, HC, hidden).float()
    w2 = weight.reshape(HC, hidden).float()
    rms = torch.rsqrt(x3.square().mean(-1, keepdim=True) + EPS)
    norm_ref = (x3 * rms * (1.0 + w2)).to(x.dtype).reshape_as(x)
    gate_ref = (
        (
            torch.sigmoid(logits.float().reshape(-1, HC, hidden))
            * normed.float().reshape(-1, HC, hidden)
        )
        .mean(-2)
        .to(normed.dtype)
    )
    alpha = 2.0 * torch.sigmoid(injection.float() / HC)
    inject_ref = (
        (
            residual.float().reshape(-1, HC, hidden)
            + block.float().unsqueeze(-2) * alpha.unsqueeze(-1)
        )
        .to(residual.dtype)
        .reshape_as(residual)
    )
    return norm_ref, gate_ref, inject_ref


@pytest.mark.qwen4_hc
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,hidden", [(1, 257), (3, 513)])
def test_qwen4_hyperconnection_matches_torch(dtype, rows, hidden):
    _require_cuda()
    torch.manual_seed(101 + rows + hidden)
    x = torch.randn((rows, HC * hidden), device=DEVICE, dtype=dtype)
    weight = torch.randn((HC * hidden,), device=DEVICE, dtype=dtype) * 0.01
    logits = torch.randn_like(x)
    normed = torch.randn_like(x)
    injection = torch.randn((rows, HC), device=DEVICE, dtype=dtype)
    block = torch.randn((rows, hidden), device=DEVICE, dtype=dtype)
    residual = torch.randn_like(x)
    norm_ref, gate_ref, inject_ref = _hc_refs(
        x, weight, logits, normed, injection, block, residual
    )

    norm_out = qwen4_grouped_gemma_rmsnorm(x, weight, HC, EPS)
    gate_out = qwen4_hc_gate_reduce(logits, normed, HC)
    inject_out = qwen4_hc_inject_combine(injection, block, residual, HC)

    atol = 3.0e-2 if dtype == torch.bfloat16 else 2.0e-2
    rtol = 2.0e-2
    torch.testing.assert_close(norm_out, norm_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(gate_out, gate_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(inject_out, inject_ref, atol=atol, rtol=rtol)

    snapshots = (norm_out.clone(), gate_out.clone(), inject_out.clone())
    for _ in range(10):
        torch.testing.assert_close(
            qwen4_grouped_gemma_rmsnorm(x, weight, HC, EPS),
            snapshots[0],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            qwen4_hc_gate_reduce(logits, normed, HC),
            snapshots[1],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            qwen4_hc_inject_combine(injection, block, residual, HC),
            snapshots[2],
            atol=0,
            rtol=0,
        )


def _qsa_mqa_reference(q, cache, table, token_to_req, positions, lengths, ratio):
    columns = table.shape[1] * cache.shape[1]
    req = token_to_req.to(torch.int64)
    valid_req = (req >= 0) & (req < table.shape[0])
    safe_req = req.clamp(0, table.shape[0] - 1)
    seq = lengths.index_select(0, safe_req)
    seq = torch.where(valid_req, seq, torch.zeros_like(seq))
    visible = torch.minimum((positions.to(torch.int64) + 1) // ratio, seq // ratio)
    col = torch.arange(columns, device=q.device, dtype=torch.int64)
    logical_page = col // cache.shape[1]
    offset = col % cache.shape[1]
    safe_logical = logical_page.clamp(0, table.shape[1] - 1)
    physical = table[safe_req[:, None], safe_logical[None, :]].to(torch.int64)
    valid = (
        valid_req[:, None]
        & (col[None, :] < visible[:, None])
        & (physical >= 0)
        & (physical < cache.shape[0])
    )
    safe_physical = physical.clamp(0, cache.shape[0] - 1)
    keys = cache[safe_physical, offset[None, :], 0].float()
    dots = torch.matmul(q.float(), keys.transpose(1, 2))
    logits = torch.relu(dots).sum(1) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~valid, -float("inf"))
    return logits, visible.to(torch.int32)


@pytest.mark.qwen4_qsa
def test_qwen4_qsa_mqa_paged_dot_matches_torch_and_invalid_pages():
    _require_cuda()
    rows, page_size, pages, columns = 8, 16, 4, 48
    torch.manual_seed(202)
    q = torch.randn((rows, 4, 128), device=DEVICE, dtype=torch.bfloat16)
    cache = torch.randn((pages, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16)
    table = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
        device=DEVICE,
        dtype=torch.int32,
    )
    table[2, 1] = -1
    token_to_req = torch.tensor(
        [0, 1, 2, 3, 0, 1, -1, 3], device=DEVICE, dtype=torch.int32
    )
    positions = torch.tensor(
        [47, 39, 31, 23, 47, 35, 47, 15], device=DEVICE, dtype=torch.int64
    )
    lengths = torch.tensor([48, 40, 32, 24], device=DEVICE, dtype=torch.int64)

    ref_logits, ref_visible = _qsa_mqa_reference(
        q, cache, table, token_to_req, positions, lengths, 4
    )
    out_logits, out_visible = qwen4_qsa_mqa_paged_dot(
        q,
        cache,
        table,
        token_to_req,
        positions,
        lengths,
        compress_ratio=4,
        num_columns=columns,
    )
    torch.testing.assert_close(out_visible, ref_visible, atol=0, rtol=0)
    torch.testing.assert_close(out_logits, ref_logits, atol=0.2, rtol=0.02)

    vendor_logits, vendor_visible = qwen4_vendor_qsa_mqa_paged(
        q,
        cache,
        table,
        token_to_req,
        positions,
        lengths,
        compress_ratio=4,
        num_columns=columns,
    )
    torch.testing.assert_close(vendor_visible, ref_visible, atol=0, rtol=0)
    torch.testing.assert_close(vendor_logits, ref_logits, atol=0.2, rtol=0.02)
    assert torch.isneginf(out_logits[6]).all()
    assert torch.isneginf(out_logits[2, 8:]).all()

    snapshot = out_logits.clone()
    for _ in range(10):
        again, visible = qwen4_qsa_mqa_paged_dot(
            q,
            cache,
            table,
            token_to_req,
            positions,
            lengths,
            compress_ratio=4,
            num_columns=columns,
        )
        torch.testing.assert_close(again, snapshot, atol=0, rtol=0)
        torch.testing.assert_close(visible, out_visible, atol=0, rtol=0)


@pytest.mark.qwen4_qsa
def test_qwen4_qsa_kv_store_matches_torch_and_skips_invalid_slots():
    _require_cuda()
    blocks, page_size, heads, dim, rows = 3, 4, 2, 17, 5
    torch.manual_seed(303)
    initial_k = torch.randn(
        (blocks, page_size, heads, dim), device=DEVICE, dtype=torch.bfloat16
    )
    initial_v = torch.randn_like(initial_k)
    key = torch.randn((rows, heads, dim), device=DEVICE, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 3, 4, 11, -1], device=DEVICE, dtype=torch.int64)

    ref_k, ref_v = initial_k.clone(), initial_v.clone()
    valid = (slots >= 0) & (slots < blocks * page_size)
    safe = slots[valid]
    ref_k.index_put_(
        (safe // page_size, safe % page_size), key[valid], accumulate=False
    )
    ref_v.index_put_(
        (safe // page_size, safe % page_size), value[valid], accumulate=False
    )

    out_k, out_v = initial_k.clone(), initial_v.clone()
    qwen4_store_qsa_kv_rows(out_k, out_v, slots, key, value)
    torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)
    torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)


@pytest.mark.qwen4_qsa
def test_qwen4_vendor_qsa_store_matches_torch_and_skips_invalid_slots():
    _require_cuda()
    blocks, page_size, rows = 3, 4, 5
    torch.manual_seed(304)
    initial = torch.randn(
        (blocks, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16
    )
    values = torch.randn((rows, 1, 128), device=DEVICE, dtype=torch.bfloat16)
    slots = torch.tensor([0, 3, 4, 11, -1], device=DEVICE, dtype=torch.int64)
    expected = initial.clone()
    valid = (slots >= 0) & (slots < blocks * page_size)
    safe = slots[valid]
    expected.index_put_(
        (safe // page_size, safe % page_size), values[valid], accumulate=False
    )

    actual = initial.clone()
    qwen4_vendor_store_qsa_rows(actual, slots, values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _compress_reference(
    raw, rope, table, requests, positions, slots, weight, cos_sin, out
):
    ratio = 4
    for row in range(requests.numel()):
        request = int(requests[row].item())
        end = int(positions[row].item())
        slot = int(slots[row].item())
        raw_rows = []
        for offset in range(ratio):
            pos = end - (ratio - 1 - offset)
            page = int(table[request, pos // raw.shape[1]].item())
            raw_rows.append(raw[page, pos % raw.shape[1], 0].float())
        pooled = (torch.stack(raw_rows).sum(0) / ratio).to(torch.bfloat16)
        pooled_fp32 = pooled.float()
        variance = pooled_fp32.square().mean() / 1.0
        normalized = (
            pooled_fp32 * torch.rsqrt(variance + EPS) * (weight.float() + 1.0)
        ).to(torch.bfloat16)
        first_pos = end - ratio + 1
        page = int(table[request, first_pos // raw.shape[1]].item())
        axes = rope[page, first_pos % rope.shape[1], 0, :3]
        freq = torch.arange(32, device=raw.device)
        use_h = (freq % 3 == 1) & (freq < 3 * 11)
        use_w = (freq % 3 == 2) & (freq < 3 * 10)
        axis_pos = torch.where(
            use_h,
            axes[1],
            torch.where(use_w, axes[2], axes[0]),
        )
        cos = cos_sin[axis_pos, freq].float()
        sin = cos_sin[axis_pos, 32 + freq].float()
        first = normalized[:32].float()
        second = normalized[32:64].float()
        rotated_first = (first * cos - second * sin).to(torch.bfloat16)
        rotated_second = (second * cos + first * sin).to(torch.bfloat16)
        stored = torch.cat((rotated_first, rotated_second, normalized[64:]), dim=0)
        block, token = slot // out.shape[1], slot % out.shape[1]
        out[block, token, 0].copy_(stored)


@pytest.mark.qwen4_qsa
def test_qwen4_qsa_fused_compress_norm_mrope_interleaved_matches_torch():
    _require_cuda()
    page_size, raw_blocks = 8, 2
    torch.manual_seed(404)
    raw = torch.randn(
        (raw_blocks, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16
    )
    rope = torch.empty((raw_blocks, page_size, 1, 3), device=DEVICE, dtype=torch.int64)
    rope[..., 0] = torch.arange(page_size, device=DEVICE).view(1, page_size, 1) % 16
    rope[..., 1] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) + 1
    ) % 16
    rope[..., 2] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) + 2
    ) % 16
    table = torch.tensor([[0, 1]], device=DEVICE, dtype=torch.int32)
    requests = torch.tensor([0, 0], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
    slots = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
    weight = torch.randn((128,), device=DEVICE, dtype=torch.bfloat16) * 0.01
    t = torch.arange(16, device=DEVICE, dtype=torch.float32)
    freq = torch.arange(32, device=DEVICE, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (freq / 32.0))
    angles = t[:, None] * inv_freq[None, :]
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    initial = torch.full_like(raw, 7)
    expected = initial.clone()
    _compress_reference(
        raw, rope, table, requests, positions, slots, weight, cos_sin, expected
    )

    actual = initial.clone()
    qwen4_compress_norm_mrope_store_groups(
        raw,
        table,
        requests,
        positions,
        slots,
        actual,
        weight,
        cos_sin,
        compress_ratio=4,
        norm_eps=EPS,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
        rope_cache=rope,
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)

    vendor_pooled, vendor_positions = qwen4_vendor_compress_qsa_groups(
        raw,
        table,
        requests,
        positions,
        slots,
        compress_ratio=4,
        rope_cache=rope,
    )
    expected_pooled = []
    expected_positions = []
    for row in range(requests.numel()):
        request = int(requests[row].item())
        end = int(positions[row].item())
        raw_rows = []
        for offset in range(4):
            position = end - (3 - offset)
            page = int(table[request, position // raw.shape[1]].item())
            raw_rows.append(raw[page, position % raw.shape[1], 0].float())
        expected_pooled.append((torch.stack(raw_rows).sum(0) / 4).to(torch.bfloat16))
        first = end - 3
        page = int(table[request, first // raw.shape[1]].item())
        expected_positions.append(rope[page, first % rope.shape[1], 0, :3])
    expected_pooled = torch.stack(expected_pooled).unsqueeze(1)
    expected_positions = torch.stack(expected_positions)
    torch.testing.assert_close(vendor_pooled, expected_pooled, atol=0.015, rtol=0)
    torch.testing.assert_close(vendor_positions, expected_positions, atol=0, rtol=0)

    snapshot = actual.clone()
    for _ in range(3):
        repeated = initial.clone()
        qwen4_compress_norm_mrope_store_groups(
            raw,
            table,
            requests,
            positions,
            slots,
            repeated,
            weight,
            cos_sin,
            compress_ratio=4,
            norm_eps=EPS,
            rotary_dim=64,
            mrope_section=(11, 11, 10),
            mrope_interleaved=True,
            rope_cache=rope,
        )
        torch.testing.assert_close(repeated, snapshot, atol=0, rtol=0)


def _strided_state(cache_rows, hidden, width):
    storage = torch.randn(
        (cache_rows, width, hidden), device=DEVICE, dtype=torch.bfloat16
    )
    return storage.transpose(1, 2)


@pytest.mark.qwen4_ple
def test_qwen4_ple_gather_preserves_strides_and_null_is_not_row_zero():
    _require_cuda()
    state = _strided_state(11, 3, 5)
    indices = torch.tensor([1, -1, 1, 99, 2], device=DEVICE, dtype=torch.int64)
    output = ple_state_gather(state, indices)
    valid = (indices >= 0) & (indices < state.shape[0])
    bounded = indices.clamp(0, state.shape[0] - 1)
    expected = torch.ops.aten.index_select.default(state, 0, bounded)
    expected = torch.where(valid.view(-1, 1, 1), expected, torch.zeros_like(expected))
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output.stride(1) == 1
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    assert not torch.equal(output[1], state[0])


@pytest.mark.qwen4_ple
def test_qwen4_ple_scatter_masked_null_and_duplicate_is_exact():
    _require_cuda()
    state = _strided_state(11, 3, 5)
    baseline = state.clone()
    indices = torch.tensor([-1, 2, 2, 99, 4], device=DEVICE, dtype=torch.int64)
    rows = torch.randn_like(state[: indices.numel()])
    write_mask = torch.tensor(
        [False, False, True, True, True], device=DEVICE, dtype=torch.bool
    )
    expected = baseline.clone()
    for row, index in enumerate(indices.tolist()):
        if bool(write_mask[row]) and 0 <= index < state.shape[0]:
            expected[index].copy_(rows[row])
    ple_state_scatter_(state, indices, rows, write_mask=write_mask)
    torch.testing.assert_close(state, expected, atol=0, rtol=0)
    torch.testing.assert_close(state[0], baseline[0], atol=0, rtol=0)
    with pytest.raises(NotImplementedError):
        ple_state_scatter_(baseline, indices, rows)


def test_qsa_exports_and_vendor_source_are_explicit():
    from flaggems_vllm.ops.qwen4 import qsa as qsa_module

    names = set(qsa_module.__all__)
    assert {
        "_qsa_mqa_paged_dot_kernel",
        "_store_qsa_kv_rows_kernel",
        "_compress_norm_mrope_store_qsa_groups_kernel",
    } <= names
    assert {
        "_compress_qsa_groups_kernel",
        "_qsa_mqa_paged_kernel",
        "_store_qsa_rows_kernel",
        "qwen4_vendor_compress_qsa_groups",
        "qwen4_vendor_qsa_mqa_paged",
        "qwen4_vendor_store_qsa_rows",
    } <= names
    assert QWEN4_VENDOR_QSA_SOURCE["repository"].endswith("vllm-plugin-FL")
    assert QWEN4_VENDOR_QSA_SOURCE["base_commit"].startswith("fadbba0")
    assert len(QWEN4_VENDOR_QSA_SOURCE["sha256"]) == 64


def test_qwen4_cpu_guards_fail_closed():
    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((16,), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        qwen4_grouped_gemma_rmsnorm(x, weight, 4, EPS)

    cache = torch.empty((1, 16, 1, 128), dtype=torch.bfloat16)
    q = torch.empty((1, 4, 128), dtype=torch.bfloat16)
    table = torch.zeros((1, 1), dtype=torch.int32)
    metadata = torch.zeros((1,), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        qwen4_qsa_mqa_paged_dot(q, cache, table, metadata, metadata, metadata)
    with pytest.raises(RuntimeError):
        qwen4_vendor_qsa_mqa_paged(q, cache, table, metadata, metadata, metadata)

    state = torch.empty((2, 3, 5), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        ple_state_gather(state, torch.zeros((1,), dtype=torch.int64))
