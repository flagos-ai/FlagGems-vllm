# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Numerical and mutation contract of the preallocated Kimi-K3 decode op."""

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert
OP = gems.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert
DTYPES = (torch.bfloat16, torch.float16)
BLOCK_SIZE = 8


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu_reference():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _views(batch, heads, dtype, layout):
    generator = torch.Generator().manual_seed(613)
    device = gems.device

    def values(shape, offset=0):
        numbers = torch.randn(shape, generator=generator) + offset
        return numbers.to(dtype=dtype, device=device)

    if layout == "dense":
        ql = values((batch, heads, 512))
        qp = values((batch, heads, 64), 4)
        kp = values((batch, 64), -8)
    else:
        # Projected queries and shared PE come from different packed buffers.
        # Also exercise independent token/head strides across the two Q inputs.
        ql = values((heads, batch, 525))[..., 3:515].transpose(0, 1)
        qp = values((batch, heads, 83), 4)[..., 5:69]
        kp = values((batch, 91), -8)[:, 1:65]
    kv = values((batch, 512), -2)

    # Include payloads which are distinguishable only by their 16-bit encoding.
    # The first entries of the NoPE rows must preserve -0 and NaN payloads.
    for source in (ql, kv):
        source.view(torch.int16)[..., 0] = -32768
        source.view(torch.int16)[..., 1] = 32705

    return ql, qp, kv, kp


def _slot_ids(batch, mode):
    if mode == "negative":
        return -1 - torch.arange(batch) % 3
    # Leave holes on both ends of ordinary pages; include first/last slots of
    # adjacent pages explicitly. Every nonnegative ID is unique.
    candidates = [0, BLOCK_SIZE - 1, BLOCK_SIZE]
    candidates.extend(range(BLOCK_SIZE + 2, BLOCK_SIZE + 2 + 2 * batch, 2))
    slots = torch.tensor(candidates[:batch], dtype=torch.long)
    if mode == "mixed":
        slots[3::4] = -7
    return slots


def _cache(batch, dtype, layout):
    pages = (BLOCK_SIZE + 2 * batch + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    row_stride = 576 if layout == "dense" else 583
    page_stride = BLOCK_SIZE * row_stride + (0 if layout == "dense" else 13)
    guard = 16 if layout == "dense" else 5
    backing = torch.full(
        (guard + pages * page_stride + 16,), -97.5, dtype=dtype, device=gems.device
    )
    return backing, backing.as_strided(
        (pages, BLOCK_SIZE, 576), (page_stride, row_stride, 1), guard
    )


def _device_indices(values, strided):
    if not strided:
        return values.to(gems.device)
    # Odd entries are guards; both slot and position guards are valid indices
    # so a wrong stride produces a numerical mismatch without a device fault.
    storage = torch.zeros(2 * values.numel(), dtype=torch.long)
    storage[::2] = values
    return storage.to(gems.device)[::2]


def _rope_table(batch, strided=False):
    positions = torch.arange(batch, 0, -1, dtype=torch.long)
    angles = torch.arange(batch + 1)[:, None] / torch.arange(1, 33)[None, :]
    table = torch.cat((angles.cos(), angles.sin()), dim=1)
    return _device_indices(positions, strided), table.to(gems.device)


def _rotate(source, positions, table):
    cos, sin = table[positions].chunk(2, dim=-1)
    if source.ndim == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    x, y = source[..., ::2].float(), source[..., 1::2].float()
    return (
        torch.stack((x * cos - y * sin, x * sin + y * cos), dim=-1)
        .flatten(-2)
        .to(source.dtype)
    )


def _bits(tensor):
    return tensor.cpu().view(torch.int16)


def _assert_rotated(actual, expected):
    # FP32 fused and unfused multiply-adds can round differently at a midpoint.
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rope", [False, True], ids=["copy", "rope"])
@pytest.mark.parametrize(
    "batch,heads,layout,slots",
    [
        (0, 1, "dense", "valid"),
        (1, 1, "dense", "valid"),
        (1, 7, "packed", "negative"),
        (7, 0, "dense", "mixed"),
        (9, 1, "packed", "negative"),
        (9, 7, "dense", "valid"),
        (63, 7, "packed", "mixed"),
        (64, 7, "packed", "mixed"),
        (65, 7, "packed", "mixed"),
    ],
)
@pytest.mark.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert
def test_fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
    dtype, rope, batch, heads, layout, slots
):
    ql, qp, kv, kp = _views(batch, heads, dtype, layout)
    if not rope:
        for source in (qp, kp):
            source.view(torch.int16)[..., 0] = -32768
            source.view(torch.int16)[..., 1] = 32705
    before = [t.cpu().clone() for t in (ql, qp, kv, kp)]
    query = torch.full((batch, heads, 576), -97.5, device=gems.device, dtype=dtype)
    backing, cache = _cache(batch, dtype, layout)
    slot_ids = _slot_ids(batch, slots)
    strided_indices = batch == 65
    device_slots = _device_indices(slot_ids, strided_indices)
    expected_backing = backing.cpu().clone()
    expected_rows = expected_backing.as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    )
    expected_qp, expected_kp = before[1], before[3]
    optional = ()
    if rope:
        positions, table = _rope_table(batch, strided_indices)
        expected_qp = _rotate(expected_qp, positions.cpu(), table.cpu())
        expected_kp = _rotate(expected_kp, positions.cpu(), table.cpu())
        optional = (positions, table)

    tokens = torch.where(slot_ids >= 0)[0]
    page, row = slot_ids[tokens] // BLOCK_SIZE, slot_ids[tokens] % BLOCK_SIZE
    expected_rows[page, row, :512] = before[2][tokens]
    expected_rows[page, row, 512:] = expected_kp[tokens]
    result = OP(ql, qp, kv, kp, query, cache, device_slots, BLOCK_SIZE, *optional)
    assert result is None
    actual_query = query.cpu()
    assert torch.equal(_bits(actual_query[..., :512]), _bits(before[0]))
    actual_backing = backing.cpu()
    if rope:
        _assert_rotated(actual_query[..., 512:], expected_qp)
        actual_rows = actual_backing.as_strided(
            cache.shape, cache.stride(), cache.storage_offset()
        )
        _assert_rotated(actual_rows[page, row, 512:], expected_kp[tokens])
        # Remove only the already-checked arithmetic lanes from the subsequent
        # bit comparison; this keeps every padding byte and unwritten row in it.
        actual_rows[page, row, 512:] = expected_kp[tokens]
    else:
        assert torch.equal(_bits(actual_query[..., 512:]), _bits(expected_qp))
    assert torch.equal(_bits(actual_backing), _bits(expected_backing))
    for actual, original in zip((ql, qp, kv, kp), before):
        assert torch.equal(_bits(actual), _bits(original))


@pytest.mark.parametrize("batch", [0, 2])
@pytest.mark.parametrize("missing", ["positions", "table"])
def test_incomplete_rope_pair_preserves_outputs(batch, missing):
    inputs = _views(batch, 3, torch.bfloat16, "packed")
    query = torch.full((batch, 3, 576), -97.5, dtype=torch.bfloat16, device=gems.device)
    backing, cache = _cache(batch, torch.bfloat16, "packed")
    initial_q, initial_cache = _bits(query).clone(), _bits(backing).clone()
    slots = _slot_ids(batch, "valid").to(gems.device)
    positions, table = _rope_table(batch)
    with pytest.raises((ValueError, RuntimeError)):
        OP(
            *inputs,
            query,
            cache,
            slots,
            BLOCK_SIZE,
            None if missing == "positions" else positions,
            None if missing == "table" else table,
        )
    assert torch.equal(_bits(query), initial_q)
    assert torch.equal(_bits(backing), initial_cache)


def test_query_grid_exceeds_65535_work_items():
    # A single decode token with 3 programs per head and one cache program
    # would need 65536 entries on grid.y. Broadcast only the read-only heads.
    heads = 21845
    dtype = torch.float16
    latent = torch.arange(512, dtype=dtype).reshape(1, 1, 512)
    pe = -torch.arange(64, dtype=dtype).reshape(1, 1, 64)
    ql = latent.to(gems.device).expand(1, heads, 512)
    qp = pe.to(gems.device).expand(1, heads, 64)
    kv = torch.full((1, 512), -17, device=gems.device, dtype=dtype)
    kp = torch.full((1, 64), 33, device=gems.device, dtype=dtype)
    out = torch.full((1, heads, 576), -91, device=gems.device, dtype=dtype)
    cache = torch.full((1, 1, 576), -91, device=gems.device, dtype=dtype)
    slots = torch.zeros(1, device=gems.device, dtype=torch.long)
    assert OP(ql, qp, kv, kp, out, cache, slots, 1) is None
    expected = torch.cat((latent, pe), dim=-1).expand(1, heads, 576)
    assert torch.equal(_bits(out), _bits(expected))
    assert torch.equal(_bits(cache[0]), _bits(torch.cat((kv.cpu(), kp.cpu()), dim=1)))


@pytest.mark.skipif(gems.vendor_name != "nvidia", reason="NVIDIA guarded launch path")
@pytest.mark.parametrize("rope", [False, True], ids=["copy", "shared-position-rope"])
@torch.inference_mode()
def test_repeated_decode_reads_current_storage(rope):
    def check():
        out.fill_(-91)
        cache.fill_(-91)
        expected_qp, expected_kp = qp.cpu(), kp.cpu()
        if rope:
            expected_qp = _rotate(expected_qp, positions.cpu(), table.cpu())
            expected_kp = _rotate(expected_kp, positions.cpu(), table.cpu())
        expected_out = torch.cat((ql.cpu(), expected_qp), dim=-1)
        expected_cache = cache.cpu().clone()
        expected_cache[0, slots.cpu()] = torch.cat((kv.cpu(), expected_kp), dim=-1)
        assert OP(ql, qp, kv, kp, out, cache, slots, 4, positions, table) is None
        assert torch.equal(_bits(out), _bits(expected_out))
        assert torch.equal(_bits(cache), _bits(expected_cache))

    # Keep both generations alive so fresh tensors cannot recycle old addresses.
    live = []
    for version in (1, 2):
        ql, qp, kv, kp = _views(2, 2, torch.float16, "dense")
        for payload in (ql, qp, kv, kp):
            payload[..., 2:].add_(version)
        out = torch.empty((2, 2, 576), dtype=ql.dtype, device=ql.device)
        cache = torch.empty((1, 4, 576), dtype=ql.dtype, device=ql.device)
        slots = torch.tensor([1, 0], device=ql.device)
        positions = table = None
        if rope:
            positions = (
                torch.tensor([2, 1], device=ql.device) if version == 1 else slots
            )
            # Quarter turns give exact FP16 results for the bit comparisons.
            table = torch.tensor(
                [[1, 0], [0, 1], [-1, 0]], dtype=torch.float32, device=ql.device
            ).repeat_interleave(32, dim=1)
        live.append((ql, qp, kv, kp, out, cache, slots, positions, table))
        check()

    aligned = qp.detach()
    # Rebind the same tensor object to fresh data with the same shape and strides.
    storage = torch.full((qp.numel() + 1,), -13.0, dtype=qp.dtype, device=qp.device)
    misaligned = storage[1:].view_as(qp)
    assert aligned.data_ptr() % 16 == 0 and misaligned.data_ptr() % 16 == 2
    qp.set_(misaligned)
    check()
    qp.set_(aligned)
    check()

    # The underlying storage remains large enough even for an erroneous launch.
    qp.set_(aligned[..., :63])
    out.fill_(-91)
    cache.fill_(-91)
    with pytest.raises(ValueError):
        OP(ql, qp, kv, kp, out, cache, slots, 4, positions, table)
    assert torch.all(out == -91).item()
    assert torch.all(cache == -91).item()


@pytest.mark.skipif(gems.vendor_name != "nvidia", reason="NVIDIA decode dispatch")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rope", [False, True], ids=["copy", "rope"])
def test_decode_switches_between_equal_numel_shapes(dtype, rope):
    # The first and second shapes have the same query numel. One has complete
    # groups of four heads and the other needs the general path.
    for batch, heads, layout in (
        (65, 64, "dense"),
        (64, 65, "dense"),
        (65, 64, "dense"),
        (65, 64, "packed"),
    ):
        test_fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
            dtype, rope, batch, heads, layout, "mixed"
        )
