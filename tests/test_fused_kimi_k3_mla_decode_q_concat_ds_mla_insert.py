# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Independent CPU reference for Kimi-K3 decode queries and 656-byte cache rows."""

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert
OP = gems.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert


def _rotate(tensor, positions, table):
    cos, sin = table[positions].chunk(2, dim=-1)
    if tensor.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    x, y = tensor.float()[..., ::2], tensor.float()[..., 1::2]
    # Use binary64 for the fused product and sum, then round once to FP32.
    first = (x.double() * cos.double() - (y * sin).double()).float()
    if tensor.ndim == 3:
        second = (x.double() * sin.double() + (y * cos).double()).float()
    else:
        second = (y.double() * cos.double() + (x * sin).double()).float()
    return torch.stack((first, second), -1).flatten(-2).to(tensor.dtype)


def expected_outputs(args):
    ql, qp, kv, kp, _, cache, slots, block_size, *rope = args
    ql, qp, kv, kp, cache, slots = (
        t.detach().cpu() for t in (ql, qp, kv, kp, cache, slots)
    )
    if rope and rope[0] is not None:
        positions, table = (t.detach().cpu() for t in rope)
        qp, kp = _rotate(qp, positions, table), _rotate(kp, positions, table)
    query = torch.cat((ql, qp), dim=-1)
    cache = cache.clone()
    groups = kv.float().reshape(kv.shape[0], 4, 128)
    peak = groups.abs().amax(dim=-1)
    scale = torch.exp2(torch.ceil(torch.log2((peak / 448).clamp_min(1e-4))))
    normalized = (groups / scale.unsqueeze(-1)).clamp(-448, 448)
    payload = normalized.to(torch.float8_e4m3fn).view(torch.uint8).flatten(1)
    records = torch.cat(
        (
            payload,
            scale.contiguous().view(torch.uint8),
            kp.contiguous().view(torch.uint8),
        ),
        dim=1,
    )
    active = slots >= 0
    selected = slots[active]
    cache[selected // block_size, selected % block_size] = records[active]
    return query, cache


def assert_cache_equal(actual, expected):
    actual = actual.detach().cpu()
    for name, start, end in (
        ("payload", 0, 512),
        ("scale", 512, 528),
        ("position", 528, 656),
    ):
        observed = actual[..., start:end]
        target = expected[..., start:end]
        if not torch.equal(observed, target):
            page, row, lane = (observed != target).nonzero()[0].tolist()
            idx = (page, row, lane)
            detail = ""
            if name == "payload":
                offset = 512 + (lane // 128) * 4

                def dequant(row):
                    scale = row[offset : offset + 4].clone().view(torch.float32).item()
                    value = (
                        row[lane : lane + 1].view(torch.float8_e4m3fn).float().item()
                    )
                    return value * scale

                error = dequant(actual[page, row]) - dequant(expected[page, row])
                detail = f" (dequantized error {error:g})"
            raise AssertionError(
                f"{name} mismatch at page {page}, row {row}, byte {start + lane}: "
                f"0x{observed[idx].item():02x} != 0x{target[idx].item():02x}{detail}"
            )


def _latent(batch, dtype):
    # Four distinct per-token scales. Groups 1 and 2 straddle the 448 boundary.
    generator = torch.Generator().manual_seed(114)
    groups = torch.randn((batch, 4, 128), generator=generator).to(dtype)
    groups[:, 0].zero_()
    groups[:, 1, 0] = 448
    groups[:, 2, 0] = 450
    groups[:, 3, 0] = 7
    return groups.flatten(1)


def _make_inputs(batch, heads, dtype, padded, rope, slot_kind):
    device = gems.device
    generator = torch.Generator().manual_seed(581)

    def values(shape):
        return torch.randn(shape, generator=generator).to(dtype=dtype, device=device)

    if padded:
        ql = values((heads, batch, 525))[..., 3:515].transpose(0, 1)
        qp = values((batch, heads, 73))[..., 1:65]
        kp = values((batch, 79))[:, 3:67]
    else:
        ql, qp, kp = (
            values(s) for s in ((batch, heads, 512), (batch, heads, 64), (batch, 64))
        )
    ql.view(torch.int16)[..., 0] = -32768
    if not rope:
        for t in (qp, kp):
            t.view(torch.int16)[..., 0] = -32768
            t.view(torch.int16)[..., 1] = 0x7FC1

    block_size = 3
    pages = max(2, batch + 1)
    row_stride = 663 if padded else 656
    page_stride = block_size * row_stride + (5 if padded else 0)
    offset = 3 if padded else 0
    backing = torch.full(
        (offset + pages * page_stride + 17,), 0xA5, device=device, dtype=torch.uint8
    )
    cache = backing.as_strided(
        (pages, block_size, 656), (page_stride, row_stride, 1), offset
    )
    query_backing = torch.full(
        (batch * heads * 576 + 16,), -91.0, dtype=dtype, device=device
    )
    query = query_backing[8:-8].view(batch, heads, 576)
    ids = torch.arange(batch, dtype=torch.long).flip(0) * block_size
    if batch >= 3:
        ids[0], ids[1] = block_size - 1, block_size
        if batch > 3:
            ids[2:-1] = torch.arange(2, batch - 1) * block_size
    if slot_kind == "mixed":
        ids[1:-1:3] = -3
    elif slot_kind == "negative":
        ids.fill_(-7)

    def indices(cpu):
        if padded:
            tmp = torch.zeros(cpu.numel() * 2, dtype=torch.long)
            tmp[::2] = cpu
            return tmp.to(device)[::2]
        return cpu.to(device)

    optional = ()
    if rope:
        positions = torch.arange(batch, 0, -1)
        # Exact quarter turns vary with BOTH position and feature pair.
        turns = (torch.arange(batch + 1)[:, None] + torch.arange(32)[None, :]) % 4
        cos = torch.tensor([1.0, 0.0, -1.0, 0.0])[turns]
        sin = torch.tensor([0.0, 1.0, 0.0, -1.0])[turns]
        optional = indices(positions), torch.cat((cos, sin), dim=1).to(device)
    args = (
        ql,
        qp,
        _latent(batch, dtype).to(device),
        kp,
        query,
        cache,
        indices(ids),
        block_size,
        *optional,
    )
    return args, query_backing, backing


def _check_case(args, q_storage, storage):
    expected_q, expected_cache = expected_outputs(args)
    expected_storage = storage.cpu().clone()
    cache = args[5]
    expected_storage.as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    ).copy_(expected_cache)
    expected_q_storage = q_storage.cpu().clone()
    expected_q_storage[8:-8].copy_(expected_q.flatten())
    readonly = [*args[:4], args[6], *args[8:]]

    def bits(t):
        t = t.detach().cpu()
        return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t

    snapshot = [bits(t).clone() for t in readonly]
    assert OP(*args) is None
    assert_cache_equal(cache, expected_cache)
    assert torch.equal(
        storage.cpu(), expected_storage
    ), "cache backing or guard corrupted"
    assert torch.equal(
        q_storage.cpu().view(torch.int16), expected_q_storage.view(torch.int16)
    ), "query/guard bytes"
    for inp, before in zip(readonly, snapshot):
        assert torch.equal(bits(inp), before), "read-only input changed"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize(
    "batch,heads,padded,slots",
    [
        (0, 3, False, "valid"),
        (1, 1, False, "valid"),
        (1, 7, True, "negative"),
        (7, 0, True, "mixed"),
        (9, 7, True, "mixed"),
        (16, 8, False, "valid"),
        (33, 5, True, "negative"),
        (65, 7, True, "valid"),
    ],
)
@pytest.mark.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert
def test_fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
    dtype, rope, batch, heads, padded, slots
):
    _check_case(*_make_inputs(batch, heads, dtype, padded, rope, slots))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_fused_rounding(dtype):
    args, query, cache = _make_inputs(1, 3, dtype, True, True, "valid")
    # From unrounded FP32 phase tables. Each dtype includes cancellation
    # cases where contracting the other product changes the stored 16 bits.
    if dtype == torch.bfloat16:
        pairs = [(-0.80078125, 1.6328125), (-2.03125, 0.2373046875)]
        cs = [
            (0.8978443741798401, -0.4403129518032074),
            (0.9932448267936707, 0.11603748053312302),
        ]
        first, query_odd, key_odd = (
            -2.849102020263672e-05,
            5.178153514862061e-07,
            5.252659320831299e-07,
        )
    else:
        pairs = [(-0.407958984375, -0.521484375), (-0.24462890625, -0.16259765625)]
        cs = [
            (0.7902212738990784, 0.612821638584137),
            (0.8301897644996643, -0.5574808716773987),
        ]
        first, query_odd, key_odd = (
            -0.0028018951416015625,
            0.0013895034790039062,
            0.0013885498046875,
        )
    values = torch.tensor(pairs, dtype=dtype).flatten()
    args[1][:, :, :4].copy_(values)
    args[3][:, :4].copy_(values)
    args[8].zero_()
    cos, sin = torch.tensor(cs, dtype=torch.float32).unbind(-1)
    args[9][0, :2].copy_(cos)
    args[9][0, 32:34].copy_(sin)
    # Round both negative zero contributions in the even lane to -0.
    args[1][:, :, 4:6].zero_()
    args[3][:, 4:6].zero_()
    args[9][0, 2] = -0.5
    args[9][0, 34] = 0.75
    _check_case(args, query, cache)
    assert args[4][0, 0, 512:516:3].tolist() == [first, query_odd]
    tail = args[5][0, 0, 528:536].cpu().contiguous().view(dtype)
    assert tail[::3].tolist() == [first, key_odd]


def test_grouped_query_fp16_rounding_midpoints():
    args, query, cache = _make_inputs(8, 64, torch.float16, False, True, "mixed")
    # Three real phase pairs hit an FP16 midpoint only after RN32 FMA.
    # Both sizes and alignment permit the eight-head HCU schedule.
    pairs = torch.tensor(
        [
            [0.5185546875, -1.380859375],
            [0.9404296875, 0.312744140625],
            [1.50390625, 0.4892578125],
        ],
        dtype=torch.float16,
    ).flatten()
    cos = torch.tensor([0.2916037440299988, -0.7258044481277466, -0.5076322555541992])
    sin = torch.tensor([-0.9565392136573792, 0.6879010796546936, 0.8615738749504089])
    args[1][..., :6].copy_(pairs)
    args[3][:, :6].copy_(pairs)
    args[8].zero_()
    args[9][0, :3].copy_(cos)
    args[9][0, 32:35].copy_(sin)
    _check_case(args, query, cache)
    words = args[4][0, 0, 512:518].cpu().view(torch.int16)
    assert words.tolist() == [-17234, -17616, -17618, 14008, -17219, 15408]


@pytest.mark.parametrize("padded", [False, True])
def test_encoding_vector(padded):
    args, q_backing, backing = _make_inputs(
        1, 3, torch.bfloat16, padded, False, "valid"
    )
    # Fixed E4M3FN bytes give a second oracle independent of torch's FP8 cast.
    vals = [
        0.0,
        -0.0,
        2**-10,
        -(2**-10),
        3 * 2**-10,
        5 * 2**-10,
        15 * 2**-10,
        1.0625,
        1.1875,
        -1.1875,
        448.0,
        -448.0,
    ]
    expected = [0, 128, 0, 128, 2, 2, 8, 56, 58, 186, 126, 254]
    args[2][0, 128 : 128 + len(vals)].copy_(torch.tensor(vals, dtype=torch.bfloat16))
    _check_case(args, q_backing, backing)
    row = args[5][0, 0].cpu().contiguous()
    assert row[128 : 128 + len(vals)].tolist() == expected
    assert row[512:528].view(torch.float32).tolist() == [2**-13, 1.0, 2.0, 2**-6]


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_positions",
        "missing_table",
        "width",
        "slot_dtype",
        "cache_dtype",
        "key_rank",
    ],
)
def test_invalid_input_preserves_outputs(invalid):
    args, q, cache = _make_inputs(2, 3, torch.bfloat16, False, True, "valid")
    values = list(args)
    if invalid == "missing_positions":
        values[8] = None
    elif invalid == "missing_table":
        values[9] = None
    elif invalid == "width":
        values[5] = values[5][..., :655]
    elif invalid == "slot_dtype":
        values[6] = values[6].to(torch.int32)
    elif invalid == "cache_dtype":
        values[5] = values[5].view(torch.int8)
    else:
        values[3] = values[3].unsqueeze(1)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError, NotImplementedError, AssertionError)
    ):
        OP(*values)
    assert torch.all(q.cpu() == -91) and torch.all(cache.cpu() == 0xA5)


def test_rebound_buffers_and_metadata():
    # Warm one shape, then alternate two sets of buffers with the same metadata.
    # Both the slot and position vectors also change in place between calls.
    batches = [
        _make_inputs(8, 8, torch.bfloat16, False, True, "valid") for _ in range(2)
    ]
    for i, idx in enumerate((0, 1, 0, 1)):
        args, query, cache = batches[idx]
        query.fill_(-91)
        cache.fill_(0xA5)
        args[0].fill_(i + 1)
        args[2].fill_(2**i)
        args[6].copy_(torch.roll(args[6].cpu(), 1).to(args[6].device))
        args[8].copy_(torch.roll(args[8].cpu(), 2).to(args[8].device))
        _check_case(args, query, cache)
    args, query, cache = batches[0]
    query.fill_(-91)
    cache.fill_(0xA5)
    args[1].transpose_(0, 2)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        OP(*args)
    assert torch.all(query.cpu() == -91) and torch.all(cache.cpu() == 0xA5)
