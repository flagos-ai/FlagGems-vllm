# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.fused_kimi_k3_mla_key_concat_ds_mla_insert


def _rotate(value, cos, sin, cache_writer=False):
    x, y = value[..., ::2].float(), value[..., 1::2].float()
    # Model the CUDA writers' fused multiply-add with a single FP32 rounding.
    even = (x.double() * cos.double() - (y * sin).double()).float()
    if cache_writer:
        odd = (y.double() * cos.double() + (x * sin).double()).float()
    else:
        odd = (x.double() * sin.double() + (y * cos).double()).float()
    return torch.stack((even, odd), dim=-1).flatten(-2).to(value.dtype)


def reference_inplace(
    q,
    k_nope,
    k_pe,
    kv_c_normed,
    k_out,
    k_cache,
    slot_mapping,
    cache_block_size,
    position_ids=None,
    cos_sin_cache=None,
):
    """Compute the three expected mutations on independent CPU inputs."""
    assert q.device.type == "cpu"
    key_tail = cache_tail = k_pe
    if position_ids is not None:
        cos, sin = cos_sin_cache[position_ids].chunk(2, dim=-1)
        q[..., 128:] = _rotate(q[..., 128:], cos[:, None], sin[:, None])
        key_tail = _rotate(k_pe, cos, sin)
        cache_tail = _rotate(k_pe, cos, sin, cache_writer=True)
    k_out[..., :128] = k_nope
    k_out[..., 128:] = key_tail[:, None]

    groups = kv_c_normed.float().reshape(kv_c_normed.size(0), 4, 128)
    peak = groups.abs().amax(dim=-1)
    scale = ((peak / 448.0).clamp_min(1e-4).log2().ceil()).exp2()
    quantized = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    rows = torch.cat(
        (
            quantized.view(torch.uint8).flatten(1),
            scale.contiguous().view(torch.uint8),
            cache_tail.contiguous().view(torch.uint8),
        ),
        dim=1,
    )
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    k_cache[slots // cache_block_size, slots % cache_block_size] = rows[valid]


# Cases keep nonnegative destinations distinct and include slot 0.
CASES = [
    (3, 4, [], False),
    (1, 1, [0], False),
    (7, 3, [-1], True),
    (3, 8, [-1, 0, -1, 5], False),
    (7, 3, [8, 0, -1, 3, 2, -7, 7, 5, 6], True),
    (0, 4, [4, -1, 0], True),
    (9, 16, [-7] * 33, True),
    (16, 16, [(31 * i) % 80 for i in range(65)], False),
]


def _views(store, heads, block, padded, rope):
    start = int(padded)
    slots = store["slots"][::2] if padded else store["slots"]
    q = store["q"].transpose(0, 1) if padded else store["q"]
    nope = store["nope"][:, :heads, start : start + 128]
    args = [
        q[..., start : start + 192],
        nope,
        store["pe"][:, start : start + 64],
        store["kv"],
        store["out"][1:-1],
        store["cache"][:, :block, start : start + 656],
        slots,
        block,
    ]
    if rope:
        ids = store["positions"][::2] if padded else store["positions"]
        args.extend([ids, store["angles"]])
    return args


def _buffers(heads, block, slots, padded, rope, dtype):
    t = len(slots)
    rng = torch.Generator().manual_seed(3181)

    def sample(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.float32).to(dtype)

    def indices(seq):
        vec = torch.tensor(seq, dtype=torch.int64)
        if padded:
            storage = torch.full((2 * len(seq),), 19, dtype=torch.int64)
            storage[::2] = vec
            return storage
        return vec

    # Both FP32 products are exact for these half inputs and dyadic phases.
    # The separate rounding test below uses the original, unrounded phase table.
    phase = torch.arange(37 * 32, dtype=torch.float32).view(37, 32) * 0.125
    pages = max(0, max(slots, default=0)) // block + 2
    store = dict(
        q=sample(heads, t, 197) if padded else sample(t, heads, 192),
        nope=sample(t, heads + int(padded), 137 if padded else 128),
        pe=sample(t, 81 if padded else 64),
        kv=sample(t, 512),
        out=torch.full((t + 2, heads, 192), -91, dtype=dtype),
        cache=torch.full(
            (pages, block + int(padded), 663 if padded else 656),
            0xA5,
            dtype=torch.uint8,
        ),
        slots=indices(slots),
        positions=indices([(19 - 11 * i) % 37 for i in range(t)]),
        angles=(torch.cat((phase.cos(), phase.sin()), dim=1) * 1024).round() / 1024,
    )
    q, nope, pe, kv, *_ = _views(store, heads, block, padded, rope)
    inf = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    special = torch.tensor([-32768, 0, inf, inf + 37], dtype=torch.int16)
    for copied in (q[..., :128], nope):
        copied.view(torch.int16)[..., :4] = special
    if not rope:
        for copied in (q[..., 128:], pe):
            copied.view(torch.int16)[..., :4] = special
    groups = kv.view(t, 4, 128)
    groups[:, 0].zero_()
    groups[:, 1, 0] = 448
    groups[:, 2, 0] = 450
    groups[:, 3, 0] = 7
    return store


def _bits(tensor):
    if tensor.element_size() == 2:
        return tensor.view(torch.int16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    return tensor


def _to_device(store):
    return {
        name: _bits(value).to(gems.device, copy=True).view(value.dtype)
        for name, value in store.items()
    }


def _check(expected, actual, layout):
    wanted = _views(expected, *layout)
    args = _views(actual, *layout)
    reference_inplace(*wanted)
    assert gems.fused_kimi_k3_mla_key_concat_ds_mla_insert(*args) is None
    for name, before in expected.items():
        torch.testing.assert_close(
            _bits(actual[name]).cpu(),
            _bits(before),
            rtol=0,
            atol=0,
            msg=lambda detail: f"{name}: {detail}",
        )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16-wire"]
)
@pytest.mark.parametrize("rope", [False, True], ids=["copy", "rope"])
@pytest.mark.parametrize(
    "heads,block,slots,padded",
    CASES,
    ids=[
        "empty",
        "slot-zero",
        "one-negative",
        "upstream-slots",
        "page-boundaries",
        "cache-only",
        "all-negative",
        "many-pages",
    ],
)
@pytest.mark.fused_kimi_k3_mla_key_concat_ds_mla_insert
def test_fused_kimi_k3_mla_key_concat_ds_mla_insert(
    dtype, rope, heads, block, slots, padded
):
    expected = _buffers(heads, block, slots, padded, rope, dtype)
    actual = _to_device(expected)
    layout = heads, block, padded, rope
    # Both first-call autotune and hot calls must rotate each Q exactly once.
    _check(expected, actual, layout)
    expected["out"].fill_(483)
    actual["out"].fill_(483)
    _check(expected, actual, layout)


@pytest.mark.parametrize("padded", [False, True])
def test_known_record_bytes(padded):
    expected = _buffers(3, 2, [0, -1], padded, False, torch.bfloat16)
    # Signed zero, subnormal ties, normal ties and signed finite extrema.
    values = [
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
    expected["kv"][0, 128 : 128 + len(values)] = torch.tensor(values)
    actual = _to_device(expected)
    layout = 3, 2, padded, False
    _check(expected, actual, layout)
    record = _views(actual, *layout)[5][0, 0].cpu().clone()
    assert record[128 : 128 + len(values)].tolist() == [
        0,
        128,
        0,
        128,
        2,
        2,
        8,
        56,
        58,
        186,
        126,
        254,
    ]
    assert record[512:528].view(torch.float32).tolist() == [2**-13, 1.0, 2.0, 2**-6]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_full_precision_rotary_rounding(dtype):
    store = _buffers(3, 2, [0, -1, 3], True, True, dtype)
    q, _, pe, *_ = _views(store, 3, 2, True, True)
    # Near cancellation crosses a stored BF16/FP16 rounding boundary.
    if dtype == torch.bfloat16:
        xy = [-0.80078125, 1.6328125, -2.03125, 0.2373046875]
        cs = [
            [0.8978443741798401, -0.4403129518032074],
            [0.9932448267936707, 0.11603748053312302],
        ]
    else:
        xy = [-0.407958984375, -0.521484375, -0.24462890625, -0.16259765625]
        cs = [
            [0.7902212738990784, 0.612821638584137],
            [0.8301897644996643, -0.5574808716773987],
        ]
    q[..., 128:132] = pe[:, :4] = torch.tensor(xy, dtype=dtype)
    store["angles"][:, :2], store["angles"][:, 32:34] = torch.tensor(cs).unbind(1)
    actual = _to_device(store)
    _check(store, actual, (3, 2, True, True))


@pytest.mark.parametrize(
    "idx,change",
    [
        (0, lambda x: x[:, :, :191]),
        (1, lambda x: x[:, :1]),
        (2, lambda x: x[:, None]),
        (2, lambda x: x.float()),
        (3, lambda x: x.t().contiguous().t()),
        (4, lambda x: x.transpose(0, 1)),
        (5, lambda x: x[:, :, :655]),
        (5, lambda x: x.view(torch.int8)),
        (6, lambda x: x.int()),
        (6, lambda x: x[:-1]),
        (7, lambda s: s + 1),
        (8, lambda x: None),
        (9, lambda x: None),
        (8, lambda x: x.int()),
        (9, lambda x: x.half()),
        (9, lambda x: x.new_empty(len(x), 65)[:, :64]),
    ],
    ids=[
        "q-width",
        "head-count",
        "shared-key-rank",
        "dtype",
        "latent-layout",
        "out-layout",
        "cache-width",
        "cache-dtype",
        "slot-dtype",
        "slot-length",
        "block-size",
        "missing-positions",
        "missing-table",
        "position-dtype",
        "table-dtype",
        "table-stride",
    ],
)
def test_rejects_invalid_metadata_before_writing(idx, change):
    store = _buffers(2, 4, [0, -1, 3], False, True, torch.bfloat16)
    actual = _to_device(store)
    args = _views(actual, 2, 4, False, True)
    args[idx] = change(args[idx])
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        gems.fused_kimi_k3_mla_key_concat_ds_mla_insert(*args)
    for name in ("q", "out", "cache"):
        assert torch.equal(_bits(actual[name]).cpu(), _bits(store[name]))


@pytest.mark.parametrize("rope", [False, True], ids=["copy", "rope"])
def test_rebound_input_views(rope):
    layout = 7, 3, True, rope
    states = []
    for iteration in range(2):
        stored = _buffers(7, 3, [5, 0, -1, 4], True, rope, torch.bfloat16)
        stored["kv"].mul_(iteration + 1)
        stored["nope"].fill_(7 - iteration)
        stored["pe"].fill_(iteration - 3)
        states.append((stored, _to_device(stored)))

    # Identical metadata can refer to different allocations on every call.
    # Changing the same slot/position vector must also take effect immediately.
    for idx in (0, 1, 0):
        expected, actual = states[idx]
        for key in ("slots", "positions"):
            expected[key][::2] = expected[key][::2].roll(1)
            actual[key].copy_(expected[key])
        _check(expected, actual, layout)
