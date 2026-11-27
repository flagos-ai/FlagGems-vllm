# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent CPU reference for DeepSeek V4 overlapping C4 cache insertion."""

import inspect
from dataclasses import dataclass

import pytest
import torch


def _raw(tensor):
    size = tensor.untyped_storage().nbytes() // tensor.element_size()
    return tensor.detach().as_strided((size,), (1,), 0).view(torch.uint8).cpu()


def _view(raw, tensor):
    return raw.view(tensor.dtype).as_strided(
        tensor.shape, tensor.stride(), tensor.storage_offset()
    )


def _layout(tensor):
    return tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.dtype


@dataclass
class Expected:
    initial: dict
    layouts: dict
    stores: list


def _compressed(state, pages, block_size, pos):
    # Keep the original eight indices when the first window starts before zero.
    history = []
    for j in range(8):
        time = pos - 7 + j
        if time >= 0:
            logical_page, offset = divmod(time, block_size)
            if logical_page >= len(pages):
                return None
            page = int(pages[logical_page])
            if not 0 <= page < state.shape[0]:
                return None
            history.append(state[page, offset].reshape(2, 2, 512)[:, j // 4])
    values, logits = torch.stack(history).unbind(dim=1)
    weights = (logits - logits.amax(dim=0)).softmax(dim=0)
    return (values * weights).sum(dim=0), (values.abs() * weights).sum(dim=0)


def _rounded_bounds(x, magnitude, weight, eps, cos_sin):
    # Permit a different rounded neighbor only where FP32 error in the
    # weighted sum, RMS reduction, or complex rotation can cross a BF16 midpoint.
    rrms = (x.square().mean() + eps).rsqrt()
    y = (x * rrms) * weight.float()
    work = magnitude * (rrms * weight.float()).abs()
    cos, sin = cos_sin.chunk(2)

    def rotate(row, c, s, signed):
        a, b = row[448::2], row[449::2]
        real = a * c - b * s if signed else a * c + b * s
        return torch.cat((row[:448], torch.stack((real, a * s + b * c), 1).flatten()))

    result = rotate(y, cos, sin, True)
    error = (
        32 * torch.finfo(torch.float32).eps * rotate(work, cos.abs(), sin.abs(), False)
    )
    return (result - error).bfloat16().float(), (result + error).bfloat16().float()


def reference(
    state_cache,
    token_to_req_indices,
    positions,
    slot_mapping,
    block_table,
    block_size,
    rms_norm_weight,
    rms_norm_eps,
    cos_sin_cache,
    k_cache,
    kv_slot_mapping,
    kv_cache_block_size,
    kv_block_stride,
    head_size=512,
    state_width=1024,
    rope_head_dim=64,
    fp8_max=448.0,
    quant_block=64,
    token_stride=576,
    scale_dim=8,
    compress_ratio=4,
    overlap=True,
    store_full_kv=False,
    store_full_fp8=False,
    fp8_scale=None,
):
    """Snapshot backing allocations and compute each active store before the call."""
    tensors = {k: x for k, x in locals().items() if isinstance(x, torch.Tensor)}
    initial = {k: _raw(x).clone() for k, x in tensors.items()}
    cpu = {k: _view(initial[k], x) for k, x in tensors.items()}
    stores = []
    for slot, pos, req, dst in zip(
        cpu["slot_mapping"].tolist(),
        cpu["positions"].tolist(),
        cpu["token_to_req_indices"].tolist(),
        cpu["kv_slot_mapping"].tolist(),
    ):
        rotary_pos = 4 * (pos // 4)
        if (
            slot < 0
            or pos < 0
            or (pos + 1) % 4
            or not 0 <= req < block_table.shape[0]
            or not 0 <= dst < k_cache.shape[0] * kv_cache_block_size
            or rotary_pos >= cos_sin_cache.shape[0]
        ):
            continue
        row = _compressed(cpu["state_cache"], cpu["block_table"][req], block_size, pos)
        if row is None:
            continue
        x, magnitude = row
        bounds = _rounded_bounds(
            x,
            magnitude,
            cpu["rms_norm_weight"],
            rms_norm_eps,
            cpu["cos_sin_cache"][rotary_pos],
        )
        stores.append((dst, *bounds))
    return Expected(initial, {k: _layout(x) for k, x in tensors.items()}, stores)


def _check_bounds(actual, lo, hi):
    valid = (lo <= actual) & (actual <= hi)
    if not valid.all():
        i = int((~valid).flatten().nonzero()[0].item())
        raise AssertionError(
            f"element {i} has {actual.flatten()[i]}, expected "
            f"[{lo.flatten()[i]}, {hi.flatten()[i]}]"
        )


def _fp8(values, limit):
    limit = min(448.0, limit)
    return values.clamp(-limit, limit).to(torch.float8_e4m3fn).float()


def _exponent(magnitude, limit):
    scale = (magnitude.amax(dim=1) / limit).clamp_min(1e-4)
    return torch.ceil(torch.log2(scale.double())) + 127


def _compare_store(raw, initial, slot, lo, hi, options):
    cache = options["k_cache"]
    size = cache.element_size()
    block = options["kv_cache_block_size"]
    page, offset = divmod(slot, block)
    page_base = (cache.storage_offset() + page * cache.stride(0)) * size
    limit = options["fp8_max"]
    if options["store_full_kv"]:
        if options["store_full_fp8"]:
            scale = options["fp8_scale"]
            inv = 1.0 if scale is None else scale.cpu().flatten()[0].reciprocal()
            lo, hi = _fp8(lo * inv, limit), _fp8(hi * inv, limit)
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.bfloat16
        start = page_base + offset * cache.stride(1) * size
        span = slice(start, start + 512 * dtype.itemsize)
        _check_bounds(raw[span].clone().view(dtype).float(), lo, hi)
        raw[span] = initial[span]
        return

    start = page_base + offset * options["token_stride"]
    scale_base = (
        page_base + block * options["token_stride"] + offset * options["scale_dim"]
    )
    scales = raw[scale_base : scale_base + 8]
    assert scales[7] == 0, "UE8M0 pad byte"
    low, high = lo[:448].reshape(7, 64), hi[:448].reshape(7, 64)
    low_abs = torch.where(low * high <= 0, 0.0, torch.minimum(low.abs(), high.abs()))
    high_abs = torch.maximum(low.abs(), high.abs())
    _check_bounds(scales[:7], _exponent(low_abs, limit), _exponent(high_abs, limit))
    scale = torch.exp2(scales[:7].float() - 127).unsqueeze(1)
    fp8 = raw[start : start + 448].view(torch.float8_e4m3fn).float().reshape(7, 64)
    _check_bounds(fp8, _fp8(low / scale, limit), _fp8(high / scale, limit))
    bf16 = raw[start + 448 : start + 576].clone().view(torch.bfloat16).float()
    _check_bounds(bf16, lo[448:], hi[448:])
    raw[start : start + 576] = initial[start : start + 576]
    scales.copy_(initial[scale_base : scale_base + 8])


def assert_result(expected, result, *args, **kwargs):
    """Check the return contract, all immutable inputs and every destination byte."""
    assert result is None
    bound = inspect.signature(reference).bind(*args, **kwargs)
    bound.apply_defaults()
    options = bound.arguments
    for name, layout in expected.layouts.items():
        assert _layout(options[name]) == layout, name
        if name != "k_cache":
            assert torch.equal(_raw(options[name]), expected.initial[name]), name
    initial = expected.initial["k_cache"]
    actual = _raw(options["k_cache"]).clone()
    for slot, lo, hi in expected.stores:
        _compare_store(actual, initial, slot, lo, hi, options)
    assert torch.equal(actual, initial), "cache bytes outside active stores"


def _build(mode="packed", weight_dtype=torch.float32, padded=False):
    rng = torch.Generator().manual_seed(804)
    block_size, page_size = 4, 4
    block_table = torch.tensor([[4, 1, 6], [2, 5, 0]], dtype=torch.int32)
    state = torch.randn(7, block_size, 2064 if padded else 2048, generator=rng)
    state = state[..., :2048]
    angles = torch.randn(12, 32, generator=rng)
    trig = torch.empty(12, 80 if padded else 64)
    trig.fill_(-33)
    trig[:, :64] = torch.cat([angles.cos(), angles.sin()], dim=1)
    dtype = dict(packed=torch.uint8, bf16=torch.bfloat16, fp8=torch.float8_e4m3fn)[mode]
    width = 584 if mode == "packed" else 512
    row_stride = width + (16 if padded and mode != "packed" else 0)
    page_stride = page_size * row_stride + (32 if padded else 0)
    raw = torch.full(
        ((5 * page_stride + 32) * dtype.itemsize,), 0xAD, dtype=torch.uint8
    )
    cache = raw.view(dtype).as_strided(
        (5, page_size, width), (page_stride, row_stride, 1), 16
    )
    return dict(
        state_cache=state,
        token_to_req_indices=torch.tensor([0, 1, 1, 0, 1, 1, 0], dtype=torch.int32),
        positions=torch.tensor([2, 3, 4, 7, 11, 7, 11]),
        slot_mapping=torch.tensor([5, 0, 3, 15, 1, -1, 12]),
        block_table=block_table,
        block_size=block_size,
        rms_norm_weight=torch.randn(512, generator=rng).to(weight_dtype),
        rms_norm_eps=1e-6,
        cos_sin_cache=trig[:, :64],
        k_cache=cache,
        kv_slot_mapping=torch.tensor([1, 11, 5, 0, 6, 2, -1]),
        kv_cache_block_size=page_size,
        kv_block_stride=page_stride,
        store_full_kv=mode != "packed",
        store_full_fp8=mode == "fp8",
    )


def _to_device(case, device):
    return {
        k: _view(_raw(x).to(device), x) if isinstance(x, torch.Tensor) else x
        for k, x in case.items()
    }


def _run(case):
    import flaggems_vllm

    case = _to_device(case, flaggems_vllm.device)
    expected = reference(**case)
    result = flaggems_vllm.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
        **case
    )
    assert_result(expected, result, **case)
    return case


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
def test_fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
    mode, weight_dtype, padded
):
    case = _build(mode, weight_dtype, padded)
    if mode == "fp8" and padded:
        case["fp8_scale"] = torch.tensor([0.375])
    _run(case)


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
@pytest.mark.parametrize(
    "short", ["positions", "slot_mapping", "token_to_req_indices", "kv_slot_mapping"]
)
def test_short_metadata(mode, short):
    case = _build(mode, padded=True)
    case[short] = case[short][:2]
    _run(case)


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
@pytest.mark.parametrize("empty_batch", [False, True])
def test_empty(mode, empty_batch):
    case = _build(mode)
    case["positions"] = case["positions"][:0]
    if empty_batch:
        for name in ("slot_mapping", "kv_slot_mapping", "token_to_req_indices"):
            case[name] = case[name][:0]
    _run(case)


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_overlapping_banks_and_featurewise_softmax():
    case = _build("bf16", padded=True)
    case["positions"] = torch.tensor([3, 7])
    case["token_to_req_indices"] = torch.tensor([0, 0], dtype=torch.int32)
    case["kv_slot_mapping"] = torch.tensor([0, 6])
    case["rms_norm_weight"].fill_(1)
    state = case["state_cache"]
    for time in range(8):
        page = case["block_table"][0, time // 4]
        rows = state[page, time % 4].view(2, 2, 512)
        rows[0, 0].fill_(2)
        rows[0, 1].fill_(10)
        rows[0, :, 1] = 4
        rows[1].zero_()
        # The two banks have different maxima in adjacent columns. Both
        # maxima overflow unstabilized exp() in FP32.
        rows[1, 0, 2:4] = torch.tensor([100, 98])
        rows[1, 1, 2:4] = torch.tensor([98, 100])
    case = _run(case)
    # Check the cache against hand-derived compressed rows. Column 1 anchors
    # the RMS scale, making the 10 (first window) vs 6 (full window) observable.
    first = torch.full((512,), 10.0)
    first[1] = 4
    second = torch.full((512,), 6.0)
    second[1] = 4
    fraction = 1 / (1 + torch.exp(torch.tensor(2.0)))
    second[2], second[3] = 2 + 8 * fraction, 10 - 8 * fraction
    raw = _view(_raw(case["k_cache"]), case["k_cache"])
    for slot, row in ((0, first), (6, second)):
        normed = row / (row.square().mean() + case["rms_norm_eps"]).sqrt()
        assert torch.equal(raw[slot // 4, slot % 4, :448], normed[:448].bfloat16())


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
def test_store_rounding(mode):
    case = _build(mode, padded=True)
    case["positions"] = torch.tensor([7])
    case["kv_slot_mapping"] = torch.tensor([5])
    case["state_cache"][:, :, :1024] = 1
    case["state_cache"][:, :, 1024:] = 0
    case["rms_norm_eps"] = 0
    weight = case["rms_norm_weight"].zero_()
    weight[64:71] = torch.tensor(
        [448, 1.0625, 1.1875, -1.0625, -1.1875, 1.06445, -1.06445]
    )
    weight[448:452] = torch.tensor([2.0, 3.0, 1.002, 1.0])
    # Row 4 rotates the first pair by a quarter turn. The second pair
    # reveals any BF16 rounding that occurred before the rotation.
    case["cos_sin_cache"][4, :32] = 0
    case["cos_sin_cache"][4, 32:] = 1
    case["cos_sin_cache"][4, [1, 33]] = 2**-0.5
    if mode == "fp8":
        weight[128:130] = torch.tensor([900.0, -800.0])
    case = _run(case)
    cache = case["k_cache"]
    raw = _raw(cache)
    page = (cache.storage_offset() + cache.stride(0)) * cache.element_size()
    ties = [0x7E, 0x38, 0x3A, 0xB8, 0xBA, 0x38, 0xB8]
    if mode == "packed":
        start = page + 576
        scales = page + 4 * 576 + 8
        assert raw[scales : scales + 8].tolist() == [
            114,
            127,
            114,
            114,
            114,
            114,
            114,
            0,
        ]
        assert raw[start : start + 64].count_nonzero() == 0
        assert raw[start + 64 : start + 71].tolist() == ties
        assert raw[start + 448 : start + 452].view(torch.bfloat16).tolist() == [-3, 2]
    elif mode == "fp8":
        start = page + cache.stride(1)
        assert raw[start + 64 : start + 71].tolist() == ties
        assert raw[start + 128 : start + 130].tolist() == [0x7E, 0xFE]
        assert raw[start + 448 : start + 450].tolist() == [0xC4, 0x40]
    else:
        row = _view(raw, cache)[1, 1]
        assert row[69:71].tolist() == [1.0625, -1.0625]
        assert row[448:450].tolist() == [-3, 2]


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_explicit_packed_strides():
    case = _build(padded=True)
    token_stride, scale_dim = 608, 16
    width = token_stride + scale_dim
    stride = 4 * width + 32
    raw = torch.full((5 * stride + 32,), 0xAD, dtype=torch.uint8)
    case.update(
        token_stride=token_stride,
        scale_dim=scale_dim,
        k_cache=raw.as_strided((5, 4, width), (stride, width, 1), 16),
        kv_block_stride=stride,
    )
    _run(case)


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_state_pages_of_five_and_strided_metadata():
    case = _build("bf16", padded=True)
    rng = torch.Generator().manual_seed(505)
    case["block_size"] = 5
    # The window ending at 11 spans rows [4], [5..9], [10..11] in
    # three different pages. Feature step two interleaves guard elements.
    case["state_cache"] = torch.randn(7, 5, 4104, generator=rng)[..., 3:4099:2]
    for name in (
        "slot_mapping",
        "kv_slot_mapping",
        "token_to_req_indices",
        "positions",
    ):
        original = case[name]
        buffer = torch.full((3 * original.numel(),), -17, dtype=original.dtype)
        case[name] = buffer[1::3].copy_(original)
    table = torch.full((2, 8), -7, dtype=torch.int32)
    case["block_table"] = table[:, 1:7:2].copy_(case["block_table"])
    _run(case)


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "fp8"])
@pytest.mark.parametrize("limit", [224.0, 896.0])
def test_custom_fp8_limit(mode, limit):
    case = _build(mode, padded=True)
    case["positions"] = torch.tensor([3])
    case["kv_slot_mapping"] = torch.tensor([0])
    case["state_cache"][:, :, :1024] = 1
    case["state_cache"][:, :, 1024:] = 0
    case["rms_norm_eps"] = 0
    case["rms_norm_weight"].fill_(1)
    case["rms_norm_weight"][:2] = torch.tensor([448, -448])
    case["fp8_max"] = limit
    cache = _run(case)["k_cache"]
    raw = _raw(cache)
    start = cache.storage_offset() * cache.element_size()
    # E4M3FN encodes 224 and 448 as 0x76 and 0x7e respectively.
    saturated = 0x76 if limit == 224 else 0x7E
    one = 0x38
    if mode == "packed":
        exponent, one = (128, 0x30) if limit == 224 else (126, 0x40)
        assert raw[start + 4 * 576] == exponent
    assert raw[start : start + 3].tolist() == [saturated, saturated | 0x80, one]


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_invalid_metadata_with_one_valid_store():
    case = _build(padded=True)
    # Only token zero may write. Request one has an invalid first and last
    # physical page. Request zero also exercises a missing logical page.
    case["block_table"][1] = torch.tensor([-1, 2, 7], dtype=torch.int32)
    case["cos_sin_cache"] = torch.cat((case["cos_sin_cache"], case["cos_sin_cache"]))
    case["positions"] = torch.tensor([3, -1, 3, 3, 3, 3, 3, 11, 15])
    case["token_to_req_indices"] = torch.tensor(
        [0, 0, -1, 2, 0, 0, 1, 1, 0], dtype=torch.int32
    )
    case["slot_mapping"] = torch.arange(9)
    case["kv_slot_mapping"] = torch.tensor([0, 1, 2, 3, 20, -1, 6, 7, 8])
    assert [slot for slot, *_ in reference(**case).stores] == [0]
    _run(case)


@pytest.mark.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize(
    "invalid",
    [
        {"compress_ratio": 128},
        {"overlap": False},
        {"state_width": 512},
        {"store_full_fp8": True},
        {"head_size": 256},
        {"rope_head_dim": 32},
        {"quant_block": 32},
    ],
)
def test_invalid_nonempty(invalid):
    import flaggems_vllm

    case = _to_device(_build(), flaggems_vllm.device)
    expected = reference(**case)
    with pytest.raises(ValueError):
        flaggems_vllm.fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            **{**case, **invalid}
        )
    for name, initial in expected.initial.items():
        assert torch.equal(_raw(case[name]), initial), name
