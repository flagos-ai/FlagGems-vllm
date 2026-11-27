# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU reference for the two public, independently predicated C128 mutations."""

import inspect
from dataclasses import dataclass

import pytest
import torch


def _storage_bytes(x):
    count = x.untyped_storage().nbytes() // x.element_size()
    return x.detach().as_strided((count,), (1,), 0).view(torch.uint8).cpu()


def _layout(x):
    return x.shape, x.stride(), x.storage_offset(), x.dtype


def _view(raw, x):
    return raw.view(x.dtype).as_strided(x.shape, x.stride(), x.storage_offset())


@dataclass
class Expected:
    original: dict
    layouts: dict
    compressed: dict
    stores: list


def reference(
    state_cache,
    token_to_req_indices,
    positions,
    slot_mapping,
    block_table,
    block_size,
    compressed_kv,
    rms_norm_weight,
    rms_norm_eps,
    cos_sin_cache,
    k_cache,
    kv_slot_mapping,
    kv_cache_block_size,
    kv_block_stride,
    head_size=512,
    state_width=512,
    rope_head_dim=64,
    fp8_max=448.0,
    quant_block=64,
    token_stride=576,
    scale_dim=8,
    compress_ratio=128,
    overlap=False,
    store_full_kv=False,
    store_full_fp8=False,
    fp8_scale=None,
):
    """Snapshot all backing storage and compute the expected workspace rows."""
    inputs = {name: x for name, x in locals().items() if isinstance(x, torch.Tensor)}
    raw = {name: _storage_bytes(x).clone() for name, x in inputs.items()}
    cpu = {name: _view(raw[name], x) for name, x in inputs.items()}
    compressed = {}
    stores = []
    reqs = cpu["token_to_req_indices"]
    kv_slots = cpu["kv_slot_mapping"]
    for t, (slot, pos) in enumerate(zip(cpu["slot_mapping"], cpu["positions"])):
        pos = int(pos)
        if slot < 0 or (pos + 1) % 128:
            continue
        start = pos - 127
        if t < reqs.numel():
            history = torch.arange(start, pos + 1)
            pages = cpu["block_table"][reqs[t], history // 8].long()
            rows = cpu["state_cache"][pages, history % 8].double()
            values, logits = rows.split(512, dim=1)
            compressed[t] = (values * logits.softmax(dim=0)).sum(dim=0).float()
        if t < kv_slots.numel() and kv_slots[t] >= 0:
            stores.append((t, start, int(kv_slots[t])))
    return Expected(
        raw, {name: _layout(x) for name, x in inputs.items()}, compressed, stores
    )


def _bf16_bounds(x, weight, eps, cos_sin):
    # RMS normalization and interleaved complex rotation are performed before
    # rounding to BF16. The bound covers FP32 tree reduction, rsqrt, products
    # and addition. Propagate magnitudes through RoPE to account for cancellation.
    x = x.double()
    y = x * torch.rsqrt(x.square().mean() + eps) * weight.double()
    c, s = cos_sin.double().chunk(2)
    a, b = y[448::2], y[449::2]
    rotated = torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten()
    mag = torch.stack(
        ((a * c).abs() + (b * s).abs(), (a * s).abs() + (b * c).abs()), dim=-1
    )
    value = torch.cat((y[:448], rotated))
    error = (
        16 * torch.finfo(torch.float32).eps * torch.cat((y[:448].abs(), mag.flatten()))
    )
    return tuple(v.float().bfloat16().float() for v in (value - error, value + error))


def _check_bounds(actual, lower, upper):
    inside = (actual >= lower) & (actual <= upper)
    if not inside.all():
        i = int(torch.where(~inside.flatten())[0][0])
        raise AssertionError(
            f"element {i}: {actual.flatten()[i]} outside "
            f"[{lower.flatten()[i]}, {upper.flatten()[i]}]"
        )


def _fp8(x, fp8_max):
    limit = min(fp8_max, 448.0)
    return x.clamp(-limit, limit).to(torch.float8_e4m3fn).float()


def _check_store(cache, initial, base, scale_base, bounds, options):
    """Consume one write, restoring its checked bytes for the final guard check."""
    lower, upper = bounds
    fp8_max = options["fp8_max"]
    if options["store_full_kv"]:
        if options["store_full_fp8"]:
            dtype = torch.float8_e4m3fn
            inv_scale = 1.0
            if options["fp8_scale"] is not None:
                inv_scale = options["fp8_scale"].cpu().float().reciprocal()
            lower, upper = _fp8(lower * inv_scale, fp8_max), _fp8(
                upper * inv_scale, fp8_max
            )
        else:
            dtype = torch.bfloat16
        span = slice(base, base + 512 * dtype.itemsize)
        _check_bounds(cache[span].clone().view(dtype).float(), lower, upper)
        cache[span] = initial[span]
        return

    scales = slice(scale_base, scale_base + 8)
    stored = cache[scales]
    assert stored[7] == 0, "UE8M0 padding byte"
    lo = lower[:448].reshape(7, 64)
    hi = upper[:448].reshape(7, 64)
    min_abs = torch.where(lo * hi <= 0, 0.0, torch.minimum(lo.abs(), hi.abs()))
    max_abs = torch.maximum(lo.abs(), hi.abs())

    def exponent(magnitudes):
        return (magnitudes.amax(dim=1) / fp8_max).clamp_min(
            1.0e-4
        ).double().log2().ceil() + 127

    _check_bounds(stored[:7], exponent(min_abs), exponent(max_abs))
    scale = torch.exp2(stored[:7].float() - 127)[:, None]
    nope = cache[base : base + 448].view(torch.float8_e4m3fn).float().reshape(7, 64)
    _check_bounds(nope, _fp8(lo / scale, fp8_max), _fp8(hi / scale, fp8_max))
    # A packed page may start at an odd byte offset in its backing storage.
    rope = cache[base + 448 : base + 576].clone().view(torch.bfloat16).float()
    _check_bounds(rope, lower[448:], upper[448:])
    cache[base : base + 576] = initial[base : base + 576]
    cache[scales] = initial[scales]


def assert_result(expected, result, *args, **kwargs):
    assert result is None
    call = inspect.signature(reference).bind(*args, **kwargs)
    call.apply_defaults()
    p = call.arguments
    for name, layout in expected.layouts.items():
        assert _layout(p[name]) == layout, name
        if name not in ("k_cache", "compressed_kv"):
            assert torch.equal(_storage_bytes(p[name]), expected.original[name]), name

    workspace = p["compressed_kv"]
    initial = expected.original["compressed_kv"]
    actual_raw = _storage_bytes(workspace).clone()
    actual = _view(actual_raw, workspace)
    for t, row in expected.compressed.items():
        # FP32 normalized weighted reduction over 128 terms.
        torch.testing.assert_close(actual[t], row, atol=2e-6, rtol=1e-5)

    original_cache = expected.original["k_cache"]
    cache = _storage_bytes(p["k_cache"]).clone()
    cache_view = p["k_cache"]
    elem_size = cache_view.element_size()
    cs = _view(expected.original["cos_sin_cache"], p["cos_sin_cache"])
    weight = _view(expected.original["rms_norm_weight"], p["rms_norm_weight"])
    for t, start, slot in expected.stores:
        page, offset = divmod(slot, p["kv_cache_block_size"])
        page_base = (
            cache_view.storage_offset() + page * cache_view.stride(0)
        ) * elem_size
        row_stride = (
            cache_view.stride(1) * elem_size
            if p["store_full_kv"]
            else p["token_stride"]
        )
        base = page_base + offset * row_stride
        scale_base = (
            page_base
            + p["kv_cache_block_size"] * p["token_stride"]
            + offset * p["scale_dim"]
        )
        # Stage two reads the validated FP32 row, including the caller's
        # original workspace when the request metadata ends before the KV slots.
        bounds = _bf16_bounds(actual[t], weight, p["rms_norm_eps"], cs[start])
        _check_store(cache, original_cache, base, scale_base, bounds, p)
    assert torch.equal(cache, original_cache), "bytes outside cache writes"

    old_workspace = _view(initial, workspace)
    for t in expected.compressed:
        actual[t].copy_(old_workspace[t])
    assert torch.equal(actual_raw, initial), "bytes outside workspace writes"


def _build(mode="packed", dtype=torch.float32, *, padded=False, shuffled=True):
    rng = torch.Generator().manual_seed(218)
    block_size = 4
    reqs = torch.tensor([1, 0, 1, 0, 0, 1, 1], dtype=torch.int32)
    positions = torch.tensor([126, 127, 128, 255, 127, 255, 127])
    slots = torch.tensor([3, 120, 5, 17, -1, 15, 12, 0])
    kv_slots = torch.tensor([4, 5, 8, 0, 1, -1])
    pages = (
        (torch.randperm(64, generator=rng) if shuffled else torch.arange(64))
        .int()
        .view(2, 32)
    )
    state = torch.randn(64, 8, 1040 if padded else 1024, generator=rng)[:, :, :1024]
    # Large per-feature offsets require a stable softmax. They cancel exactly
    # in the mathematical reference. Different logits select different rows.
    state[..., 512:] += torch.linspace(-96, 96, 512)
    workspace_storage = torch.arange(slots.numel() * 512 + 8, dtype=torch.float32)
    workspace = workspace_storage[4:-4].view(-1, 512).mul_(0.01).add_(0.25)
    weight = torch.randn(512, generator=rng).to(dtype)
    phases = torch.randn(256, 32, generator=rng)
    cos_sin = torch.cat((phases.cos(), phases.sin()), dim=1)
    cache_dtype = {
        "packed": torch.uint8,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }[mode]
    row_width = 584 if mode == "packed" else 512
    row_stride = row_width + (16 if padded and mode != "packed" else 0)
    page_stride = block_size * row_stride + (32 if padded else 0)
    raw = torch.full(
        ((3 * page_stride + 32) * cache_dtype.itemsize,), 0xAD, dtype=torch.uint8
    )
    cache = raw.view(cache_dtype).as_strided(
        (3, block_size, row_width), (page_stride, row_stride, 1), 16
    )
    return dict(
        state_cache=state,
        token_to_req_indices=reqs,
        positions=positions,
        slot_mapping=slots,
        block_table=pages,
        block_size=8,
        compressed_kv=workspace,
        rms_norm_weight=weight,
        rms_norm_eps=1e-6,
        cos_sin_cache=cos_sin,
        k_cache=cache,
        kv_slot_mapping=kv_slots,
        kv_cache_block_size=block_size,
        kv_block_stride=cache.stride(0),
        store_full_kv=mode != "packed",
        store_full_fp8=mode == "fp8",
    )


def _to_device(case, device):
    return {
        name: (
            _view(_storage_bytes(x).to(device), x) if isinstance(x, torch.Tensor) else x
        )
        for name, x in case.items()
    }


def _check(case):
    import flaggems_vllm

    case = _to_device(case, flaggems_vllm.device)
    expected = reference(**case)
    result = flaggems_vllm.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
        **case
    )
    assert_result(expected, result, **case)
    return case


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
def test_split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(mode, dtype, padded):
    case = _build(mode, dtype, padded=padded, shuffled=padded)
    if mode == "fp8" and padded:
        case["fp8_scale"] = torch.tensor([0.03125])
    _check(case)


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
@pytest.mark.parametrize(
    "scenario", ["short_req", "negative_kv", "empty_positions", "empty_batch"]
)
def test_split_predicates(mode, scenario):
    case = _build(mode, padded=True)
    if scenario == "short_req":
        # Token three stores its unchanged caller-provided workspace at slot zero.
        case["token_to_req_indices"] = case["token_to_req_indices"][:2]
    elif scenario == "negative_kv":
        case["positions"] = torch.tensor([126, 127, 128])
        case["kv_slot_mapping"] = torch.tensor([1, -1, 5])
    else:
        case["positions"] = case["positions"][:0]
        if scenario == "empty_batch":
            for name in (
                "slot_mapping",
                "token_to_req_indices",
                "kv_slot_mapping",
                "compressed_kv",
            ):
                case[name] = case[name][:0]
    _check(case)


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
def test_exact_c128_bytes(mode):
    case = _build(mode, padded=True)
    case["positions"] = torch.tensor([127])
    case["kv_slot_mapping"] = torch.tensor([5])
    req = int(case["token_to_req_indices"][0])
    pages = case["block_table"][req, :16].long()
    case["state_cache"][pages, :, :512] = torch.arange(128.0).view(16, 8, 1)
    case["state_cache"][pages, :, 512:] = 0
    case["rms_norm_eps"] = 0
    weight = case["rms_norm_weight"].zero_()
    # 448 forces UE8M0 exponent 127 in group one. 1.0625 and 1.1875
    # are E4M3 midpoints with opposite even neighbors. 1.064453125 first
    # rounds to BF16 1.0625, testing the intermediate conversion.
    weight[64:71] = torch.tensor(
        [448, 1.0625, 1.1875, -1.0625, -1.1875, 1.064453125, -1.064453125]
    )
    if mode == "fp8":
        weight[128:130] = torch.tensor([900, -800])
    weight[448:452] = torch.tensor([2, 3, 1.002, 1])
    case["cos_sin_cache"][0, :32] = 0
    case["cos_sin_cache"][0, 32:] = 1
    # Rounding (1.002, 1) before rotation would erase the real component.
    case["cos_sin_cache"][0, [1, 33]] = 2**-0.5
    case = _check(case)
    assert torch.equal(case["compressed_kv"][0].cpu(), torch.full((512,), 63.5))
    cache = case["k_cache"]
    raw = _storage_bytes(cache)
    page = (cache.storage_offset() + cache.stride(0)) * cache.element_size()
    ties = torch.tensor([0x7E, 0x38, 0x3A, 0xB8, 0xBA, 0x38, 0xB8], dtype=torch.uint8)
    if mode == "packed":
        base = page + 576
        scale = page + 4 * 576 + 8
        assert raw[scale : scale + 8].tolist() == [
            0x72,
            0x7F,
            0x72,
            0x72,
            0x72,
            0x72,
            0x72,
            0,
        ]
        assert not raw[base : base + 64].any()
        assert torch.equal(raw[base + 64 : base + 71], ties)
        assert raw[base + 448 : base + 452].view(torch.bfloat16).tolist() == [-3, 2]
    elif mode == "fp8":
        base = page + cache.stride(1)
        assert torch.equal(raw[base + 64 : base + 71], ties)
        assert raw[base + 128 : base + 130].tolist() == [0x7E, 0xFE]
        assert raw[base + 448 : base + 450].tolist() == [0xC4, 0x40]
    else:
        row = raw.view(torch.bfloat16)
        base = cache.storage_offset() + cache.stride(0) + cache.stride(1)
        assert row[base + 69 : base + 71].tolist() == [1.0625, -1.0625]
        assert row[base + 448 : base + 450].tolist() == [-3, 2]


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_legacy_packed_strides():
    case = _build(padded=True)
    token_stride, scale_dim = 608, 16
    row_width = token_stride + scale_dim
    block_size = case["kv_cache_block_size"]
    page_stride = block_size * row_width + 32
    raw = torch.full((3 * page_stride + 32,), 0xAD, dtype=torch.uint8)
    case.update(
        k_cache=raw.as_strided(
            (3, block_size, row_width), (page_stride, row_width, 1), 16
        ),
        kv_block_stride=page_stride,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )
    _check(case)
    # The explicit-geometry mirror also accepts a custom quantization limit.
    # Start at an odd byte to exercise the BF16 payload inside the raw page.
    case.update(
        fp8_max=224.0,
        k_cache=raw.as_strided(
            (3, block_size, row_width), (page_stride, row_width, 1), 17
        ),
    )
    _check(case)


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
def test_featurewise_softmax_across_state_pages():
    case = _build(padded=True)
    case["positions"] = torch.tensor([255])
    case["kv_slot_mapping"] = torch.tensor([0])
    req = int(case["token_to_req_indices"][0])
    pages = case["block_table"][req, 16:32].long()
    case["state_cache"][pages, :, :512] = torch.arange(128.0).view(16, 8, 1)
    features = torch.arange(512)
    first = (features * 37) % 128
    second = (first + 64) % 128
    # Each feature has two competing maxima in different tiles. exp(100)
    # overflows FP32; independently averaging the tiles also gives a wrong row.
    logits = torch.full((128, 512), -160.0)
    logits[first, features] = 100
    logits[second, features] = 98
    case["state_cache"][pages, :, 512:] = logits.view(16, 8, 512)
    case = _check(case)
    expected = first + (second - first) / (1.0 + torch.exp(torch.tensor(2.0)))
    torch.testing.assert_close(
        case["compressed_kv"][0].cpu(), expected, atol=2e-6, rtol=1e-5
    )


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize(
    "change",
    [
        {"compress_ratio": 4},
        {"overlap": True},
        {"block_size": 16},
        {"store_full_fp8": True},
        {"head_size": 256},
        {"state_width": 1024},
        {"rope_head_dim": 32},
        {"quant_block": 32},
        {"kv_cache_block_size": 8},
        {"rms_norm_weight": torch.ones(512, dtype=torch.float16)},
    ],
)
def test_split_invalid_nonempty(change):
    import flaggems_vllm

    case = _build()
    case.update(change)
    with pytest.raises(ValueError):
        flaggems_vllm.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
            **_to_device(case, flaggems_vllm.device)
        )


@pytest.mark.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl
@pytest.mark.parametrize("mode", ["packed", "bf16", "fp8"])
def test_empty_positions_reject_invalid_ratio(mode):
    import flaggems_vllm

    case = _build(mode)
    case["positions"] = case["positions"][:0]
    case["compress_ratio"] = 4
    case = _to_device(case, flaggems_vllm.device)
    expected = reference(**case)
    with pytest.raises(ValueError):
        flaggems_vllm.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(**case)
    assert_result(expected, None, **case)
