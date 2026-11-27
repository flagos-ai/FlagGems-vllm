# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm as gems


def _rotate(x, cos, sin):
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    return (
        torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        .flatten(-2)
        .to(x.dtype)
    )


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
    """Apply the prefill epilogue to CPU views, preserving their backing storage."""
    assert q.device.type == "cpu"
    if position_ids is not None:
        cos, sin = cos_sin_cache[position_ids].split(32, dim=-1)
        q[..., 128:] = _rotate(q[..., 128:], cos[:, None], sin[:, None])
        k_pe = _rotate(k_pe, cos, sin)
    k_out[..., :128] = k_nope
    k_out[..., 128:] = k_pe[:, None, :]
    for token, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            page, offset = divmod(slot, cache_block_size)
            k_cache[page, offset, :512] = kv_c_normed[token]
            k_cache[page, offset, 512:] = k_pe[token]


def _shuffled_slots(tokens, capacity):
    return [(-3 if t % 7 == 2 else (31 * t) % capacity) for t in range(tokens)]


CASES = [
    pytest.param(3, 8, [], False, id="empty"),
    pytest.param(1, 1, [0], False, id="single"),
    pytest.param(3, 8, [-1], True, id="single-negative"),
    pytest.param(2, 8, [-1, 0, -1, 5], False, id="upstream-mixed"),
    pytest.param(5, 3, [8, 0, -1, 3, 2, -7, 7, 5, 6], True, id="page-boundaries"),
    pytest.param(3, 4, [-1, -2, -1, -7, -3, -1, -1], True, id="all-negative"),
    pytest.param(0, 4, [4, -1, 0], True, id="cache-only"),
    pytest.param(9, 16, _shuffled_slots(33, 48), True, id="tail-heads"),
    pytest.param(16, 16, _shuffled_slots(65, 80), False, id="multi-page"),
]


def _arguments(buffers, heads, block_size, packed, rope):
    offset = 8 if packed else 0
    args = {
        "q": buffers["q"][:, :heads, offset : offset + 192],
        "k_nope": buffers["k_nope"][:, :heads, :128],
        "k_pe": buffers["k_pe"][:, -64:],
        "kv_c_normed": buffers["kv_c_normed"],
        "k_out": buffers["k_out"][1:-1],
        "k_cache": buffers["k_cache"][:, :block_size, offset : offset + 576],
        "slot_mapping": buffers["slot_mapping"],
        "cache_block_size": block_size,
    }
    if rope:
        args.update(
            position_ids=buffers["position_ids"], cos_sin_cache=buffers["cos_sin_cache"]
        )
    return args


def _make_buffers(heads, block_size, slots, packed, rope, dtype):
    tokens = len(slots)
    pages = max(0, max(slots, default=0)) // block_size + 1
    rng = torch.Generator(device="cpu").manual_seed(8732)

    def sample(*shape):
        return torch.randint(-127, 128, shape, generator=rng, device="cpu").to(dtype)

    buffers = {
        "q": sample(tokens, heads + packed, 216 if packed else 192),
        "k_nope": sample(tokens, heads + packed, 256 if packed else 128),
        "k_pe": sample(tokens, 576 if packed else 64),
        "kv_c_normed": sample(tokens, 512),
        "k_out": torch.full((tokens + 2, heads, 192), -811, dtype=dtype, device="cpu"),
        "k_cache": sample(pages, block_size + packed, 608 if packed else 576),
        "slot_mapping": torch.tensor(slots, dtype=torch.int64, device="cpu"),
    }
    if rope:
        positions = (torch.arange(tokens, device="cpu") * 11 + 7) % 37
        angles = torch.outer(
            torch.arange(1, 38, dtype=torch.float32, device="cpu"),
            torch.arange(1, 33, dtype=torch.float32, device="cpu") / 41,
        )
        # Dyadic coefficients keep each product exact in FP32 for either input
        # dtype, so an FMA and separate FP32 operations agree even on replay.
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        buffers.update(
            position_ids=positions, cos_sin_cache=(table * 4096).round() / 4096
        )

    args = _arguments(buffers, heads, block_size, packed, rope)
    # Signed zero, infinities and distinct NaN payloads in the bit-copy paths.
    inf = 0x7C00 if dtype == torch.float16 else 0x7F80
    specials = torch.tensor(
        [-32768, 0, inf, inf - 32768, inf + 65, inf + 71], dtype=torch.int16
    )
    for name in ("q", "k_nope", "kv_c_normed"):
        args[name].view(torch.int16)[..., :6] = specials
    if not rope:
        args["q"].view(torch.int16)[..., 128:134] = specials
        args["k_pe"].view(torch.int16)[..., :6] = specials
    return buffers


def _device_buffers(buffers):
    # Transfer 16-bit payloads verbatim, including the non-canonical NaNs.
    return {
        name: (
            value.view(torch.int16).to(gems.device, copy=True).view(value.dtype)
            if value.element_size() == 2
            else value.to(gems.device, copy=True)
        )
        for name, value in buffers.items()
    }


def _bits(tensor):
    if tensor.element_size() == 2:
        return tensor.view(torch.int16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    return tensor


@pytest.fixture(scope="module", autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("heads,block_size,slots,packed", CASES)
@pytest.mark.parametrize("rope", [False, True], ids=["copy", "rope"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_prefill(heads, block_size, slots, packed, rope, dtype):
    expected = _make_buffers(heads, block_size, slots, packed, rope, dtype)
    actual = _device_buffers(expected)
    cpu_args = _arguments(expected, heads, block_size, packed, rope)
    args = _arguments(actual, heads, block_size, packed, rope)
    for poison in (-521, 789):
        # Keep the query from the preceding invocation; every call rotates it once.
        args["k_out"].fill_(poison)
        cpu_args["k_out"].fill_(poison)
        reference_inplace(**cpu_args)
        assert gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args) is None
        for name in expected:
            torch.testing.assert_close(
                _bits(actual[name]).cpu(),
                _bits(expected[name]),
                rtol=0,
                atol=0,
                msg=lambda detail: f"{name}: {detail}",
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_full_precision_rope_with_strided_indices(dtype):
    expected = _make_buffers(3, 4, [4, -1, 0], True, True, dtype)
    angles = torch.arange(37 * 32, dtype=torch.float32).reshape(37, 32) / 29
    expected["cos_sin_cache"] = torch.cat((angles.cos(), angles.sin()), dim=-1)
    # Isolate each product so the exact rounded result also holds with FMA.
    before = _arguments(expected, 3, 4, True, True)
    before["q"][..., 129::2] = 0
    before["k_pe"][:, 1::2] = 0
    for name in ("slot_mapping", "position_ids"):
        values = expected[name]
        storage = torch.zeros((len(values), 2), dtype=torch.int64)
        storage[:, 1] = values
        expected[name] = storage
    actual = _device_buffers(expected)
    args = _arguments(actual, 3, 4, True, True)
    cpu_args = _arguments(expected, 3, 4, True, True)
    for name in ("slot_mapping", "position_ids"):
        cpu_args[name] = expected[name][:, 1]
        args[name] = actual[name][:, 1]
    reference_inplace(**cpu_args)
    assert gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args) is None
    for name in expected:
        torch.testing.assert_close(
            _bits(actual[name]).cpu(), _bits(expected[name]), rtol=0, atol=0
        )


def _strided_last_dim(tensor):
    return tensor.new_empty((*tensor.shape[:-1], 2 * tensor.shape[-1]))[..., ::2]


@pytest.mark.parametrize(
    "name,invalid",
    [
        pytest.param("q", lambda x: x[:, 0], id="q-rank"),
        pytest.param("q", lambda x: x[:, :-1], id="q-heads"),
        pytest.param("q", lambda x: x[:-1], id="q-tokens"),
        pytest.param("q", lambda x: x[..., :191], id="q-width"),
        pytest.param("q", _strided_last_dim, id="q-inner-stride"),
        pytest.param("k_nope", lambda x: x[..., :127], id="nope-width"),
        pytest.param("k_nope", _strided_last_dim, id="nope-inner-stride"),
        pytest.param("k_pe", lambda x: x[:, None], id="shared-pe-rank"),
        pytest.param("k_pe", lambda x: x[:, :63], id="pe-width"),
        pytest.param("k_pe", _strided_last_dim, id="pe-inner-stride"),
        pytest.param("k_pe", lambda x: x.bfloat16(), id="pe-dtype"),
        pytest.param("k_pe", lambda x: x.cpu(), id="pe-device"),
        pytest.param("kv_c_normed", lambda x: x[:, :511], id="latent-width"),
        pytest.param("kv_c_normed", _strided_last_dim, id="latent-contiguity"),
        pytest.param("k_out", lambda x: x[:, :1].contiguous(), id="out-heads"),
        pytest.param("k_out", _strided_last_dim, id="out-contiguity"),
        pytest.param("k_cache", lambda x: x[..., :575], id="cache-width"),
        pytest.param("k_cache", _strided_last_dim, id="cache-inner-stride"),
        pytest.param("k_cache", lambda x: x.bfloat16(), id="cache-dtype"),
        pytest.param("cache_block_size", lambda n: n + 1, id="block-size"),
        pytest.param("cache_block_size", lambda n: float(n), id="block-size-type"),
        pytest.param("slot_mapping", lambda x: x.int(), id="slots-dtype"),
        pytest.param("slot_mapping", lambda x: x[:-1], id="slots-length"),
        pytest.param("slot_mapping", lambda x: x[:, None], id="slots-rank"),
        pytest.param("position_ids", lambda x: x.int(), id="positions-dtype"),
        pytest.param("position_ids", lambda x: x[:-1], id="positions-length"),
        pytest.param("cos_sin_cache", lambda x: x.half(), id="cos-sin-dtype"),
        pytest.param("cos_sin_cache", lambda x: x[..., :63], id="cos-sin-width"),
        pytest.param("cos_sin_cache", _strided_last_dim, id="cos-sin-inner-stride"),
        pytest.param(
            "cos_sin_cache",
            lambda x: x.new_empty((len(x), 72))[:, :64],
            id="cos-sin-row-stride",
        ),
    ],
)
@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_invalid_inputs(name, invalid):
    buffers = _make_buffers(2, 4, [-1, 0, 3], False, True, torch.float16)
    args = _arguments(_device_buffers(buffers), 2, 4, False, True)
    args[name] = invalid(args[name])
    before = {key: _bits(args[key]).cpu().clone() for key in ("q", "k_out", "k_cache")}
    with pytest.raises((ValueError, TypeError)):
        gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args)
    for key, saved in before.items():
        assert torch.equal(_bits(args[key]).cpu(), saved)


@pytest.mark.parametrize("slots", [[], [0, -1]], ids=["empty", "nonempty"])
@pytest.mark.parametrize("missing", ["position_ids", "cos_sin_cache"])
@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_optional_rope_pair(slots, missing):
    expected = _make_buffers(1, 4, slots, False, True, torch.float16)
    actual = _device_buffers(expected)
    args = _arguments(actual, 1, 4, False, True)
    del args[missing]
    with pytest.raises((ValueError, TypeError)):
        gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args)
    for name in ("q", "k_out", "k_cache"):
        assert torch.equal(_bits(actual[name]).cpu(), _bits(expected[name]))


@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_unsupported_dtype():
    buffers = _make_buffers(2, 4, [0, -1], False, True, torch.float16)
    args = _arguments(_device_buffers(buffers), 2, 4, False, True)
    for name in ("q", "k_nope", "k_pe", "kv_c_normed", "k_out", "k_cache"):
        args[name] = args[name].float()
    with pytest.raises(TypeError):
        gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args)


@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_cpu_unsupported():
    buffers = _make_buffers(2, 4, [0, -1], False, True, torch.float16)
    args = _arguments(buffers, 2, 4, False, True)
    with pytest.raises(NotImplementedError):
        gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args)


@pytest.mark.parametrize("rope", [False, True], ids=["copy", "rope"])
@pytest.mark.fused_kimi_k3_mla_key_concat_kv_cache_insert
def test_rebound_projection_views(rope):
    # Keep every allocation alive. Matching shape/strides/dtype must still read
    # today's pointers; the last PE view also has a different alignment.
    live = []
    for attempt in range(3):
        expected = _make_buffers(3, 4, [4, -1, 0, 3], True, rope, torch.bfloat16)
        for i, name in enumerate(("q", "k_nope", "k_pe", "kv_c_normed")):
            expected[name].fill_(3 * attempt + i + 1)
        actual = _device_buffers(expected)
        live.append(actual)
        cpu_args = _arguments(expected, 3, 4, True, rope)
        args = _arguments(actual, 3, 4, True, rope)
        if attempt == 2:
            cpu_args["k_pe"] = expected["k_pe"][:, -65:-1]
            args["k_pe"] = actual["k_pe"][:, -65:-1]
        reference_inplace(**cpu_args)
        assert gems.fused_kimi_k3_mla_key_concat_kv_cache_insert(**args) is None
        for name in expected:
            assert torch.equal(_bits(actual[name]).cpu(), _bits(expected[name])), name
