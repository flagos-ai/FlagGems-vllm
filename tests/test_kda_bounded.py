# SPDX-License-Identifier: Apache-2.0
"""GLM bounded KDA regressions against the unchanged vLLM FLA core."""

import pytest
import torch
import triton
import triton.language as tl

from flaggems_vllm.ops import kda_bounded as bounded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@triton.jit
def _reference_cumsum(
    G,
    A,
    BIAS,
    CU,
    INDICES,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    VARLEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Original PR safe-gate cumsum: the FP32 tl.dot accumulation is deliberate.
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    if VARLEN:
        seq = tl.load(INDICES + block * 2)
        block = tl.load(INDICES + block * 2 + 1)
        start, end = tl.load(CU + seq), tl.load(CU + seq + 1)
        count = end - start
    else:
        start, count = batch * T, T
    ti = block * 64 + tl.arange(0, 64)
    di = tl.arange(0, 128)
    raw = tl.load(
        G + ((start + ti[:, None]) * H + head) * D + di[None, :],
        (ti[:, None] < count) & (di[None, :] < D),
        other=0,
    ).to(tl.float32)
    if HAS_BIAS:
        raw += tl.load(BIAS + head * D + di, di < D, other=0).to(tl.float32)[None, :]
    gate = -5.0 / (1.0 + tl.exp(-(tl.exp(tl.load(A + head).to(tl.float32)) * raw)))
    i = tl.arange(0, 64)
    triangle = tl.where(i[:, None] >= i[None, :], 1.0, 0.0)
    cumulative = tl.dot(triangle, gate, allow_tf32=False) * 1.4426950408889634
    tl.store(
        OUT + ((start + ti[:, None]) * H + head) * D + di[None, :],
        cumulative,
        (ti[:, None] < count) & (di[None, :] < D),
    )


def reference_cumsum(raw, a, bias, cu=None, indices=None):
    b, t, h, d = raw.shape
    out = torch.empty_like(raw, dtype=torch.float32)
    nt = triton.cdiv(t, 64) if cu is None else indices.shape[0]
    if nt:
        _reference_cumsum[(nt, b * h)](
            raw,
            a,
            bias,
            cu,
            indices,
            out,
            t,
            h,
            d,
            cu is not None,
            bias is not None,
            num_warps=4,
        )
    return out


def metadata(lengths):
    from vllm.model_executor.layers.fla.ops.chunk_delta_h import prepare_chunk_offsets
    from vllm.model_executor.layers.fla.ops.kda import prepare_chunk_indices

    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    return cu, prepare_chunk_indices(cu, 64), prepare_chunk_offsets(cu, 64)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("varlen", [False, True])
def test_gate_cumsum_preserves_original_dot_order(dtype, varlen):
    torch.manual_seed(7354)
    raw = torch.randn((1, 131, 3, 64), dtype=dtype, device="cuda")
    a = torch.randn(3, dtype=torch.float32, device="cuda")
    bias = torch.randn(3 * 64, dtype=dtype, device="cuda")
    cu, indices, _ = metadata([1, 65, 65]) if varlen else (None, None, None)
    actual = bounded.safe_kda_gate_chunk_cumsum(
        raw, a, bias, cu_seqlens=cu, chunk_indices=indices
    )
    expected = reference_cumsum(raw, a, bias, cu, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 63, 64, 65, 129])
def test_prefill_matches_original_64_core(dtype, tokens, key_dim=64, value_dim=32):
    from vllm.model_executor.layers.fla.ops import kda

    torch.manual_seed(454 + tokens)
    q = torch.randn((1, tokens, 2, key_dim), dtype=dtype, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn((1, tokens, 2, value_dim), dtype=dtype, device="cuda")
    raw = torch.randn_like(q)
    beta = torch.rand((1, tokens, 2), dtype=torch.float32, device="cuda")
    a = torch.randn(2, dtype=torch.float32, device="cuda")
    bias = torch.randn(2 * key_dim, dtype=dtype, device="cuda")
    initial = (
        torch.randn((1, 2, value_dim, key_dim), dtype=torch.float32, device="cuda")
        * 0.1
    )
    gate = reference_cumsum(raw, a, bias)
    expected, expected_state = kda._chunk_kda_fwd_with_cumulative_g(
        q=kda.l2norm_fwd(q),
        k=kda.l2norm_fwd(k),
        v=v.clone(),
        g=gate,
        beta=beta,
        scale=key_dim**-0.5,
        initial_state=initial.clone(),
        output_final_state=True,
        chunk_size=64,
    )
    actual, state = bounded.chunk_kda_with_safe_gate(
        q,
        k,
        v.clone(),
        raw,
        beta,
        a,
        bias,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert state.shape == (1, 2, value_dim, key_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("beta_value", [0.0, 1.0])
def test_prefill_varlen_keeps_activated_beta_and_vk_state(beta_value):
    from vllm.model_executor.layers.fla.ops import kda

    torch.manual_seed(534)
    dtype = torch.bfloat16
    q = torch.randn((1, 131, 2, 64), dtype=dtype, device="cuda")
    k, raw = torch.randn_like(q), torch.randn_like(q)
    v = torch.randn_like(q)
    beta = torch.full((1, 131, 2), beta_value, device="cuda")
    a = torch.randn(2, device="cuda")
    initial = torch.randn((3, 2, 64, 64), device="cuda") * 0.1
    cu, indices, offsets = metadata([1, 65, 65])
    gate = reference_cumsum(raw, a, None, cu, indices)
    expected, expected_state = kda._chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v.clone(),
        g=gate,
        beta=beta,
        scale=64**-0.5,
        initial_state=initial.clone(),
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=indices,
        chunk_size=64,
    )
    actual, state = bounded.chunk_kda_with_safe_gate(
        q,
        k,
        v.clone(),
        raw,
        beta,
        a,
        None,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=indices,
        chunk_offsets=offsets,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
def test_decode_matches_vllm_vk_state_slots(dtype, inplace, dim=32):
    from vllm.model_executor.layers.fla.ops import kda

    torch.manual_seed(136)
    q = torch.randn((1, 3, 2, dim), dtype=dtype, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    gate = -torch.rand_like(q, dtype=torch.float32)
    beta = torch.rand((1, 3, 2), device="cuda")
    initial = torch.randn((5, 2, dim, dim), dtype=torch.float32, device="cuda")
    actual_state, reference_state = initial.clone(), initial.clone()
    cu = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    slots = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32, device="cuda")
    expected, expected_final = kda.fused_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=reference_state,
        cu_seqlens=cu,
        ssm_state_indices=slots,
        inplace_final_state=inplace,
    )
    actual, final = bounded.fused_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=actual_state,
        cu_seqlens=cu,
        ssm_state_indices=slots,
        inplace_final_state=inplace,
    )
    assert final.shape == expected_final.shape
    assert (final is actual_state) == inplace
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)


def test_prefill_requires_metadata_before_core(monkeypatch):
    q = torch.empty((1, 1, 2, 32), device="cuda", dtype=torch.bfloat16)
    beta = torch.empty((1, 1, 2), device="cuda")
    a = torch.empty(2, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        bounded,
        "_prefill_core",
        lambda: pytest.fail("core loaded before metadata validation"),
    )
    with pytest.raises(ValueError, match="framework-prepared"):
        bounded.chunk_kda_with_safe_gate(q, q, q, q, beta, a, None, cu_seqlens=cu)


def test_decode_validates_vk_state_before_core(monkeypatch):
    q = torch.empty((1, 1, 2, 32), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((1, 1, 2, 16), device="cuda", dtype=torch.bfloat16)
    beta = torch.empty((1, 1, 2), device="cuda")
    wrong = torch.empty((1, 2, 32, 16), device="cuda")
    monkeypatch.setattr(
        bounded,
        "_recurrent_core",
        lambda: pytest.fail("core loaded before state validation"),
    )
    with pytest.raises(ValueError, match="V,K"):
        bounded.fused_recurrent_kda(q, q, v, q, beta, initial_state=wrong)


def test_prefill_execution_error_propagates(monkeypatch):
    q = torch.zeros((1, 1, 2, 32), device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros((1, 1, 2), device="cuda")
    a = torch.zeros(2, device="cuda")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("submitted KDA failed")

    monkeypatch.setattr(bounded, "prefill_kda64", fail)
    with pytest.raises(RuntimeError, match="submitted KDA failed"):
        bounded.chunk_kda_with_safe_gate(q, q, q, q, beta, a, None)
    assert calls == [1]


def test_prefill_model_head_dimension_128():
    test_prefill_matches_original_64_core(torch.bfloat16, 65, 128, 128)


def test_decode_model_head_dimension_128():
    test_decode_matches_vllm_vk_state_slots(torch.bfloat16, True, 128)


def test_safe_gate_chunk_cumsum_resets_at_sequence_and_tiny_chunk():
    import math

    raw = torch.zeros((1, 7, 1, 1), device="cuda")
    a_log = torch.zeros(1, device="cuda")
    boundaries = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    actual = bounded.safe_kda_gate_chunk_cumsum(
        raw, a_log, cu_seqlens=boundaries, chunk_size=2
    ).flatten()
    gate = -2.5 / math.log(2.0)
    expected = torch.tensor(
        [gate, 2 * gate, gate, gate, 2 * gate, gate, 2 * gate], device="cuda"
    )
    torch.testing.assert_close(actual, expected)


def test_safe_gate_matches_fp32_definition():
    import math

    raw = torch.tensor([[1.0, -2.0, 0.5, 3.0]], device="cuda")
    a_log = torch.tensor([math.log(0.5), math.log(2.0)], device="cuda")
    bias = torch.tensor([[0.25, -0.5], [0.0, 1.0]], device="cuda")
    actual = bounded.safe_kda_gate(raw, a_log, 2, bias)
    expected = -5.0 * torch.sigmoid(
        torch.exp(a_log).reshape(1, 2, 1) * (raw.reshape(1, 2, 2) + bias)
    )
    torch.testing.assert_close(actual, expected)


def test_empty_prefill_returns_vk_state_without_native_core(monkeypatch):
    q = torch.empty((1, 0, 2, 32), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((1, 0, 2, 16), device="cuda", dtype=torch.bfloat16)
    beta = torch.empty((1, 0, 2), device="cuda")
    a = torch.empty(2, device="cuda")
    monkeypatch.setattr(
        bounded, "_prefill_core", lambda: pytest.fail("empty input loaded core")
    )
    out, state = bounded.chunk_kda_with_safe_gate(
        q, q, v, q, beta, a, None, output_final_state=True
    )
    assert out.shape == v.shape and out.dtype == v.dtype
    assert state.shape == (1, 2, 16, 32) and state.dtype == torch.float32
    assert torch.count_nonzero(state) == 0


def test_empty_decode_preserves_live_state_without_core(monkeypatch):
    q = torch.empty((1, 0, 2, 32), device="cuda", dtype=torch.bfloat16)
    beta = torch.empty((1, 0, 2), device="cuda")
    initial = torch.randn((1, 2, 32, 32), device="cuda")
    monkeypatch.setattr(
        bounded, "_recurrent_core", lambda: pytest.fail("empty input loaded core")
    )
    out, state = bounded.fused_recurrent_kda(q, q, q, q, beta, initial_state=initial)
    assert out.shape == q.shape and state is initial


def test_multi_token_decode_rejects_1d_slots_before_core(monkeypatch):
    q = torch.empty((1, 3, 2, 32), device="cuda", dtype=torch.bfloat16)
    beta = torch.empty((1, 3, 2), device="cuda")
    initial = torch.empty((5, 2, 32, 32), device="cuda")
    cu = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        bounded, "_recurrent_core", lambda: pytest.fail("invalid slots loaded core")
    )
    with pytest.raises(ValueError, match="Multi-token"):
        bounded.fused_recurrent_kda(
            q,
            q,
            q,
            q,
            beta,
            initial_state=initial,
            cu_seqlens=cu,
            ssm_state_indices=slots,
        )


def test_model_decode_one_token_requests_preserve_1d_slots():
    from vllm.model_executor.layers.fla.ops import kda

    q = torch.randn((1, 2, 2, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    gate = -torch.rand_like(q, dtype=torch.float32)
    beta = torch.rand((1, 2, 2), device="cuda")
    initial = torch.randn((5, 2, 128, 128), device="cuda")
    actual_state, expected_state = initial.clone(), initial.clone()
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    expected, expected_final = kda.fused_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=expected_state,
        cu_seqlens=cu,
        ssm_state_indices=slots,
    )
    actual, actual_final = bounded.fused_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=actual_state,
        cu_seqlens=cu,
        ssm_state_indices=slots,
    )
    assert actual_final is actual_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_final, expected_final, rtol=0, atol=0)


def test_decode_preserves_padded_mamba_cache_slot_stride():
    from vllm.model_executor.layers.fla.ops import kda

    torch.manual_seed(341)
    h, dim, slots, padding = 2, 128, 5, 256
    slot_stride = h * dim * dim + padding
    raw_seed = torch.full((slots * slot_stride,), 1234.0, device="cuda")
    state_shape, state_stride = (slots, h, dim, dim), (slot_stride, dim * dim, dim, 1)
    seed = raw_seed.as_strided(state_shape, state_stride)
    seed.copy_(torch.randn(state_shape, device="cuda"))
    expected_raw, actual_raw = raw_seed.clone(), raw_seed.clone()
    expected_state = expected_raw.as_strided(state_shape, state_stride)
    actual_state = actual_raw.as_strided(state_shape, state_stride)
    assert not actual_state.is_contiguous()
    q = torch.randn((1, 2, h, dim), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    g = -torch.rand_like(q, dtype=torch.float32)
    beta = torch.rand((1, 2, h), device="cuda")
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    expected, _ = kda.fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=expected_state,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )
    actual, final = bounded.fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=actual_state,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )
    assert final is actual_state and final.stride(0) == slot_stride
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_raw, expected_raw, rtol=0, atol=0)
    untouched_padding = actual_raw.as_strided(
        (slots, padding), (slot_stride, 1), storage_offset=h * dim * dim
    )
    assert torch.all(untouched_padding == 1234.0)


def test_prefill_materializes_padded_initial_state_like_original():
    from vllm.model_executor.layers.fla.ops import kda

    torch.manual_seed(642)
    q = torch.randn((1, 66, 2, 64), device="cuda", dtype=torch.bfloat16)
    k, v, raw = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    beta = torch.rand((1, 66, 2), device="cuda")
    a = torch.randn(2, device="cuda")
    initial = torch.empty_strided(
        (2, 2, 64, 64), (2 * 64 * 64 + 256, 64 * 64, 64, 1), device="cuda"
    )
    initial.copy_(torch.randn(initial.shape, device="cuda"))
    cu, indices, offsets = metadata([1, 65])
    gate = reference_cumsum(raw, a, None, cu, indices)
    expected, expected_state = kda._chunk_kda_fwd_with_cumulative_g(
        q=q,
        k=k,
        v=v.clone(),
        g=gate,
        beta=beta,
        scale=64**-0.5,
        initial_state=initial.contiguous(),
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=indices,
        chunk_size=64,
    )
    actual, actual_state = bounded.chunk_kda_with_safe_gate(
        q,
        k,
        v.clone(),
        raw,
        beta,
        a,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=indices,
        chunk_offsets=offsets,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,resolver",
    [
        ("chunk_kda_with_safe_gate", "_prefill_core"),
        ("fused_recurrent_kda", "_recurrent_core"),
    ],
)
def test_public_capability_hook_checks_core_before_execution(
    monkeypatch, name, resolver
):
    calls = []

    def missing():
        calls.append(1)
        raise NotImplementedError("missing ABI")

    monkeypatch.setattr(bounded, resolver, missing)
    hook = getattr(bounded, name)._is_available
    hook.cache_clear()
    try:
        assert hook() is False
        assert hook() is False
        assert calls == [1]
    finally:
        hook.cache_clear()


def test_public_core_capabilities_are_available_in_validation_environment():
    assert bounded.chunk_kda_with_safe_gate._is_available()
    assert bounded.fused_recurrent_kda._is_available()
