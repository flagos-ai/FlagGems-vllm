# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import triton

import flaggems_vllm
from flaggems_vllm import gdn_w8a16, vendor_name
from flaggems_vllm.ops.gdn_w8a16 import dequantize_gdn_state_fp8, quantize_gdn_state_fp8
from flaggems_vllm.runtime import torch_device_fn
from tests import conftest as cfg
from tests.accuracy_utils import gems_assert_close, gems_assert_equal

pytestmark = pytest.mark.gdn_w8a16

requires_fp8 = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 GDN requires NVIDIA SM90 or newer",
)

DECODE_N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384]
PREFILL_NT = [
    (1, 2),
    (1, 8),
    (1, 32),
    (4, 4),
    (4, 16),
    (16, 4),
    (1, 256),
    (1, 512),
    (2, 192),
    (4, 256),
    (8, 128),
    (16, 128),
    (32, 64),
    (64, 32),
    (128, 24),
]
SHAPES = [(n, 1) for n in DECODE_N] + PREFILL_NT
if cfg.QUICK_MODE:
    SHAPES = [(1, 1), (32, 1), (128, 1), (1, 8), (2, 192)]


@requires_fp8
def test_gdn_w8a16_registration():
    import flaggems_vllm.ops as ops

    assert "gdn_w8a16" in ops.__all__
    assert flaggems_vllm.gdn_w8a16 is ops.gdn_w8a16
    entries = [entry for entry in flaggems_vllm._FULL_CONFIG if entry[0] == "gdn_w8a16"]
    assert entries == [("gdn_w8a16", flaggems_vllm.gdn_w8a16)]
    assert flaggems_vllm.FULL_CONFIG_BY_FUNC["gdn_w8a16"] == entries
    registered_names = {entry[0] for entry in flaggems_vllm._FULL_CONFIG}
    assert "quantize_gdn_state_fp8" not in registered_names
    assert "dequantize_gdn_state_fp8" not in registered_names


def make_inputs(n, t, h, hv, dtype, packed=True):
    shape = (1, n * t) if packed else (n, t)
    q = torch.randn(*shape, h, 128, device=flaggems_vllm.device, dtype=dtype)
    k = torch.randn_like(q)
    v = 0.125 * torch.randn(*shape, hv, 128, device=q.device, dtype=dtype)
    g = torch.empty(*shape, hv, device=q.device, dtype=dtype).uniform_(
        math.log(0.98), math.log(0.995)
    )
    beta = torch.rand(*shape, hv, device=q.device, dtype=dtype)
    cu = torch.arange(0, n * t + 1, t, device=q.device, dtype=torch.int32)
    indices = torch.arange(n, device=q.device, dtype=torch.int32)
    return q, k, v, g, beta, cu if packed else None, indices


def reference(q, k, v, g, beta, state, n, t):
    h, hv = q.shape[2], v.shape[2]
    q = q.float().reshape(n, t, h, 128)
    k = k.float().reshape(n, t, h, 128)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // h, dim=2)
    k = k.repeat_interleave(hv // h, dim=2)
    v_view = v.reshape(n, t, hv, 128)
    g = g.reshape(n, t, hv)
    beta = beta.reshape(n, t, hv)
    output = torch.empty_like(v_view)
    state = state.float().clone()
    # Bound reference temporary memory for the largest decode batches.
    for start in range(0, n, 128):
        end = min(start + 128, n)
        acc = state[start:end]
        for token in range(t):
            acc = acc * g[start:end, token].float().exp()[..., None, None]
            key = k[start:end, token]
            prediction = torch.einsum("nhk,nhkv->nhv", key, acc)
            residual = (v_view[start:end, token].float() - prediction) * beta[
                start:end, token
            ].float()[..., None]
            acc = acc + key[..., None] * residual[..., None, :]
            output[start:end, token] = torch.einsum(
                "nhk,nhkv->nhv", q[start:end, token], acc
            ).to(v.dtype)
        state[start:end] = acc
    return output.reshape_as(v), state


@pytest.mark.parametrize("h,hv", [(2, 4), (2, 6)])
@pytest.mark.parametrize("n,t", SHAPES)
@requires_fp8
def test_gdn_w8a16_accuracy(h, hv, n, t):
    torch.manual_seed(42)
    q, k, v, g, beta, cu, indices = make_inputs(n, t, h, hv, torch.bfloat16)
    initial = torch.zeros(n, hv, 128, 128, device=q.device, dtype=q.dtype)
    fp8, scales = quantize_gdn_state_fp8(initial)
    expected, expected_state = reference(q, k, v, g, beta, initial, n, t)
    actual, returned_fp8, returned_scale = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )
    assert returned_fp8 is fp8 and returned_scale is scales
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("n,t", [(3, 4), (2, 192)])
@requires_fp8
def test_gdn_w8a16_nonzero_state(dtype, packed, n, t):
    torch.manual_seed(43)
    h, hv = 2, 6
    q, k, v, g, beta, cu, _ = make_inputs(n, t, h, hv, dtype, packed)
    # Strided q/k exercise recurrent address calculations.
    q = torch.stack((q, q), dim=-1)[..., 0]
    k = torch.stack((k, k), dim=-1)[..., 0]
    initial = 0.02 * torch.randn(2 * n, hv, 128, 128, device=q.device, dtype=dtype)
    # Permuted, strided slots must select state and scale together in both paths.
    indices = torch.arange(2 * n - 1, -1, -1, device=q.device, dtype=torch.int64)[::2]
    fp8, scales = quantize_gdn_state_fp8(initial)
    initial = dequantize_gdn_state_fp8(fp8, scales, output_dtype=dtype)
    expected, updated_state = reference(q, k, v, g, beta, initial[indices], n, t)
    expected_state = initial.float().clone()
    expected_state[indices] = updated_state
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)


@pytest.mark.parametrize("n,t", [(4, 1), (32, 1), (3, 3)])
@requires_fp8
def test_gdn_w8a16_speculative(n, t):
    torch.manual_seed(44)
    h, hv, slots = 4, 8, 3
    q, k, v, g, beta, cu, _ = make_inputs(n, t, h, hv, torch.bfloat16)
    indices = torch.randperm(n * slots, device=q.device).reshape(n, slots)
    accepted = torch.arange(n, device=q.device) % slots + 1
    initial = 0.02 * torch.randn(
        n * slots, hv, 128, 128, device=q.device, dtype=q.dtype
    )
    fp8, scales = quantize_gdn_state_fp8(initial)
    before_fp8, before_scale = fp8.view(torch.uint8).clone(), scales.clone()
    initial = dequantize_gdn_state_fp8(fp8, scales, output_dtype=q.dtype)
    expected = torch.empty_like(v)
    expected_state = initial.float().clone()
    for seq in range(n):
        acc = initial[indices[seq, accepted[seq] - 1]][None]
        for token in range(t):
            offset = seq * t + token
            out, acc = reference(
                q[:, offset : offset + 1],
                k[:, offset : offset + 1],
                v[:, offset : offset + 1],
                g[:, offset : offset + 1],
                beta[:, offset : offset + 1],
                acc,
                1,
                1,
            )
            expected[:, offset : offset + 1] = out
            expected_state[indices[seq, token]] = acc[0]
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)
    written = torch.zeros(n * slots, device=q.device, dtype=torch.bool)
    written[indices[:, :t].reshape(-1)] = True
    assert torch.equal(fp8.view(torch.uint8)[~written], before_fp8[~written])
    assert torch.equal(scales[~written], before_scale[~written])


@requires_fp8
def test_gdn_w8a16_empty_and_invalid():
    q, k, v, g, beta, cu, indices = make_inputs(2, 1, 2, 4, torch.bfloat16)
    state = torch.zeros(2, 4, 128, 128, device=q.device, dtype=q.dtype)
    fp8, scales = quantize_gdn_state_fp8(state)
    with pytest.raises(ValueError, match="2D per-token"):
        flaggems_vllm.gdn_w8a16(
            q,
            k,
            v,
            g,
            beta,
            128**-0.5,
            fp8,
            scales,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=indices,
        )
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q[:, :0],
        k[:, :0],
        v[:, :0],
        g[:, :0],
        beta[:, :0],
        128**-0.5,
        fp8,
        scales,
    )
    assert actual.numel() == 0


@requires_fp8
def test_gdn_w8a16_variable_lengths_and_padding():
    torch.manual_seed(45)
    q, k, v, g, beta, _, _ = make_inputs(1, 5, 2, 4, torch.bfloat16)
    initial = 0.02 * torch.randn(3, 4, 128, 128, device=q.device, dtype=q.dtype)
    fp8, scales = quantize_gdn_state_fp8(initial)
    initial = dequantize_gdn_state_fp8(fp8, scales, output_dtype=q.dtype)
    cu = torch.tensor([0, 0, 1, 5, 5], device=q.device, dtype=torch.int32)
    indices = torch.tensor([2, 1, 0, -1], device=q.device, dtype=torch.int32)
    expected = torch.empty_like(v)
    expected_state = initial.float().clone()
    for start, end, slot in [(0, 1, 1), (1, 5, 0)]:
        out, state = reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial[slot : slot + 1],
            1,
            end - start,
        )
        expected[:, start:end] = out
        expected_state[slot] = state[0]
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=4,
    )
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)


@requires_fp8
def test_gdn_w8a16_dynamic_scale_range():
    torch.manual_seed(46)
    amplitude = torch.logspace(-2, 3, 128, device=flaggems_vllm.device)
    state = (
        torch.randn(2, 4, 128, 128, device=amplitude.device) * amplitude
    ).bfloat16()
    fp8, scale = quantize_gdn_state_fp8(state)
    restored = dequantize_gdn_state_fp8(fp8, scale)
    error = (restored - state.float()).abs() / state.float().abs().amax(2, keepdim=True)
    assert error.max().item() < 0.08
    assert error.mean().item() < 0.02
    assert restored.abs().max().item() > 448


def torch_gdn_decode(
    mixed,
    a,
    b,
    a_log,
    bias,
    scale,
    state,
    output,
    indices,
    normalize,
    *,
    round_beta=False,
):
    """Vectorized PyTorch recurrence with FP32 state and one token per slot."""
    _, heads, columns, reduction = state.shape
    if isinstance(mixed, tuple):
        q, k, v = mixed
        tokens, query_heads = q.shape[1:3]
    else:
        tokens, packed = mixed.shape
        if tokens == 0:
            return output, state
        query_heads = (packed - heads * columns) // (2 * reduction)
        query, key, value = mixed.split(
            (query_heads * reduction, query_heads * reduction, heads * columns),
            dim=-1,
        )
        q = query.reshape(1, tokens, query_heads, reduction)
        k = key.reshape(1, tokens, query_heads, reduction)
        v = value.reshape(1, tokens, heads, columns)
    q, k = q[0].float(), k[0].float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(heads // query_heads, 1) * scale
    k = k.repeat_interleave(heads // query_heads, 1)
    slots = indices.clamp_min(0)
    original = state[slots].float()
    decay = torch.exp(-a_log.exp() * torch.nn.functional.softplus(a.float() + bias))
    current = original * decay[..., None, None]
    beta = b.float().sigmoid()
    if round_beta:
        beta = beta.to(b.dtype).float()
    delta = (v[0].float() - (current * k[..., None, :]).sum(-1)) * beta[..., None]
    current = current + delta[..., None] * k[..., None, :]
    valid = indices > 0
    result = (current * q[..., None, :]).sum(-1)
    output.copy_(
        torch.where(valid[:, None, None], result, 0).reshape(tokens, 1, heads, columns)
    )
    state.index_copy_(
        0,
        slots,
        torch.where(valid[:, None, None, None], current, original).to(state.dtype),
    )
    return output, state


def torch_gdn_prefill(
    q, k, v, g, beta, initial_state, bounds, *, normalize=True, output_dtype=None
):
    """PyTorch BF16/FP16-activation reference with FP32 recurrent state.

    The chunk algebra follows FlagGems PR 7279's PyTorch reference. Sequence
    boundaries are supplied on the host so timing contains no device reads.
    """
    dtype = q.dtype
    if normalize:
        q = (
            q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
        k = (
            k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
    state = initial_state
    group = v.shape[2] // q.shape[2]
    q = q[0].float().repeat_interleave(group, 1).transpose(0, 1)
    k = k[0].float().repeat_interleave(group, 1).transpose(0, 1)
    v = v[0].float().transpose(0, 1)
    g = g[0].transpose(0, 1)
    beta = beta[0].float().transpose(0, 1)
    outputs, states = [], []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        length = stop - start
        if length == 0:
            outputs.append(v[:, start:stop])
            states.append(state[seq])
            continue
        chunks = (length + 63) // 64
        padding = chunks * 64 - length
        qq = torch.nn.functional.pad(q[:, start:stop], (0, 0, 0, padding)).reshape(
            q.shape[0], chunks, 64, q.shape[-1]
        )
        kk = torch.nn.functional.pad(k[:, start:stop], (0, 0, 0, padding)).reshape(
            k.shape[0], chunks, 64, k.shape[-1]
        )
        vv = torch.nn.functional.pad(v[:, start:stop], (0, 0, 0, padding)).reshape(
            v.shape[0], chunks, 64, v.shape[-1]
        )
        gg = (
            torch.nn.functional.pad(g[:, start:stop], (0, padding))
            .reshape(g.shape[0], chunks, 64)
            .cumsum(-1)
        )
        bb = torch.nn.functional.pad(beta[:, start:stop], (0, padding)).reshape(
            beta.shape[0], chunks, 64
        )
        decay = (gg[..., :, None] - gg[..., None, :]).exp().tril()
        lhs = (kk @ kk.transpose(-1, -2) * decay * bb[..., None]).tril(-1)
        # Strictly lower 64x64 L is nilpotent: the finite geometric series
        # gives (I+L)^-1 exactly, using only PyTorch batched matmuls.
        power = -lhs
        inverse = torch.eye(64, device=q.device) + power
        for _ in range(5):
            power = power @ power
            inverse = inverse + inverse @ power
        rhs = bb[..., None] * torch.cat((vv, gg.exp()[..., None] * kk), dim=-1)
        solved = inverse @ rhs
        u, w = solved.split((vv.shape[-1], kk.shape[-1]), dim=-1)
        q_exp = qq * gg.exp()[..., None]
        qk = (qq @ kk.transpose(-1, -2)) * decay
        weighted_k = kk * (gg[..., -1:] - gg).exp()[..., None]
        last_decay = gg[..., -1].exp()
        s = state[seq].transpose(-1, -2)
        chunk_outputs = []
        for i in range(chunks):
            update = u[:, i] - w[:, i] @ s
            chunk_outputs.append(q_exp[:, i] @ s + qk[:, i] @ update)
            s = (
                last_decay[:, i, None, None] * s
                + weighted_k[:, i].transpose(-1, -2) @ update
            )
        outputs.append(
            torch.stack(chunk_outputs, dim=1).reshape(
                q.shape[0], chunks * 64, v.shape[-1]
            )[:, :length]
        )
        states.append(s.transpose(-1, -2))
    output = torch.cat(outputs, dim=1).transpose(0, 1).unsqueeze(0) * (
        q.shape[-1] ** -0.5
    )
    return output.to(output_dtype or dtype), torch.stack(states).contiguous()


def recurrent_gdn_reference(q, k, v, g, beta, initial, boundaries, normalize, scale):
    """Token recurrence independent of the chunked prefill reference."""
    group = v.shape[2] // q.shape[2]
    q, k = q[0].float(), k[0].float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(group, 1) * scale
    k = k.repeat_interleave(group, 1)
    output = torch.empty_like(v)
    final = []
    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial[sequence].float().clone()
        for token in range(begin, end):
            state = state * g[0, token].exp()[:, None, None]
            key = k[token, :, None, :]
            residual = v[0, token].float() - (state * key).sum(-1)
            state = (
                state + (residual * beta[0, token].float()[:, None])[..., None] * key
            )
            output[0, token] = (state * q[token, :, None, :]).sum(-1)
        final.append(state)
    return output, torch.stack(final)


def torch_int8_state_gdn_decode(
    mixed, a, b, a_log, bias, scale, state, output, indices, normalize
):
    return torch_gdn_decode(
        mixed, a, b, a_log, bias, scale, state, output, indices, normalize
    )


def build_int8_gdn_decode_inputs(shape, dtype, reduction=128):
    tokens, packed, heads, columns = shape
    device = flaggems_vllm.device
    mixed = torch.randn((tokens, packed), device=device, dtype=dtype)
    a = torch.randn((tokens, heads), device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.zeros(heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    original = (
        torch.randn((tokens + 1, heads, columns, reduction), device=device, dtype=dtype)
        * 0.1
    )
    state, scales = flaggems_vllm.quantize_gdn_state_int8(original)
    # Start both paths from exactly the same representable state.
    reference_state = state.float() * scales[..., None]
    output = torch.empty((tokens, 1, heads, columns), device=device, dtype=dtype)
    reference_output = torch.empty_like(output)
    indices = torch.arange(1, tokens + 1, device=device, dtype=torch.int64)
    shared = (mixed, a, b, A_log, dt_bias, reduction**-0.5)
    reference_args = (*shared, reference_state, reference_output, indices, True)
    gems_args = (*shared, state, scales, output, indices, True)
    return torch_int8_state_gdn_decode, reference_args, gems_args


def torch_int8_state_gdn_prefill(
    q, k, v, g, beta, initial_state, cu_seqlens, raw_cu_seqlens
):
    return torch_gdn_prefill(q, k, v, g, beta, initial_state, raw_cu_seqlens)


def build_int8_gdn_prefill_inputs(shape, dtype, boundaries=None):
    batch, tokens, query_heads, reduction, heads, columns, sequences = shape
    device = flaggems_vllm.device
    q = torch.randn((batch, tokens, query_heads, reduction), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn((batch, tokens, heads, columns), device=device, dtype=dtype) * 0.125
    g = -torch.rand((batch, tokens, heads), device=device, dtype=torch.float32) * 0.1
    beta = torch.rand((batch, tokens, heads), device=device, dtype=dtype)
    original = (
        torch.randn((sequences, heads, columns, reduction), device=device, dtype=dtype)
        * 0.1
    )
    state, scales = flaggems_vllm.quantize_gdn_state_int8(original)
    initial = state.float() * scales[..., None]
    if boundaries is None:
        # The shape exports omit boundaries; uniform splits are supplemental.
        boundaries = [0]
        for sequence in range(sequences):
            length = tokens // sequences + (sequence < tokens % sequences)
            boundaries.append(boundaries[-1] + length)
    pairs, starts = [], [0]
    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        pairs.extend((sequence, chunk) for chunk in range(triton.cdiv(end - begin, 64)))
        starts.append(len(pairs))
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    indices = torch.tensor(pairs, device=device, dtype=torch.int32)
    offsets = torch.tensor(starts, device=device, dtype=torch.int32)
    reference = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial,
        cu_seqlens=cu,
        raw_cu_seqlens=tuple(boundaries),
    )
    return (
        torch_int8_state_gdn_prefill,
        reference,
        (q, k, v, g, beta, state, scales, cu, indices, offsets),
    )


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    "chunk_gated_delta_rule_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state prefill",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16, 128, 32, 128, 1),
        (1, 17, 16, 128, 32, 128, 1),
        (1, 32, 16, 128, 32, 128, 3),
        (1, 43, 16, 128, 32, 128, 1),
        (1, 109, 16, 128, 32, 128, 1),
        (1, 128, 16, 128, 32, 128, 2),
        (1, 129, 16, 128, 32, 128, 2),
        (1, 1036, 16, 128, 32, 128, 1),
        (1, 2113, 16, 128, 32, 128, 2),
        (1, 16357, 16, 128, 32, 128, 5),
        (1, 16352, 16, 128, 32, 128, 17),
    ],
)
def test_int8_state_gdn_prefill(dtype, shape, caplog):
    torch.manual_seed(42)
    baseline, reference_args, args = build_int8_gdn_prefill_inputs(shape, dtype)
    expected, expected_state = baseline(**reference_args)
    entry = flaggems_vllm.chunk_gated_delta_rule_w8a16_int8
    with caplog.at_level("DEBUG", logger=entry.__module__):
        actual, state = entry(*args)
    assert (
        f"GEMS_{flaggems_vllm.vendor_name.upper()} CHUNK_GATED_DELTA_RULE_W8A16_INT8"
        in caplog.text
    )
    gems_assert_close(actual, expected, dtype, reduce_dim=1, atol=2e-3)
    restored = (state.float() * args[6][..., None]).to(dtype)
    gems_assert_close(
        restored, expected_state.to(dtype), dtype, reduce_dim=1, atol=3e-3
    )


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    "chunk_gated_delta_rule_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state prefill",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,boundaries,normalize,metadata_dtype",
    [
        (17, None, True, torch.int32),
        (32, [0, 3, 32], False, torch.int64),
        (109, [0, 1, 65, 109], True, torch.int32),
    ],
)
def test_int8_state_gdn_prefill_independent_calls(
    dtype, tokens, boundaries, normalize, metadata_dtype
):
    torch.manual_seed(42)
    device = flaggems_vllm.device
    sequences = 1 if boundaries is None else len(boundaries) - 1
    reference_boundaries = [0, tokens] if boundaries is None else boundaries

    streams = [torch_device_fn.Stream(), torch_device_fn.Stream()]
    for alignment_offset in (0, 1, 0):
        calls = []
        for stream in streams:
            _, _, args = build_int8_gdn_prefill_inputs(
                (1, tokens, 2, 128, 4, 128, sequences),
                dtype,
                boundaries=reference_boundaries,
            )
            inputs = list(args)
            # Keep the recurrence stable when in-kernel normalization is disabled.
            inputs[1] = torch.nn.functional.normalize(inputs[1].float(), dim=-1).to(
                dtype
            )
            if boundaries is None:
                inputs[7:] = (None, None, None)
            else:
                inputs[7:] = [tensor.to(metadata_dtype) for tensor in inputs[7:]]
            if alignment_offset:
                for index, tensor in enumerate(inputs):
                    if tensor is not None:
                        storage = torch.empty(
                            tensor.numel() + 1, device=device, dtype=tensor.dtype
                        )
                        shifted = storage[1:].view(tensor.shape)
                        shifted.copy_(tensor)
                        inputs[index] = shifted
            q, k, v, g, beta, state, scales, _, _, _ = inputs
            expected, expected_state = recurrent_gdn_reference(
                q,
                k,
                v,
                g,
                beta,
                state.float() * scales[..., None],
                reference_boundaries,
                normalize,
                0.125,
            )
            stream.wait_stream(torch_device_fn.current_stream())
            with torch_device_fn.stream(stream):
                actual, updated = flaggems_vllm.chunk_gated_delta_rule_w8a16_int8(
                    *inputs,
                    scale=0.125,
                    use_qk_l2norm_in_kernel=normalize,
                )
            calls.append((stream, inputs, actual, updated, expected, expected_state))
        for stream, inputs, actual, updated, expected, expected_state in calls:
            stream.synchronize()
            gems_assert_close(actual, expected, dtype, reduce_dim=1, atol=2e-3)
            restored = updated.float() * inputs[6][..., None]
            gems_assert_close(
                restored.to(dtype),
                expected_state.to(dtype),
                dtype,
                reduce_dim=1,
                atol=3e-3,
            )
        assert calls[0][2].data_ptr() != calls[1][2].data_ptr()


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state GDN",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 3, 49])
@pytest.mark.parametrize("steps", [1, 32])
@pytest.mark.parametrize("dimensions", [(8192, 32, 128, 128), (100, 2, 17, 33)])
def test_int8_state_gdn_decode(dtype, tokens, steps, dimensions, caplog):
    torch.manual_seed(42)
    packed, heads, columns, reduction = dimensions
    baseline, reference_args, gems_args = build_int8_gdn_decode_inputs(
        (tokens, packed, heads, columns), dtype, reduction=reduction
    )
    reference_args = list(reference_args)
    reference_args[6] = reference_args[6].float()
    for _ in range(steps):
        # Isolate each update's arithmetic from accumulated quantization drift.
        reference_args[6].copy_(gems_args[6].float() * gems_args[7][..., None])
        expected, reference_state = baseline(*reference_args)
        entry = flaggems_vllm.gdn_w8a16_int8
        with caplog.at_level("DEBUG", logger=entry.__module__):
            actual, state = entry(*gems_args)
        assert f"GEMS_{flaggems_vllm.vendor_name.upper()} GDN_W8A16_INT8" in caplog.text
        gems_assert_close(actual, expected, dtype, reduce_dim=1, atol=1e-6)
        scales = gems_args[7][..., None]
        # Nearest rounding must stay within half one INT8 quantization step.
        gems_assert_close(
            state.float(),
            reference_state / scales,
            torch.float32,
            reduce_dim=1,
            atol=0.501,
        )


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state GDN",
)
@pytest.mark.parametrize("tokens", [0, 2])
def test_int8_state_gdn_null_slots(tokens):
    torch.manual_seed(42)
    baseline, reference_args, gems_args = build_int8_gdn_decode_inputs(
        (tokens, 8192, 32, 128), torch.bfloat16
    )
    gems_args[-2].zero_()
    old_state = gems_args[6].clone()
    old_scales = gems_args[7].clone()
    expected, _ = baseline(*reference_args)
    actual, state = flaggems_vllm.gdn_w8a16_int8(*gems_args)
    gems_assert_equal(actual, expected)
    gems_assert_equal(state, old_state)
    gems_assert_equal(gems_args[7], old_scales)


requires_metax = pytest.mark.skipif(
    vendor_name != "metax", reason="MetaX FP32-state GDN contract"
)


@pytest.fixture
def metax_ieee_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def fp32_reference(q, k, v, g, beta, state, bounds):
    return torch_gdn_prefill(
        q, k, v, g, beta, state, bounds, normalize=False, output_dtype=torch.float32
    )


def inputs(row, dtype=torch.bfloat16):
    shape = row["shape"]
    packed = len(shape) == 4
    if packed:
        tokens, _, hv, dv = shape
        b, heads, dk, n = 1, 16, 128, tokens
    else:
        b, tokens, heads, dk, hv, dv, n = shape
    bounds = [i * tokens // n for i in range(n + 1)]
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    if packed:
        mixed = torch.randn(
            tokens, 2 * heads * dk + hv * dv, device="cuda", dtype=dtype
        )
        q = mixed[:, : heads * dk].view(1, tokens, heads, dk)
        k = mixed[:, heads * dk : 2 * heads * dk].view(1, tokens, heads, dk)
        v = mixed[:, 2 * heads * dk :].view(1, tokens, hv, dv)
        q.copy_(torch.nn.functional.normalize(q.float(), dim=-1).to(dtype))
        k.copy_(torch.nn.functional.normalize(k.float(), dim=-1).to(dtype))
    else:
        q = torch.nn.functional.normalize(
            torch.randn(b, tokens, heads, dk, device="cuda"), dim=-1
        ).to(dtype)
        k = torch.nn.functional.normalize(
            torch.randn(b, tokens, heads, dk, device="cuda"), dim=-1
        ).to(dtype)
        v = torch.randn(b, tokens, hv, dv, device="cuda", dtype=dtype)
    g = -torch.rand(b, tokens, hv, device="cuda") * 0.1
    beta = torch.rand(b, tokens, hv, device="cuda")
    state = torch.randn(n, hv, dv, dk, device="cuda") * 0.1
    return (q, k, v, g, beta, state, cu), bounds


def check(row, dtype):
    args, bounds = inputs(row, dtype)
    before = args[5].clone()
    out, state = gdn_w8a16(*args)
    ref_out, ref_state = fp32_reference(*args[:6], bounds)
    torch.testing.assert_close(out.float(), ref_out, atol=0.002, rtol=0.02)
    torch.testing.assert_close(state, ref_state, atol=1e-4, rtol=0.002)
    torch.testing.assert_close(args[5], before, atol=0, rtol=0)
    return (
        args,
        float((out.float() - ref_out).abs().max()) if out.numel() else 0.0,
        float((state - ref_state).abs().max()),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,sequences", [(0, 1), (1, 1), (17, 3), (129, 2)])
@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16(dtype, tokens, sequences):
    torch.manual_seed(1234)
    check(
        dict(
            op="chunk_gated_delta_rule", shape=[1, tokens, 16, 128, 32, 128, sequences]
        ),
        dtype,
    )


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_packed():
    torch.manual_seed(1234)
    check(dict(op="packed", shape=[7, 8192, 32, 128]), torch.bfloat16)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_reject_dtype():
    args, _ = inputs(dict(op="packed", shape=[1, 8192, 32, 128]))
    with pytest.raises(NotImplementedError):
        gdn_w8a16(args[0].float(), *args[1:])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,sequences", [(512, 1), (1025, 3)])
@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_chunk(dtype, tokens, sequences):
    torch.manual_seed(1234)
    check(dict(shape=[1, tokens, 16, 128, 32, 128, sequences]), dtype)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_chunk_graph_boundaries():
    torch.manual_seed(1234)
    args, bounds = inputs(dict(shape=[1, 513, 16, 128, 32, 128, 3]))
    before = args[5].clone()
    for _ in range(3):
        gdn_w8a16(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gdn_w8a16(*args)
    # Update contents at the same address: no identity-cached chunk metadata.
    bounds = [0, 0, 1, 513]
    args[-1].copy_(torch.tensor(bounds, device=args[0].device, dtype=torch.int32))
    graph.replay()
    expected = fp32_reference(*args[:6], bounds)
    torch.testing.assert_close(actual[0].float(), expected[0], atol=0.002, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)
    torch.testing.assert_close(args[5], before, atol=0, rtol=0)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_single_dispatch_nonuniform():
    # T == N does not imply one token per sequence.
    args, _ = inputs(dict(shape=[1, 4, 16, 128, 32, 128, 4]))
    bounds = [0, 0, 1, 1, 4]
    args[-1].copy_(torch.tensor(bounds, device=args[0].device, dtype=torch.int32))
    actual = gdn_w8a16(*args)
    expected = fp32_reference(*args[:6], bounds)
    torch.testing.assert_close(actual[0].float(), expected[0], atol=0.002, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_chunk_packed_custom_scale():
    torch.manual_seed(1234)
    args, bounds = inputs(dict(shape=[1, 513, 16, 128, 32, 128, 3]))
    q, k, v, g, beta, state, cu = args
    mixed = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
    q = mixed[..., :2048].view_as(q)
    k = mixed[..., 2048:4096].view_as(k)
    v = mixed[..., 4096:].view_as(v)
    actual = gdn_w8a16(q, k, v, g, beta, state, cu, scale=0.25)
    expected = fp32_reference(*args[:6], bounds)
    torch.testing.assert_close(
        actual[0].float(), expected[0] * (0.25 / 128**-0.5), atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)


@pytest.mark.parametrize("sequences", [3, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_bounded_workspace_replay(monkeypatch, dtype, sequences):
    from flaggems_vllm import runtime

    if runtime.device.vendor_name != "metax":
        pytest.skip("MetaX windowed chunk implementation")
    import importlib

    chunk = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.gdn_w8a16"
    )

    # A small budget forces several windows without a large CI allocation.
    budget = 4 * 1024**2
    monkeypatch.setattr(chunk, "_WORKSPACE_LIMIT_BYTES", budget)
    args, _ = inputs(dict(shape=[1, 641, 2, 128, 4, 128, sequences]), dtype)
    original = args[5].clone()
    for _ in range(3):
        gdn_w8a16(*args)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = gdn_w8a16(*args)
    torch.cuda.synchronize()
    result_bytes = sum(x.numel() * x.element_size() for x in actual)
    assert torch.cuda.max_memory_allocated() - allocated <= budget + result_bytes + 4096
    del actual
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gdn_w8a16(*args)
    # Move empty and long requests across windows at the same Cu address.
    for bounds in (
        [0] * (sequences - 1) + [1, 641],
        [0, 321] + [321] * (sequences - 2) + [641],
    ):
        args[-1].copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        expected = fp32_reference(*args[:6], bounds)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(
                actual[0].float(), expected[0], atol=0.002, rtol=0.02
            )
            torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=0.002)
            torch.testing.assert_close(args[5], original, atol=0, rtol=0)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_compile_hook_reload(monkeypatch):
    import importlib
    import importlib.util
    import sys

    chunk = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.gdn_w8a16"
    )
    calls = []

    def translate(src, mxcc, maca, options):
        calls.append((src, options))
        return b"compiled"

    monkeypatch.setattr(chunk.backend.metax, "translate_llvmir_to_mcfatbin", translate)
    monkeypatch.setattr(chunk.backend, "maca_get_kernel_name", lambda src: src)
    spec = importlib.util.spec_from_file_location("_gdn_reload_test", chunk.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    installed = chunk.backend.metax.translate_llvmir_to_mcfatbin
    # Re-execute the same module namespace, as reload does, without replacing
    # the public package entry used by other tests.
    spec.loader.exec_module(module)
    assert chunk.backend.metax.translate_llvmir_to_mcfatbin is installed
    for name in ("_gdn_chunk_scan", "_gdn_fp32_chunk_scan", "unrelated_kernel"):
        assert installed(name, None, None, "-O3") == b"compiled"
    flag = " -mllvm -metaxgpu-disable-licm=true"
    assert calls == [
        ("_gdn_chunk_scan", "-O3" + flag),
        ("_gdn_fp32_chunk_scan", "-O3" + flag),
        ("unrelated_kernel", "-O3"),
    ]


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_mixed_window_replay():
    import importlib

    chunk = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.gdn_w8a16"
    )
    torch.manual_seed(9616)
    tokens, sequences = 8191, 16
    args, bounds = inputs(dict(shape=[1, tokens, 16, 128, 32, 128, sequences]))
    # Use the stable raw-input distribution of the mixed-precision benchmark.
    args[0].normal_()
    args[1].normal_()
    args[4].div_(128)
    original = args[5].clone()

    def call():
        return gdn_w8a16(*args, mixed_precision=True)

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = call()
    torch.cuda.synchronize()
    result_bytes = sum(t.numel() * t.element_size() for t in actual)
    assert (
        torch.cuda.max_memory_allocated() - allocated
        <= chunk._WORKSPACE_LIMIT_BYTES + result_bytes + 4096
    )
    del actual
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    # Same graph, different numbers of active chunks and requests, including
    # the maximum chunk count and empty requests across the 96-chunk window.
    for bounds in (
        bounds,
        [*range(sequences), tokens],
        [0] * 8 + [i * tokens // 8 for i in range(9)],
    ):
        args[-1].copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        expected = fp32_reference(*args[:6], bounds)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(
                actual[0].float(), expected[0], atol=0.002, rtol=0.02
            )
            torch.testing.assert_close(actual[1], expected[1], atol=0.002, rtol=0.02)
            torch.testing.assert_close(args[5], original, atol=0, rtol=0)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_backend_registration():
    import flaggems_vllm
    from flaggems_vllm.runtime.backend._metax.ops.gdn_w8a16 import (
        gdn_w8a16 as metax_gdn,
    )

    assert flaggems_vllm.gdn_w8a16 is metax_gdn
    entries = [e for e in flaggems_vllm._FULL_CONFIG if e[0] == "gdn_w8a16"]
    assert entries == [("gdn_w8a16", metax_gdn)]


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
@pytest.mark.parametrize("real_packed", [False, True])
def test_gdn_w8a16_graph_contract(real_packed):
    """Multi-step decode and mixed-precision prefill with dynamic graph metadata."""
    from types import SimpleNamespace

    low = SimpleNamespace(
        gdn_w8a16_chunk=lambda args, scale: gdn_w8a16(
            *args, scale=scale, mixed_precision=True
        )
    )

    def public_decode(qkv, a, b, al, bias, st, ids, out, normalize=True, **kwargs):
        q, k, v = qkv if isinstance(qkv, tuple) else (qkv, None, None)
        return gdn_w8a16(
            q,
            k,
            v,
            initial_state=st,
            decode=True,
            a=a,
            b=b,
            A_log=al,
            dt_bias=bias,
            state_indices=ids,
            out=out,
            normalize=normalize,
        )

    decode = SimpleNamespace(decode=public_decode)
    torch.manual_seed(537)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []

    def close(a, b, atol=0.002, rtol=0.02):
        torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)

    def ref_decode(q, k, v, a, b, al, bias, st, ids, norm):
        n, hv, dv = v.shape[1:]
        out = torch.empty((n, 1, hv, dv), dtype=v.dtype, device=v.device)
        torch_gdn_decode(
            (q, k, v),
            a,
            b,
            al,
            bias,
            q.shape[-1] ** -0.5,
            st,
            out,
            ids,
            norm,
            round_beta=True,
        )
        return out.view_as(v)

    cases = (
        [(n, 128, 128) for n in (1, 2, 3, 7, 9, 31)]
        if real_packed
        else [(0, 128, 128), (1, 128, 128), (9, 128, 128), (3, 33, 17)]
    )
    for dtype in (torch.bfloat16, torch.float16):
        for norm in (True,) if real_packed else (False, True):
            for n, dk, dv in cases:
                h, hv = (16, 32) if real_packed else (2, 4)
                q = torch.randn((1, n, h, dk), device="cuda", dtype=dtype) * 0.2
                k = torch.randn_like(q) * 0.2
                v = torch.randn((1, n, hv, dv), device="cuda", dtype=dtype) * 0.2
                a = torch.randn((n, hv), device="cuda", dtype=dtype)
                b = torch.randn_like(a)
                al = torch.randn(hv, device="cuda") * 0.1
                bias = torch.randn_like(al) * 0.1
                ids = torch.arange(n, 0, -1, device="cuda", dtype=torch.int64)
                if n > 1:
                    ids[-1] = -1 if real_packed else 0
                original = torch.randn((n + 2, hv, dv, dk), device="cuda") * 0.1
                mixed = torch.cat(
                    (q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=1
                )
                for packed in (True,) if real_packed else (False, True):
                    for factor, fma in [(False, False)]:
                        st = original.clone()
                        ref = original.clone()
                        out = torch.empty_like(v)

                        def call():
                            return decode.decode(
                                mixed if packed else (q, k, v),
                                a,
                                b,
                                al,
                                bias,
                                st,
                                ids,
                                out,
                                normalize=norm,
                                factor=factor,
                                fma=fma,
                            )

                        call()
                        st.copy_(original)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            call()
                        st.copy_(original)
                        for step in range(32 if n else 1):
                            expected = ref_decode(
                                q, k, v, a, b, al, bias, ref, ids, norm
                            )
                            graph.replay()
                            close(out, expected.to(dtype))
                            close(st, ref, atol=1e-5, rtol=0.002)
                        # Replay must consume changed slot mappings, not cached metadata values.
                        if n > 1:
                            ids.copy_(torch.arange(1, n + 1, device="cuda"))
                            st.copy_(original)
                            ref.copy_(original)
                            expected = ref_decode(
                                q, k, v, a, b, al, bias, ref, ids, norm
                            )
                            graph.replay()
                            close(out, expected.to(dtype))
                            close(st, ref, atol=1e-5, rtol=0.002)
                            ids.copy_(torch.arange(n, 0, -1, device="cuda"))
                            ids[-1] = 0
                        records.append(
                            dict(
                                kind="decode",
                                dtype=str(dtype),
                                normalize=norm,
                                n=n,
                                dk=dk,
                                dv=dv,
                                packed=packed,
                                factor=factor,
                                fma=fma,
                                passed=True,
                            )
                        )
                        print("PASS", records[-1], flush=True)

    if real_packed:
        assert len(records) == 12
        return

    # Validate GPU mappings, window boundaries, empty and skewed requests, both dtypes.
    for dtype in (torch.bfloat16, torch.float16):
        for lengths in (
            [0],
            [17],
            [43],
            [109],
            [0, 65, 0, 2050],
            [1, 2049, 2],
            [64] * 33,
        ):
            n = len(lengths)
            t = sum(lengths)
            h, hv, dk, dv = 2, 4, 128, 128
            q = torch.randn((1, t, h, dk), device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn((1, t, hv, dv), device="cuda", dtype=dtype)
            g = -torch.rand((1, t, hv), device="cuda") * 0.1
            b = torch.rand_like(g) / 128
            st = torch.randn((n, hv, dv, dk), device="cuda") * 0.1
            bounds = [0]
            for length in lengths:
                bounds.append(bounds[-1] + length)
            cu = torch.tensor(bounds, device="cuda", dtype=torch.int32)
            original = st.clone()

            # Independent per-token FP32 recurrence, including empty requests.
            def fp32_reference():
                return recurrent_gdn_reference(
                    q, k, v, g, b, st, bounds, normalize=False, scale=dk**-0.5
                )

            expected = fp32_reference()
            if not t:
                actual = gdn_w8a16(q, k, v, g, b, st, cu)
            else:
                actual = low.gdn_w8a16_chunk((q, k, v, g, b, st, cu), dk**-0.5)
            close(actual[0], expected[0])
            close(actual[1], expected[1])
            torch.testing.assert_close(st, original, atol=0, rtol=0)
            if t:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = low.gdn_w8a16_chunk((q, k, v, g, b, st, cu), dk**-0.5)
                graph.replay()
                close(actual[0], expected[0])
                close(actual[1], expected[1])
                # Change request lengths while retaining metadata shape and total tokens.
                lengths = list(reversed(lengths))
                bounds = [0]
                for length in lengths:
                    bounds.append(bounds[-1] + length)
                cu.copy_(torch.tensor(bounds, device="cuda", dtype=cu.dtype))
                expected = fp32_reference()
                graph.replay()
                close(actual[0], expected[0])
                close(actual[1], expected[1])
            records.append(
                dict(kind="prefill", dtype=str(dtype), lengths=lengths, passed=True)
            )
            print("PASS", records[-1], flush=True)

    assert len(records) == 46


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
def test_gdn_w8a16_small_packed_scale_and_dynamic_padding(n, dtype):
    torch.manual_seed(991 + n)
    x = torch.randn(n, 8192, device="cuda", dtype=dtype)
    a = torch.randn(n, 32, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    al = torch.randn(32, device="cuda") * 0.1
    bias = torch.randn_like(al) * 0.1
    base = torch.randn(n + 2, 32, 128, 128, device="cuda") * 0.1
    state = base.clone()
    ids = torch.arange(n, 0, -1, device="cuda", dtype=torch.int64)
    out = torch.empty(1, n, 32, 128, device="cuda", dtype=dtype)
    for scale in (0.37, -0.21, 0.0):

        def call():
            return gdn_w8a16(
                x,
                initial_state=state,
                decode=True,
                a=a,
                b=b,
                A_log=al,
                dt_bias=bias,
                state_indices=ids,
                out=out,
                scale=scale,
            )

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        for padding in (False, True, False):
            ids.copy_(torch.arange(n, 0, -1, device="cuda", dtype=torch.int64))
            if padding:
                ids[-1] = -(2**40)
            state.copy_(base)
            expected = base.clone()
            ref = torch.empty_like(out)
            torch_gdn_decode(
                x,
                a,
                b,
                al,
                bias,
                scale,
                expected,
                ref.view(n, 1, 32, 128),
                ids,
                True,
                round_beta=True,
            )
            graph.replay()
            assert actual[0] is out and actual[1] is state
            torch.testing.assert_close(out, ref, atol=0.002, rtol=0.02)
            torch.testing.assert_close(state, expected, atol=1e-5, rtol=0.002)


@requires_metax
@pytest.mark.usefixtures("metax_ieee_reference")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_metax_fp32_state_gdn_decode(dtype, packed, normalize):
    torch.manual_seed(537)
    n, h, hv, dk, dv = 3, 2, 4, 33, 17
    q = torch.randn((1, n, h, dk), device="cuda", dtype=dtype) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn((1, n, hv, dv), device="cuda", dtype=dtype) * 0.2
    a = torch.randn((n, hv), device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    al = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn_like(al) * 0.1
    ids = torch.tensor([3, 1, 0], device="cuda", dtype=torch.int64)
    state = torch.randn((n + 2, hv, dv, dk), device="cuda") * 0.1
    expected = state.clone()
    mixed = torch.cat((q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=-1)
    ref_out = torch.empty((n, 1, hv, dv), device=q.device, dtype=dtype)
    torch_gdn_decode(
        mixed if packed else (q, k, v),
        a,
        b,
        al,
        bias,
        dk**-0.5,
        expected,
        ref_out,
        ids,
        normalize,
        round_beta=True,
    )
    out = torch.empty_like(v)
    actual, cache = gdn_w8a16(
        mixed if packed else q,
        None if packed else k,
        None if packed else v,
        initial_state=state,
        decode=True,
        a=a,
        b=b,
        A_log=al,
        dt_bias=bias,
        state_indices=ids,
        out=out,
        normalize=normalize,
    )
    assert actual is out and cache is state
    torch.testing.assert_close(out[0], ref_out[:, 0], atol=0.002, rtol=0.02)
    torch.testing.assert_close(state, expected, atol=1e-5, rtol=0.002)
