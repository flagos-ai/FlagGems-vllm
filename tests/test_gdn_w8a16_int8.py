# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import triton

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close, gems_assert_equal


def torch_int8_state_gdn_decode(
    mixed, a, b, a_log, bias, scale, state, output, indices, normalize
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
    delta = (v[0].float() - (current * k[..., None, :]).sum(-1)) * b.float().sigmoid()[
        ..., None
    ]
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
    """PyTorch BF16/FP16-activation reference with FP32 recurrent state.

    The chunk algebra follows FlagGems PR 7279's PyTorch reference. Sequence
    boundaries are supplied on the host so timing contains no device reads.
    """
    dtype = q.dtype
    q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(
        dtype
    )
    k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(
        dtype
    )
    state, bounds = initial_state, raw_cu_seqlens
    group = v.shape[2] // q.shape[2]
    q = q[0].float().repeat_interleave(group, 1).transpose(0, 1)
    k = k[0].float().repeat_interleave(group, 1).transpose(0, 1)
    v = v[0].float().transpose(0, 1)
    g = g[0].transpose(0, 1)
    beta = beta[0].float().transpose(0, 1)
    outputs, states = [], []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        length = stop - start
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
    return output.to(dtype), torch.stack(states).contiguous()


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


@pytest.mark.gdn_w8a16_int8
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


@pytest.mark.gdn_w8a16_int8
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

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
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
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
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


@pytest.mark.gdn_w8a16_int8
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


@pytest.mark.gdn_w8a16_int8
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
