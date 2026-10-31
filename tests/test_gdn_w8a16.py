# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.gdn_w8a16 import dequantize_gdn_state_fp8, quantize_gdn_state_fp8
from tests import conftest as cfg

INT8_STATE_GDN = flaggems_vllm.gdn_w8a16.__module__ == (
    "flaggems_vllm.runtime.backend._hygon.ops.gdn_w8a16"
)
pytestmark = [
    pytest.mark.gdn_w8a16,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not (
            INT8_STATE_GDN
            or (
                flaggems_vllm.vendor_name == "nvidia"
                and torch.cuda.get_device_capability()[0] >= 9
            )
        ),
        reason="backend does not export a supported GDN state implementation",
    ),
]

if INT8_STATE_GDN:
    import triton

    from flaggems_vllm.runtime.backend._hygon.ops.gdn_w8a16 import (
        quantize_gdn_state_int8,
    )
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
        delta = (
            v[0].float() - (current * k[..., None, :]).sum(-1)
        ) * b.float().sigmoid()[..., None]
        current = current + delta[..., None] * k[..., None, :]
        valid = indices > 0
        result = (current * q[..., None, :]).sum(-1)
        output.copy_(
            torch.where(valid[:, None, None], result, 0).reshape(
                tokens, 1, heads, columns
            )
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
            torch.randn(
                (tokens + 1, heads, columns, reduction), device=device, dtype=dtype
            )
            * 0.1
        )
        state, scales = quantize_gdn_state_int8(original)
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
        q = (
            q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
        k = (
            k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
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
        q = torch.randn(
            (batch, tokens, query_heads, reduction), device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = (
            torch.randn((batch, tokens, heads, columns), device=device, dtype=dtype)
            * 0.125
        )
        g = (
            -torch.rand((batch, tokens, heads), device=device, dtype=torch.float32)
            * 0.1
        )
        beta = torch.rand((batch, tokens, heads), device=device, dtype=dtype)
        original = (
            torch.randn(
                (sequences, heads, columns, reduction), device=device, dtype=dtype
            )
            * 0.1
        )
        state, scales = quantize_gdn_state_int8(original)
        initial = state.float() * scales[..., None]
        if boundaries is None:
            # The shape exports omit boundaries; uniform splits are supplemental.
            boundaries = [0]
            for sequence in range(sequences):
                length = tokens // sequences + (sequence < tokens % sequences)
                boundaries.append(boundaries[-1] + length)
        pairs, starts = [], [0]
        for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
            pairs.extend(
                (sequence, chunk) for chunk in range(triton.cdiv(end - begin, 64))
            )
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

    def recurrent_gdn_reference(
        q, k, v, g, beta, initial, boundaries, normalize, scale
    ):
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
                    state
                    + (residual * beta[0, token].float()[:, None])[..., None] * key
                )
                output[0, token] = (state * q[token, :, None, :]).sum(-1)
            final.append(state)
        return output, torch.stack(final)

    def test_int8_state_gdn_registration():
        provider = "flaggems_vllm.runtime.backend._hygon.ops.gdn_w8a16"
        assert flaggems_vllm.gdn_w8a16.__module__ == provider
        assert flaggems_vllm.chunk_gated_delta_rule_w8a16.__module__ == provider
        entries = [
            entry for entry in flaggems_vllm._FULL_CONFIG if entry[0] == "gdn_w8a16"
        ]
        assert entries == [("gdn_w8a16", flaggems_vllm.gdn_w8a16)]

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("tokens", [1, 3, 49])
    @pytest.mark.parametrize("dimensions", [(8192, 32, 128, 128), (100, 2, 17, 33)])
    @pytest.mark.parametrize("separate", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_int8_state_gdn_decode(dtype, tokens, dimensions, separate, normalize):
        torch.manual_seed(42)
        packed, heads, columns, reduction = dimensions
        baseline, reference_args, args = build_int8_gdn_decode_inputs(
            (tokens, packed, heads, columns), dtype, reduction
        )
        reference_args = (*reference_args[:-1], normalize)
        args = (*args[:-1], normalize)
        if not normalize:
            # Keep eight recurrent updates stable without in-kernel normalization;
            # random unit-variance K would amplify the state exponentially.
            query_width = (packed - heads * columns) // 2
            args[0][:, query_width : 2 * query_width].mul_(0.05)
        if separate:
            query_heads = (packed - heads * columns) // (2 * reduction)
            values = args[0].split(
                (query_heads * reduction, query_heads * reduction, heads * columns), -1
            )
            qkv = tuple(
                value.view(1, tokens, width, feature).contiguous()
                for value, width, feature in zip(
                    values,
                    (query_heads, query_heads, heads),
                    (reduction, reduction, columns),
                )
            )
            args = (qkv, *args[1:])
        for _ in range(2 if cfg.QUICK_MODE else 8):
            reference_args[6].copy_(args[6].float() * args[7][..., None])
            expected, expected_state = baseline(*reference_args)
            actual, state = flaggems_vllm.gdn_w8a16(*args)
            assert actual is args[8] and state is args[6]
            gems_assert_close(actual, expected, dtype, atol=1e-5)
            gems_assert_close(
                state.float(),
                expected_state / args[7][..., None],
                torch.float32,
                atol=0.501,
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_int8_state_gdn_stream(dtype):
        torch.manual_seed(47)
        # Launch the same specialization with fresh buffers and a different stream.
        # The warm-launch cache must resolve that stream on every invocation.
        for _ in range(2):
            baseline, reference_args, args = build_int8_gdn_decode_inputs(
                (3, 8192, 32, 128), dtype
            )
            expected, expected_state = baseline(*reference_args)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual, state = flaggems_vllm.gdn_w8a16(*args)
            torch.cuda.current_stream().wait_stream(stream)
            gems_assert_close(actual, expected, dtype, atol=1e-5)
            gems_assert_close(
                state.float(),
                expected_state / args[7][..., None],
                torch.float32,
                atol=0.501,
            )

    @pytest.mark.parametrize("case", ["zero_key", "overflow_norm"])
    def test_int8_state_gdn_normalization_extremes(case):
        torch.manual_seed(48)
        baseline, reference_args, args = build_int8_gdn_decode_inputs(
            (3, 8192, 32, 128), torch.bfloat16
        )
        args[0].zero_()
        if case == "zero_key":
            args[0][:, 4096:].fill_(1e37)
        else:
            args[0][:, :2048].fill_(1e30)
            args[7].mul_(1e36)
        reference_args[6].copy_(args[6].float() * args[7][..., None])
        expected, expected_state = baseline(*reference_args)
        actual, state = flaggems_vllm.gdn_w8a16(*args)
        gems_assert_close(actual, expected, torch.bfloat16, atol=1e-5)
        gems_assert_close(
            state.float(),
            expected_state / args[7][..., None],
            torch.float32,
            atol=0.501,
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        "tokens,sequences", [(1, 1), (17, 1), (109, 3), (129, 2), (1036, 1)]
    )
    def test_int8_state_gdn_prefill(dtype, tokens, sequences):
        torch.manual_seed(43)
        baseline, reference_args, args = build_int8_gdn_prefill_inputs(
            (1, tokens, 2, 128, 4, 128, sequences), dtype
        )
        expected, expected_state = baseline(**reference_args)
        actual, state = flaggems_vllm.chunk_gated_delta_rule_w8a16(*args)
        gems_assert_close(actual, expected, dtype, atol=2e-3)
        restored = (state.float() * args[6][..., None]).to(dtype)
        gems_assert_close(restored, expected_state.to(dtype), dtype, atol=3e-3)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_int8_state_gdn_prefill_ragged(dtype, normalize):
        torch.manual_seed(44)
        bounds = [0, 1, 65, 109]
        _, _, args = build_int8_gdn_prefill_inputs(
            (1, 109, 2, 128, 4, 128, 3), dtype, bounds
        )
        inputs = list(args)
        inputs[1] = torch.nn.functional.normalize(inputs[1].float(), dim=-1).to(dtype)
        inputs[7:] = [tensor.to(torch.int64) for tensor in inputs[7:]]
        expected, expected_state = recurrent_gdn_reference(
            *inputs[:5],
            inputs[5].float() * inputs[6][..., None],
            bounds,
            normalize,
            0.125,
        )
        actual, state = flaggems_vllm.chunk_gated_delta_rule_w8a16(
            *inputs,
            scale=0.125,
            use_qk_l2norm_in_kernel=normalize,
        )
        gems_assert_close(actual, expected, dtype, atol=2e-3)
        restored = (state.float() * inputs[6][..., None]).to(dtype)
        gems_assert_close(restored, expected_state.to(dtype), dtype, atol=3e-3)

    @pytest.mark.parametrize("tokens", [0, 2])
    def test_int8_state_gdn_null_slots(tokens):
        torch.manual_seed(45)
        baseline, reference_args, args = build_int8_gdn_decode_inputs(
            (tokens, 8192, 32, 128), torch.bfloat16
        )
        args[-2].zero_()
        old_state, old_scale = args[6].clone(), args[7].clone()
        expected, _ = baseline(*reference_args)
        actual, state = flaggems_vllm.gdn_w8a16(*args)
        gems_assert_equal(actual, expected)
        gems_assert_equal(state, old_state)
        gems_assert_equal(args[7], old_scale)

    @pytest.mark.parametrize("tokens", [0, 17])
    def test_int8_state_gdn_dense_prefill(tokens):
        torch.manual_seed(46)
        baseline, reference_args, args = build_int8_gdn_prefill_inputs(
            (1, tokens, 2, 128, 4, 128, 1), torch.bfloat16
        )
        old_state, old_scale = args[5].clone(), args[6].clone()
        actual, state = flaggems_vllm.chunk_gated_delta_rule_w8a16(*args[:7])
        if tokens == 0:
            assert actual.shape == args[2].shape
            gems_assert_equal(state, old_state)
            gems_assert_equal(args[6], old_scale)
        else:
            expected, expected_state = baseline(**reference_args)
            gems_assert_close(actual, expected, torch.bfloat16, atol=2e-3)
            restored = (state.float() * args[6][..., None]).bfloat16()
            gems_assert_close(
                restored, expected_state.bfloat16(), torch.bfloat16, atol=3e-3
            )

else:
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

    def test_gdn_w8a16_registration():
        import flaggems_vllm.ops as ops

        assert "gdn_w8a16" in ops.__all__
        assert flaggems_vllm.gdn_w8a16 is ops.gdn_w8a16
        entries = [
            entry for entry in flaggems_vllm._FULL_CONFIG if entry[0] == "gdn_w8a16"
        ]
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
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-3, rtol=5e-2
        )
        torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("packed", [False, True])
    @pytest.mark.parametrize("n,t", [(3, 4), (2, 192)])
    def test_gdn_w8a16_nonzero_state(dtype, packed, n, t):
        torch.manual_seed(43)
        h, hv = 2, 6
        q, k, v, g, beta, cu, _ = make_inputs(n, t, h, hv, dtype, packed)
        # Strided q/k exercise recurrent address calculations.
        q = torch.stack((q, q), dim=-1)[..., 0]
        k = torch.stack((k, k), dim=-1)[..., 0]
        initial = 0.02 * torch.randn(2 * n, hv, 128, 128, device=q.device, dtype=dtype)
        # Permuted, strided slots must select state and scale together in both paths.
        indices = torch.arange(2 * n - 1, -1, -1, device=q.device, dtype=torch.int64)[
            ::2
        ]
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
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-3, rtol=5e-2
        )
        torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)

    @pytest.mark.parametrize("n,t", [(4, 1), (32, 1), (3, 3)])
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
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-3, rtol=5e-2
        )
        torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)
        written = torch.zeros(n * slots, device=q.device, dtype=torch.bool)
        written[indices[:, :t].reshape(-1)] = True
        assert torch.equal(fp8.view(torch.uint8)[~written], before_fp8[~written])
        assert torch.equal(scales[~written], before_scale[~written])

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
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-3, rtol=5e-2
        )
        torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)

    def test_gdn_w8a16_dynamic_scale_range():
        torch.manual_seed(46)
        amplitude = torch.logspace(-2, 3, 128, device=flaggems_vllm.device)
        state = (
            torch.randn(2, 4, 128, 128, device=amplitude.device) * amplitude
        ).bfloat16()
        fp8, scale = quantize_gdn_state_fp8(state)
        restored = dequantize_gdn_state_fp8(fp8, scale)
        error = (restored - state.float()).abs() / state.float().abs().amax(
            2, keepdim=True
        )
        assert error.max().item() < 0.08
        assert error.mean().item() < 0.02
        assert restored.abs().max().item() > 448
