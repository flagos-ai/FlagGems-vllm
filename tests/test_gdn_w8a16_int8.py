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
from tests.gdn_reference import (
    recurrent_gdn_reference,
    torch_gdn_decode,
    torch_gdn_prefill,
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
    return torch_gdn_decode, reference_args, gems_args


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
        torch_gdn_prefill,
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


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX FP32 state")
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
    actual, cache = flaggems_vllm.gdn_w8a16(
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
