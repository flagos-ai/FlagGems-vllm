# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 recurrence; fixed tolerances shared with native validation."""

import os

import pytest
import torch
import torch.nn.functional as F

CASES = [
    (1, 1, 2, 64, 64, None),
    (2, 15, 2, 64, 32, None),
    (1, 16, 2, 64, 64, None),
    (1, 17, 2, 33, 40, None),
    (1, 31, 2, 64, 64, None),
    (1, 32, 2, 128, 128, None),
    (1, 33, 2, 64, 80, None),
    (1, 63, 2, 128, 64, None),
    (1, 64, 2, 128, 128, None),
    (1, 65, 2, 128, 128, None),
    (1, 127, 2, 256, 80, None),
    (1, 129, 2, 64, 96, None),
    (1, 83, 2, 64, 64, (65, 1, 17)),
    (1, 198, 2, 128, 128, (63, 64, 71)),
]


def get_op():
    if os.environ.get("FG_KDA_PROVIDER") == "native":
        from vllm.model_executor.layers.fla.ops.kda import chunk_kda_with_fused_gate
    else:
        from flaggems_vllm import chunk_kda_with_fused_gate
    return chunk_kda_with_fused_gate


def device_name():
    explicit = os.environ.get("FG_KDA_DEVICE")
    if explicit is not None:
        return explicit
    if os.environ.get("FG_KDA_PROVIDER") == "native":
        # The isolated native runner supplies its backend explicitly.
        return "cuda"
    import flaggems_vllm

    return flaggems_vllm.device


def build(case, dtype, norm=True, state=True, bias=True, final=True, seed=20261029):
    torch.manual_seed(seed)
    b, t, h, dk, dv, lengths = case
    dev = device_name()
    q = (torch.randn(b, t, h, dk, device=dev) * 0.2).to(dtype)
    k = torch.randn_like(q) * 0.2
    v = (torch.randn(b, t, h, dv, device=dev) * 0.2).to(dtype)
    g = (torch.randn(b, t, h, dk, device=dev) * 0.5).to(dtype)
    beta = torch.sigmoid(torch.randn(b, t, h, device=dev))
    a = torch.full((h,), -2.0, device=dev)
    gb = torch.randn(h, dk, device=dev) * 0.1 if bias else None
    n = len(lengths) if lengths is not None else b
    h0 = torch.randn(n, h, dv, dk, device=dev) * 0.1 if state else None
    cu = (
        torch.tensor(
            [0] + list(__import__("itertools").accumulate(lengths)),
            device=dev,
            dtype=torch.long,
        )
        if lengths is not None
        else None
    )
    return (q, k, v, g, beta, a, gb), dict(
        scale=dk**-0.5,
        initial_state=h0,
        output_final_state=final,
        use_qk_l2norm_in_kernel=norm,
        cu_seqlens=cu,
    )


def reference(args, kwargs):
    q, k, v, raw, beta, a, bias = [
        x.detach().cpu() if x is not None else None for x in args
    ]
    dtype = q.dtype
    if kwargs["use_qk_l2norm_in_kernel"]:
        # Match public norm materialization before independent recurrence.
        q = (
            q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
        k = (
            k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
    q, k, v, raw, beta, a = [x.double() for x in (q, k, v, raw, beta, a)]
    gate = -a.exp()[None, None, :, None] * F.softplus(
        raw + (bias.double()[None, None] if bias is not None else 0)
    )
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    cu = kwargs["cu_seqlens"]
    bounds = [0, t] if cu is None else cu.cpu().tolist()
    initial = kwargs["initial_state"]
    initial = initial.detach().cpu().double() if initial is not None else None
    out = torch.empty_like(v)
    finals = []
    n = b if cu is None else len(bounds) - 1
    for seq in range(n):
        batch = seq if cu is None else 0
        start, end = (0, t) if cu is None else bounds[seq : seq + 2]
        state = (
            torch.zeros(h, dv, dk, dtype=torch.float64)
            if initial is None
            else initial[seq].clone()
        )
        for i in range(start, end):
            state *= gate[batch, i].exp()[:, None, :]
            residual = v[batch, i] - torch.einsum("hvk,hk->hv", state, k[batch, i])
            state += (
                beta[batch, i, :, None, None]
                * residual[:, :, None]
                * k[batch, i, :, None, :]
            )
            out[batch, i] = (
                torch.einsum("hvk,hk->hv", state, q[batch, i]) * kwargs["scale"]
            )
        finals.append(state)
    final = torch.stack(finals).float() if finals else torch.empty((0, h, dv, dk))
    return out.to(dtype), final if kwargs["output_final_state"] else None


def check(actual, expected, dtype):
    atol, rtol = (0.008, 0.03) if dtype == torch.bfloat16 else (0.002, 0.006)
    torch.testing.assert_close(
        actual.cpu(), expected, atol=atol, rtol=rtol, check_dtype=False
    )


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("flags", [False, True])
def test_chunk_kda_with_fused_gate_reference(case, dtype, flags):
    args, kw = build(case, dtype, norm=flags, state=flags, bias=flags, final=flags)
    expected = reference(args, kw)
    saved = [
        x.clone() if x is not None else None
        for x in (*args[:2], *args[3:], kw["initial_state"])
    ]
    original_v = args[2]
    actual = get_op()(*args, **kw)
    check(actual[0], expected[0], dtype)
    assert actual[0].data_ptr() == original_v.data_ptr()
    if flags:
        assert actual[1].dtype == torch.float32
        check(actual[1], expected[1], dtype)
    else:
        assert actual[1] is None
    for old, new in zip(saved, (*args[:2], *args[3:], kw["initial_state"])):
        if old is not None:
            torch.testing.assert_close(old, new, atol=0, rtol=0)


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontig", [False, True])
def test_chunk_kda_with_fused_gate_layout(dtype, noncontig):
    args, kw = build((1, 65, 2, 64, 40, None), dtype)
    args = list(args)
    if noncontig:
        # A real gapped last stride; expected output does not alias original v.
        base = torch.empty(
            (*args[2].shape[:-1], 80), device=args[2].device, dtype=dtype
        )
        base[..., ::2] = args[2]
        args[2] = base[..., ::2]
    expected = reference(args, kw)
    result = get_op()(*args, **kw)
    check(result[0], expected[0], dtype)
    check(result[1], expected[1], dtype)
    assert (result[0].data_ptr() == args[2].data_ptr()) != noncontig


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_kda_with_fused_gate_beta_zero(dtype):
    args, kw = build((1, 65, 2, 64, 64, None), dtype)
    args[4].zero_()
    expected = reference(args, kw)
    actual = get_op()(*args, **kw)
    check(actual[0], expected[0], dtype)
    check(actual[1], expected[1], dtype)


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("flags", range(16))
def test_chunk_kda_with_fused_gate_independent_flags(dtype, flags):
    args, kw = build(
        (1, 65, 2, 64, 40, None),
        dtype,
        norm=bool(flags & 1),
        state=bool(flags & 2),
        bias=bool(flags & 4),
        final=bool(flags & 8),
    )
    expected = reference(args, kw)
    actual = get_op()(*args, **kw)
    check(actual[0], expected[0], dtype)
    if expected[1] is None:
        assert actual[1] is None
    else:
        check(actual[1], expected[1], dtype)


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [None, (0, 0)])
@pytest.mark.parametrize("state_dtype", [None, torch.float16, torch.float32])
def test_chunk_kda_with_fused_gate_empty(dtype, lengths, state_dtype):
    args, kw = build((1, 0, 2, 64, 40, lengths), dtype, state=state_dtype is not None)
    if state_dtype is not None:
        kw["initial_state"] = kw["initial_state"].to(state_dtype)
        saved = kw["initial_state"].clone()
    actual = get_op()(*args, **kw)
    assert actual[0].shape == args[2].shape
    assert actual[1].dtype == torch.float32
    expected = torch.zeros_like(actual[1]) if state_dtype is None else saved.float()
    torch.testing.assert_close(actual[1], expected, atol=0, rtol=0)
    if state_dtype is not None:
        torch.testing.assert_close(kw["initial_state"], saved, atol=0, rtol=0)


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_kda_with_fused_gate_low_precision_state_and_empty_sequence(dtype):
    args, kw = build((1, 66, 2, 33, 40, (0, 65, 0, 1, 0)), dtype)
    kw["initial_state"] = kw["initial_state"].to(dtype)
    expected = reference(args, kw)
    saved = kw["initial_state"].clone()
    actual = get_op()(*args, **kw)
    check(actual[0], expected[0], dtype)
    check(actual[1], expected[1], dtype)
    torch.testing.assert_close(kw["initial_state"], saved, atol=0, rtol=0)


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [None, (0, 33, 1, 0, 33)])
@pytest.mark.parametrize("noncontig", [False, True])
def test_chunk_kda_with_fused_gate_wide_key_metadata(dtype, lengths, noncontig):
    args, kw = build((1, 67, 2, 256, 40, lengths), dtype, norm=False)
    args = list(args)
    if noncontig:
        values = torch.empty(
            (*args[2].shape[:-1], 80), device=args[2].device, dtype=dtype
        )
        values[..., ::2] = args[2]
        args[2] = values[..., ::2]
    expected = reference(args, kw)
    saved_state = kw["initial_state"].clone()
    actual = get_op()(*args, **kw)
    check(actual[0], expected[0], dtype)
    check(actual[1], expected[1], dtype)
    assert (actual[0].data_ptr() == args[2].data_ptr()) != noncontig
    torch.testing.assert_close(kw["initial_state"], saved_state, atol=0, rtol=0)


@pytest.mark.chunk_kda_with_fused_gate
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("flags", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        (1, 17, 2, 1, 1, None),
        (1, 65, 2, 7, 9, None),
        (1, 83, 2, 16, 17, (65, 0, 18)),
    ],
)
def test_chunk_kda_with_fused_gate_small_dimensions(case, dtype, flags):
    args, kwargs = build(case, dtype, norm=flags, state=flags, bias=flags, final=True)
    expected = reference(args, kwargs)
    initial = kwargs["initial_state"]
    saved = initial.clone() if initial is not None else None
    actual = get_op()(*args, **kwargs)
    check(actual[0], expected[0], dtype)
    check(actual[1], expected[1], dtype)
    assert actual[0].data_ptr() == args[2].data_ptr()
    assert actual[1].dtype == torch.float32
    if saved is not None:
        torch.testing.assert_close(initial, saved, atol=0, rtol=0)
