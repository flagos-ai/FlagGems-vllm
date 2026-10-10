# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [(1, 1, 3, 5), (7, 4, 31, 17), (128, 64, 512, 64)]
if cfg.QUICK_MODE:
    SHAPES = SHAPES[:2]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@gpu
@pytest.mark.concat_mla_q
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_concat_mla_q_matches_reference(dtype, shape, broadcast, strided):
    tokens, heads, nope, rope = shape
    step = 2 if strided else 1
    qn = torch.randn(tokens, heads, nope * step, dtype=dtype, device="cuda")[
        ..., ::step
    ]
    qr = torch.randn(
        tokens, 1 if broadcast else heads, rope * step, dtype=dtype, device="cuda"
    )[..., ::step]
    backing = torch.full(
        (tokens, heads, (nope + rope) * step), 123, dtype=dtype, device="cuda"
    )
    out = backing[..., ::step]
    before_n, before_r = qn.clone(), qr.clone()
    address = out.data_ptr()
    assert flaggems_vllm.concat_mla_q(qn, qr, out) is None
    expected = torch.cat([qn, qr.expand(tokens, heads, rope)], dim=-1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out.data_ptr() == address
    torch.testing.assert_close(qn, before_n, rtol=0, atol=0)
    torch.testing.assert_close(qr, before_r, rtol=0, atol=0)
    if strided:
        assert torch.all(backing[..., 1::2] == 123)


@gpu
@pytest.mark.concat_mla_q
def test_empty_and_nonfinite_values():
    qn = torch.empty(0, 4, 3, device="cuda")
    qr = torch.empty(0, 1, 5, device="cuda")
    out = torch.empty(0, 4, 8, device="cuda")
    assert flaggems_vllm.concat_mla_q(qn, qr, out) is None
    qn = torch.tensor([[[float("nan"), float("inf"), -0.0]]], device="cuda")
    qr = torch.tensor([[[-float("inf"), 0.0]]], device="cuda")
    out = torch.empty(1, 1, 5, device="cuda")
    flaggems_vllm.concat_mla_q(qn, qr, out)
    expected = torch.cat([qn, qr], dim=-1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(out), torch.signbit(expected))


@gpu
@pytest.mark.concat_mla_q
@pytest.mark.parametrize("broken", ["shape", "head", "dtype", "alias", "overlap"])
def test_invalid_input_rejected_before_output_mutation(broken):
    qn = torch.randn(2, 4, 3, device="cuda")
    qr = torch.randn(2, 1, 5, device="cuda")
    out = torch.full((2, 4, 8), 77.0, device="cuda")
    flaggems_vllm.concat_mla_q(qn, qr, out)
    out.fill_(77)
    if broken == "shape":
        out = out[..., :-1]
    elif broken == "head":
        qr = torch.randn(2, 2, 5, device="cuda")
    elif broken == "dtype":
        qr = qr.half()
    elif broken == "alias":
        qn = out[..., :3]
    else:
        out = out[:, :1].expand(2, 4, 8)
    before = out.clone()
    with pytest.raises(ValueError):
        flaggems_vllm.concat_mla_q(qn, qr, out)
    torch.testing.assert_close(out, before, rtol=0, atol=0)


@pytest.mark.concat_mla_q
def test_cpu_is_unsupported_without_torch_compute_fallback():
    qn, qr, out = [torch.ones(2, 1, n) for n in (3, 5, 8)]
    with pytest.raises(NotImplementedError, match="CUDA"):
        flaggems_vllm.concat_mla_q(qn, qr, out)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@gpu
@pytest.mark.concat_mla_q
def test_graph_replay_consumes_new_inputs_and_writes_output():
    qn = torch.randn(7, 4, 31, device="cuda", dtype=torch.bfloat16)
    qr = torch.randn(7, 1, 17, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(7, 4, 48, device="cuda", dtype=torch.bfloat16)
    flaggems_vllm.concat_mla_q(qn, qr, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.concat_mla_q(qn, qr, out)
    address = out.data_ptr()
    for seed in (11, 23):
        torch.manual_seed(seed)
        qn.normal_()
        qr.normal_()
        expected = torch.cat([qn, qr.expand(7, 4, 17)], dim=-1)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert out.data_ptr() == address


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least two CUDA GPUs"
)
@pytest.mark.concat_mla_q
@pytest.mark.parametrize("input_device", [0, 1])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("broadcast", [False, True])
def test_concat_mla_q_uses_input_device_and_restores_current(
    input_device, dtype, broadcast
):
    device = torch.device("cuda", input_device)
    qn = torch.randn(7, 4, 31, device=device, dtype=dtype)
    qr = torch.randn(7, 1 if broadcast else 4, 17, device=device, dtype=dtype)
    out = torch.empty(7, 4, 48, device=device, dtype=dtype)
    with torch.cuda.device(1 - input_device):
        current = torch.cuda.current_device()
        for _ in range(2):
            qn.normal_()
            qr.normal_()
            out.fill_(float("nan"))
            expected = torch.cat([qn, qr.expand(7, 4, 17)], dim=-1)
            assert torch.cuda.current_device() == current
            assert flaggems_vllm.concat_mla_q(qn, qr, out) is None
            assert torch.cuda.current_device() == current
            torch.cuda.synchronize(device)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
