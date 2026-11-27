# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BF16 SiTU, using the existing capped/non-inferior FP32 reference gate."""
import flag_gems
import pytest
import torch

from flaggems_vllm import runtime, situ_and_mul

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name != "metax", reason="MetaX implementation"
)


def _formula(x, beta, linear_beta):
    gate, up = x.float().chunk(2, -1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def _errors(actual, ref):
    actual, ref = actual.detach().cpu().float(), ref.detach().cpu().float()
    if not actual.numel():
        return 0.0, 0.0, 0.0
    delta = actual - ref
    near = ref.abs() <= 1e-3
    return (
        float(delta.abs().max()),
        float(delta.square().mean().sqrt()),
        float(delta.abs()[near].max()) if near.any() else 0.0,
    )


@pytest.mark.parametrize(
    "m,width", [(0, 1536), (1, 768), (3, 1536), (17, 8448), (64, 1536), (4096, 8448)]
)
@pytest.mark.parametrize("pattern", ["normal", "zero", "tiny", "saturated"])
@pytest.mark.parametrize("beta,linear_beta", [(4.0, 25.0), (1.0, -1.0)])
def test_situ_precision_and_shape(m, width, pattern, beta, linear_beta):
    torch.manual_seed(851 + m)
    cpu = torch.randn(m, width).bfloat16()
    if pattern == "zero":
        cpu.zero_()
    elif pattern == "tiny":
        cpu.mul_(1e-5)
    elif pattern == "saturated":
        cpu.mul_(30)
    x = cpu.cuda()
    out = torch.empty((m, width // 2), dtype=x.dtype, device=x.device)
    with flag_gems.use_gems():
        baseline = _formula(x, beta, linear_beta).to(x.dtype)
        situ_and_mul(out, x, beta, linear_beta)
    ref = _formula(cpu, beta, linear_beta)
    a, b = _errors(out, ref), _errors(baseline, ref)
    cap = float(ref.abs().max()) / 128 + 1e-6 if ref.numel() else 1e-6
    assert torch.isfinite(out).all()
    assert a[0] <= min(cap, b[0] * 1.02 + 1e-6)
    assert a[1] <= b[1] * 1.02 + 1e-7
    assert a[2] <= b[2] * 1.02 + 1e-6
    assert torch.equal(x.cpu().view(torch.int16), cpu.view(torch.int16))


def test_situ_changing_graph_and_positive_zero():
    x = torch.randn(17, 1536, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(17, 768, dtype=x.dtype, device=x.device)
    for _ in range(3):
        situ_and_mul(out, x, 4.0, 25.0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        situ_and_mul(out, x, 4.0, 25.0)
    for _ in range(3):
        x.normal_()
        graph.replay()
        eager = torch.empty_like(out)
        situ_and_mul(eager, x, 4.0, 25.0)
        assert torch.equal(eager.view(torch.int16), out.view(torch.int16))
    x.fill_(-0.0)
    graph.replay()
    assert (out.view(torch.int16) == 0).all()


def test_situ_rejects_alias_dtype_and_stride():
    x = torch.zeros(1, 1536, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="share storage"):
        situ_and_mul(x[:, :768], x, 4.0, 25.0)
    with pytest.raises(NotImplementedError):
        situ_and_mul(torch.empty(1, 768, device="cuda"), x)
    with pytest.raises(ValueError):
        situ_and_mul(torch.empty(1, 384, dtype=x.dtype, device=x.device), x[:, ::2])
    with pytest.raises(ValueError):
        situ_and_mul(torch.empty(1, 768, dtype=x.dtype, device=x.device), x, 0.0)
