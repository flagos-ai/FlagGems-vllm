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

"""FP32 projection and BF16 premix oracle for the three-kernel decode path."""
import pytest
import torch

import flaggems_vllm as gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

pytestmark = pytest.mark.skipif(gems.vendor_name != "ascend", reason="Ascend only")


def make_inputs(n=2, pattern="random"):
    torch.manual_seed(991)
    x = torch.randn(n, 4, 4096).bfloat16()
    if pattern == "zero":
        x.zero_()
    if pattern == "small":
        x *= 0.001
    if pattern == "large":
        x *= 100.0
    fn = torch.randn(24, 16384) * 0.005
    scales = torch.tensor([0.1, 0.3, 0.2])
    base = torch.randn(24) * 0.1
    gamma = (torch.randn(4096) * 0.3 + 1).bfloat16()
    return tuple(t.to(gems.device) for t in (x, fn, scales, base, gamma))


def reference(
    x,
    fn,
    scales,
    base,
    gamma,
    *,
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    norm_eps=1e-5,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
):
    x, fn, scales, base, gamma = [
        t.cpu().double() for t in (x, fn, scales, base, gamma)
    ]
    flat = x.flatten(1)
    logits = flat @ fn.t()
    inv = torch.rsqrt(flat.square().mean(1, keepdim=True) + rms_eps)
    logits *= inv
    pre = (logits[:, :4] * scales[0] + base[:4]).sigmoid() + hc_pre_eps
    post = (logits[:, 4:8] * scales[1] + base[4:8]).sigmoid() * hc_post_mult_value
    comb = (logits[:, 8:] * scales[2] + base[8:]).reshape(-1, 4, 4).softmax(
        -1
    ) + hc_sinkhorn_eps
    comb /= comb.sum(-2, keepdim=True) + hc_sinkhorn_eps
    for _ in range(sinkhorn_repeat - 1):
        comb /= comb.sum(-1, keepdim=True) + hc_sinkhorn_eps
        comb /= comb.sum(-2, keepdim=True) + hc_sinkhorn_eps
    # Required native rounding boundary: do not normalize the FP32 premix.
    value = (x * pre.unsqueeze(-1)).sum(1).bfloat16().double()
    out = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm_eps) * gamma
    return post.float().unsqueeze(-1), comb.float(), out.bfloat16()


def assert_outputs(actual, expected):
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.shape == e.shape and a.dtype == e.dtype
        assert torch.isfinite(a).all()
        torch.testing.assert_close(
            a.cpu(),
            e.cpu(),
            rtol=0.016 if i == 2 else 2e-5,
            atol=0.016 if i == 2 else 2e-6,
        )


@pytest.mark.mhc_pre_norm_small
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
@pytest.mark.parametrize(
    "pattern",
    ["random", "zero"] if QUICK_MODE else ["random", "zero", "small", "large"],
)
def test_small_outputs(n, pattern):
    args = make_inputs(n, pattern)
    before = [t.clone() for t in args]
    assert_outputs(gems.mhc_pre_norm_small(*args), reference(*args))
    for a, b in zip(args, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.mhc_pre_norm_small
@pytest.mark.parametrize("repeats", [1, 5, 20])
def test_independent_eps_and_iterations(repeats):
    args = make_inputs(3)
    kw = dict(
        rms_eps=1e-4,
        hc_pre_eps=0.03,
        hc_sinkhorn_eps=0.002,
        norm_eps=1e-3,
        hc_post_mult_value=1.5,
        sinkhorn_repeat=repeats,
    )
    assert_outputs(gems.mhc_pre_norm_small(*args, **kw), reference(*args, **kw))


@pytest.mark.mhc_pre_norm_small
@pytest.mark.parametrize(
    "invalid",
    [
        "batch",
        "shape",
        "fn",
        "gamma",
        "dtype",
        "layout",
        "device",
        "repeat",
        "eps",
        "grad",
        "eps_tensor",
        "repeat_tensor",
    ],
)
def test_small_rejects_unsupported(invalid):
    args = list(make_inputs(5 if invalid == "batch" else 2))
    kw = {}
    error = ValueError
    if invalid == "batch":
        error = NotImplementedError
    if invalid == "shape":
        args[0] = args[0][:, :, :128]
    if invalid == "fn":
        args[1] = args[1][:23]
    if invalid == "gamma":
        args[4] = args[4][:128]
    if invalid == "dtype":
        args[1] = args[1].bfloat16()
        error = NotImplementedError
    if invalid == "layout":
        args[1] = args[1].t().contiguous().t()
        error = NotImplementedError
    if invalid == "device":
        args[4] = args[4].cpu()
    if invalid == "repeat":
        kw["sinkhorn_repeat"] = 0
    if invalid == "eps":
        kw["norm_eps"] = -1.0
    if invalid == "grad":
        args[1].requires_grad_()
        error = NotImplementedError
    if invalid == "eps_tensor":
        kw["norm_eps"] = torch.tensor(1e-5, device=gems.device)
    if invalid == "repeat_tensor":
        kw["sinkhorn_repeat"] = torch.tensor(20, device=gems.device)
    with pytest.raises(error):
        gems.mhc_pre_norm_small(*args, **kw)


@pytest.mark.mhc_pre_norm_small
@pytest.mark.parametrize("n", [1, 4])
def test_small_changed_input_graph_replay(n):
    args = make_inputs(n)
    gems.mhc_pre_norm_small(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = gems.mhc_pre_norm_small(*args)
    for factor in (0.0, 0.5, 1.25):
        args[0].fill_(factor)
        args[1].fill_(factor * 0.001)
        graph.replay()
        torch.npu.synchronize()
        assert_outputs(out, reference(*args))


@pytest.mark.mhc_pre_norm_small
def test_small_native_parity():
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "npu_hc_pre_v2"):
        pytest.skip("GLM5Next native extension unavailable")
    args = make_inputs(4)
    x, fn, s, b, gamma = args
    value, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        x, fn, s, b, 4, 20, 1e-5, 1e-6
    )
    out = torch.ops.npu.npu_rms_norm(value, gamma, 1e-5)[0]
    assert_outputs(gems.mhc_pre_norm_small(*args), (post.unsqueeze(-1), comb, out))


@pytest.mark.mhc_pre_norm_small
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.0])
def test_small_nonfinite_and_signed_zero(value):
    args = make_inputs(1)
    args[0].fill_(value)
    actual = gems.mhc_pre_norm_small(*args)
    expected = reference(*args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(
            a.cpu(), e.cpu(), rtol=0.016, atol=0.016, equal_nan=True
        )


@pytest.mark.mhc_pre_norm_small
def test_small_public_dispatch_log(caplog):
    args = make_inputs(2, "zero")
    entry = gems.mhc_pre_norm_small
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(entry)):
        actual = entry(*args)
    assert f"{utils.gems_log_prefix(entry)} MHC_PRE_NORM_SMALL" in caplog.text
    expected = reference(*args)
    utils.gems_assert_close(actual[2].cpu(), expected[2], dtype=torch.bfloat16)
    assert_outputs(actual, expected)
