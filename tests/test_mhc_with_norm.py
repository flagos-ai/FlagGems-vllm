# SPDX-License-Identifier: Apache-2.0
"""MHC oracle keeps the BF16 projection and weighted-sum boundaries."""

import importlib

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _norm_reference(x, weight, eps):
    if weight is None:
        return x
    value = x.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)


def _pre_reference(residual, fn, scales, base, eps=1e-6, repeats=5, projected=None):
    hc = residual.shape[-2]
    flat = residual.flatten(-2).bfloat16()
    # Both GEMM output and the weighted layer input materialize in BF16.
    if projected is None:
        projected = (flat @ fn.bfloat16().t()).bfloat16().float()
    rms = torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + eps)
    expanded = torch.cat(
        (scales[0].expand(hc), scales[1].expand(hc), scales[2].expand(hc * hc))
    )
    mix = projected * rms * expanded + base
    pre = mix[:, :hc].sigmoid().unsqueeze(-1) + eps
    post = mix[:, hc : hc * 2].sigmoid().unsqueeze(-1)
    comb = mix[:, hc * 2 :].view(-1, hc, hc).softmax(-1) + eps
    comb /= comb.sum(-2, keepdim=True) + eps
    for _ in range(repeats - 1):
        comb /= comb.sum(-1, keepdim=True) + eps
        comb /= comb.sum(-2, keepdim=True) + eps
    value = (residual.float() * pre).sum(-2).bfloat16()
    return post, comb, value


def _inputs(tokens=3, hc=4, hidden=128):
    torch.manual_seed(731)
    residual = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16, device=gems.device)
    fn = torch.randn(2 * hc + hc * hc, hc * hidden, device=gems.device) * 0.003
    scale = torch.tensor([0.1, 0.3, 0.2], device=gems.device)
    base = torch.randn(2 * hc + hc * hc, device=gems.device) * 0.1
    norm = torch.randn(hidden, dtype=torch.bfloat16, device=gems.device) * 0.3 + 1
    return residual, fn, scale, base, norm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [127, 128, 257, 1280])
def test_rms_norm_strides_and_materialized_dtype(dtype, hidden):
    torch.manual_seed(42)
    x = torch.randn(3, hidden * 2, dtype=dtype, device=gems.device)[:, ::2]
    weight = torch.randn(hidden * 2, dtype=dtype, device=gems.device)[::2]
    actual = gems.mhc_rms_norm(x, weight, 1e-6)
    expected = _norm_reference(x, weight, 1e-6)
    assert actual.dtype == dtype
    tolerance = 1e-5 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_rms_norm_optional_identity_and_empty():
    x = torch.empty(0, 127, dtype=torch.bfloat16, device=gems.device)
    assert gems.mhc_rms_norm(x, None, 1e-6) is x
    actual = gems.mhc_rms_norm(x, torch.ones(127, device=gems.device), 1e-6)
    assert actual.shape == x.shape and actual.dtype == x.dtype


@pytest.mark.parametrize("hc,hidden", [(2, 127), (4, 128), (4, 257)])
def test_mhc_pre_norm_preserves_bf16_weighted_sum(hc, hidden):
    residual, fn, scale, base, norm = _inputs(hc=hc, hidden=hidden)
    expected = _pre_reference(residual, fn, scale, base)
    actual = gems.mhc_pre_with_norm(
        residual,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
        norm_weight=norm,
        norm_eps=1e-6,
    )
    for observed, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-5)
    reference_norm = _norm_reference(expected[2], norm, 1e-6)
    assert actual[2].dtype == torch.bfloat16
    torch.testing.assert_close(actual[2], reference_norm, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("tokens,hidden", [(3, 128)])
def test_mhc_fused_post_pre_norm_order(tokens, hidden):
    residual, fn, scale, base, norm = _inputs(tokens=tokens, hidden=hidden)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=gems.device)
    post = torch.randn(tokens, 4, 1, device=gems.device) * 0.3
    comb = torch.randn(tokens, 4, 4, device=gems.device) * 0.2
    # The unchanged public post is the shared stage under both pipelines.
    after_post = gems.mhc_post(x, residual, post, comb)
    expected = _pre_reference(after_post, fn, scale, base)
    actual = gems.mhc_fused_post_pre_with_norm(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
        norm_weight=norm,
        norm_eps=1e-6,
    )
    torch.testing.assert_close(actual[0], after_post, rtol=0, atol=0)
    for observed, reference in zip(actual[1:3], expected[:2]):
        torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        actual[3], _norm_reference(expected[2], norm, 1e-6), rtol=0.01, atol=0.01
    )


def test_mhc_hot_projection_materialization_and_mix_norm_stages(monkeypatch):
    """Validate projection accuracy before its discontinuous BF16 boundary."""
    module = importlib.import_module("flaggems_vllm.ops.mhc.mhc_pre")
    residual, fn, scale, base, norm = _inputs(tokens=512, hidden=1280)
    x = torch.randn(512, 1280, dtype=torch.bfloat16, device=gems.device)
    post = torch.randn(512, 4, 1, device=gems.device) * 0.3
    comb = torch.randn(512, 4, 4, device=gems.device) * 0.2
    after_post = gems.mhc_post(x, residual, post, comb)
    projections, mix_inputs = [], []
    original_bmm = module.flag_gems.bmm_out
    original_kernel = module.mhc_pre_fused_kernel_hc_mult_4

    def observe_bmm(a, b, out):
        assert a.dtype == b.dtype == torch.bfloat16
        assert out.dtype == torch.float32
        torch.testing.assert_close(a[0], after_post.flatten(-2), rtol=0, atol=0)
        torch.testing.assert_close(b[0], fn.bfloat16().t(), rtol=0, atol=0)
        result = original_bmm(a, b, out)
        projections.append(out[0].clone())
        return result

    class ObserveMix:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def observe(*args, **kwargs):
                mix_inputs.append(args[0].clone())
                return launch(*args, **kwargs)

            return observe

    monkeypatch.setattr(module.flag_gems, "bmm_out", observe_bmm)
    monkeypatch.setattr(module, "mhc_pre_fused_kernel_hc_mult_4", ObserveMix())
    actual = gems.mhc_fused_post_pre_with_norm(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
        norm_weight=norm,
        norm_eps=1e-6,
    )
    assert len(projections) == len(mix_inputs) == 1
    exact_projection = after_post.flatten(-2).double() @ fn.bfloat16().t().double()
    torch.testing.assert_close(
        projections[0].double(), exact_projection, rtol=1e-5, atol=1e-5
    )
    materialized_projection = projections[0].bfloat16().float()
    assert mix_inputs[0].dtype == torch.float32
    torch.testing.assert_close(mix_inputs[0], materialized_projection, rtol=0, atol=0)
    expected = _pre_reference(
        after_post, fn, scale, base, projected=materialized_projection
    )
    torch.testing.assert_close(actual[0], after_post, rtol=0, atol=0)
    for observed, reference in zip(actual[1:3], expected[:2]):
        torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        actual[3], _norm_reference(expected[2], norm, 1e-6), rtol=0.01, atol=0.01
    )


@pytest.mark.parametrize("invalid", ["shape", "device"])
def test_mhc_norm_preflight_before_post_and_pre(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.mhc_with_norm")
    calls = []
    monkeypatch.setattr(module, "mhc_pre", lambda *args: calls.append("pre"))
    monkeypatch.setattr(module, "mhc_post", lambda *args: calls.append("post"))
    residual, fn, scale, base, norm = _inputs()
    norm = norm[:127] if invalid == "shape" else norm.cpu()
    before = residual.clone()
    with pytest.raises(ValueError):
        gems.mhc_pre_with_norm(
            residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 5, norm_weight=norm
        )
    with pytest.raises(ValueError):
        gems.mhc_fused_post_pre_with_norm(
            residual[:, 0],
            residual,
            torch.ones(3, 4, 1, device=gems.device),
            torch.ones(3, 4, 4, device=gems.device),
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            5,
            norm_weight=norm,
        )
    assert not calls
    torch.testing.assert_close(residual, before, rtol=0, atol=0)


@torch.no_grad()
def test_mhc_pre_inference_repeats_and_weight_version_change():
    residual, fn, scale, base, norm = _inputs()
    for factor in (1.0, 1.25):
        fn.mul_(factor)
        expected = _pre_reference(residual, fn, scale, base)
        for _ in range(2):
            actual = gems.mhc_pre_with_norm(
                residual,
                fn,
                scale,
                base,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                5,
                norm_weight=norm,
                norm_eps=1e-6,
            )
            for observed, reference in zip(actual[:2], expected[:2]):
                torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(
                actual[2],
                _norm_reference(expected[2], norm, 1e-6),
                rtol=0.01,
                atol=0.01,
            )


@pytest.mark.parametrize("hc", [2, 4])
@pytest.mark.parametrize("with_norm", [False, True])
def test_mhc_pre_empty_preserves_output_contract(hc, with_norm):
    residual, fn, scale, base, norm = _inputs(tokens=0, hc=hc)
    actual = gems.mhc_pre_with_norm(
        residual,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
        norm_weight=norm if with_norm else None,
        norm_eps=1e-6,
    )
    for out, shape, dtype in zip(
        actual,
        ((0, hc, 1), (0, hc, hc), (0, 128)),
        (torch.float32, torch.float32, torch.bfloat16),
    ):
        assert (
            out.shape == shape and out.dtype == dtype and out.device == residual.device
        )


@torch.inference_mode()
def test_mhc_inference_tensor_without_version_counter():
    residual, fn, scale, base, norm = _inputs()
    assert fn.is_inference()
    expected = _pre_reference(residual, fn, scale, base)
    for _ in range(2):
        actual = gems.mhc_pre_with_norm(
            residual,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            5,
            norm_weight=norm,
            norm_eps=1e-6,
        )
        for observed, reference in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            actual[2],
            _norm_reference(expected[2], norm, 1e-6),
            rtol=0.01,
            atol=0.01,
        )
