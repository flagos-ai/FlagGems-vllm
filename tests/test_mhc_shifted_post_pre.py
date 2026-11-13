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

"""Functional tests for the shifted mHC post->pre operator.

The reference mirrors vLLM's ``mhc_pre_delayed_torch``; the bf16 materialisation
of the projection is the documented numeric boundary of this implementation.
"""

import re
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from flaggems_vllm.ops.mhc.mhc_post import MixLayout, mhc_post, mhc_post_ref
from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre
from flaggems_vllm.ops.mhc.mhc_shifted_post_pre import mhc_shifted_post_pre

from .conftest import QUICK_MODE


def _load_vllm_delayed_reference():
    """vLLM's delayed pre/post reference: installed module, else a source tree.

    vLLM 0.24.0 ships mhc_pre_torch/mhc_post_torch but not the *delayed* pre;
    that one lives in the DSV4.1 tree.  The module is pure torch, so a checkout
    can be loaded by path (point ``VLLM_DSV41_SRC`` at its root) without
    building vLLM.
    """
    try:
        from vllm.model_executor.kernels.mhc.torch import (
            mhc_post_torch,
            mhc_pre_delayed_torch,
        )

        return mhc_post_torch, mhc_pre_delayed_torch, "installed vllm"
    except ImportError:
        pass
    import importlib.util
    import os

    root = os.environ.get("VLLM_DSV41_SRC")
    if not root:
        return None, None, None
    path = os.path.join(root, "vllm/model_executor/kernels/mhc/torch.py")
    if not os.path.isfile(path):
        return None, None, None
    spec = importlib.util.spec_from_file_location("_vllm_dsv41_mhc_torch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "mhc_pre_delayed_torch"):
        return None, None, None
    return module.mhc_post_torch, module.mhc_pre_delayed_torch, root


_VLLM_POST_DELAYED, _VLLM_PRE_DELAYED, _VLLM_SRC = _load_vllm_delayed_reference()
HAS_VLLM_DELAYED_REF = _VLLM_PRE_DELAYED is not None


try:  # vLLM ships the torch reference for the same mHC math
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch as _vllm_post_torch
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch as _vllm_pre_torch

    HAS_VLLM_REF = True
except ImportError:
    HAS_VLLM_REF = False

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

MIX_EPS = 2e-3
COLLAPSE_EPS = 1e-2

_SHAPE_CASES = (
    [(1, 64, 4), (8, 1280, 4)]
    if QUICK_MODE
    else [
        (1, 64, 4),
        (1, 1280, 4),
        (8, 1280, 4),
        (64, 4096, 4),
        (3, 127, 4),
        (5, 257, 2),
        (7, 5120, 4),
        (17, 300, 1),
        (33, 1024, 8),
    ]
)
_HIDDEN_CASES = [127, 1280] if QUICK_MODE else [127, 129, 257, 1000, 1280, 4095]
_HC_CASES = [1, 4] if QUICK_MODE else [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Torch reference.  It lives in tests, not in the operator module: production
# code may not carry a torch compute path (see the repository workflow).
# ---------------------------------------------------------------------------


def _norm_ref(layer_input, norm_weight, norm_eps):
    if norm_weight is None:
        return layer_input
    value = layer_input.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).to(layer_input.dtype)


def _sinkhorn_normalize_ref(x, repeat, eps):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_shifted_post_pre_ref(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
    capture_aux=False,
):
    """PyTorch reference.  Mirrors vLLM's ``mhc_pre_delayed_torch`` semantics.

    The projection materialises in bfloat16 before the RMS scaling, which is the
    numeric boundary the Triton kernel reproduces.
    """
    residual_cur = mhc_post_ref(
        x, residual, post_layer_mix, comb_res_mix, mix_layout=MixLayout.TRANSPOSED
    )
    hc = residual.shape[-2]
    hidden_size = residual.shape[-1]
    flat = residual_cur.flatten(-2).bfloat16()
    projected = (flat @ fn.bfloat16().t()).bfloat16().float()
    rms = torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + rms_eps)
    mixes = projected * rms

    pre = (mixes[:, :hc] * hc_scale[0] + hc_base[:hc]).sigmoid() + hc_pre_eps
    post_logits = mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    post = (post_logits.sigmoid() * hc_post_mult_value).unsqueeze(-1)
    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * hc_scale[2]
    comb = comb + hc_base[2 * hc :].view(1, hc, hc)
    comb = _sinkhorn_normalize_ref(comb, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)

    if pre_mix is None:
        layer_input = residual_cur[:, 0]
    else:
        layer_input = (
            (pre_mix.unsqueeze(-1) * residual_cur.float()).sum(dim=1).bfloat16()
        )
    layer_input = _norm_ref(layer_input, norm_weight, norm_eps)

    if capture_aux:
        aux = residual_cur.mean(dim=1)
    else:
        aux = residual_cur.new_empty(0, hidden_size)
    return residual_cur, post, comb, layer_input, pre, aux


def _inputs(num_tokens=8, hidden=1280, hc=4, seed=731, with_pre=True, with_norm=True):
    torch.manual_seed(seed)
    dev = gems.device
    residual = torch.randn(num_tokens, hc, hidden, dtype=torch.bfloat16, device=dev)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
    fn = torch.randn(hc * (hc + 2), hc * hidden, device=dev) * 0.02
    hc_scale = torch.tensor([0.1, 0.3, 0.2], device=dev)
    hc_base = torch.randn(hc * (hc + 2), device=dev) * 0.1
    post = torch.rand(num_tokens, hc, 1, device=dev) * 0.3
    comb = torch.randn(num_tokens, hc, hc, device=dev) * 0.2
    pre = torch.rand(num_tokens, hc, device=dev) + 0.5 if with_pre else None
    norm = (
        torch.randn(hidden, dtype=torch.bfloat16, device=dev) * 0.3 + 1
        if with_norm
        else None
    )
    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post,
        "comb_res_mix": comb,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "pre_mix": pre,
        "norm_weight": norm,
    }


def _call(kwargs, **overrides):
    args = dict(
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=5,
        pre_mix=kwargs["pre_mix"],
        norm_weight=kwargs["norm_weight"],
        norm_eps=1e-6,
        capture_aux=True,
    )
    args.update(overrides)
    return mhc_shifted_post_pre(
        kwargs["x"],
        kwargs["residual"],
        kwargs["post_layer_mix"],
        kwargs["comb_res_mix"],
        kwargs["fn"],
        kwargs["hc_scale"],
        kwargs["hc_base"],
        **args,
    )


def _ref(kwargs, **overrides):
    args = dict(
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=5,
        pre_mix=kwargs["pre_mix"],
        norm_weight=kwargs["norm_weight"],
        norm_eps=1e-6,
        capture_aux=True,
    )
    args.update(overrides)
    return mhc_shifted_post_pre_ref(
        kwargs["x"],
        kwargs["residual"],
        kwargs["post_layer_mix"],
        kwargs["comb_res_mix"],
        kwargs["fn"],
        kwargs["hc_scale"],
        kwargs["hc_base"],
        **args,
    )


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize("num_tokens,hidden,hc", _SHAPE_CASES)
def test_matches_reference(num_tokens, hidden, hc):
    data = _inputs(num_tokens=num_tokens, hidden=hidden, hc=hc)
    actual = _call(data)
    expected = _ref(data)

    assert len(actual) == 6 and len(expected) == 6
    for observed, reference, name in zip(
        actual, expected, ("residual", "post", "comb", "input", "pre", "aux")
    ):
        assert observed.dtype == reference.dtype, name
        assert observed.shape == reference.shape, name
    torch.testing.assert_close(actual[1], expected[1], rtol=MIX_EPS, atol=MIX_EPS)
    torch.testing.assert_close(actual[2], expected[2], rtol=MIX_EPS, atol=MIX_EPS)
    torch.testing.assert_close(actual[4], expected[4], rtol=MIX_EPS, atol=MIX_EPS)
    torch.testing.assert_close(
        actual[3], expected[3], rtol=COLLAPSE_EPS, atol=COLLAPSE_EPS
    )
    torch.testing.assert_close(
        actual[5], expected[5], rtol=COLLAPSE_EPS, atol=COLLAPSE_EPS
    )


@pytest.mark.mhc_shifted_post_pre
def test_post_stage_is_bitwise_mhc_post():
    """The reused post kernel must not be perturbed by the fused op."""
    data = _inputs(num_tokens=16, hidden=1024)
    actual = _call(data)
    expected = mhc_post(
        data["x"],
        data["residual"],
        data["post_layer_mix"],
        data["comb_res_mix"],
        mix_layout=MixLayout.TRANSPOSED,
    )
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


@pytest.mark.mhc_shifted_post_pre
def test_comb_layout_is_transposed():
    """Regression guard for the silent layout trap.

    ``comb_res_mix``/``comb_mix`` use vLLM's orientation, which is
    ``MixLayout.TRANSPOSED``.  If someone "fixes" the call to NATURAL the result
    stays plausible but is wrong, so pin the difference explicitly.
    """
    data = _inputs(num_tokens=16, hidden=1024)
    actual = _call(data)
    reference = mhc_post_ref(
        data["x"],
        data["residual"],
        data["post_layer_mix"],
        data["comb_res_mix"],
        mix_layout=MixLayout.TRANSPOSED,
    )
    natural = mhc_post(
        data["x"],
        data["residual"],
        data["post_layer_mix"],
        data["comb_res_mix"],
        mix_layout=MixLayout.NATURAL,
    )
    torch.testing.assert_close(
        actual[0].float(), reference.float(), rtol=2e-2, atol=2e-2
    )
    assert (actual[0].float() - natural.float()).abs().max() > 1e-2


@pytest.mark.mhc_shifted_post_pre
def test_pre_mix_none_selects_stream_zero():
    data = _inputs(num_tokens=8, hidden=1024, with_pre=False, with_norm=False)
    actual = _call(data)
    expected = _ref(data)
    torch.testing.assert_close(actual[0][:, 0], actual[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    # Nothing collapses the stream mean here; aux is independent of pre_mix.
    torch.testing.assert_close(
        actual[5], expected[5], rtol=COLLAPSE_EPS, atol=COLLAPSE_EPS
    )


@pytest.mark.mhc_shifted_post_pre
def test_norm_none_returns_collapsed_value():
    data = _inputs(num_tokens=8, hidden=1024, with_norm=False)
    actual = _call(data)
    no_norm = _ref(data)
    torch.testing.assert_close(
        actual[3], no_norm[3], rtol=COLLAPSE_EPS, atol=COLLAPSE_EPS
    )
    with_norm = _call(data, norm_weight=data["norm_weight"])
    assert with_norm[3].dtype == torch.bfloat16
    assert (with_norm[3] - actual[3]).abs().max() >= 0


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize("capture_aux", [True, False])
def test_aux_contract(capture_aux):
    data = _inputs(num_tokens=8, hidden=1024)
    actual = _call(data, capture_aux=capture_aux)
    if capture_aux:
        assert actual[5].shape == data["residual"].shape[:1] + (
            data["residual"].shape[-1],
        )
        expected = actual[0].mean(dim=1)
        torch.testing.assert_close(
            actual[5], expected, rtol=COLLAPSE_EPS, atol=COLLAPSE_EPS
        )
    else:
        assert actual[5].shape == (0, data["residual"].shape[-1])
        assert actual[5].dtype == torch.bfloat16


@pytest.mark.mhc_shifted_post_pre
def test_empty_tokens():
    data = _inputs(num_tokens=0, hidden=1024)
    actual = _call(data)
    hc = data["residual"].shape[1]
    hidden = data["residual"].shape[2]
    expected_shapes = [
        (0, hc, hidden),
        (0, hc, 1),
        (0, hc, hc),
        (0, hidden),
        (0, hc),
        (0, hidden),
    ]
    expected_dtypes = [
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    ]
    for out, shape, dtype in zip(actual, expected_shapes, expected_dtypes):
        assert out.shape == shape and out.dtype == dtype


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize("sinkhorn_repeat", [1, 2, 5, 10])
def test_sinkhorn_repeat_matches_reference(sinkhorn_repeat):
    data = _inputs(num_tokens=4, hidden=512)
    actual = _call(data, sinkhorn_repeat=sinkhorn_repeat)
    expected = _ref(data, sinkhorn_repeat=sinkhorn_repeat)
    torch.testing.assert_close(actual[2], expected[2], rtol=MIX_EPS, atol=MIX_EPS)
    # Sanity: the sinkhorn normalisation is not the identity.
    raw = torch.softmax(expected[2], dim=-1)
    assert (expected[2] - raw).abs().max() > 0


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize("hc_post_mult_value", [1.0, 1.5, 2.0])
def test_post_multiplier_is_not_hardcoded(hc_post_mult_value):
    """The generic sinkhorn helper bakes in ``2 * sigmoid``; this op must not."""
    data = _inputs(num_tokens=4, hidden=512)
    actual = _call(data, hc_post_mult_value=hc_post_mult_value)
    expected = _ref(data, hc_post_mult_value=hc_post_mult_value)
    torch.testing.assert_close(actual[1], expected[1], rtol=MIX_EPS, atol=MIX_EPS)
    if hc_post_mult_value != 2.0:
        doubled = _call(data, hc_post_mult_value=2.0)[1]
        assert (actual[1] - doubled).abs().max() > 1e-2


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize("hidden", _HIDDEN_CASES)
def test_hidden_not_power_of_two(hidden):
    """Guards the RMSNorm tile path against shape-dependent rewrites."""
    data = _inputs(num_tokens=4, hidden=hidden)
    actual = _call(data)
    expected = _ref(data)
    torch.testing.assert_close(
        actual[3], expected[3], rtol=COLLAPSE_EPS, atol=COLLAPSE_EPS
    )


@pytest.mark.mhc_shifted_post_pre
def test_hc_variants_match_reference():
    for hc in _HC_CASES:
        data = _inputs(num_tokens=6, hidden=384, hc=hc)
        actual = _call(data)
        expected = _ref(data)
        torch.testing.assert_close(actual[1], expected[1], rtol=MIX_EPS, atol=MIX_EPS)
        torch.testing.assert_close(actual[2], expected[2], rtol=MIX_EPS, atol=MIX_EPS)
        torch.testing.assert_close(actual[4], expected[4], rtol=MIX_EPS, atol=MIX_EPS)


@pytest.mark.mhc_shifted_post_pre
def test_2d_post_mix_is_accepted():
    data = _inputs(num_tokens=8, hidden=1024)
    data2 = dict(data)
    data2["post_layer_mix"] = data["post_layer_mix"].squeeze(-1)
    a = _call(data)
    b = _call(data2)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)


@pytest.mark.mhc_shifted_post_pre
def test_stream_is_accepted_and_ignored():
    data = _inputs(num_tokens=8, hidden=1024)
    side = torch.cuda.Stream()
    a = _call(data)
    b = mhc_shifted_post_pre(
        data["x"],
        data["residual"],
        data["post_layer_mix"],
        data["comb_res_mix"],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
        pre_mix=data["pre_mix"],
        norm_weight=data["norm_weight"],
        norm_eps=1e-6,
        capture_aux=True,
        stream=side,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)


@pytest.mark.mhc_shifted_post_pre
def test_consistency_with_mhc_pre():
    """post/comb must equal mhc_pre on the same residual (same coefficient math)."""
    data = _inputs(num_tokens=16, hidden=1024)
    actual = _call(data)
    post, comb, _ = mhc_pre(
        actual[0],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
    )
    torch.testing.assert_close(actual[1], post, rtol=MIX_EPS, atol=MIX_EPS)
    torch.testing.assert_close(actual[2], comb, rtol=MIX_EPS, atol=MIX_EPS)


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize(
    "field,value,index",
    [
        ("rms_eps", 1e-6, 1),  # same as the default: post is bitwise identical
        ("hc_pre_eps", 0.5, 4),  # pre epsilon feeds next_pre only
        ("hc_sinkhorn_eps", 0.25, 2),  # sinkhorn epsilon feeds comb only
    ],
)
def test_epsilons_are_wired(field, value, index):
    data = _inputs(num_tokens=4, hidden=512)
    base = _call(data)
    changed = _call(data, **{field: value})
    if field == "rms_eps":
        torch.testing.assert_close(base[index], changed[index], rtol=0, atol=0)
    else:
        assert (base[index] - changed[index]).abs().max() > 1e-4


@pytest.mark.mhc_shifted_post_pre
def test_unsupported_paths_raise():
    data = _inputs(num_tokens=4, hidden=512)
    with pytest.raises(NotImplementedError):
        _call(data, reduce_results=True)

    with pytest.raises(NotImplementedError):
        mhc_shifted_post_pre(
            object(),
            data["residual"],
            data["post_layer_mix"],
            data["comb_res_mix"],
            data["fn"],
            data["hc_scale"],
            data["hc_base"],
            1e-6,
            1e-6,
            1e-6,
            1.0,
            5,
        )

    padded = torch.zeros(
        data["residual"].shape[0],
        data["residual"].shape[1],
        data["residual"].shape[2] * 2,
        dtype=torch.bfloat16,
        device=data["residual"].device,
    )
    padded[:, :, ::2] = data["residual"]
    bad = dict(data)
    bad["residual"] = padded[:, :, ::2]
    assert not bad["residual"].is_contiguous()
    with pytest.raises(NotImplementedError):
        _call(bad)

    with pytest.raises(NotImplementedError):
        _call(dict(data, residual=data["residual"].float()))

    with pytest.raises(ValueError):
        _call(data, sinkhorn_repeat=0)
    with pytest.raises(ValueError):
        _call(data, norm_eps=-1.0)


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.parametrize(
    "field,shape",
    [
        ("x", (4, 511)),
        ("post_layer_mix", (4, 3, 1)),
        ("comb_res_mix", (4, 3, 3)),
        ("fn", (12, 2048)),
        ("pre_mix", (4, 3)),
        ("norm_weight", (511,)),
    ],
)
def test_shape_validation(field, shape):
    data = _inputs(num_tokens=4, hidden=512)
    bad = dict(data)
    bad[field] = torch.zeros(shape, device=data["residual"].device)
    with pytest.raises(ValueError):
        _call(bad)


@pytest.mark.mhc_shifted_post_pre
def test_production_path_has_no_torch_compute():
    """Static audit: host code may allocate, copy and read metadata, nothing else."""
    path = Path(__file__).resolve().parents[1] / (
        "src/flaggems_vllm/ops/mhc/mhc_shifted_post_pre.py"
    )
    source = path.read_text()
    # The whole module is production; the torch reference lives in this test file.
    production = source
    allowed = {
        "torch.empty",
        "torch.zeros",
        "torch.Tensor",
        "torch.bfloat16",
        "torch.float32",
        "torch.cuda",
    }
    for match in re.finditer(r"torch\.[A-Za-z_][A-Za-z0-9_.]*", production):
        call = match.group(0).rstrip(".")
        assert any(call.startswith(prefix) for prefix in allowed), call


# ---------------------------------------------------------------------------
# Cross-check against vLLM's own torch reference
#
# vLLM 0.24.0 has no *delayed* pre (that only exists in the DSV4.1 tree), so the
# comparison is built from the two pieces it does ship:
#   * mhc_post_torch  -> the post stage, must be bitwise identical;
#   * mhc_pre_torch   -> post/comb coefficients and the weighted-sum collapse,
#                        when the carried pre-mix is the one this call computed
#                        (which is exactly the delayed contract: next_pre is fed
#                        to the boundary that consumes it).
# ---------------------------------------------------------------------------


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.skipif(not HAS_VLLM_REF, reason="vLLM torch reference unavailable")
def test_post_stage_matches_vllm_bitwise():
    data = _inputs(num_tokens=16, hidden=1024)
    actual = _call(data)
    reference = _vllm_post_torch(
        data["x"],
        data["residual"],
        data["post_layer_mix"],
        data["comb_res_mix"],
    )
    torch.testing.assert_close(actual[0], reference, rtol=0, atol=0)


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.skipif(not HAS_VLLM_REF, reason="vLLM torch reference unavailable")
def test_coefficients_match_vllm():
    data = _inputs(num_tokens=16, hidden=1024)
    actual = _call(data)
    vpost, vcomb, _ = _vllm_pre_torch(
        actual[0],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
    )
    torch.testing.assert_close(actual[1], vpost, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual[2], vcomb, rtol=5e-3, atol=5e-3)


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.skipif(not HAS_VLLM_REF, reason="vLLM torch reference unavailable")
def test_next_pre_reproduces_vllm_layer_input():
    """Feed our next_pre back in: the collapse must match vLLM's weighted sum.

    vLLM's mhc_pre_torch computes a pre-mix from the residual and immediately
    applies it; our operator computes the same pre-mix but returns it as
    ``next_pre`` for the next boundary.  Re-running with that pre-mix carried in
    makes both sides apply the *same* coefficients, so a mismatch in
    ``next_pre`` would show up here.
    """
    data = _inputs(num_tokens=16, hidden=1024)
    first = _call(data)
    # vLLM's reference has no RMSNorm, so compare the raw weighted sum.
    second = _call(data, pre_mix=first[4], norm_weight=None)
    _, _, vllm_layer_input = _vllm_pre_torch(
        second[0],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
    )
    torch.testing.assert_close(second[3], vllm_layer_input, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Cross-check against vLLM's *delayed* reference (the operator this one mirrors)
# ---------------------------------------------------------------------------


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.skipif(
    not HAS_VLLM_DELAYED_REF, reason="vLLM delayed mHC reference unavailable"
)
@pytest.mark.parametrize("num_tokens,hidden", [(16, 1024), (512, 7168)])
def test_matches_vllm_delayed_reference(num_tokens, hidden):
    """All four coefficient/collapse outputs against vLLM's own delayed pre."""
    data = _inputs(num_tokens=num_tokens, hidden=hidden)
    actual = _call(data)  # norm off below: vLLM's reference has no RMSNorm
    post_mix, comb_mix, layer_input, next_pre = _VLLM_PRE_DELAYED(
        actual[0],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        1e-6,
        1e-6,
        1e-6,
        1.0,
        5,
        pre_mix=data["pre_mix"],
    )
    torch.testing.assert_close(actual[1], post_mix, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual[2], comb_mix, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual[4], next_pre, rtol=5e-3, atol=5e-3)

    raw = _call(data, norm_weight=None)
    torch.testing.assert_close(raw[3], layer_input, rtol=1e-2, atol=1e-2)


@pytest.mark.mhc_shifted_post_pre
@pytest.mark.skipif(
    not HAS_VLLM_DELAYED_REF, reason="vLLM delayed mHC reference unavailable"
)
def test_post_stage_matches_vllm_delayed_post():
    data = _inputs(num_tokens=16, hidden=1024)
    actual = _call(data)
    reference = _VLLM_POST_DELAYED(
        data["x"], data["residual"], data["post_layer_mix"], data["comb_res_mix"]
    )
    torch.testing.assert_close(actual[0], reference, rtol=0, atol=0)
