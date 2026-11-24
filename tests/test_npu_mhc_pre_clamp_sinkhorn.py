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

"""Accuracy tests for the Ascend standalone ``npu_mhc_pre_clamp_sinkhorn`` operator.

Style mirrors the other platform-operator tests (e.g.
tests/test_scaled_int8_quant.py): call through the package-level name
``flaggems_vllm.npu_mhc_pre_clamp_sinkhorn`` so the vendor registry resolution is what is
actually exercised, guarded by ``flaggems_vllm.vendor_name``.

``flaggems_vllm.mhc_pre`` stays the generic variant on Ascend (``npu_mhc_pre_clamp_sinkhorn``
registers under its own name), so the existing generic tests are unaffected.

The regular-shape test is parametrized over two reference sources:

* ``cpu`` - the clamped PyTorch reference, always available;
* ``vllm_ascend`` - ``torch.ops._C_ascend.npu_hc_pre_v2``, the AscendC kernel
  the model actually runs on Ascend. That case skips itself when vllm-ascend's
  private-use ops are not registered, and also when the vendor kernel returns
  non-finite values for a shape (measured: ``hidden_size <= 2048``; our own
  output is finite there, so it is a vendor limitation, not ours).

"""

import pytest
import torch

import flaggems_vllm

try:
    import torch_npu

    from flaggems_vllm.runtime.backend._ascend.ops.mhc_pre import CLAMP_MAX, CLAMP_MIN

    HAS_NPU = True
except ImportError:  # pragma: no cover - non-NPU environment
    torch_npu = None  # noqa: F841
    HAS_NPU = False

from .test_mhc_ops import MHC_PRE_CONFIGS, generate_mhc_pre_data, mhc_pre_ref

requires_ascend = pytest.mark.skipif(
    not HAS_NPU or flaggems_vllm.vendor_name != "ascend",
    reason="npu_mhc_pre_clamp_sinkhorn requires torch_npu on an Ascend platform",
)

NORMAL_SHAPES = [
    (512, 4, 1280),
    (1024, 4, 2560),
    (256, 4, 4096),
    (2048, 4, 7168),
]


def _common_stages(residual, fn, hc_scale, hc_base, rms_eps):
    """Returns (pre_logits, post_logits, comb_logits, residual_flat)."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # projection: x @ fn.T
    x_flat = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x_flat, fn.t())

    # RMS
    sqrsum = x_flat.square().sum(dim=-1)
    rms_inv = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    mixes = mixes * rms_inv.unsqueeze(-1)

    # affine
    scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * scale_expanded + hc_base

    pre_logits = mixes[:, :hc_mult]
    post_logits = mixes[:, hc_mult : 2 * hc_mult]
    comb_logits = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    return pre_logits, post_logits, comb_logits, residual_flat


def _sinkhorn(x, clamp_before, hc_sinkhorn_eps, sinkhorn_repeat):
    if clamp_before:
        x = x.clamp(CLAMP_MIN, CLAMP_MAX)
    x = x.softmax(-1) + hc_sinkhorn_eps
    x = x / (x.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        x = x / (x.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        x = x / (x.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    return x


def mhc_pre_ref_clamped(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    pre_logits, post_logits, comb_logits, residual_flat = _common_stages(
        residual, fn, hc_scale, hc_base, rms_eps
    )
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value
    comb_mix = _sinkhorn(
        comb_logits, True, hc_sinkhorn_eps, sinkhorn_repeat
    )  # model-required clamp
    hin = (residual_flat.to(torch.float32) * pre_mix.unsqueeze(-1)).sum(dim=1)
    hin = hin.to(torch.bfloat16)
    return post_mix, comb_mix, hin


# model-spec kwargs, shared by every call below
MODEL_KWARGS = dict(
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=1.0,
    sinkhorn_repeat=20,
)


def _assert_close(exp, ref, case, rtol=1e-2, atol=1e-2):
    torch.testing.assert_close(
        exp.detach().float().cpu(),
        ref.detach().float().cpu(),
        rtol=rtol,
        atol=atol,
        msg=lambda m, c=case: "[{}] {}".format(c, m),
    )


# ---------------------------------------------------------------------------
# vllm-ascend reference (``torch.ops._C_ascend.npu_hc_pre_v2``)
# ---------------------------------------------------------------------------
# vllm-ascend ships the AscendC mHC kernel behind its private-use operator
# namespace; that is what the model runs on Ascend, so it is the reference this
# functional test compares against when it registers. The resolver stays in this
# file on purpose: the tests own their references, and nothing here participates
# in ``flaggems_vllm`` dispatch (which is why the operator under test is called
# through ``flaggems_vllm.npu_mhc_pre_clamp_sinkhorn`` rather than by module path).
#
# Contract mapping (vllm-ascend -> FlagGems) for ``npu_hc_pre_v2``::
#
#     y, post, comb = npu_hc_pre_v2(x, hc_fn, hc_scale, hc_base,
#                                   n, sinkhorn_iters, norm_eps, hc_eps)
#
# ``y`` is our ``hin``, and their ``post`` is ``2 * sigmoid(...)`` while ours is
# ``hc_post_mult_value * sigmoid(...)``, hence the rescale. They take one shared
# ``hc_eps`` (exact only when ``hc_pre_eps == hc_sinkhorn_eps``), do not clamp
# the comb logits, and support ``n == 4`` only on Atlas A2/A3.
_VLLM_ASCEND_PRE = None
_VLLM_ASCEND_PRE_ERROR = None
_VLLM_ASCEND_PRE_RESOLVED = False


def _resolve_vllm_ascend_pre():
    """Resolve once, lazily; never at collection time."""
    global _VLLM_ASCEND_PRE, _VLLM_ASCEND_PRE_ERROR, _VLLM_ASCEND_PRE_RESOLVED
    if _VLLM_ASCEND_PRE_RESOLVED:
        return _VLLM_ASCEND_PRE
    _VLLM_ASCEND_PRE_RESOLVED = True
    try:
        import torch_npu  # noqa: F401 - registers the PrivateUse1 backend

        if not torch.npu.is_available():
            raise RuntimeError("no NPU device available")
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("vllm_ascend.utils.enable_custom_op() failed")
        ops = getattr(torch.ops, "_C_ascend", None)
        if ops is None or not hasattr(ops, "npu_hc_pre_v2"):
            raise ImportError("torch.ops._C_ascend.npu_hc_pre_v2 is not registered")
        _VLLM_ASCEND_PRE = ops.npu_hc_pre_v2
    except Exception as exc:  # noqa: BLE001 - availability is environment-specific
        _VLLM_ASCEND_PRE_ERROR = exc
    return _VLLM_ASCEND_PRE


def _vllm_ascend_pre(residual, fn, hc_scale, hc_base, **kwargs):
    """``npu_hc_pre_v2`` behind the FlagGems ``mhc_pre`` signature and shapes."""
    op = _resolve_vllm_ascend_pre()
    if op is None:
        raise RuntimeError(
            "vllm-ascend npu_hc_pre_v2 unavailable: {!r}".format(_VLLM_ASCEND_PRE_ERROR)
        )
    hc_mult = residual.shape[-2]
    if hc_mult != 4:
        raise NotImplementedError(
            "npu_hc_pre_v2 supports hc_mult == 4, got {}".format(hc_mult)
        )
    if kwargs["hc_pre_eps"] != kwargs["hc_sinkhorn_eps"]:
        raise NotImplementedError(
            "npu_hc_pre_v2 takes one shared hc_eps; "
            "hc_pre_eps={} != hc_sinkhorn_eps={}".format(
                kwargs["hc_pre_eps"], kwargs["hc_sinkhorn_eps"]
            )
        )
    out, post, comb = op(
        residual,
        fn,
        hc_scale,
        hc_base,
        hc_mult,
        kwargs["sinkhorn_repeat"],
        kwargs["rms_eps"],
        kwargs["hc_sinkhorn_eps"],
    )
    post = (post * (kwargs["hc_post_mult_value"] / 2.0)).unsqueeze(-1)
    return post, comb, out


def _reference_outputs(ref_source, data):
    """Reference ``(post, comb, hin)`` from ``ref_source``, or ``None`` + reason.

    ``vllm_ascend`` is resolved on first use (never at collection time); a
    missing registration is a skip, not a failure. The vendor kernel is also
    skipped when it returns non-finite values for the shape (``hidden_size <=
    2048`` on 910B4, while ``npu_mhc_pre_clamp_sinkhorn`` stays finite), so a vendor
    limitation never reds our suite.
    """
    if ref_source == "cpu":
        return (
            mhc_pre_ref_clamped(
                data["residual"],
                data["fn"],
                data["hc_scale"],
                data["hc_base"],
                **MODEL_KWARGS,
            ),
            None,
        )

    if _resolve_vllm_ascend_pre() is None:
        return None, "vllm-ascend npu_hc_pre_v2 unavailable: {!r}".format(
            _VLLM_ASCEND_PRE_ERROR
        )

    ref_post, ref_comb, ref_hin = _vllm_ascend_pre(
        data["residual"],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        **MODEL_KWARGS,
    )
    torch.npu.synchronize()
    if not all(torch.isfinite(t.float()).all() for t in (ref_post, ref_comb, ref_hin)):
        return None, (
            "vllm-ascend npu_hc_pre_v2 returned non-finite values for "
            "hidden_size={} (vendor kernel limitation; npu_mhc_pre_clamp_sinkhorn is finite)".format(
                data["residual"].shape[-1]
            )
        )
    return (ref_post, ref_comb, ref_hin), None


# ``vllm_ascend`` tolerances come from measuring our bf16 Sinkhorn against the
# vendor HF32 kernel: post/comb agree within 1e-2, while ``hin`` carries a bf16
# output-resolution floor (one ULP near |x| = 4 is 3.1e-2), so it needs
# atol = 4e-2. All three pass at 100% of elements on 910B4.
_REFERENCE_TOLERANCES = {
    "cpu": dict(rtol=1e-2, atol=1e-2),
    "vllm_ascend": dict(rtol=1e-2, atol=4e-2),
}


@requires_ascend
@pytest.mark.parametrize("ref_source", ["cpu", "vllm_ascend"])
@pytest.mark.parametrize(
    "num_tokens, hc_mult, hidden_size",
    NORMAL_SHAPES,
    ids=["n{}_hc{}_h{}".format(n, hc, h) for n, hc, h in NORMAL_SHAPES],
)
def test_npu_mhc_pre_clamp_sinkhorn_regular_shapes(
    num_tokens, hc_mult, hidden_size, ref_source
):
    """Regular real shapes must agree with the CPU and vllm-ascend references."""
    case = "n{}_hc{}_h{}".format(num_tokens, hc_mult, hidden_size)
    data = generate_mhc_pre_data(num_tokens, hc_mult, hidden_size)
    reference, skip_reason = _reference_outputs(ref_source, data)
    if reference is None:
        pytest.skip(skip_reason)
    ref_post, ref_comb, ref_hin = reference

    out_post, out_comb, out_hin = flaggems_vllm.npu_mhc_pre_clamp_sinkhorn(
        **{**data, **MODEL_KWARGS}
    )
    torch.npu.synchronize()

    tol = _REFERENCE_TOLERANCES[ref_source]
    _assert_close(out_post.squeeze(-1), ref_post.squeeze(-1), case + "/post", **tol)
    _assert_close(out_comb, ref_comb, case + "/comb", **tol)
    _assert_close(out_hin, ref_hin, case + "/hin", **tol)


@requires_ascend
def test_npu_mhc_pre_clamp_sinkhorn_shape_validation():
    """Invalid shapes must raise ValueError; hc_mult=6 must be accepted
    (the mhc_post hc_mult==4 restriction does not apply here)."""
    data = generate_mhc_pre_data(8, 4, 1280)
    passed = failed = 0

    def expect_value_error(case, **overrides):
        nonlocal passed, failed
        kwargs = dict(data)
        kwargs.update(overrides)
        try:
            flaggems_vllm.npu_mhc_pre_clamp_sinkhorn(**{**kwargs, **MODEL_KWARGS})
        except ValueError:
            print("  [PASS] {}".format(case))
            passed += 1
            return
        except Exception as exc:  # noqa: BLE001
            print(
                "  [FAIL] {}: expected ValueError, got {}: {}".format(
                    case, type(exc).__name__, exc
                )
            )
            failed += 1
            return
        print("  [FAIL] {}: no error raised".format(case))
        failed += 1

    expect_value_error("fn wrong rows", fn=data["fn"][:-1])
    expect_value_error("fn wrong cols", fn=data["fn"][:, :-1])
    expect_value_error(
        "hc_scale shape (4,)",
        hc_scale=torch.full((4,), 0.1, dtype=torch.float32, device=data["fn"].device),
    )
    expect_value_error("hc_base wrong length", hc_base=data["hc_base"][:-1])
    expect_value_error(
        "residual not bf16",
        residual=data["residual"].to(torch.float32),
    )
    expect_value_error(
        "residual rank too low",
        residual=data["residual"][0],
    )

    # hc_mult=6: accepted and numerically sound (no mhc_post reuse)
    device = data["fn"].device
    torch.manual_seed(7)
    hc_mult, hidden = 6, 64
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    residual6 = torch.randn(
        (8, hc_mult, hidden), dtype=torch.float32, device=device
    ).bfloat16()
    fn6 = (
        torch.randn((hc_mult3, hc_mult * hidden), dtype=torch.float32, device=device)
        * 1e-4
    )
    scale6 = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
    base6 = torch.randn((hc_mult3,), dtype=torch.float32, device=device) * 0.1
    try:
        ref6 = mhc_pre_ref_clamped(residual6, fn6, scale6, base6, **MODEL_KWARGS)
        out6 = flaggems_vllm.npu_mhc_pre_clamp_sinkhorn(
            residual6, fn6, scale6, base6, **MODEL_KWARGS
        )
        torch.npu.synchronize()
        _assert_close(out6[1], ref6[1], "hc_mult6/comb")
        print("  [PASS] hc_mult=6 accepted and matches reference")
        passed += 1
    except Exception as exc:  # noqa: BLE001
        print("  [FAIL] hc_mult=6: {}: {}".format(type(exc).__name__, exc))
        failed += 1

    print("  result: {}/{} passed".format(passed, passed + failed))
    assert failed == 0, "{} shape-validation cases failed".format(failed)


@requires_ascend
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_PRE_CONFIGS,
    ids=[f"n{n}_h{h}_hc_mult{hc}" for n, h, hc in MHC_PRE_CONFIGS],
)
def test_npu_mhc_pre_clamp_sinkhorn_generic_parity(n, hidden_size, hc_mult):
    """Ordinary generic-suite cases replayed on the specialized op.

    Same inputs as tests/test_mhc_ops.py (seed 42, sinkhorn_repeat=10),
    same CPU reference (mhc_pre_ref, no clamp - inert on realistic data),
    same 1e-2 tolerance: proves the vendor op satisfies the exact assertion
    the ordinary suite makes for every one of its 24 cases.
    """
    case = "n{}_h{}_hc_mult{}".format(n, hidden_size, hc_mult)
    data = generate_mhc_pre_data(n, hc_mult, hidden_size)
    data_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in data.items()}
    post_ref, comb_ref, li_ref = mhc_pre_ref(**data_cpu)
    out_post, out_comb, out_li = flaggems_vllm.npu_mhc_pre_clamp_sinkhorn(**data)
    torch.npu.synchronize()
    _assert_close(out_post, post_ref, case + "/post")
    _assert_close(out_comb, comb_ref, case + "/comb")
    _assert_close(out_li, li_ref, case + "/li")


@requires_ascend
def test_npu_mhc_pre_clamp_sinkhorn_tle_unavailable_still_clamps(monkeypatch):
    """hc_mult == 4 without TLE must still apply the model-contract clamp.

    The fallback used to delegate to the generic ``mhc_pre``, whose Ascend
    kernels do not clamp, so a stock-Triton build silently dropped the clamp.
    The comb scale is inflated until the logits leave the +-30 band, which makes
    the clamp observable: the result must track the clamped reference and must
    stay clearly away from the unclamped one.
    """
    from flaggems_vllm.runtime.backend._ascend.ops import mhc_pre as mhc_mod

    data = dict(generate_mhc_pre_data(256, 4, 1280))
    device = data["fn"].device
    data["hc_scale"] = torch.tensor(
        [0.0, 0.0, 5000.0], dtype=torch.float32, device=device
    )
    data["hc_base"] = torch.zeros_like(data["hc_base"])
    data.update(MODEL_KWARGS)
    data_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in data.items()}

    _, ref_comb_clamped, _ = mhc_pre_ref_clamped(**data_cpu)
    _, _, comb_logits, _ = _common_stages(
        data_cpu["residual"],
        data_cpu["fn"],
        data_cpu["hc_scale"],
        data_cpu["hc_base"],
        MODEL_KWARGS["rms_eps"],
    )
    ref_comb_raw = _sinkhorn(
        comb_logits,
        False,
        MODEL_KWARGS["hc_sinkhorn_eps"],
        MODEL_KWARGS["sinkhorn_repeat"],
    )
    # The stress data must actually leave the clamp band, otherwise this test
    # would pass even with the clamp missing.
    assert int((comb_logits.abs() > CLAMP_MAX).sum()) > 0

    monkeypatch.setattr(mhc_mod, "_HAS_TLE", False)
    _, out_comb, _ = flaggems_vllm.npu_mhc_pre_clamp_sinkhorn(**data)
    torch.npu.synchronize()
    out_comb = out_comb.float().cpu()

    d_clamped = (out_comb - ref_comb_clamped.float()).abs()
    d_raw = (out_comb - ref_comb_raw.float()).abs()
    # bf16 logits near the clamp boundary cost a few 1e-2; the clamp itself is
    # worth ~1.0 here, so the two checks cannot be confused.
    assert (
        d_clamped.max() < 5e-2
    ), "TLE-less fallback deviates from the clamped reference: {}".format(
        d_clamped.max()
    )
    assert d_raw.max() > 0.2, (
        "TLE-less fallback did not apply clamp(-30, 30): max diff vs unclamped "
        "reference is only {}".format(d_raw.max())
    )
