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

"""Shared mHC pre contract tests with explicit backend-specific coverage.

Ascend's standalone npu_mhc_pre uses its clamped model contract; the MetaX
public mhc_pre path is checked against the independent generic CPU reference.
Backend requirements are attached to individual tests, never the whole module.
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
    reason="npu_mhc_pre requires torch_npu on an Ascend platform",
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


@requires_ascend
@pytest.mark.parametrize(
    "num_tokens, hc_mult, hidden_size",
    NORMAL_SHAPES,
    ids=["n{}_hc{}_h{}".format(n, hc, h) for n, hc, h in NORMAL_SHAPES],
)
def test_npu_mhc_pre_regular_shapes(num_tokens, hc_mult, hidden_size):
    """Regular real shapes: must agree with the clamped reference (4/4)."""
    case = "n{}_hc{}_h{}".format(num_tokens, hc_mult, hidden_size)
    data = generate_mhc_pre_data(num_tokens, hc_mult, hidden_size)
    ref_post, ref_comb, ref_hin = mhc_pre_ref_clamped(
        data["residual"],
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        **MODEL_KWARGS,
    )
    out_post, out_comb, out_hin = flaggems_vllm.npu_mhc_pre(**{**data, **MODEL_KWARGS})
    torch.npu.synchronize()

    _assert_close(out_post.squeeze(-1), ref_post, case + "/post")
    _assert_close(out_comb, ref_comb, case + "/comb")
    _assert_close(out_hin, ref_hin, case + "/hin")


@requires_ascend
def test_npu_mhc_pre_shape_validation():
    """Invalid shapes must raise ValueError; hc_mult=6 must be accepted
    (the mhc_post hc_mult==4 restriction does not apply here)."""
    data = generate_mhc_pre_data(8, 4, 1280)
    passed = failed = 0

    def expect_value_error(case, **overrides):
        nonlocal passed, failed
        kwargs = dict(data)
        kwargs.update(overrides)
        try:
            flaggems_vllm.npu_mhc_pre(**{**kwargs, **MODEL_KWARGS})
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
        out6 = flaggems_vllm.npu_mhc_pre(residual6, fn6, scale6, base6, **MODEL_KWARGS)
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
def test_npu_mhc_pre_generic_parity(n, hidden_size, hc_mult):
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
    out_post, out_comb, out_li = flaggems_vllm.npu_mhc_pre(**data)
    torch.npu.synchronize()
    _assert_close(out_post, post_ref, case + "/post")
    _assert_close(out_comb, comb_ref, case + "/comb")
    _assert_close(out_li, li_ref, case + "/li")


# MetaX graph/layout checks share this file with the Ascend accuracy cases.
requires_metax = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax", reason="MetaX public-path contract"
)


@requires_metax
@pytest.mark.mhc_pre
@pytest.mark.parametrize(
    "rows,hidden",
    [(1, 4096), (32, 4096), (64, 4096), (65, 4096), (128, 4096), (129, 4096), (3, 128)],
)
def test_public_mhc_pre_reference_and_new_input_replay(rows, hidden):
    from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre_ref

    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(rows, 4, hidden)
    result = []

    def call():
        result[:] = flaggems_vllm.mhc_pre(**data)

    def check():
        cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        expected = mhc_pre_ref(**cpu)
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-2, atol=1e-2)

    call()
    check()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        data["residual"].fill_(value)
        for output in result:
            output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@requires_metax
@pytest.mark.mhc_pre
def test_mhc_pre_empty_batch_preserves_shape():
    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(0, 4, 4096)
    post, comb, layer = flaggems_vllm.mhc_pre(**data)
    assert post.shape == (0, 4, 1) and comb.shape == (0, 4, 4)
    assert layer.shape == (0, 4096) and layer.dtype == torch.bfloat16


@requires_metax
@pytest.mark.mhc_pre
def test_mhc_pre_mixed_device_rejected_before_mutation():
    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(1, 4, 4096)
    data["fn"] = data["fn"].cpu()
    before = {
        key: value.clone()
        for key, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    with pytest.raises(ValueError, match="share an accelerator"):
        flaggems_vllm.mhc_pre(**data)
    for key, value in before.items():
        assert torch.equal(data[key], value)


@requires_metax
@pytest.mark.mhc_pre
def test_mhc_pre_nondefault_splits_preserves_common_warning():
    from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre_ref

    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(1, 4, 4096)
    data["n_splits"] = 2
    with pytest.warns(RuntimeWarning, match="n_splits"):
        actual = flaggems_vllm.mhc_pre(**data)
    reference = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
        if key != "n_splits"
    }
    for result, expected in zip(actual, mhc_pre_ref(**reference)):
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.fixture(autouse=True)
def restore_mhc_test_rng_state():
    # The upstream data helper deliberately seeds globally; contain that effect
    # within each test rather than leaking it into another operator's tests.
    if flaggems_vllm.vendor_name != "metax":
        yield
        return
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    yield
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(gpu)
