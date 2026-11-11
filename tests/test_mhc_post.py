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

"""Accuracy tests for the Ascend ``mhc_post`` operator.

On Ascend the vendor registry replaces the public ``flaggems_vllm.mhc_post``
with the implementation in ``_ascend.ops.mhc_post``.  That implementation
keeps the generic vLLM signature and delegates non-``hc_mult=4`` cases to
the generic kernel, so these tests exercise the public signature first and
then the private AscendC TLE entry point directly.
"""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.mhc.mhc_post import MixLayout
from flaggems_vllm.ops.mhc.mhc_post import mhc_post_ref as generic_mhc_post_ref
from flaggems_vllm.runtime.backend._ascend.ops.mhc_post import _mhc_post_ascend

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except ImportError:  # pragma: no cover - non-NPU environment
    HAS_NPU = False

requires_ascend = pytest.mark.skipif(
    not HAS_NPU or flaggems_vllm.vendor_name != "ascend",
    reason="mhc_post requires torch_npu on an Ascend platform",
)

DEVICE = flaggems_vllm.device
HC_MULT = 4


def _flatten_bsn(x, h_res, h_out, h_post):
    """Return (x, h_res, h_out, h_post, restore_shape) in (T, n, D) layout."""
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        x = x.reshape(T, N, D).contiguous()
        h_res = h_res.reshape(T, N, N).contiguous()
        h_out = h_out.reshape(T, D).contiguous()
        h_post = h_post.reshape(T, N).contiguous()
        return x, h_res, h_out, h_post, (B, S, N, D)
    if x.dim() == 3:
        T, N, D = x.shape
        return (
            x.contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            (T, N, D),
        )
    raise ValueError(f"unsupported x.dim()={x.dim()}, expect 3 or 4")


def mhc_post_ref(x, h_res, h_out, h_post):
    """AscendC/aclnn reference for the private TLE entry point.

    output[t,i,d] = sum_j h_res[t,j,i] * x[t,j,d]
                  + h_post[t,i] * h_out[t,d]
    """
    orig_dtype = x.dtype
    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    x_f = xf.float()
    hout_f = hout.float()
    mix = torch.einsum("tji,tjd->tid", hres.float(), x_f)
    outer = hpost.float().unsqueeze(-1) * hout_f.unsqueeze(1)
    y = (mix + outer).to(orig_dtype)
    return y.reshape(shape)


def _make_generic_data(tokens, hc, hidden, seed=42):
    torch.manual_seed(seed)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn((tokens, hc, hidden), dtype=torch.bfloat16, device=DEVICE)
    post_layer_mix = torch.randn((tokens, hc, 1), dtype=torch.float32, device=DEVICE)
    comb_res_mix = torch.randn((tokens, hc, hc), dtype=torch.float32, device=DEVICE)
    return x, residual, post_layer_mix, comb_res_mix


def _make_ascend_data(tokens, hidden, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    x = torch.randn((tokens, HC_MULT, hidden), dtype=dtype, device=DEVICE)
    h_res = torch.randn((tokens, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE)
    h_out = torch.randn((tokens, hidden), dtype=dtype, device=DEVICE)
    h_post = torch.randn((tokens, HC_MULT), dtype=torch.float32, device=DEVICE)
    return x, h_res, h_out, h_post


def _assert_close(actual, expected, case):
    torch.testing.assert_close(
        actual.detach().float().cpu(),
        expected.detach().float().cpu(),
        rtol=1e-2,
        atol=1e-2,
        msg=lambda m, c=case: "[{}] {}".format(c, m),
    )


def _run_public(data, case, mix_layout, post_is_2d=False):
    x, residual, post_layer_mix, comb_res_mix = data
    if post_is_2d:
        post_layer_mix = post_layer_mix.squeeze(-1)
    out = flaggems_vllm.mhc_post(
        x, residual, post_layer_mix, comb_res_mix, mix_layout=mix_layout
    )
    torch.npu.synchronize()
    # The generic reference predates the (N, hc) form; unsqueeze it back.
    ref_post = (
        post_layer_mix.unsqueeze(-1) if post_layer_mix.dim() == 2 else post_layer_mix
    )
    ref = generic_mhc_post_ref(
        x.cpu(),
        residual.cpu(),
        ref_post.cpu(),
        comb_res_mix.cpu(),
        mix_layout=mix_layout,
    )
    _assert_close(out, ref, case)
    return out


@requires_ascend
@pytest.mark.parametrize("hidden", [128, 1280], ids=["h128", "h1280_tail"])
@pytest.mark.parametrize(
    "mix_layout",
    [MixLayout.TRANSPOSED, MixLayout.NATURAL],
    ids=["transposed", "natural"],
)
def test_mhc_post_generic_signature(mix_layout, hidden):
    """Public signature must match the generic reference for both layouts."""
    data = _make_generic_data(8, HC_MULT, hidden)
    _run_public(data, "h{}".format(hidden), mix_layout)


@requires_ascend
def test_mhc_post_post_mix_2d():
    """``post_layer_mix`` may be (N, hc) instead of (N, hc, 1)."""
    data = _make_generic_data(8, HC_MULT, 256)
    _run_public(data, "post2d", MixLayout.TRANSPOSED, post_is_2d=True)


@requires_ascend
def test_mhc_post_transposed_view():
    """A non-contiguous TRANSPOSED view must match the reference."""
    tokens, hidden = 8, 128
    x, residual, post_layer_mix, comb_res_mix = _make_generic_data(
        tokens, HC_MULT, hidden
    )
    wider = torch.randn(
        (tokens, HC_MULT, HC_MULT * 2),
        dtype=torch.float32,
        device=DEVICE,
    )
    view = wider[:, :, :HC_MULT]
    assert not view.is_contiguous()
    out = flaggems_vllm.mhc_post(
        x,
        residual,
        post_layer_mix,
        view,
        mix_layout=MixLayout.TRANSPOSED,
    )
    torch.npu.synchronize()
    ref = generic_mhc_post_ref(
        x.cpu(),
        residual.cpu(),
        post_layer_mix.cpu(),
        view.cpu(),
        mix_layout=MixLayout.TRANSPOSED,
    )
    _assert_close(out, ref, "transposed_view")


@requires_ascend
def test_mhc_post_hc2_delegates():
    """hc_mult != 4 keeps using the generic implementation."""
    data = _make_generic_data(8, 2, 256)
    _run_public(data, "hc2", MixLayout.TRANSPOSED)


@requires_ascend
@pytest.mark.parametrize(
    "variant",
    ["pipeline", "rows", "concat_reduce"],
    ids=["pipeline", "rows", "concat_reduce"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_mhc_post_ascend_variants(dtype, variant):
    """All three TLE kernels must agree with the AscendC reference."""
    kwargs = {
        "pipeline": {},
        "rows": {"use_pipeline": False},
        "concat_reduce": {"use_concat_reduce": True},
    }[variant]
    data = _make_ascend_data(8, 1024, dtype=dtype)
    out = _mhc_post_ascend(*data, **kwargs)
    torch.npu.synchronize()
    ref = mhc_post_ref(*(t.cpu() for t in data))
    _assert_close(out, ref, variant)


@requires_ascend
def test_mhc_post_ascend_tail_and_4d():
    """The private AscendC entry still supports D tails and BSND inputs."""
    data = _make_ascend_data(8, 1280)
    out = _mhc_post_ascend(*data)
    ref = mhc_post_ref(*(t.cpu() for t in data))
    _assert_close(out, ref, "d_tail")

    torch.manual_seed(7)
    x = torch.randn((2, 3, HC_MULT, 256), dtype=torch.bfloat16, device=DEVICE)
    h_res = torch.randn((2, 3, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE)
    h_out = torch.randn((2, 3, 256), dtype=torch.bfloat16, device=DEVICE)
    h_post = torch.randn((2, 3, HC_MULT), dtype=torch.float32, device=DEVICE)
    out4 = _mhc_post_ascend(x, h_res, h_out, h_post)
    ref4 = mhc_post_ref(x.cpu(), h_res.cpu(), h_out.cpu(), h_post.cpu())
    _assert_close(out4, ref4, "bsnd")


@requires_ascend
def test_mhc_post_validation():
    """Invalid layout/dtype/rank/shape must raise the generic errors."""
    data = _make_generic_data(2, HC_MULT, 128)
    x, residual, post_layer_mix, comb_res_mix = data

    with pytest.raises(TypeError):
        flaggems_vllm.mhc_post(
            x, residual, post_layer_mix, comb_res_mix, mix_layout="transposed"
        )

    with pytest.raises(NotImplementedError):
        flaggems_vllm.mhc_post(x, residual.float(), post_layer_mix, comb_res_mix)

    with pytest.raises(ValueError):
        flaggems_vllm.mhc_post(x, residual, post_layer_mix, comb_res_mix[:, :3, :3])

    with pytest.raises(ValueError):
        flaggems_vllm.mhc_post(residual, residual, post_layer_mix, comb_res_mix)
