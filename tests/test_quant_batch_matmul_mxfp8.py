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

"""Accuracy tests for the A5 MX-FP8 quant matmul (QuantBatchMatmulV3).

The hardware MX-MAC path is bit-exact with the vendor operator whenever
split-K is off, which the DSV4 shape asserts strictly.  Shapes that enable
split-K reorder the K accumulation, so they are only required to stay inside
the BF16 output tolerance; in practice they have measured bit-exact as well.
ND and NZ weights must agree element for element, and a pure-torch fp32
reference bounds the vendor baseline's own BF16 error.

Usage:
  PYTHONPATH=src pytest -q tests/test_quant_batch_matmul_mxfp8.py
"""

import os

# TMPDIR must point to /tmp before importing torch_npu.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from flaggems_vllm.runtime.backend._ascend.a5.ops.quant_batch_matmul_mxfp8 import (  # noqa: E402
    ACL_FORMAT_FRACTAL_NZ,
    MX_BLOCK,
    quant_batch_matmul_mxfp8,
)

DEV = "npu"

# Deployment shapes, K/N as the operator receives them (after the weight
# transpose in process_weights_after_loading).  GLM-W8 shares GLM-W4's shape
# and is therefore not repeated here.
SHAPES = {
    "dsv4": dict(M=1, K=1024, N=32768, nz=False, name="DeepSeek V4"),
    "glm": dict(M=1, K=2048, N=16384, nz=True, name="GLM W4/W8"),
    "m3": dict(M=8, K=2048, N=6144, nz=True, name="MiniMax M3.1"),
    "qwen": dict(M=1, K=2048, N=512, nz=False, name="Qwen3.6"),
}

TOL = 2e-2  # BF16 output plus K-length fp32 accumulation


def to_nz(w):
    w = torch_npu.npu_format_cast(
        w.contiguous(), ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
    )
    assert torch_npu.get_npu_format(w) == ACL_FORMAT_FRACTAL_NZ
    return w


def build_inputs(M, K, N, nz=True, seed=0):
    """Build MX-FP8 inputs in the layouts process_weights_after_loading emits.

    Quantising with the NPU op keeps the e8m0 exponent distribution close to
    real inference; hand-filled uint8 scales give an unrealistic exponent range
    and can hide overflow bugs.

      weight        [N,K] checkpoint -> transpose -> [K,N] as received
      weight_scale  [N,K/32] -> [N,K/64,2] -> transpose(0,1) -> [K/64,N,2]
    """
    torch.manual_seed(seed)
    x_bf = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w_bf = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.1

    x, sa = torch_npu.npu_dynamic_mx_quant(x_bf, dst_type=torch.float8_e4m3fn)
    wq, sq = torch_npu.npu_dynamic_mx_quant(w_bf, dst_type=torch.float8_e4m3fn)

    w_nd = wq.t().contiguous()  # [K,N]
    sb = sq.reshape(N, K // 64, 2).transpose(0, 1).contiguous()  # [K/64,N,2]

    del x_bf, w_bf, wq, sq
    torch.npu.empty_cache()
    return x, sa.contiguous(), (to_nz(w_nd) if nz else w_nd), w_nd, sb


def vendor(x, sa, w, sb, bias=None, output_dtype=torch.bfloat16):
    """Vendor baseline in the exact form vllm-ascend calls it
    (AscendW8A8MXFP8LinearMethod.apply, w8a8_mxfp8.py:143)."""
    return torch_npu.npu_quant_matmul(
        x,
        w,
        sb,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=sa,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=bias,
        output_dtype=output_dtype,
        group_sizes=[1, 1, MX_BLOCK],
    )


def quant_batch_matmul_mxfp8_ref(
    x, weight, weight_scale, pertoken_scale, bias=None, output_dtype=torch.bfloat16
):
    """Pure-torch fp32 reference; test-only, never imported by the operator.

    ND weight only, since NZ has no torch-side view.  Scale indexing follows
    the measured layouts: pertoken_scale is simply flattened while
    weight_scale's trailing 2 is truly interleaved with N.
    """
    M, K = x.shape
    _, N = weight.shape
    nb = K // MX_BLOCK

    def e8m0(v):
        vi = v.to(torch.int32)
        return torch.where(
            vi == 0,
            torch.zeros_like(vi, dtype=torch.float32),
            torch.exp2((vi - 127).to(torch.float32)),
        )

    sa = pertoken_scale.reshape(-1)[: M * nb].reshape(M, nb)
    # [K/64,N,2] -> logical [K/32,N]: b maps to (b//2, n, b%2).
    sb = weight_scale.reshape(-1).reshape(nb // 2, N, 2).permute(0, 2, 1).reshape(nb, N)

    xd = x.float() * e8m0(sa).repeat_interleave(MX_BLOCK, dim=-1)
    wd = weight.float() * e8m0(sb).repeat_interleave(MX_BLOCK, dim=0)
    out = xd @ wd
    if bias is not None:
        out = out + bias.reshape(-1).float()
    return out.to(output_dtype)


def mismatch(actual, expected):
    return int((actual != expected).sum().item())


def max_absdiff(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def max_rel(actual, expected):
    peak = expected.float().abs().max().clamp(min=1e-6)
    return max_absdiff(actual, expected) / peak.item()


@pytest.mark.quant_batch_matmul_mxfp8
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_matches_vendor_on_deployment_shapes(shape):
    """Deployment shapes in their profiled weight layout."""
    cfg = SHAPES[shape]
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    x, sa, w, _, sb = build_inputs(M, K, N, nz=cfg["nz"])

    expected = vendor(x, sa, w, sb)
    actual = quant_batch_matmul_mxfp8(x, w, sb, sa)

    assert actual.shape == (M, N)
    assert actual.dtype == torch.bfloat16
    rel = max_rel(actual, expected)
    assert rel <= TOL, (
        f"{cfg['name']} {'NZ' if cfg['nz'] else 'ND'}: max_rel={rel:.3e}, "
        f"max_absdiff={max_absdiff(actual, expected):g}"
    )


@pytest.mark.quant_batch_matmul_mxfp8
def test_bitwise_equal_to_vendor_without_split_k():
    """Bit-exactness proves this path reaches the same hardware MX-MAC.

    The DSV4 shape fills every core from its n-tiles alone, so _pick_tiling
    leaves split-K off and the K accumulation order matches the vendor's.
    """
    cfg = SHAPES["dsv4"]
    x, sa, w, _, sb = build_inputs(cfg["M"], cfg["K"], cfg["N"], nz=cfg["nz"])
    expected = vendor(x, sa, w, sb)
    actual = quant_batch_matmul_mxfp8(x, w, sb, sa, split_k=1)
    bad = mismatch(actual, expected)
    assert bad == 0, (
        f"{bad}/{actual.numel()} elements differ, "
        f"max_absdiff={max_absdiff(actual, expected):g}"
    )


@pytest.mark.quant_batch_matmul_mxfp8
@pytest.mark.parametrize("shape", ["m3", "qwen"])
def test_nd_and_nz_agree(shape):
    """Both layouts read the same bytes, so the outputs must be identical."""
    cfg = SHAPES[shape]
    x, sa, _, w_nd, sb = build_inputs(cfg["M"], cfg["K"], cfg["N"], nz=False)
    nd = quant_batch_matmul_mxfp8(x, w_nd, sb, sa)
    nz = quant_batch_matmul_mxfp8(x, to_nz(w_nd), sb, sa)
    bad = mismatch(nd, nz)
    assert bad == 0, f"{cfg['name']}: ND and NZ differ in {bad}/{nd.numel()} elements"


@pytest.mark.quant_batch_matmul_mxfp8
def test_matches_torch_fp32_reference_within_vendor_error():
    """Deviation from fp32 must not exceed the vendor baseline's own error."""
    cfg = SHAPES["qwen"]
    x, sa, w, w_nd, sb = build_inputs(cfg["M"], cfg["K"], cfg["N"], nz=cfg["nz"])
    reference = quant_batch_matmul_mxfp8_ref(x, w_nd, sb, sa)

    rel_ours = max_rel(quant_batch_matmul_mxfp8(x, w, sb, sa), reference)
    rel_base = max_rel(vendor(x, sa, w, sb), reference)
    assert rel_ours <= max(
        rel_base * 1.05, TOL
    ), f"vs fp32 reference: ours {rel_ours:.3e} exceeds vendor {rel_base:.3e}"


def run_edge_case(M, K, N, nz, bias=False, output_dtype=torch.bfloat16, **kwargs):
    x, sa, w, w_nd, sb = build_inputs(M, K, N, nz=nz)
    b = torch.randn(N, device=x.device, dtype=torch.float32) if bias else None
    expected = vendor(x, sa, w, sb, bias=b, output_dtype=output_dtype)
    actual = quant_batch_matmul_mxfp8(
        x, w, sb, sa, bias=b, output_dtype=output_dtype, **kwargs
    )
    assert actual.shape == (M, N)
    assert actual.dtype == output_dtype
    rel = max_rel(actual, expected)
    assert rel <= TOL, (
        f"M={M} K={K} N={N} {'NZ' if nz else 'ND'} {kwargs}: max_rel={rel:.3e}, "
        f"max_absdiff={max_absdiff(actual, expected):g}"
    )


@pytest.mark.quant_batch_matmul_mxfp8
@pytest.mark.parametrize(
    "M",
    [
        16,  # exactly one BLOCK_M, no padding and no mask
        17,  # two m-tiles, so split-K is disabled and the store mask is dynamic
        128,  # prefill-like
        4,  # 1 < M < BLOCK_M: M becomes a constexpr and the mask is static
    ],
)
def test_m_boundaries(M):
    run_edge_case(M, 2048, 2048, nz=True)


@pytest.mark.quant_batch_matmul_mxfp8
@pytest.mark.parametrize("N", [32, 512])
def test_narrow_n(N):
    run_edge_case(1, 2048, N, nz=False)


@pytest.mark.quant_batch_matmul_mxfp8
def test_bias_and_fp16_output():
    run_edge_case(1, 2048, 2048, nz=True, bias=True)
    run_edge_case(1, 2048, 2048, nz=True, output_dtype=torch.float16)


@pytest.mark.quant_batch_matmul_mxfp8
def test_shortest_k_against_torch_reference():
    """K=64 has no vendor cross-check: CANN itself rejects it with ERR01001
    'Input x2 tensor and scale tensor's transpose are not same'."""
    x, sa, _, w_nd, sb = build_inputs(1, 64, 2048, nz=False)
    reference = quant_batch_matmul_mxfp8_ref(x, w_nd, sb, sa)
    rel = max_rel(quant_batch_matmul_mxfp8(x, w_nd, sb, sa), reference)
    assert rel <= TOL, f"K=64: max_rel={rel:.3e}"


@pytest.mark.quant_batch_matmul_mxfp8
def test_tuning_overrides():
    """Explicit tiling, and multibuffer forced off, must not change the result."""
    x, sa, w, _, sb = build_inputs(8, 2048, 6144, nz=True)
    expected = vendor(x, sa, w, sb)
    for kwargs in (
        dict(block_n=1024, split_k=4, block_k=64),
        dict(block_n=512, split_k=1),
        dict(multibuffer=False),
    ):
        rel = max_rel(quant_batch_matmul_mxfp8(x, w, sb, sa, **kwargs), expected)
        assert rel <= TOL, f"{kwargs}: max_rel={rel:.3e}"


@pytest.mark.quant_batch_matmul_mxfp8
def test_unsupported_and_invalid_inputs():
    x, sa, _, w_nd, sb = build_inputs(17, 2048, 2048, nz=False)
    with pytest.raises(NotImplementedError, match="output_dtype"):
        quant_batch_matmul_mxfp8(x, w_nd, sb, sa, output_dtype=torch.float32)
    with pytest.raises(TypeError, match="fp8_e4m3fn"):
        quant_batch_matmul_mxfp8(x.to(torch.bfloat16), w_nd, sb, sa)
    with pytest.raises(ValueError, match="K mismatch"):
        quant_batch_matmul_mxfp8(x[:, :1024], w_nd, sb, sa)
    with pytest.raises(ValueError, match="must divide N"):
        quant_batch_matmul_mxfp8(x, w_nd, sb, sa, block_n=768)
    with pytest.raises(ValueError, match="must be 2-D"):
        quant_batch_matmul_mxfp8(x.unsqueeze(0), w_nd, sb, sa)
    with pytest.raises(ValueError, match="contiguous"):
        quant_batch_matmul_mxfp8(x.t().contiguous().t(), w_nd, sb, sa)
    with pytest.raises(ValueError, match="bias needs"):
        quant_batch_matmul_mxfp8(x, w_nd, sb, sa, bias=torch.zeros(8, device=x.device))
    # Two m-tiles cannot use split-K: the row mask would have to be dynamic,
    # which aborts the compiler on the partials store.
    with pytest.raises(ValueError, match="split_k"):
        quant_batch_matmul_mxfp8(x, w_nd, sb, sa, split_k=2)
