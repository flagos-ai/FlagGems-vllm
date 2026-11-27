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

"""Accuracy tests for the A5 MX-FP8 grouped matmul.

The default hardware path must be bit-exact against the vendor operator, for
both ND and NZ weights, and both layouts must agree element for element.  A
pure-torch fp32 reference bounds the vendor baseline's own BF16 output error.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src pytest -q \\
      tests/test_grouped_matmul_mxfp8.py
"""

import os

# TMPDIR must point to /tmp before importing torch_npu.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from flaggems_vllm.runtime.backend._ascend.a5.ops.grouped_matmul_mxfp8 import (  # noqa: E402
    ACL_FORMAT_FRACTAL_NZ,
    MX_BLOCK,
    grouped_matmul_mxfp8,
)

DEV = "npu"

# Deployment shapes, K/N as received by the operator (post weight transpose).
SHAPES = {
    "m3": dict(M=64, K=3072, N=6144, E=16, name="MiniMax M3.1"),
    "glm_w8": dict(M=64, K=2048, N=6144, E=32, name="GLM W8A8C8"),
    "qwen": dict(M=16, K=512, N=2048, E=128, name="Qwen3.6"),
}

TUNED = dict(block_n=1024, block_k=64)


def build_inputs(M, K, N, E, seed=0):
    """Build MX-FP8 inputs through npu_dynamic_mx_quant.

    Quantising with the NPU op keeps the scale exponent distribution close to
    real inference; random uint8 scales can hide overflow and saturation bugs.
    """
    torch.manual_seed(seed)
    x_bf = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w_bf = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.1

    x, sa = torch_npu.npu_dynamic_mx_quant(x_bf, dst_type=torch.float8_e4m3fn)

    # npu_dynamic_mx_quant blocks along the last dim, but the weight must be
    # blocked along K, so transpose, quantise, and transpose back.
    # torch.stack has no fp8 kernel, hence the pre-allocated per-expert fill.
    w = torch.empty(E, K, N, device=DEV, dtype=torch.float8_e4m3fn)
    sb_log = torch.empty(E, N, K // MX_BLOCK, device=DEV, dtype=torch.uint8)
    for e in range(E):
        wq, sq = torch_npu.npu_dynamic_mx_quant(
            w_bf[e].t().contiguous(), dst_type=torch.float8_e4m3fn
        )
        w[e] = wq.t()
        sb_log[e] = sq.reshape(N, K // MX_BLOCK)
    # [E,N,K/32] -> physical [E,K/64,N,2]: logical b maps to (b//2, n, b%2).
    sb = sb_log.permute(0, 2, 1).contiguous()
    sb = sb.reshape(E, K // 64, 2, N).permute(0, 1, 3, 2).contiguous()

    base, rem = M // E, M % E
    sizes = [base + (1 if i < rem else 0) for i in range(E)]
    gl = torch.tensor(sizes, device=DEV, dtype=torch.int64)

    del w_bf, x_bf, sb_log
    torch.npu.empty_cache()
    return x, w, sb, sa.contiguous(), gl


def vendor(x, w, sb, sa, gl, group_list_type=1):
    """Vendor baseline matching vllm-ascend apply_gmm1 (w8a8_mxfp8.py:585)."""
    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[w],
        scale=[sb],
        per_token_scale=[sa],
        bias=None,
        split_item=2,
        group_type=0,
        group_list=gl,
        group_list_type=group_list_type,
        output_dtype=torch.bfloat16,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]


def grouped_matmul_mxfp8_ref(
    x, weight, weight_scale, per_token_scale, group_list, group_list_type=1
):
    """Pure-torch fp32 reference; test-only, never imported by the operator.

    Scale indexing follows the measured layouts: per_token_scale is flat and
    weight_scale's trailing 2 is truly interleaved with N.
    """
    M, K = x.shape
    E, _, N = weight.shape
    n_kblk = K // MX_BLOCK

    sa = per_token_scale.reshape(M, n_kblk).to(torch.int32)
    sa_f = torch.where(
        sa == 0,
        torch.zeros_like(sa, dtype=torch.float32),
        torch.pow(2.0, sa.float() - 127.0),
    )

    sb = weight_scale.reshape(E, n_kblk // 2, N, 2).to(torch.int32)
    sb = sb.permute(0, 1, 3, 2).reshape(E, n_kblk, N)  # (b//2, b%2) -> b
    sb_f = torch.where(
        sb == 0,
        torch.zeros_like(sb, dtype=torch.float32),
        torch.pow(2.0, sb.float() - 127.0),
    )

    xf = x.to(torch.float32) * sa_f.repeat_interleave(MX_BLOCK, dim=1)
    wf = weight.to(torch.float32) * sb_f.repeat_interleave(MX_BLOCK, dim=1)

    gl = group_list.tolist()
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    start = 0
    for e in range(E):
        end = gl[e] if group_list_type == 0 else start + gl[e]
        if end > start:
            out[start:end] = xf[start:end] @ wf[e]
        start = end
    return out.to(torch.bfloat16)


def to_nz(w):
    return torch_npu.npu_format_cast(
        w.contiguous(), ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
    )


def mismatch(actual, expected):
    return int((actual != expected).sum().item())


def max_absdiff(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("layout", ["ND", "NZ"])
def test_bitwise_equal_to_vendor(shape, layout):
    """Bit-exactness also proves both paths reach the same MX-MAC instruction."""
    cfg = SHAPES[shape]
    M, K, N, E = cfg["M"], cfg["K"], cfg["N"], cfg["E"]
    x, w, sb, sa, gl = build_inputs(M, K, N, E)
    if layout == "NZ":
        w = to_nz(w)
        assert torch_npu.get_npu_format(w) == ACL_FORMAT_FRACTAL_NZ

    expected = vendor(x, w, sb, sa, gl)
    actual = grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=1)

    assert actual.shape == (M, N)
    assert actual.dtype == torch.bfloat16
    bad = mismatch(actual, expected)
    assert bad == 0, (
        f"{cfg['name']} {layout}: {bad}/{actual.numel()} elements differ, "
        f"max_absdiff={max_absdiff(actual, expected):g}"
    )


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_nd_and_nz_agree(shape):
    """Both layouts read the same bytes, so the outputs must be identical."""
    cfg = SHAPES[shape]
    x, w, sb, sa, gl = build_inputs(cfg["M"], cfg["K"], cfg["N"], cfg["E"])
    nd = grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=1)
    nz = grouped_matmul_mxfp8(x, to_nz(w), sb, sa, gl, group_list_type=1)
    bad = mismatch(nd, nz)
    assert bad == 0, f"{cfg['name']}: ND and NZ differ in {bad}/{nd.numel()} elements"


@pytest.mark.grouped_matmul_mxfp8
def test_matches_torch_fp32_reference_within_vendor_error():
    """Deviation from fp32 must not exceed the vendor baseline's own error.

    The operator is bit-exact with the vendor, so this bounds the shared BF16
    output and K-accumulation error rather than an error of this kernel.
    """
    cfg = SHAPES["qwen"]
    x, w, sb, sa, gl = build_inputs(cfg["M"], cfg["K"], cfg["N"], cfg["E"])
    reference = grouped_matmul_mxfp8_ref(x, w, sb, sa, gl, group_list_type=1)
    peak = reference.float().abs().max().clamp(min=1e-6)

    ours = grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=1)
    base = vendor(x, w, sb, sa, gl)
    rel_ours = max_absdiff(ours, reference) / peak
    rel_base = max_absdiff(base, reference) / peak
    assert rel_ours <= max(
        rel_base * 1.05, 2e-2
    ), f"vs fp32 reference: ours {rel_ours:.3e} exceeds vendor {rel_base:.3e}"


def run_edge_case(M, K, N, sizes, group_list_type, **kwargs):
    E = len(sizes)
    x, w, sb, sa, _ = build_inputs(M, K, N, E)
    assert sum(sizes) == M
    if group_list_type == 0:
        ends, end = [], 0
        for size in sizes:
            end += size
            ends.append(end)
        values = ends
    else:
        values = sizes
    gl = torch.tensor(values, device=DEV, dtype=torch.int64)

    expected = vendor(x, w, sb, sa, gl, group_list_type=group_list_type)
    actual = grouped_matmul_mxfp8(
        x, w, sb, sa, gl, group_list_type=group_list_type, **kwargs
    )
    bad = mismatch(actual, expected)
    assert bad == 0, (
        f"M={M} K={K} N={N} sizes={sizes} glt={group_list_type} {kwargs}: "
        f"{bad}/{actual.numel()} elements differ, "
        f"max_absdiff={max_absdiff(actual, expected):g}"
    )


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize(
    "sizes",
    [
        [0, 0, 16, 1, 3, 4, 0, 8, 2, 5, 0, 7, 6, 1, 11, 0],  # empty and large groups
        [64] + [0] * 15,  # everything in the first group
        [0] * 15 + [64],  # everything in the last group
    ],
)
def test_empty_and_uneven_groups(sizes):
    run_edge_case(64, 3072, 6144, sizes, 1, **TUNED)


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize(
    "sizes",
    [
        [4] * 16,
        [0, 7, 1, 12, 0, 3, 9, 2, 5, 0, 6, 4, 8, 1, 6, 0],
    ],
)
def test_cumulative_group_list(sizes):
    run_edge_case(64, 3072, 6144, sizes, 0, **TUNED)


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize(
    "M,sizes",
    [
        (17, [1, 2, 3, 4, 7]),
        (33, [11, 0, 9, 13]),
        (7, [7]),
        (1, [1]),
        (100, [33, 0, 7, 60]),
    ],
)
def test_unaligned_m_and_group_starts(M, sizes):
    run_edge_case(M, 3072, 6144, sizes, 1, **TUNED)


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize(
    "K,N,block_n,block_k",
    [
        (1024, 2048, 1024, 64),
        (512, 1024, 1024, 64),
        (2048, 4096, 1024, 128),
        (64, 512, 512, 64),
        (3072, 6144, 1024, 192),
    ],
)
def test_other_k_n_alignments(K, N, block_n, block_k):
    run_edge_case(32, K, N, [8] * 4, 1, block_n=block_n, block_k=block_k)


@pytest.mark.grouped_matmul_mxfp8
@pytest.mark.parametrize("M,sizes", [(64, [4] * 16), (17, [1, 2, 3, 4, 7])])
def test_default_tiling(M, sizes):
    """No tuning parameters: exercises the production default configuration."""
    run_edge_case(M, 3072, 6144, sizes, 1)


@pytest.mark.grouped_matmul_mxfp8
def test_fp16_output_dtype():
    cfg = SHAPES["qwen"]
    x, w, sb, sa, gl = build_inputs(cfg["M"], cfg["K"], cfg["N"], cfg["E"])
    out = grouped_matmul_mxfp8(
        x, w, sb, sa, gl, group_list_type=1, output_dtype=torch.float16
    )
    assert out.dtype == torch.float16
    # BF16 and FP16 round the same fp32 accumulator differently; bound the gap
    # by the coarser of the two output formats.
    reference = grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=1)
    peak = reference.float().abs().max().clamp(min=1e-6)
    assert max_absdiff(out, reference) / peak <= 8e-3


@pytest.mark.grouped_matmul_mxfp8
def test_unsupported_and_invalid_inputs():
    x, w, sb, sa, gl = build_inputs(16, 512, 2048, 128)
    with pytest.raises(NotImplementedError, match="output_dtype"):
        grouped_matmul_mxfp8(x, w, sb, sa, gl, output_dtype=torch.float32)
    with pytest.raises(ValueError, match="group_list_type"):
        grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=2)
    with pytest.raises(TypeError, match="fp8_e4m3fn"):
        grouped_matmul_mxfp8(x.to(torch.bfloat16), w, sb, sa, gl)
    with pytest.raises(ValueError, match="must divide N"):
        grouped_matmul_mxfp8(x, w, sb, sa, gl, block_n=768)
    with pytest.raises(ValueError, match="K mismatch"):
        grouped_matmul_mxfp8(x[:, :256], w, sb, sa, gl)
    with pytest.raises(ValueError, match="group_list length"):
        grouped_matmul_mxfp8(x, w, sb, sa, gl[:4])
    with pytest.raises(ValueError, match="contiguous"):
        grouped_matmul_mxfp8(x.t().contiguous().t(), w, sb, sa, gl)
