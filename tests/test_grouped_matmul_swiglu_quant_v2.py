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

"""Accuracy test for MXFP8 GroupedMatmulSwigluQuantV2.

Compares the Triton kernel and the vendor operator against a pure-torch
fp32 reference implementation.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src pytest -q \\
      tests/test_grouped_matmul_swiglu_quant_v2.py
"""

import os

# TMPDIR must point to /tmp before importing torch_npu.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from flaggems_vllm.runtime.backend._ascend.a5.ops.grouped_matmul_swiglu_quant_v2 import (  # noqa: E402
    E4M3_EMAX,
    MX_BLOCK,
    grouped_matmul_swiglu_quant_v2,
)

DEV = "npu"

SHAPES = {
    "glm_w8": dict(M=64, K=2048, N=6144, E=32, name="GLM W8A8C8"),
    "qwen": dict(M=16, K=512, N=2048, E=128, name="Qwen3.6"),
}


def grouped_matmul_swiglu_quant_v2_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    scale_alg: int = 0,
):
    """Pure-torch fp32 reference implementation for accuracy comparison.

    Scale index layout follows the same conventions as the kernel:
    x_scale is flat; weight_scale has a truly-interleaved trailing-2 dimension.
    """
    M, K = x.shape
    E, _, N = weight.shape
    NH = N // 2
    nkb = K // MX_BLOCK

    sa = x_scale.reshape(M, nkb)
    sa = (sa.view(torch.uint8) if sa.dtype != torch.uint8 else sa).to(torch.int32)
    sa_f = torch.where(
        sa == 0,
        torch.zeros_like(sa, dtype=torch.float32),
        torch.pow(2.0, sa.float() - 127.0),
    )

    sb = (
        weight_scale.view(torch.uint8)
        if weight_scale.dtype != torch.uint8
        else weight_scale
    )
    sb = sb.reshape(E, nkb // 2, N, 2).to(torch.int32)
    sb = sb.permute(0, 1, 3, 2).reshape(E, nkb, N)  # (b//2, b%2) -> b
    sb_f = torch.where(
        sb == 0,
        torch.zeros_like(sb, dtype=torch.float32),
        torch.pow(2.0, sb.float() - 127.0),
    )

    xf = x.to(torch.float32) * sa_f.repeat_interleave(MX_BLOCK, dim=1)
    wf = weight.to(torch.float32) * sb_f.repeat_interleave(MX_BLOCK, dim=1)

    gl = group_list.tolist()
    c = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    start = 0
    for e in range(E):
        end = gl[e] if group_list_type == 0 else start + gl[e]
        if end > start:
            c[start:end] = xf[start:end] @ wf[e]
        start = end
    valid = start

    act, gate = c[:, :NH], c[:, NH:]
    s = torch.nn.functional.silu(act) * gate

    # MX quant: round to bf16 first, then block-quantise in groups of 32
    sbf = s.to(torch.bfloat16).to(torch.float32)
    blk = sbf.reshape(M, NH // MX_BLOCK, MX_BLOCK)
    amax = blk.abs().amax(dim=-1)
    bits = amax.view(torch.int32)
    eb = ((bits >> 23) & 0xFF) - E4M3_EMAX
    if scale_alg == 1:
        eb = eb + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    ebs = torch.clamp(eb, min=0)
    ebs = torch.where(amax == 0, torch.zeros_like(ebs), ebs)
    rc = torch.pow(2.0, -(ebs.float() - 127.0)).unsqueeze(-1)
    q = torch.clamp(blk * rc, -448.0, 448.0)
    yq = q.reshape(M, NH).to(torch.float8_e4m3fn)
    return yq, ebs.to(torch.uint8), valid


def build_inputs(M, K, N, E, seed=0):
    """Build MXFP8 inputs via npu_dynamic_mx_quant.

    Using the NPU quantisation op ensures that the scale exponent
    distribution matches real inference; random uint8 scales can hide
    overflow/saturation issues.
    """
    torch.manual_seed(seed)
    x_bf = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w_bf = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.1

    x, sa = torch_npu.npu_dynamic_mx_quant(x_bf, dst_type=torch.float8_e4m3fn)

    # npu_dynamic_mx_quant scales along the last dim; weight must be scaled
    # along K (second-to-last), so transpose, quantise, transpose back.
    # torch.stack does not support fp8 (aclnnStack lacks DT_FLOAT8_E4M3FN),
    # so pre-allocate and fill expert by expert.
    w = torch.empty(E, K, N, device=DEV, dtype=torch.float8_e4m3fn)
    sb_log = torch.empty(E, N, K // MX_BLOCK, device=DEV, dtype=torch.uint8)
    for e in range(E):
        wt = w_bf[e].t().contiguous()
        wq, sq = torch_npu.npu_dynamic_mx_quant(wt, dst_type=torch.float8_e4m3fn)
        w[e] = wq.t()
        sb_log[e] = sq.reshape(N, K // MX_BLOCK)
    # [E, N, K/32] -> physical [E, K/64, N, 2]: logical index b maps to
    # (b//2, n, b%2).
    sb = sb_log.permute(0, 2, 1).contiguous()
    sb = sb.reshape(E, K // 64, 2, N).permute(0, 1, 3, 2).contiguous()

    # group_list: distribute M rows evenly across E experts; pass as cumsum
    # (group_list_type=0, matching vllm usage).
    base, rem = M // E, M % E
    sizes = [base + (1 if i < rem else 0) for i in range(E)]
    gl_count = torch.tensor(sizes, device=DEV, dtype=torch.int64)
    gl = torch.cumsum(gl_count, dim=0)

    del w_bf, x_bf, sb_log
    torch.npu.empty_cache()
    return x, w, sb, sa.contiguous(), gl


def vendor(x, w, sb, sa, gl):
    """Vendor baseline matching vllm-ascend apply_gmm1_act_quant (w8a8_mxfp8.py:562)."""
    return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=x,
        weight=[w],
        group_list=gl,
        weight_scale=[sb],
        x_scale=sa,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        x_dtype=None,
        weight_dtype=None,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
    )


def compare(tag, y, ys, ry, rys, valid, nh):
    """Accuracy check: bitwise equality also confirms the same hardware path."""
    yf = y[:valid].float()
    rf = ry[:valid].float()
    dy = (yf - rf).abs().max().item()
    nbad = (yf != rf).sum().item()
    a = ys.reshape(ys.shape[0], -1)[:valid, : nh // MX_BLOCK].int()
    b = rys.reshape(rys.shape[0], -1)[:valid, : nh // MX_BLOCK].int()
    ds = (a - b).abs().max().item()
    sbad = (a != b).sum().item()
    verdict = "bitwise equal" if (nbad == 0 and sbad == 0) else "differ"
    print(
        f"  {tag:20s} y max_absdiff {dy:<10g} y diff {nbad}/{yf.numel()}  "
        f"scale max_absdiff {ds} diff {sbad}/{a.numel()}   {verdict}"
    )
    return nbad == 0 and sbad == 0


@pytest.mark.parametrize("shape", ["glm_w8", "qwen"])
@pytest.mark.parametrize("scale_alg", [0, 1])
def test_grouped_matmul_swiglu_quant_v2(shape, scale_alg, capsys):
    cfg = SHAPES[shape]
    M, K, N, E = cfg["M"], cfg["K"], cfg["N"], cfg["E"]
    NH = N // 2
    print(f"== {cfg['name']}: x[{M},{K}] w[{E},{K},{N}] -> y[{M},{NH}] ==")

    x, w, sb, sa, gl = build_inputs(M, K, N, E)
    valid = int(gl[-1])
    print(
        f"[accuracy] valid rows [0,{valid}) (rows outside group_list are not initialised)"
    )

    ty, tys = grouped_matmul_swiglu_quant_v2(
        x, w, sb, sa, gl, group_list_type=0, scale_alg=scale_alg
    )
    ry, rys, rvalid = grouped_matmul_swiglu_quant_v2_ref(
        x, w, sb, sa, gl, group_list_type=0, scale_alg=scale_alg
    )
    n = min(valid, rvalid)
    ok = compare("triton vs torch", ty, tys, ry, rys, n, NH)

    vy, vys = vendor(x, w, sb, sa, gl)
    ok &= compare("vendor vs torch", vy, vys, ry, rys, n, NH)
    ok &= compare("triton vs vendor", ty, tys, vy, vys, valid, NH)

    assert ok, capsys.readouterr().out
