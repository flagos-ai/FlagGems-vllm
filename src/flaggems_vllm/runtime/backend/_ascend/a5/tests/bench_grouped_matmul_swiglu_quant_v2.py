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

"""Single-shape debug script for MXFP8 GroupedMatmulSwigluQuantV2.

Default shape (GLM W8A8C8):

  x  [64, 2048]        fp8_e4m3fn
  w  [32, 2048, 6144]  fp8_e4m3fn   gate|up concatenated, output width 3072
  y  [64, 3072]        fp8_e4m3fn
  ys [64, 48, 2]       e8m0

Usage:
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src python \\
      src/flaggems_vllm/runtime/backend/_ascend/a5/tests/bench_grouped_matmul_swiglu_quant_v2.py
  ... --shape qwen
  ... --block-n 256
"""

import argparse
import csv
import glob
import os
import shutil
import tempfile

# TMPDIR must point to /tmp before importing torch_npu; otherwise the profiler
# cannot write kernel_details.csv and device_time_us silently returns None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import torch
import torch_npu  # noqa: F401

from flaggems_vllm.runtime.backend._ascend.a5.ops.grouped_matmul_swiglu_quant_v2 import (
    MX_BLOCK,
    grouped_matmul_swiglu_quant_v2,
)

DEV = "npu"

# K and N are post-transpose values as received by the operator (gate|up
# concatenated along N; output width is N/2).
SHAPES = {
    "glm_w8": dict(M=64, K=2048, N=6144, E=32, name="GLM W8A8C8 (7.37%)"),
    "qwen": dict(M=16, K=512, N=2048, E=128, name="Qwen3.6 (8.22%)"),
}


def build_inputs(M, K, N, E, seed=0):
    """Build MXFP8 inputs via npu_dynamic_mx_quant.

    Using the NPU quantisation op ensures that the scale exponent distribution
    matches real inference; random uint8 scales can hide overflow/saturation
    issues.
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


def as_u8(t):
    """View a float8_e8m0fnu scale tensor as uint8.

    aclnnInplaceCopy (error 561103) rejects .int() on e8m0 tensors; a raw
    byte view is safe.
    """
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def device_time_us(fn, warmup=5, iters=20):
    """Return mean device kernel time in us via the NPU profiler.

    Returns (total_mean_us, {kernel_name: mean_us}).
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    out_dir = tempfile.mkdtemp(prefix="gmmsq_prof_")
    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
        ):
            for _ in range(iters):
                fn()
            torch.npu.synchronize()

        cands = glob.glob(
            os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True
        )
        if not cands:
            return None, {}
        per_kernel = {}
        with open(cands[0]) as f:
            for row in csv.DictReader(f):
                name = row.get("Name") or row.get("Op Name") or "?"
                dur = (
                    row.get("Duration(us)")
                    or row.get("Duration (us)")
                    or row.get("Duration")
                )
                if dur is None:
                    continue
                try:
                    per_kernel[name] = per_kernel.get(name, 0.0) + float(dur)
                except ValueError:
                    continue
        total = sum(per_kernel.values()) / iters
        return total, {k: v / iters for k, v in per_kernel.items()}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def compare(tag, y, ys, ry, rys, valid, nh):
    """Accuracy check: bitwise equality also confirms the same hardware path."""
    yf = y[:valid].float()
    rf = ry[:valid].float()
    dy = (yf - rf).abs().max().item()
    nbad = (yf != rf).sum().item()
    a = as_u8(ys).reshape(ys.shape[0], -1)[:valid, : nh // MX_BLOCK].int()
    b = as_u8(rys).reshape(rys.shape[0], -1)[:valid, : nh // MX_BLOCK].int()
    ds = (a - b).abs().max().item()
    sbad = (a != b).sum().item()
    verdict = "bitwise equal" if (nbad == 0 and sbad == 0) else "differ"
    print(
        f"  {tag:20s} y max_absdiff {dy:<10g} y diff {nbad}/{yf.numel()}  "
        f"scale max_absdiff {ds} diff {sbad}/{a.numel()}   {verdict}"
    )
    return nbad == 0 and sbad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="glm_w8", choices=sorted(SHAPES))
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument(
        "--block-n",
        type=int,
        default=None,
        help="default: largest value fitting two accumulators in L0C",
    )
    ap.add_argument(
        "--block-k",
        type=int,
        default=64,
        help="must be a multiple of 64 (B-side scale interleaved layout constraint)",
    )
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument(
        "--scale-alg", type=int, default=0, help="0=OCP (vendor default), 1=cuBLAS"
    )
    ap.add_argument(
        "--nd",
        action="store_true",
        help="force ND layout (default: NZ, the deployment layout)",
    )
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    cfg = SHAPES[args.shape]
    M, K, N, E = cfg["M"], cfg["K"], cfg["N"], cfg["E"]
    NH = N // 2
    print(f"== {cfg['name']}: x[{M},{K}] w[{E},{K},{N}] -> y[{M},{NH}] ==")
    print(
        f"   weight {E * K * N / 2**20:.0f}MB   " f"layout {'ND' if args.nd else 'NZ'}"
    )

    x, w, sb, sa, gl = build_inputs(M, K, N, E)
    if not args.nd:
        w = torch_npu.npu_format_cast(w, 29, customize_dtype=torch.float8_e4m3fn)
        assert torch_npu.get_npu_format(w) == 29

    kw = dict(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_cores=args.cores,
        scale_alg=args.scale_alg,
    )

    # ---- correctness ----
    vy, vys = vendor(x, w, sb, sa, gl)
    valid = int(gl[-1])
    print(
        f"\n[accuracy] valid rows [0,{valid}) (rows outside group_list are not initialised)"
    )
    try:
        ty, tys = grouped_matmul_swiglu_quant_v2(
            x, w, sb, sa, gl, group_list_type=0, **kw
        )
    except Exception as err:
        print(f"  FAILED: {type(err).__name__}: {err}")
        raise
    ok = compare("triton vs vendor", ty, tys, vy, vys, valid, NH)

    # ---- performance ----
    tv, kv = device_time_us(lambda: vendor(x, w, sb, sa, gl), iters=args.iters)
    tt, kt = device_time_us(
        lambda: grouped_matmul_swiglu_quant_v2(
            x, w, sb, sa, gl, group_list_type=0, **kw
        ),
        iters=args.iters,
    )
    if tv is None or tt is None:
        print(f"\n[perf] profiler did not produce kernel_details.csv (v={tv}, t={tt})")
        return
    print(f"\n[perf] vendor {tv:.2f}us   triton {tt:.2f}us   ratio {tv / tt:.3f}x")
    for tag, ks in (("vendor", kv), ("triton", kt)):
        for name, us in sorted(ks.items(), key=lambda kv2: -kv2[1])[:4]:
            print(f"   {tag:6s} {us:8.2f}us  {name}")
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
