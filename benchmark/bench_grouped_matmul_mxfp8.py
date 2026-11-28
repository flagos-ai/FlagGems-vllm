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

"""Benchmark the A5 MX-FP8 grouped matmul against npu_grouped_matmul.

Default is one shape (MiniMax M3.1, the largest GroupedMatmul share) for
iteration; ``--all`` sweeps every deployment shape in both layouts and is
meant for final measurement only.

  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src python \\
      benchmark/bench_grouped_matmul_mxfp8.py
  ... --shape qwen --nd
  ... --all --iters 20

Latency is device kernel time summed from the profiler's
kernel_details.csv, including the operator's own pre-processing kernels.
Host wall time is not used.  SpeedUp = latency_vendor / latency_flaggems.
"""

import argparse
import csv
import glob
import os
import shutil
import tempfile

# TMPDIR must point to /tmp before importing torch_npu, or the profiler cannot
# write kernel_details.csv and device_time_us silently returns None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import torch
import torch_npu  # noqa: F401

from flaggems_vllm.runtime.backend._ascend.a5.ops.grouped_matmul_mxfp8 import (
    ACL_FORMAT_FRACTAL_NZ,
    MX_BLOCK,
    grouped_matmul_mxfp8,
)

DEV = "npu"

# K and N are the values the operator receives (after the weight transpose in
# process_weights_after_loading).  share is the op's share of model time.
SHAPES = {
    "m3": dict(M=64, K=3072, N=6144, E=16, share=21.59, name="MiniMax M3.1"),
    "glm_w8": dict(M=64, K=2048, N=6144, E=32, share=3.37, name="GLM W8A8C8"),
    "qwen": dict(M=16, K=512, N=2048, E=128, share=4.19, name="Qwen3.6"),
}


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


def vendor(x, w, sb, sa, gl):
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
        group_list_type=1,
        output_dtype=torch.bfloat16,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
    )[0]


def device_time_us(fn, warmup=5, iters=20):
    """Return mean device kernel time in us via the NPU profiler.

    Returns (total_mean_us, {kernel_name: mean_us}).
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    out_dir = tempfile.mkdtemp(prefix="gmm_prof_")
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


def to_nz(w):
    w = torch_npu.npu_format_cast(
        w.contiguous(), ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
    )
    assert torch_npu.get_npu_format(w) == ACL_FORMAT_FRACTAL_NZ
    return w


def measure(x, w, sb, sa, gl, kw, iters):
    """Return (vendor_us, ours_us, ours_kernels, mismatching_elements)."""
    expected = vendor(x, w, sb, sa, gl)
    actual = grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=1, **kw)
    bad = int((actual != expected).sum().item())
    del expected, actual

    t_vendor, _ = device_time_us(lambda: vendor(x, w, sb, sa, gl), iters=iters)
    t_ours, kernels = device_time_us(
        lambda: grouped_matmul_mxfp8(x, w, sb, sa, gl, group_list_type=1, **kw),
        iters=iters,
    )
    return t_vendor, t_ours, kernels, bad


def bench_single(name, cfg, layout, kw, iters, breakdown=True):
    M, K, N, E = cfg["M"], cfg["K"], cfg["N"], cfg["E"]
    x, w, sb, sa, gl = build_inputs(M, K, N, E)
    if layout == "NZ":
        w = to_nz(w)
    t_vendor, t_ours, kernels, bad = measure(x, w, sb, sa, gl, kw, iters)
    verdict = "bitwise equal" if bad == 0 else f"DIFFER in {bad} elements"
    print(f"\n== {cfg['name']} {layout}: x[{M},{K}] w[{E},{K},{N}] -> out[{M},{N}] ==")
    print(f"   accuracy vs vendor: {verdict}")
    if t_vendor is None or t_ours is None:
        print("   profiler produced no kernel_details.csv")
    else:
        print(
            f"   vendor {t_vendor:8.2f}us   flaggems {t_ours:8.2f}us   "
            f"SpeedUp {t_vendor / t_ours:.3f}x"
        )
        if breakdown:
            for kname, us in sorted(kernels.items(), key=lambda kv: -kv[1]):
                print(f"      {us:8.2f}us {us / t_ours * 100:5.1f}%  {kname}")
    del x, w, sb, sa, gl
    torch.npu.empty_cache()
    return t_vendor, t_ours, bad


def bench_all(kw, iters):
    print("\nMX-FP8 GroupedMatmul: every deployment shape, ND and NZ weights")
    print("NZ is detected from the ACL format, so callers need no change")
    header = (
        f"{'case':<14} {'share':>6} {'shape':<26} "
        f"{'vendor-ND':>10} {'ours-ND':>9} {'ND':>7} "
        f"{'vendor-NZ':>10} {'ours-NZ':>9} {'NZ':>7}  acc"
    )
    print("\n" + header)
    print("-" * len(header))
    rows = []
    for key in ("m3", "glm_w8", "qwen"):
        cfg = SHAPES[key]
        M, K, N, E = cfg["M"], cfg["K"], cfg["N"], cfg["E"]
        x, w, sb, sa, gl = build_inputs(M, K, N, E)
        wnz = to_nz(w)
        result = {}
        outs = {}
        for layout, weight in (("ND", w), ("NZ", wnz)):
            outs[layout] = grouped_matmul_mxfp8(
                x, weight, sb, sa, gl, group_list_type=1, **kw
            )
            t_vendor, t_ours, _, bad = measure(x, weight, sb, sa, gl, kw, iters)
            result[layout] = (t_vendor, t_ours, bad)
        cross = int((outs["ND"] != outs["NZ"]).sum().item())
        (cnd, tnd, bnd), (cnz, tnz, bnz) = result["ND"], result["NZ"]
        ok = bnd == 0 and bnz == 0 and cross == 0
        print(
            f"{cfg['name']:<14} {cfg['share']:5.2f}% "
            f"{f'[{M},{K}]x[{E},{K},{N}]':<26} "
            f"{cnd:10.2f} {tnd:9.2f} {cnd / tnd:6.3f}x "
            f"{cnz:10.2f} {tnz:9.2f} {cnz / tnz:6.3f}x  "
            f"{'bitwise' if ok else 'DIFFER'}"
        )
        rows.append((cfg, cnd, tnd, cnz, tnz, ok))
        del x, w, wnz, sb, sa, gl, outs
        torch.npu.empty_cache()

    print("-" * len(header))
    for layout, ci, ti in (("ND", 1, 2), ("NZ", 3, 4)):
        share = sum(r[0]["share"] for r in rows)
        weighted = sum(r[0]["share"] * r[ci] / r[ti] for r in rows) / share
        print(
            f"  {layout}: share-weighted SpeedUp {weighted:.3f}x over "
            f"{share:.2f}% of model time"
        )
    if not all(r[5] for r in rows):
        print("  WARNING: at least one case is not bitwise equal to the vendor")
    print("\nFP4 (FLOAT4_E2M1) shapes are out of scope: DSV4 8.77%, GLM-W4 2.45%.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="m3", choices=sorted(SHAPES))
    ap.add_argument(
        "--all",
        action="store_true",
        help="sweep every shape in both layouts (final measurement only)",
    )
    ap.add_argument(
        "--nd",
        action="store_true",
        help="force ND layout (default: NZ, the deployment layout)",
    )
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument(
        "--block-n",
        type=int,
        default=None,
        help="must divide N; default 1024 when possible (scale loads are mask-free)",
    )
    ap.add_argument(
        "--block-k",
        type=int,
        default=None,
        help="multiple of 64; default 64 (the interleaved B scale degrades above it)",
    )
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument(
        "--variant",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="2=hardware MX-MAC (default); 0/1 are the software-scale fallbacks",
    )
    ap.add_argument(
        "--no-multibuffer",
        action="store_true",
        help="disable B double-buffering (default: enabled when the UB budget fits)",
    )
    args = ap.parse_args()

    kw = dict(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_cores=args.cores,
        variant=args.variant,
    )
    if args.no_multibuffer:
        kw["multibuffer"] = False
    if args.all:
        bench_all(kw, args.iters)
    else:
        bench_single(
            args.shape,
            SHAPES[args.shape],
            "ND" if args.nd else "NZ",
            kw,
            args.iters,
        )


if __name__ == "__main__":
    main()
