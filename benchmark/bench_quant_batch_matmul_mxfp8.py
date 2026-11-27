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

"""Benchmark the A5 MX-FP8 quant matmul against npu_quant_matmul.

Default is one shape (GLM W4, the largest share of this operator's calls) for
iteration; ``--all`` sweeps every deployment shape and is meant for final
measurement only.

  PYTHONPATH=src python benchmark/bench_quant_batch_matmul_mxfp8.py
  ... --shape qwen --iters 20
  ... --all --rounds 5

Latency is device time read from the profiler's ``task_time.csv`` rather than
``kernel_details.csv``, because the latter excludes memcpy: any host-side input
rearrangement on this side shows up as MEMCPY_ASYNC, while the vendor baseline
is a single C++ dispatch with no memcpy at all, so kernel_details would
under-count this side only.  Host wall time is not comparable: it is dominated
by the Python launch path (shape checks, format query, tiling choice) that the
vendor's dispatch does not have.

The first profiler measurement of any function is unreliable, so every
measurement discards a settle round, and each case takes the minimum over
several rounds (interference only ever adds time).  Absolute values drift
across processes on a shared machine; only a ratio measured by alternating both
sides within one run is meaningful.  SpeedUp = latency_vendor / latency_flaggems.
"""

import argparse
import csv
import glob
import os
import shutil
import statistics
import tempfile

# TMPDIR must point to /tmp before importing torch_npu, or the profiler cannot
# write task_time.csv and device_time_us silently returns None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import torch
import torch_npu  # noqa: F401

from flaggems_vllm.runtime.backend._ascend.a5.ops.quant_batch_matmul_mxfp8 import (
    ACL_FORMAT_FRACTAL_NZ,
    MX_BLOCK,
    _cube_core_num,
    _pick_tiling,
    quant_batch_matmul_mxfp8,
)

DEV = "npu"

# K and N are the values the operator receives (after the weight transpose in
# process_weights_after_loading).  share is the op's share of model time; this
# operator ranks in the top three of all five models.
SHAPES = {
    "dsv4": dict(M=1, K=1024, N=32768, nz=False, share=12.01, name="DeepSeek V4"),
    "glm_w4": dict(M=1, K=2048, N=16384, nz=True, share=12.41, name="GLM W4"),
    "glm_w8": dict(M=1, K=2048, N=16384, nz=True, share=11.54, name="GLM W8"),
    "m3": dict(M=8, K=2048, N=6144, nz=True, share=9.49, name="MiniMax M3.1"),
    "qwen": dict(M=1, K=2048, N=512, nz=False, share=8.63, name="Qwen3.6"),
}

# DSV4's profiled weight is [N,K] with a [N,K/64,2] scale, while this operator
# takes [K,N].  That case is built as [K,N] below: the same arithmetic over the
# same number of bytes, but not the byte layout of that profiled call, so treat
# its number as a same-scale reference rather than a drop-in replacement.


def to_nz(w):
    w = torch_npu.npu_format_cast(
        w.contiguous(), ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
    )
    assert torch_npu.get_npu_format(w) == ACL_FORMAT_FRACTAL_NZ
    return w


def build_inputs(M, K, N, nz=True, seed=0):
    """Build MX-FP8 inputs in the layouts process_weights_after_loading emits.

    Quantising with the NPU op keeps the e8m0 exponent distribution close to
    real inference; hand-filled uint8 scales give an unrealistic exponent range.

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
    return x, sa.contiguous(), (to_nz(w_nd) if nz else w_nd), sb


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


def _read_task_time(out_dir):
    """Sum device time per entry from task_time.csv (memcpy included)."""
    per = {}
    cands = glob.glob(os.path.join(out_dir, "**", "task_time.csv"), recursive=True)
    if cands:
        with open(cands[0]) as f:
            for row in csv.DictReader(f):
                ktype = row.get("kernel_type") or ""
                name = row.get("kernel_name") or "?"
                if "MEMCPY" in ktype:
                    name = f"[memcpy] {ktype}"
                try:
                    per[name] = per.get(name, 0.0) + float(
                        row.get("task_time(us)") or 0.0
                    )
                except ValueError:
                    continue
        if per:
            return per
    # Fallback for profiler versions without task_time.csv; this one excludes
    # memcpy, so it under-counts any side that rearranges inputs on the host.
    cands = glob.glob(os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True)
    if not cands:
        return per
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
                per[name] = per.get(name, 0.0) + float(dur)
            except ValueError:
                continue
    return per


def device_time_us(fn, warmup=5, iters=20, _settle=True):
    """Return (mean total device us, {entry: mean us}).

    The first profiler measurement of a function is unreliable even when the
    function is warm and the process has already profiled something else, so a
    settle measurement is taken and discarded first.
    """
    if _settle:
        device_time_us(fn, warmup=2, iters=3, _settle=False)

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    out_dir = tempfile.mkdtemp(prefix="qbmm_prof_")
    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
        ):
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
        per = _read_task_time(out_dir)
        if not per:
            return None, {}
        return sum(per.values()) / iters, {k: v / iters for k, v in per.items()}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def measure(x, sa, w, sb, kw, iters, rounds):
    """Alternate both sides within one run; minimum over rounds."""
    expected = vendor(x, sa, w, sb)
    actual = quant_batch_matmul_mxfp8(x, w, sb, sa, **kw)
    peak = expected.float().abs().max().clamp(min=1e-6).item()
    absdiff = (actual.float() - expected.float()).abs().max().item()
    bad = int((actual != expected).sum().item())
    numel = actual.numel()
    del expected, actual

    rb = [
        device_time_us(lambda: vendor(x, sa, w, sb), iters=iters) for _ in range(rounds)
    ]
    ro = [
        device_time_us(
            lambda: quant_batch_matmul_mxfp8(x, w, sb, sa, **kw), iters=iters
        )
        for _ in range(rounds)
    ]
    bl = [r[0] for r in rb]
    ol = [r[0] for r in ro]
    t_base, t_ours = min(bl), min(ol)
    return dict(
        t_base=t_base,
        t_ours=t_ours,
        base_kernels=rb[bl.index(t_base)][1],
        our_kernels=ro[ol.index(t_ours)][1],
        base_med=statistics.median(bl),
        base_rng=max(bl) - min(bl),
        our_med=statistics.median(ol),
        our_rng=max(ol) - min(ol),
        absdiff=absdiff,
        rel=absdiff / peak,
        bad=bad,
        numel=numel,
    )


def tiling_of(cfg, kw):
    bn, sk, bk = _pick_tiling(
        cfg["M"],
        cfg["K"],
        cfg["N"],
        kw.get("num_cores") or _cube_core_num(),
        kw.get("block_n"),
        kw.get("split_k"),
        kw.get("block_k"),
    )
    tiles = ((cfg["M"] + 15) // 16) * (cfg["N"] // bn)
    return bn, sk, bk, tiles * sk


def describe(cfg, layout, bn, sk, bk, programs):
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    print(
        f"\n== {cfg['name']} {layout}: x[{M},{K}] w[{K},{N}] -> out[{M},{N}], "
        f"weight {K * N / 2**20:.0f}MB =="
    )
    print(f"   tiling BLOCK_N={bn} SPLIT_K={sk} BLOCK_K={bk}, {programs} programs")


def report(res, breakdown=True):
    if res["t_base"] is None or res["t_ours"] is None:
        print("   profiler produced no device timing")
        return
    verdict = (
        "bitwise equal"
        if res["bad"] == 0
        else f"max_rel={res['rel']:.3e} max_absdiff={res['absdiff']:.6g} "
        f"({res['bad']}/{res['numel']} elements differ; split-K reorders the "
        f"K accumulation)"
    )
    print(f"   accuracy vs vendor: {verdict}")
    print(
        f"   vendor {res['t_base']:8.2f}us (median {res['base_med']:.2f}, "
        f"range {res['base_rng']:.2f})   "
        f"flaggems {res['t_ours']:8.2f}us (median {res['our_med']:.2f}, "
        f"range {res['our_rng']:.2f})   "
        f"SpeedUp {res['t_base'] / res['t_ours']:.3f}x"
    )
    if breakdown:
        for name, us in sorted(res["our_kernels"].items(), key=lambda kv: -kv[1]):
            print(f"      {us:8.2f}us {us / res['t_ours'] * 100:5.1f}%  {name[:60]}")


def bench_single(cfg, layout, kw, iters, rounds):
    bn, sk, bk, programs = tiling_of(cfg, kw)
    describe(cfg, layout, bn, sk, bk, programs)
    x, sa, w, sb = build_inputs(cfg["M"], cfg["K"], cfg["N"], nz=layout == "NZ")
    res = measure(x, sa, w, sb, kw, iters, rounds)
    report(res)
    del x, sa, w, sb
    torch.npu.empty_cache()
    return res


def bench_all(kw, iters, rounds):
    print("\nMX-FP8 QuantBatchMatmulV3: every deployment shape, profiled layout")
    print("NZ is detected from the ACL format, so callers need no change")
    header = (
        f"{'case':<13} {'share':>6} {'shape':<30} {'w':>5} "
        f"{'vendor':>8} {'flaggems':>9} {'main':>7} {'SpeedUp':>8} {'max_rel':>9}  acc"
    )
    print("\n" + header)
    print("-" * len(header))
    rows = []
    for key in SHAPES:
        cfg = SHAPES[key]
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        x, sa, w, sb = build_inputs(M, K, N, nz=cfg["nz"])
        res = measure(x, sa, w, sb, kw, iters, rounds)
        main = max(res["our_kernels"].values()) if res["our_kernels"] else 0.0
        ratio = res["t_base"] / res["t_ours"]
        print(
            f"{cfg['name']:<13} {cfg['share']:5.2f}% "
            f"{f'x[{M},{K}]xw[{K},{N}]':<30} {'NZ' if cfg['nz'] else 'ND':>5} "
            f"{res['t_base']:8.2f} {res['t_ours']:9.2f} {main:7.2f} "
            f"{ratio:7.3f}x {res['rel']:9.3e}  "
            f"{'bitwise' if res['bad'] == 0 else 'in tol'}"
        )
        rows.append((cfg, res, ratio))
        del x, sa, w, sb
        torch.npu.empty_cache()

    print("-" * len(header))
    share = sum(c["share"] for c, _, _ in rows)
    weighted = sum(c["share"] * r for c, _, r in rows) / share
    print(f"  share-weighted SpeedUp {weighted:.3f}x over {share:.2f}% of model time")
    worst = max(res["rel"] for _, res, _ in rows)
    print(f"  worst max_rel {worst:.3e} (tolerance 2e-2)")
    print(
        "\nThe 'main' column is the largest single kernel; the gap to the total "
        "is the split-K combine epilogue."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="glm_w4", choices=sorted(SHAPES))
    ap.add_argument(
        "--all",
        action="store_true",
        help="sweep every deployment shape (final measurement only)",
    )
    ap.add_argument(
        "--nd", action="store_true", help="force ND weight instead of the shape default"
    )
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="rounds per side, minimum is taken; single-round noise reaches 15%%",
    )
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument(
        "--block-n",
        type=int,
        default=None,
        help="must divide N; default is its largest power-of-two divisor up to 1024",
    )
    ap.add_argument(
        "--block-k",
        type=int,
        default=None,
        help="multiple of 64; default 64 (the interleaved B scale degrades above it)",
    )
    ap.add_argument(
        "--split-k",
        type=int,
        default=None,
        help="default fills the cores without going past them",
    )
    ap.add_argument("--cores", type=int, default=None)
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
        split_k=args.split_k,
        num_cores=args.cores,
    )
    if args.no_multibuffer:
        kw["multibuffer"] = False
    if args.all:
        bench_all(kw, args.iters, args.rounds)
    else:
        cfg = SHAPES[args.shape]
        layout = "ND" if (args.nd or not cfg["nz"]) else "NZ"
        bench_single(cfg, layout, kw, args.iters, args.rounds)


if __name__ == "__main__":
    main()
