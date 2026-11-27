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

"""Benchmark the A5 HcPre op against vllm-ascend's ``npu_hc_pre_v2``.

    source <your CANN env>
    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src python benchmark/bench_hc_pre.py
    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src python benchmark/bench_hc_pre.py --all

Default is the single DSV4 decode shape (T=1), for iterating; ``--all`` adds 12
more shapes and is meant for a final measurement only.

Why this is a script and not ``benchmark/test_hc_pre.py``: the baseline is a
vllm-ascend *vendor* op, and reaching it requires
``bootstrap_custom_op_env(include_vendor_lib=True)`` to run before ``torch_npu``
is imported.  A pytest module cannot guarantee that, because any earlier
collected module may import ``torch_npu`` first.  The other A5 ops in this
directory (``bench_swiglu_group_quant.py``,
``bench_grouped_matmul_swiglu_quant_v2.py``) are standalone for the same kind of
reason.

Measurement protocol, which this op's numbers depend on:

* Device kernel time only, read from the profiler's ``task_time.csv`` -- *not*
  ``kernel_details.csv``, which excludes memcpy.  Our wrapper may issue a
  memcpy for non-contiguous input and the vendor op never does, so reading only
  the latter would under-count our own side.
* ``task_time.csv`` has one row per kernel *call*, not per program.
* Cross-process numbers are not comparable.  On a shared machine two
  identically shaped runs can differ by 27%, so both sides are measured
  alternately inside one process and only same-run ratios are reported.
* Several rounds per shape, taking the minimum (contention only adds time), and
  the first profiler measurement of each function is discarded: a cold first
  measurement has been seen at 89.61us where the steady state was 47.8-48.1us.
"""

import argparse
import csv
import glob
import os
import shutil
import tempfile

# Must run before torch_npu is imported, or the vendor op is unreachable:
# the aclnnHcPre symbol lives in vllm_ascend's custom_transformer vendor dir
# and is found through ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH.
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)

import torch  # noqa: E402
import torch_npu  # noqa: E402
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401  registers _C_ascend

from flaggems_vllm.runtime.backend._ascend.a5.ops.hc_pre import (  # noqa: E402
    HC_POST_MULT,
    hc_pre,
)

ATOL = RTOL = 4e-2  # the vendor op's own error level; see hc_pre.py
# (name, T, n, D, sinkhorn iterations, share of the DSV4 trace, note)
DSV4 = ("DSV4 decode", 1, 4, 4096, 20, 4.48, "the only variant in the profile")
CASES = [
    DSV4,
    ("T=8", 8, 4, 4096, 20, 0.0, "below the mix crossover (vector mix)"),
    ("T=16", 16, 4, 4096, 20, 0.0, "above the mix crossover (cube mix)"),
    ("T=64", 64, 4, 4096, 20, 0.0, "chunked prefill"),
    ("T=128", 128, 4, 4096, 20, 0.0, "cube mix + fused epilogue"),
    ("T=256", 256, 4, 4096, 20, 0.0, "cube mix + comb/y epilogue"),
    ("T=1000", 1000, 4, 4096, 20, 0.0, "not a power of two, row mask"),
    ("T=2048", 2048, 4, 4096, 20, 0.0, "whole prefill chunk"),
    ("D=16", 1, 4, 16, 20, 0.0, "smallest 16-aligned hidden size"),
    ("D=2560", 1, 4, 2560, 20, 0.0, "the aclnn doc's example"),
    ("D=7168", 1, 4, 7168, 20, 0.0, "wider hidden size"),
    ("sinkhorn=1", 1, 4, 4096, 1, 0.0, "loop boundary"),
    ("sinkhorn=5", 1, 4, 4096, 5, 0.0, "loop boundary"),
]


def build_inputs(T, n, D, seed=0, device="npu"):
    """Build HcPre inputs, returning the device tensors and the CPU originals.

    ``hc_fn`` is scaled to 0.02 to match real weights; at unit scale the mixes
    saturate every sigmoid, pre/post degenerate into constants and numerical
    problems stop being visible.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    j = n * n + 2 * n
    x = torch.randn(T, n, D, generator=g, dtype=torch.bfloat16)
    fn = torch.randn(j, n * D, generator=g, dtype=torch.float32) * 0.02
    scale = torch.rand(3, generator=g, dtype=torch.float32) + 0.5
    base = torch.randn(j, generator=g, dtype=torch.float32) * 0.1
    cpu = (x, fn, scale, base)
    return tuple(t.to(device) for t in cpu), cpu


def vendor_baseline(x, fn, scale, base, n, iters, norm_eps=1e-6, hc_eps=1e-6):
    """The baseline as ``DeepseekV4DecoderLayer.hc_pre`` calls it."""
    return torch.ops._C_ascend.npu_hc_pre_v2(
        x, fn, scale, base, n, iters, norm_eps, hc_eps
    )


def reference(x, fn, scale, base, n, iters, norm_eps=1e-6, hc_eps=1e-6):
    """fp32 torch reference, from vllm's ``mhc_pre_torch``.  Benchmark only."""
    D = x.shape[-1]
    outer = x.shape[:-2]
    xf = x.reshape(-1, n * D).float()
    mixes = xf @ fn.t().float()
    mixes = mixes * torch.rsqrt(xf.square().sum(-1, keepdim=True) / (n * D) + norm_eps)
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + hc_eps
    post = (
        torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n]) * HC_POST_MULT
    )
    c = mixes[:, 2 * n :].view(-1, n, n) * scale[2] + base[2 * n :].view(1, n, n)
    c = torch.softmax(c, dim=-1) + hc_eps
    c = c / (c.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + hc_eps)
        c = c / (c.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * x.reshape(-1, n, D).float()).sum(1).to(x.dtype)
    return y.view(*outer, D), post.view(*outer, n), c.view(*outer, n, n)


def device_time_us(fn, warmup=5, iters=20, settle=True):
    """Device time in us from the profiler: ``(total mean, kernel name -> mean)``.

    ``task_time.csv`` includes memcpy (``MEMCPY_ASYNC`` in ``kernel_type``) and
    ``kernel_details.csv`` does not; if our wrapper issues a memcpy to reorder
    inputs, reading the latter would under-count our own side.

    ``settle``: the first profiler measurement of any function is untrustworthy,
    so one throwaway round runs first.
    """
    if settle:
        device_time_us(fn, warmup=2, iters=3, settle=False)

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    out_dir = tempfile.mkdtemp(prefix="hc_pre_prof_")
    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
        ):
            for _ in range(iters):
                fn()
            torch.npu.synchronize()

        per = {}
        found = glob.glob(os.path.join(out_dir, "**", "task_time.csv"), recursive=True)
        if found:
            with open(found[0]) as handle:
                for row in csv.DictReader(handle):
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
        if not per:
            return None, {}
        return sum(per.values()) / iters, {k: v / iters for k, v in per.items()}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def worst_abs(got, want):
    got = got.float().cpu()
    want = want.float().cpu().reshape(got.shape)
    delta = (got - want).abs()
    ok = bool((delta <= ATOL + RTOL * want.abs()).all())
    return ok, float(delta.max()) if delta.numel() else 0.0


def fmt(value):
    return f"{value:8.3f}" if value else "       -"


def run_case(case, args):
    name, T, n, D, sinkhorn, share, note = case
    dev, cpu = build_inputs(T, n, D)

    ry, rpost, rcomb = reference(*cpu, n, sinkhorn)
    vendor_out = vendor_baseline(*dev, n, sinkhorn)
    ours_out = hc_pre(*dev, n, sinkhorn)

    ok = True
    worst_ours = worst_vendor = 0.0
    for mine, theirs, want in zip(ours_out, vendor_out, (ry, rpost, rcomb)):
        passed, delta = worst_abs(mine, want)
        ok &= passed
        worst_ours = max(worst_ours, delta)
        _, delta = worst_abs(theirs, want)
        worst_vendor = max(worst_vendor, delta)

    vendor_us = ours_us = None
    breakdown = {}
    if not args.no_perf:
        for _ in range(args.rounds):
            total, _ = device_time_us(
                lambda: vendor_baseline(*dev, n, sinkhorn), iters=args.iters
            )
            if total and (vendor_us is None or total < vendor_us):
                vendor_us = total
            total, per = device_time_us(
                lambda: hc_pre(*dev, n, sinkhorn), iters=args.iters
            )
            if total and (ours_us is None or total < ours_us):
                ours_us, breakdown = total, per

    ratio = f"{vendor_us / ours_us:.3f}x" if (vendor_us and ours_us) else "-"
    print(
        f"  {'OK' if ok else 'XX'} {name:<12} vendor {fmt(vendor_us)} "
        f"ours {fmt(ours_us)} {ratio:>8}  max_abs vendor {worst_vendor:.1e} / "
        f"ours {worst_ours:.1e}   {note}",
        flush=True,
    )
    return name, share, vendor_us, ours_us, ok, breakdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all 13 shapes (final measurement); default is DSV4 only",
    )
    parser.add_argument("--only", help="run only cases whose name contains this")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-perf", action="store_true")
    args = parser.parse_args()

    cases = CASES if (args.all or args.only) else [DSV4]
    if args.only:
        cases = [c for c in cases if args.only in c[0]]

    rows = [run_case(case, args) for case in cases]

    passed = sum(1 for row in rows if row[4])
    print(f"\n{passed}/{len(rows)} cases within atol={ATOL} rtol={RTOL}")

    weight = sum(r[1] for r in rows if r[1] and r[2] and r[3])
    if weight:
        total = sum(r[1] * (r[2] / r[3]) for r in rows if r[1] and r[2] and r[3])
        print(f"Weighted ratio over {weight:.2f}% of the trace: {total / weight:.3f}x")

    if any(r[5] for r in rows):
        print("\n--- per-kernel breakdown (us) ---")
        for name, _, _, _, _, breakdown in rows:
            if not breakdown:
                continue
            parts = "  ".join(
                f"{k.replace('_hc_', '').replace('_kernel', '')}={v:.2f}"
                for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])
            )
            print(f"  {name:<12} {parts}")


if __name__ == "__main__":
    main()
